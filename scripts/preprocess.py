"""Videos -> per-frame landmark features (+ augmented copies for the train split).

For every video of every sign in the config, runs MediaPipe hands + pose and
saves a (T, 108) float32 array. Train-split videos also get augmented copies
(see isl/augment.py) until each sign has ~``augment.target_per_class``
samples. Splitting happens *before* augmentation, so no augmented copy of a
validation/test video ever lands in training.

Writes ``<processed_dir>/manifest.csv`` and is resumable (existing .npy files
are skipped).

    python scripts/preprocess.py --config configs/isl24.yaml --workers 4
"""

import argparse
import csv
import random
import re
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.augment import RECIPES, VideoAugmentation, copies_needed  # noqa: E402
from isl.config import load_config  # noqa: E402
from isl.data import MANIFEST_FIELDS  # noqa: E402

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}


def normalize(path: str) -> str:
    """Case-, slash- and underscore-insensitive path key ("Days_and_Time" == "Days and Time")."""
    parts = str(path).replace("\\", "/").replace("_", " ").split("/")
    return "/".join(" ".join(p.lower().split()) for p in parts if p.strip())


VIDEO_PATH = re.compile(r".+?\.(?:mov|mp4|avi|mkv)", re.IGNORECASE)


def official_splits(include_root: Path, lists: str = "include") -> dict:
    """normalized 'Category/NN. Sign/[Extra/]file.MOV' -> train|val|test.

    lists: "include" (all 263 signs) or "include50". A few lines in the published
    files are missing their newline ("...a.MOVCategory/..."), so paths are split on
    the video extension rather than on lines.
    """
    splits = {}
    for split in ("train", "val", "test"):
        path = include_root / "splits" / f"{lists}_{split}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/download_include.py first")
        for line in path.read_text(encoding="utf-8").splitlines():
            for video in VIDEO_PATH.findall(line.strip()):
                splits[normalize(video)] = split
    return splits


def random_splits(videos, val_frac: float, test_frac: float, seed: int) -> dict:
    """Stratified per-sign split of original videos."""
    by_label = defaultdict(list)
    for label, rel, *_ in videos:
        by_label[label].append(rel)
    rng = random.Random(seed)
    splits = {}
    for rels in by_label.values():
        rels = sorted(rels)
        rng.shuffle(rels)
        n_test = round(len(rels) * test_frac)
        n_val = max(1, round(len(rels) * val_frac))
        for i, rel in enumerate(rels):
            splits[normalize(rel)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return splits


def move_train_to_val(videos, splits: dict, fraction: float, seed: int) -> dict:
    """Move a stratified fraction of each sign's train videos to val (test is untouched).

    For vocabularies where the official val split misses some signs entirely.
    """
    splits = dict(splits)
    by_label = defaultdict(list)
    for label, rel, *_ in videos:
        if splits.get(normalize(rel), "train") == "train":
            by_label[label].append(rel)
    rng = random.Random(seed)
    for rels in by_label.values():
        rels = sorted(rels)
        rng.shuffle(rels)
        for rel in rels[:max(1, round(len(rels) * fraction)) if len(rels) >= 3 else 0]:
            splits[normalize(rel)] = "val"
    return splits


def find_videos(cfg):
    """[(label, 'Category/NN. Sign/[Extra/]file.MOV', absolute path, '[Extra/]file.MOV')], incl. sub-folders."""
    videos = []
    for sign in cfg.signs:
        folder = cfg.include_root / sign.include
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS) if folder.exists() else []
        if not files:
            print(f"WARNING: no videos for '{sign.label}' in {folder}")
        for p in files:
            inner = p.relative_to(folder).as_posix()
            videos.append((sign.label, f"{sign.include}/{inner}", p, inner))
    return videos


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.strip()).strip("_").lower()


def process(task):
    """Worker: extract features for one (video, augmentation) pair."""
    from isl.landmarks import extract_video_features

    video, out_path, recipe, seed, fps = task
    transform = VideoAugmentation(recipe, seed) if recipe else None
    feats = extract_video_features(video, fps=fps, transform=transform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feats)
    return out_path, len(feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="show split/sample counts without processing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.processed_dir
    videos = find_videos(cfg)
    if not videos:
        sys.exit("No videos found. Run scripts/download_include.py first.")

    seed = int(cfg.train.get("seed", 42))
    if cfg.data.get("split", "official") == "official":
        splits = official_splits(cfg.include_root, cfg.data.get("split_lists", "include"))
    else:
        splits = random_splits(videos, cfg.data.get("val_frac", 0.2), cfg.data.get("test_frac", 0.0), seed)

    extra_val = float(cfg.data.get("val_from_train", 0))
    if extra_val > 0:
        splits = move_train_to_val(videos, splits, extra_val, seed)

    train_counts = defaultdict(int)
    for label, rel, *_ in videos:
        train_counts[label] += splits.get(normalize(rel), "train") == "train"

    augment = cfg.augment.get("enabled", True) and not args.no_augment
    target = int(cfg.augment.get("target_per_class", 100))
    fps = cfg.data.get("fps")

    rows, tasks, unlisted = [], [], 0
    for label, rel, path, inner in videos:
        split = splits.get(normalize(rel))
        if split is None:
            unlisted += 1
            split = "train"
        base = Path(slug(label)) / slug(str(Path(inner).with_suffix("")))
        variants = [None]
        if augment and split == "train":
            n = copies_needed(train_counts[label], target)
            variants += [RECIPES[k % len(RECIPES)] for k in range(n)]
        for k, recipe in enumerate(variants):
            name = base.with_name(base.name + (f"__aug{k:02d}_{'+'.join(recipe)}" if recipe else "") + ".npy")
            rows.append({"file": name.as_posix(), "label": label, "split": split,
                         "source": rel, "augmentation": "+".join(recipe) if recipe else ""})
            if not (out_dir / name).exists():
                task_seed = zlib.crc32(f"{seed}:{rel}:{k}".encode())
                tasks.append((path, out_dir / name, recipe, task_seed, fps))

    n_split = defaultdict(int)
    for r in rows:
        n_split[r["split"]] += 1
    print(f"{len(videos)} videos, {len(rows)} samples ({dict(n_split)}); {len(tasks)} to process")
    if unlisted:
        print(f"note: {unlisted} videos not in the split lists were assigned to train")
    if args.dry_run:
        per_sign = defaultdict(lambda: defaultdict(int))
        for r in rows:
            if not r["augmentation"]:
                per_sign[r["label"]][r["split"]] += 1
        for label in cfg.labels:
            c = per_sign[label]
            print(f"  {label:15s} train {c['train']:3d}  val {c['val']:3d}  test {c['test']:3d}")
        return

    failed = []
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="video"):
                try:
                    _, n_frames = fut.result()
                    if n_frames == 0:
                        failed.append(futures[fut][1])
                except Exception as e:
                    print(f"failed: {futures[fut][0]}: {e}")
                    failed.append(futures[fut][1])

    failed = {Path(p).resolve() for p in failed}
    rows = [r for r in rows if (out_dir / r["file"]).resolve() not in failed]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} samples -> {out_dir / 'manifest.csv'}" +
          (f" ({len(failed)} failed, excluded)" if failed else ""))


if __name__ == "__main__":
    main()
