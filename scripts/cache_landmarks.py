"""Cache raw (un-normalised) landmarks for the val/test videos of a config.

scripts/eval_continuous.py needs raw positions (wrists relative to shoulders), which
the training features no longer have after per-part normalisation. Writes one .npz
per video with pose (T, 12, 2), left (T, 21, 2) and right (T, 21, 2); missing parts
are NaN.

    python scripts/cache_landmarks.py --config configs/isl60.yaml --out C:/isl-data/raw/isl60 --workers 8
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import load_config  # noqa: E402

NAN_POSE = np.full((12, 2), np.nan, np.float32)
NAN_HAND = np.full((21, 2), np.nan, np.float32)


def extract(video: Path, out: Path):
    from isl.landmarks import LandmarkExtractor, iter_video_frames

    poses, lefts, rights = [], [], []
    with LandmarkExtractor() as ex:
        for ts, frame in iter_video_frames(video):
            pose, left, right = ex.detect(frame, ts)
            poses.append(NAN_POSE if pose is None else pose)
            lefts.append(NAN_HAND if left is None else left)
            rights.append(NAN_HAND if right is None else right)
    import cv2
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 25.0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pose=np.stack(poses), left=np.stack(lefts), right=np.stack(rights), fps=fps)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    rows = [r for r in csv.DictReader(open(cfg.processed_dir / "manifest.csv", encoding="utf-8"))
            if r["split"] in args.splits and not r["augmentation"]]
    index = []
    tasks = []
    for r in rows:
        dest = out / r["split"] / (Path(r["file"]).with_suffix(".npz"))
        index.append({"file": dest.relative_to(out).as_posix(), "label": r["label"], "split": r["split"], "source": r["source"]})
        if not dest.exists():
            tasks.append((cfg.include_root / r["source"], dest))
    print(f"{len(rows)} videos, {len(tasks)} to extract")
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(extract, v, d) for v, d in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), unit="video"):
            f.result()
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "split", "source"])
        w.writeheader()
        w.writerows(index)
    print(f"index -> {out / 'index.csv'}")


if __name__ == "__main__":
    main()
