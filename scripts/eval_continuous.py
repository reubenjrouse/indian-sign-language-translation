"""Continuous-signing benchmark: how well are signs cut out of a stream and recognised?

INCLUDE only has isolated signs, so sentences are made by joining held-out clips
(2-4 signs each) end to end at the app's processing rate (15 frames/s):

* natural  - clips as recorded, so each sign starts and ends with the signer's arms at rest
* no-rest  - the resting frames at the joins are cut, so the hands go straight from one
             sign to the next (much closer to fluent signing, and much harder)

Each segmenter's segments go through the same classifier; a sign is kept at 50%
confidence or more and when it differs from the previous one, as in the app. Scored
with word error rate (WER: substitutions + deletions + insertions over reference signs)
and the share of sentences recognised exactly.

Joining isolated clips is only a proxy for continuous signing: real signers blend one
sign into the next (coarticulation), which these sentences do not have.

    python scripts/cache_landmarks.py --config configs/isl60.yaml --out C:/isl-data/raw/isl60
    python scripts/eval_continuous.py --raw C:/isl-data/raw/isl60 --split test
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import ROOT, load_config  # noqa: E402
from isl.features import frame_features, resample_sequence  # noqa: E402
from isl.model import load_checkpoint  # noqa: E402
from isl.segmenter import (HandSegmenter, HandSegmenterConfig, MotionSegmenter,  # noqa: E402
                           SegmenterConfig, wrist_signals)

FPS = 15.0
MIN_CONFIDENCE = 0.5


def load_clips(raw: Path, split: str):
    clips = []
    for r in csv.DictReader(open(raw / "index.csv", encoding="utf-8")):
        if r["split"] != split:
            continue
        d = np.load(raw / r["file"])
        step = float(d["fps"]) / FPS
        idx = np.arange(0, len(d["pose"]), step).astype(int)
        clips.append({"label": r["label"], "pose": d["pose"][idx], "left": d["left"][idx], "right": d["right"][idx]})
    return clips


def rest_mask(pose, raise_off):
    """True where the higher wrist is down at rest."""
    out = []
    for p in pose:
        h, _, _ = wrist_signals(p)
        out.append(h is not None and h > raise_off)
    return np.array(out)


def trim_rest(clip, raise_off, keep=2):
    """Cut the resting frames at both ends of a clip, keeping `keep` frames of each."""
    rest = rest_mask(clip["pose"], raise_off)
    moving = np.flatnonzero(~rest)
    if not len(moving):
        return clip
    a, b = max(0, moving[0] - keep), min(len(rest), moving[-1] + 1 + keep)
    return {k: (v[a:b] if isinstance(v, np.ndarray) else v) for k, v in clip.items()}


def make_sentences(clips, n, seed, condition, raise_off):
    rng = np.random.default_rng(seed)
    sentences = []
    for _ in range(n):
        k = int(rng.integers(2, 5))
        picked = []
        while len(picked) < k:
            c = clips[int(rng.integers(len(clips)))]
            if not picked or picked[-1]["label"] != c["label"]:
                picked.append(trim_rest(c, raise_off) if condition == "no-rest" else c)
        sentences.append({
            "labels": [c["label"] for c in picked],
            "pose": np.concatenate([c["pose"] for c in picked]),
            "left": np.concatenate([c["left"] for c in picked]),
            "right": np.concatenate([c["right"] for c in picked]),
        })
    return sentences


def nan_to_none(a):
    return None if np.isnan(a).any() else a


def frame_stream(s):
    for pose, left, right in zip(s["pose"], s["left"], s["right"]):
        pose, left, right = nan_to_none(pose), nan_to_none(left), nan_to_none(right)
        yield frame_features(pose, left, right), pose


def segments(sentence, method):
    seg = method()
    out = []
    for feats, pose in frame_stream(sentence):
        s = seg.push(feats, pose) if isinstance(seg, HandSegmenter) else seg.push(feats)
        if s is not None:
            out.append(s)
    last = seg.flush()
    if last is not None:
        out.append(last)
    return out


def recognise(model, labels, segs, seq_len):
    if not segs:
        return []
    x = torch.from_numpy(np.stack([resample_sequence(s, seq_len) for s in segs]))
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=1).numpy()
    words = []
    for p in probs:
        i = int(p.argmax())
        if p[i] >= MIN_CONFIDENCE and (not words or words[-1] != labels[i]):
            words.append(labels[i])
    return words


def edit_distance(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return int(d[-1])


# Settings for both tuned on validation sentences (grid search, mean WER of both conditions).
METHODS = {
    "motion (2024 app)": lambda: MotionSegmenter(SegmenterConfig(fps=FPS)),
    "motion, tuned": lambda: MotionSegmenter(SegmenterConfig(fps=FPS, start_threshold=0.03,
                                                              static_threshold=0.03, static_frames=4)),
    "hand position (new)": lambda: HandSegmenter(HandSegmenterConfig(fps=FPS)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="output folder of scripts/cache_landmarks.py")
    ap.add_argument("--config", default=str(ROOT / "configs" / "isl60.yaml"))
    ap.add_argument("--checkpoint", default=str(ROOT / "models" / "isl60" / "best.pt"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--sentences", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write results JSON here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, ckpt = load_checkpoint(args.checkpoint, cfg.model)
    labels = ckpt.get("labels") or cfg.labels
    clips = load_clips(Path(args.raw), args.split)
    raise_off = HandSegmenterConfig().raise_off
    print(f"{len(clips)} {args.split} clips -> {args.sentences} sentences of 2-4 signs per condition")

    results = {}
    for condition in ("natural", "no-rest"):
        sentences = make_sentences(clips, args.sentences, args.seed, condition, raise_off)
        n_ref = sum(len(s["labels"]) for s in sentences)
        for name, method in METHODS.items():
            errors = exact = n_hyp = 0
            for s in sentences:
                hyp = recognise(model, labels, segments(s, method), cfg.seq_len)
                errors += edit_distance(s["labels"], hyp)
                exact += hyp == s["labels"]
                n_hyp += len(hyp)
            r = {"wer": errors / n_ref, "exact": exact / len(sentences), "signs_per_ref": n_hyp / n_ref}
            results[f"{condition} / {name}"] = r
            print(f"{condition:8s} {name:22s} WER {r['wer']:6.1%}   exact sentences {r['exact']:6.1%}   "
                  f"signs found per real sign {r['signs_per_ref']:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"split": args.split, "sentences": args.sentences,
                                              "seed": args.seed, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
