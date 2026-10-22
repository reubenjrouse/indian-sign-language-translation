"""Run the full pipeline on a video file (no browser needed).

--mode isolated   classify the whole clip as one sign (like an INCLUDE video)
--mode continuous segment the clip by motion, classify each sign, then
                  translate the gloss sequence with the LLM

    python scripts/infer_video.py clip.mp4 --mode isolated
    python scripts/infer_video.py sentence.mp4 --mode continuous --translate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import ROOT, load_config  # noqa: E402
from isl.features import resample_sequence  # noqa: E402
from isl.features import frame_features  # noqa: E402
from isl.landmarks import LandmarkExtractor, extract_video_features, iter_video_frames  # noqa: E402
from isl.model import load_checkpoint  # noqa: E402
from isl.segmenter import HandSegmenter, HandSegmenterConfig  # noqa: E402


def classify(model, segment, labels, seq_len, top_k=3):
    x = torch.from_numpy(resample_sequence(segment, seq_len)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=1)[0].numpy()
    order = np.argsort(probs)[::-1][:top_k]
    return [(labels[i], float(probs[i])) for i in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--config", default=str(ROOT / "configs" / "isl60.yaml"))
    ap.add_argument("--checkpoint", default=str(ROOT / "models" / "isl60" / "best.pt"))
    ap.add_argument("--mode", choices=["isolated", "continuous"], default="isolated")
    ap.add_argument("--fps", type=float, default=HandSegmenterConfig.fps,
                    help="processing rate for continuous mode (segmenter thresholds assume this)")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--translate", action="store_true", help="send detected glosses to the LLM")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, ckpt = load_checkpoint(args.checkpoint, cfg.model)
    labels = ckpt.get("labels") or cfg.labels

    if args.mode == "isolated":
        feats = extract_video_features(args.video)
        print(f"{len(feats)} frames")
        for label, p in classify(model, feats, labels, cfg.seq_len):
            print(f"  {label:15s} {p:.1%}")
        return

    with LandmarkExtractor() as ex:  # raw landmarks: the segmenter needs wrist positions
        frames = [ex.detect(frame, ts) for ts, frame in iter_video_frames(args.video, fps=args.fps)]
    segmenter = HandSegmenter(HandSegmenterConfig(fps=args.fps))
    words = []
    for t, raw in enumerate([*frames, None]):
        if raw is None:
            segment = segmenter.flush()  # the clip may end mid-sign
        else:
            pose, left, right = raw
            segment = segmenter.push(frame_features(pose, left, right), pose)
        if segment is None:
            continue
        top = classify(model, segment, labels, cfg.seq_len)
        label, p = top[0]
        accepted = p >= args.min_confidence and (not words or words[-1] != label)
        alts = ", ".join(f"{l} {q:.0%}" for l, q in top[1:])
        print(f"[{t / args.fps:6.2f}s] {label:15s} {p:.0%}  ({alts}){'' if accepted else '  - skipped'}")
        if accepted:
            words.append(label)

    print("glosses:", " ".join(words) or "(none)")
    if args.translate and words:
        from isl.translate import translate

        load_dotenv(ROOT / ".env")
        result = translate(words)
        print("sentence:", result["sentence"], "" if result["source"] == "llm" else f"[fallback: {result['error']}]")


if __name__ == "__main__":
    main()
