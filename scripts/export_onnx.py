"""Export a checkpoint to ONNX for the browser app, and verify it against PyTorch.

Writes ``<out>.onnx`` (input ``frames`` [batch, 65, 108] -> output ``probs``
[batch, classes]) and ``<out>.json`` with labels and input shape.

    python scripts/export_onnx.py --checkpoint models/isl60/best.pt --config configs/isl60.yaml \
        --metrics models/isl60/metrics_test.json --out web/model/isl60
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import load_config  # noqa: E402
from isl.features import FEATURE_DIM  # noqa: E402
from isl.model import load_checkpoint  # noqa: E402


class ProbabilityHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames):
        logits, _ = self.model(frames)
        return torch.softmax(logits, dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--metrics", help="metrics_test.json from scripts/evaluate.py, shown in the app")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, ckpt = load_checkpoint(args.checkpoint, cfg.model)
    labels = ckpt.get("labels") or cfg.labels
    seq_len = ckpt.get("seq_len", cfg.seq_len)
    if len(labels) != model.classifier[-1].out_features:
        sys.exit(f"{len(labels)} labels vs {model.classifier[-1].out_features} model outputs")

    wrapped = ProbabilityHead(model).eval()
    example = torch.randn(1, seq_len, FEATURE_DIM)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, (example,), str(out.with_suffix(".onnx")),
        input_names=["frames"], output_names=["probs"],
        dynamic_axes={"frames": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    import onnxruntime as ort
    session = ort.InferenceSession(str(out.with_suffix(".onnx")), providers=["CPUExecutionProvider"])
    test = torch.randn(4, seq_len, FEATURE_DIM)
    with torch.no_grad():
        expected = wrapped(test).numpy()
    actual = session.run(None, {"frames": test.numpy()})[0]
    diff = float(np.abs(expected - actual).max())
    if diff > 1e-4:
        sys.exit(f"ONNX output differs from PyTorch by {diff:.2e}")

    meta = {"labels": labels, "seq_len": seq_len, "feature_dim": FEATURE_DIM,
            "source_checkpoint": Path(args.checkpoint).name, "config": cfg.name}
    if labels == cfg.labels:  # INCLUDE category per sign, for grouping the vocabulary in the app
        meta["categories"] = [s.include.split("/")[0].replace("_", " ") for s in cfg.signs]
    if args.metrics:
        m = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        meta["test"] = {k: m[k] for k in ("accuracy", "macro_f1", "samples")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    size = out.with_suffix(".onnx").stat().st_size / 1e6
    print(f"exported {out.with_suffix('.onnx')} ({size:.1f} MB), max |onnx - torch| = {diff:.1e}")


if __name__ == "__main__":
    main()
