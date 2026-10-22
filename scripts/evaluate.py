"""Evaluate a checkpoint on a split: accuracy, macro F1, per-class report, confusion matrix.

Only original (non-augmented) videos are evaluated.

    python scripts/evaluate.py --config configs/isl24.yaml --checkpoint runs/isl24/best.pt --split test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import load_config  # noqa: E402
from isl.data import SequenceDataset  # noqa: E402
from isl.model import load_checkpoint  # noqa: E402


def plot_confusion(cm, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = max(6, 0.4 * len(labels))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", help="output directory (default: next to the checkpoint)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, ckpt = load_checkpoint(args.checkpoint, cfg.model)
    labels = ckpt.get("labels") or cfg.labels
    if len(labels) != model.classifier[-1].out_features:
        sys.exit(f"config has {len(labels)} labels but checkpoint has {model.classifier[-1].out_features} classes")

    ds = SequenceDataset(cfg.processed_dir, labels, args.split, cfg.seq_len, include_augmented=False)
    with torch.no_grad():
        logits, _ = model(ds.x)
    y_true, y_pred = ds.y.numpy(), logits.argmax(1).numpy()

    metrics = {
        "split": args.split,
        "samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=range(len(labels)), zero_division=0),
        "checkpoint": str(args.checkpoint),
    }
    report = classification_report(y_true, y_pred, labels=range(len(labels)), target_names=labels, zero_division=0)
    print(report)
    print(f"{args.split}: accuracy {metrics['accuracy']:.4f}, macro F1 {metrics['macro_f1']:.4f} ({len(y_true)} videos)")

    out = Path(args.out) if args.out else Path(args.checkpoint).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / f"report_{args.split}.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plot_confusion(cm, labels, out / f"confusion_{args.split}.png",
                   f"{cfg.name} - {args.split} (acc {metrics['accuracy']:.1%})")
    print(f"saved metrics and confusion matrix -> {out}")


if __name__ == "__main__":
    main()
