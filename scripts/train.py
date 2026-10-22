"""Train the BiLSTM + attention classifier on preprocessed landmark sequences.

Follows the paper's setup: Adam-family optimizer (AdamW, lr 1e-3, weight decay 0.01), cross-entropy,
batch 32, ReduceLROnPlateau (x0.5, patience 5) and early stopping (patience 15)
on validation accuracy. Saves the best checkpoint, a metrics CSV and training
curves to ``runs/<config name>/``.

    python scripts/train.py --config configs/isl24.yaml
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import ROOT, load_config  # noqa: E402
from isl.data import SequenceDataset  # noqa: E402
from isl.model import SignLanguageModel  # noqa: E402


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
    return total_loss / count, correct / count


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="validation")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="output directory (default runs/<name>)")
    ap.add_argument("--epochs", type=int, help="override train.max_epochs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg.train
    seed_everything(int(tc.get("seed", 42)))
    out = Path(args.out) if args.out else ROOT / "runs" / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    train_ds = SequenceDataset(cfg.processed_dir, cfg.labels, "train", cfg.seq_len)
    val_ds = SequenceDataset(cfg.processed_dir, cfg.labels, "val", cfg.seq_len, include_augmented=False)
    print(f"train {len(train_ds)} samples, val {len(val_ds)} samples, {len(cfg.labels)} classes, device {args.device}")

    generator = torch.Generator().manual_seed(int(tc.get("seed", 42)))
    train_loader = DataLoader(train_ds, batch_size=tc.get("batch_size", 32), shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=256)

    model_config = {**cfg.model, "num_classes": len(cfg.labels)}
    model = SignLanguageModel(**model_config).to(args.device)
    criterion = nn.CrossEntropyLoss()
    # AdamW (decoupled weight decay). With plain Adam, weight_decay=0.01 is added to the
    # gradients and swamps the LSTM's small gradients: on the 60-sign set the model never
    # left chance level (loss stuck at ln 60), while AdamW reaches >90% val in 4 epochs.
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.get("lr", 1e-3), weight_decay=tc.get("weight_decay", 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=tc.get("lr_factor", 0.5), patience=tc.get("lr_patience", 5))

    max_epochs = args.epochs or tc.get("max_epochs", 200)
    patience = tc.get("early_stop_patience", 15)
    best_acc, best_epoch, history = -1.0, 0, []
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, args.device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, args.device)
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, lr=lr))
        improved = val_acc > best_acc
        print(f"epoch {epoch:3d}  train {train_loss:.4f}/{train_acc:.3f}  "
              f"val {val_loss:.4f}/{val_acc:.3f}  lr {lr:.1e}  {time.time() - start:.1f}s" + ("  *" if improved else ""))
        if improved:
            best_acc, best_epoch = val_acc, epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "value": val_acc,
                "labels": cfg.labels,
                "model_config": cfg.model,
                "seq_len": cfg.seq_len,
                "config": cfg.name,
            }, out / "best.pt")
        elif epoch - best_epoch >= patience:
            print(f"early stopping: no improvement for {patience} epochs")
            break

    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_curves(history, out / "training_curves.png")
    (out / "summary.json").write_text(json.dumps(
        {"best_val_acc": best_acc, "best_epoch": best_epoch, "epochs_run": len(history),
         "train_samples": len(train_ds), "val_samples": len(val_ds)}, indent=2), encoding="utf-8")
    print(f"best val acc {best_acc:.4f} at epoch {best_epoch} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
