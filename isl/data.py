"""Dataset over preprocessed landmark sequences.

``scripts/preprocess.py`` writes one ``.npy`` (T x 108) per video or
augmented copy, plus ``manifest.csv`` with columns:
    file, label, split, source, augmentation
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import resample_sequence

MANIFEST_FIELDS = ["file", "label", "split", "source", "augmentation"]


def read_manifest(processed_dir: Path) -> list[dict]:
    with open(Path(processed_dir) / "manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class SequenceDataset(Dataset):
    """All sequences of one split, resampled to ``seq_len`` and held in memory."""

    def __init__(self, processed_dir, labels: list[str], split: str, seq_len: int = 65,
                 include_augmented: bool = True):
        processed_dir = Path(processed_dir)
        index = {label: i for i, label in enumerate(labels)}
        rows = [r for r in read_manifest(processed_dir)
                if r["split"] == split and r["label"] in index
                and (include_augmented or not r["augmentation"])]
        if not rows:
            raise ValueError(f"no samples for split '{split}' in {processed_dir}")
        self.rows = rows
        self.x = torch.from_numpy(np.stack([
            resample_sequence(np.load(processed_dir / r["file"]), seq_len) for r in rows]))
        self.y = torch.tensor([index[r["label"]] for r in rows])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
