"""Experiment config loading (``configs/*.yaml``)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Sign:
    label: str      # display name / class name
    include: str    # INCLUDE folder, e.g. "Greetings/48. Hello"


@dataclass
class Config:
    name: str
    signs: list[Sign]
    data: dict = field(default_factory=dict)
    augment: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.signs]

    @property
    def include_root(self) -> Path:
        return ROOT / self.data.get("include_root", "data/include")

    @property
    def processed_dir(self) -> Path:
        return ROOT / self.data.get("processed_dir", f"data/processed/{self.name}")

    @property
    def seq_len(self) -> int:
        return int(self.data.get("seq_len", 65))


def load_config(path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    signs = [Sign(**s) for s in raw.pop("signs")]
    labels = [s.label for s in signs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate labels in {path}")
    return Config(signs=signs, path=path, **raw)
