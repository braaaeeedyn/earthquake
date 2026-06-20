"""Bridge from the Phase-1 dataset (data/processed/dataset.npz) to torch tensors.

Also exposes the station adjacency (for the GNN) and the train-set ``pos_weight``
(for imbalance-aware BCE). All splits stay chronological — this module never reshuffles
across the train/val/test boundary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import Config
from ..graph import build_station_graph, normalized_adjacency

DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "dataset.npz"


def _split(npz, prefix: str):
    x = torch.tensor(npz[f"X{prefix}"], dtype=torch.float32)
    y = torch.tensor(npz[f"y{prefix}"], dtype=torch.float32)
    return x, y


def load_dataset(path: Path = DATASET_PATH) -> dict:
    """Return ``{"train": (X, y), "val": (X, y), "test": (X, y)}`` as torch tensors."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/build_dataset.py` first."
        )
    npz = np.load(path)
    return {"train": _split(npz, "tr"), "val": _split(npz, "va"), "test": _split(npz, "te")}


def make_loaders(data: dict, batch_size: int = 32) -> dict:
    """DataLoaders for each split. Only the train loader is shuffled (within-split only)."""
    loaders = {}
    for name, (x, y) in data.items():
        loaders[name] = DataLoader(
            TensorDataset(x, y), batch_size=batch_size, shuffle=(name == "train")
        )
    return loaders


def adjacency_tensor(cfg: Config | None = None) -> torch.Tensor:
    """Symmetric-normalized station adjacency (with self-loops) as a float tensor."""
    cfg = cfg or Config()
    a = normalized_adjacency(build_station_graph(cfg.stations, cfg))
    return torch.tensor(a, dtype=torch.float32)


def pos_weight(y: torch.Tensor) -> torch.Tensor:
    """neg/pos ratio for BCEWithLogitsLoss — upweights the rare positive class."""
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return torch.tensor([neg / pos if pos > 0 else 1.0])
