"""Classifier wrappers (MVP §4 fusion).

``Classifier`` adds a linear head to a single backbone so each sub-model can be trained
and overfit-tested in isolation. ``FusedClassifier`` concatenates the embeddings of
several backbones (late fusion) and learns a small MLP head over them — the same
backbone instances can be reused, so nothing is rebuilt going standalone -> fused.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Backbone + linear head -> per-sample logit (B,)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class FusedClassifier(nn.Module):
    """Late fusion of several backbones -> per-sample logit (B,)."""

    def __init__(self, backbones: Sequence[nn.Module], hidden: int = 32):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        total = sum(b.out_dim for b in backbones)
        self.head = nn.Sequential(
            nn.Linear(total, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([b(x) for b in self.backbones], dim=-1)
        return self.head(emb).squeeze(-1)
