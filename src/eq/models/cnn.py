"""CNN sub-model (MVP §4): spatial structure within each station's 24x27 matrix.

Input (B, S, 24, 27) is treated as an S-channel 2D image; 2D convolutions learn
local hour x day patterns, global pooling yields a fixed-size embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatialCNN(nn.Module):
    def __init__(self, n_stations: int, n_features: int = 1, out_dim: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(n_stations * n_features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, 24, 27) or (B, S, F, 24, 27) -> (B, S*F, 24, 27)
        if x.ndim == 5:
            x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return self.net(x)
