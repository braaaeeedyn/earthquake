"""Transformer sub-model (MVP §4): temporal structure across the 27 days.

Each of the 27 days is a token whose features are the flattened (stations x 24 hours)
field. A learned positional embedding + a small Transformer encoder model day-to-day
dependencies; a mean over days yields the embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(self, n_stations: int, n_features: int = 1, hours: int = 24, days: int = 27,
                 d_model: int = 64, nhead: int = 4, layers: int = 2, out_dim: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(n_stations * n_features * hours, d_model)
        self.pos = nn.Parameter(torch.zeros(1, days, d_model))
        encoder = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder, layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:  # (B, S, 24, 27) -> add feature axis
            x = x.unsqueeze(2)
        b, s, f, h, d = x.shape
        z = x.permute(0, 4, 1, 2, 3).reshape(b, d, s * f * h)  # (B, days, S*F*hours)
        z = self.proj(z) + self.pos
        z = self.encoder(z)
        return torch.relu(self.head(self.drop(z.mean(dim=1))))  # pool over days -> (B, out_dim)
