"""The two networks the live system runs, plus the classic baseline the detector is measured against.

  DetectorNet        single-station CNN -> Transformer: P(earthquake) for a 30 s vertical window
  MultiStationModel  per-station 3-C CNN -> GNN over the station graph -> Transformer -> magnitude
  adjacency          the station graph (Gaussian distance weights, symmetrically normalized)
  sta_lta_scores     max STA/LTA ratio per window (the textbook trigger, detection baseline)

Trained by scripts/demo_detect.py and scripts/demo_magnitude.py; run by src/eq/pipeline.py (live + replay).
Layer attribute names are part of the saved checkpoints' state_dict keys -- don't rename them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .locate import haversine_km

SR = 100.0


class WaveBackbone(nn.Module):
    """4 strided 1-D convolutions -> 2-layer Transformer encoder -> mean pool -> projection."""

    def __init__(self, out_dim=64):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
        )
        enc = nn.TransformerEncoderLayer(64, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):                      # x: (B, NPTS)
        h = self.conv(x.unsqueeze(1)).transpose(1, 2)   # (B, L, 64)
        h = self.tr(h).mean(dim=1)                       # (B, 64)
        return torch.relu(self.proj(h))


class DetectorNet(nn.Module):
    """Detect: the backbone with a detection head. Input = pipeline.det_prep windows, output = logits."""

    def __init__(self, dim=64):
        super().__init__()
        self.backbone = WaveBackbone(dim)
        self.det = nn.Linear(dim, 1)

    def forward(self, x):                      # x: (B, NPTS) std-normalized -> logits (B,)
        return self.det(self.backbone(x)).squeeze(-1)


def sta_lta_scores(waves, sta_s=0.5, lta_s=5.0):
    """Max STA/LTA ratio per window (classic detector score)."""
    nsta, nlta = int(sta_s * SR), int(lta_s * SR)
    out = np.zeros(len(waves))
    for i, w in enumerate(waves):
        x = w.astype(float) ** 2
        cs = np.cumsum(np.insert(x, 0, 0))
        sta = (cs[nsta:] - cs[:-nsta]) / nsta
        lta = (cs[nlta:] - cs[:-nlta]) / nlta
        m = min(len(sta), len(lta))
        ratio = sta[:m] / (lta[:m] + 1e-9)
        out[i] = np.nanmax(ratio[nlta:]) if m > nlta else np.nanmax(ratio)
    return out


def adjacency(coords, connect_km=150.0, sigma_km=50.0):
    """Station graph: Gaussian distance weights (sigma 50 km, cut off at 150 km), symmetrically normalized."""
    lat, lon = coords[:, 0], coords[:, 1]
    S = len(coords)
    d = np.zeros((S, S))
    for i in range(S):
        d[i] = haversine_km(lat[i], lon[i], lat, lon)
    A = np.exp(-d ** 2 / (2 * sigma_km ** 2))
    A[d > connect_km] = 0.0
    np.fill_diagonal(A, 1.0)
    dinv = 1.0 / np.sqrt(np.maximum(A.sum(1), 1e-9))
    return (dinv[:, None] * A * dinv[None, :]).astype(np.float32)


class WaveCNN3(nn.Module):
    """Per-station 3-component CNN embedding."""

    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 16, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        return torch.relu(self.proj(self.net(x)))


class GCN(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = nn.Linear(din, dout)

    def forward(self, H, Ahat):
        return torch.relu(self.lin(torch.einsum("st,btd->bsd", Ahat, H)))


class MultiStationModel(nn.Module):
    """Size. amp_feature=True (v2): waveforms arrive unit-peak-normalized per station and each station's
    standardized log10 peak velocity joins log-distance as a node feature, so the CNN reads SHAPE and the
    graph reads SIZE (amplitude vs distance). hybrid=True adds 4 network amp/distance statistics to the head."""

    def __init__(self, Ahat, dim=64, hybrid=False, n_aux=4, amp_feature=False):
        super().__init__()
        self.register_buffer("Ahat", torch.tensor(Ahat))
        self.hybrid = hybrid
        self.amp_feature = amp_feature
        self.cnn = WaveCNN3(dim)
        self.g1 = GCN(dim + 1 + int(amp_feature), dim)
        self.g2 = GCN(dim, dim)
        enc = nn.TransformerEncoderLayer(dim, 4, 128, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, 1)
        self.head = nn.Sequential(nn.Linear(dim + (n_aux if hybrid else 0), 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, mask, logdist, aux=None, logamp=None):
        B, S = x.shape[:2]
        e = self.cnn(x.reshape(B * S, 3, -1)).reshape(B, S, -1)
        parts = [e, logdist.unsqueeze(-1)] + ([logamp.unsqueeze(-1)] if self.amp_feature else [])
        feat = torch.cat(parts, -1) * mask.unsqueeze(-1).float()
        h = self.g2(self.g1(feat, self.Ahat), self.Ahat)
        pad = ~mask
        h = self.tr(h, src_key_padding_mask=pad).masked_fill(pad.unsqueeze(-1), 0.0)
        pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        if self.hybrid:
            pooled = torch.cat([pooled, aux], -1)
        return self.head(pooled).squeeze(-1)
