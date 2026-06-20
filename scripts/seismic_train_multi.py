"""Phase 2a, Steps 2-3: multi-station CNN+GNN+Transformer for network magnitude.

Per-station 3-component CNN -> GNN over the station graph -> Transformer over station tokens
-> masked pool -> magnitude. Amplitudes are response-removed (physical m/s), so magnitude is
actually learnable here (unlike Phase 1). Must beat the amp+dist network baseline; the
single-station ablation shows the GNN/multi-station information adds value.

  python scripts/seismic_train_multi.py --overfit
  python scripts/seismic_train_multi.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.linear_model import LinearRegression  # noqa: E402

from eq.seismic import haversine_km  # noqa: E402

NPZ = Path(__file__).resolve().parents[1] / "data" / "processed" / "seismic_phase2a.npz"


def adjacency(coords, connect_km=150.0, sigma_km=50.0):
    lat, lon = coords[:, 0], coords[:, 1]
    S = len(coords); d = np.zeros((S, S))
    for i in range(S):
        d[i] = haversine_km(lat[i], lon[i], lat, lon)
    A = np.exp(-d ** 2 / (2 * sigma_km ** 2)); A[d > connect_km] = 0.0
    np.fill_diagonal(A, 1.0)
    deg = A.sum(1); dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    return (dinv[:, None] * A * dinv[None, :]).astype(np.float32)


class WaveCNN3(nn.Module):
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
        super().__init__(); self.lin = nn.Linear(din, dout)

    def forward(self, H, Ahat):
        return torch.relu(self.lin(torch.einsum("st,btd->bsd", Ahat, H)))


class MultiStationModel(nn.Module):
    def __init__(self, Ahat, dim=64, hybrid=False, n_aux=4):
        super().__init__()
        self.register_buffer("Ahat", torch.tensor(Ahat))
        self.hybrid = hybrid
        self.cnn = WaveCNN3(dim)
        self.g1 = GCN(dim + 1, dim); self.g2 = GCN(dim, dim)
        enc = nn.TransformerEncoderLayer(dim, 4, 128, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, 1)
        self.head = nn.Sequential(nn.Linear(dim + (n_aux if hybrid else 0), 32),
                                  nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, mask, logdist, aux=None):
        B, S = x.shape[:2]
        e = self.cnn(x.reshape(B * S, 3, -1)).reshape(B, S, -1)
        feat = torch.cat([e, logdist.unsqueeze(-1)], -1) * mask.unsqueeze(-1).float()
        h = self.g2(self.g1(feat, self.Ahat), self.Ahat)
        pad = ~mask
        h = self.tr(h, src_key_padding_mask=pad).masked_fill(pad.unsqueeze(-1), 0.0)
        pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        if self.hybrid:
            pooled = torch.cat([pooled, aux], -1)
        return self.head(pooled).squeeze(-1)


def prep(d):
    X = d["X"].astype(np.float32); mask = d["mask"]
    scale = X[mask].std() + 1e-12                      # global amplitude scale (preserve relative amp)
    Xn = X / scale
    logdist = np.where(d["dist"] > 0, np.log10(np.maximum(d["dist"], 1.0)), 0.0).astype(np.float32)
    return Xn, mask, logdist


def baseline_features(d):
    """Network amp+dist features per event: mean/max log peak velocity + mean log distance."""
    X, mask, dist = d["X"], d["mask"], d["dist"]
    feats = []
    for i in range(len(X)):
        m = mask[i]
        peak = np.log10(np.abs(X[i][m]).reshape(m.sum(), -1).max(1) + 1e-12)
        ld = np.log10(np.maximum(dist[i][m], 1.0))
        feats.append([peak.mean(), peak.max(), ld.mean(), ld.min()])
    return np.asarray(feats, np.float32)


def split_chrono(wtime, fr=(0.7, 0.15)):
    o = np.argsort(wtime); n = len(o)
    return o[:int(n*fr[0])], o[int(n*fr[0]):int(n*(fr[0]+fr[1]))], o[int(n*(fr[0]+fr[1])):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--npz", default=str(NPZ))
    args = ap.parse_args()
    torch.manual_seed(0)
    d = dict(np.load(args.npz, allow_pickle=True))
    Xn, mask, logdist = prep(d)
    mag = d["mag"].astype(np.float32)
    Ahat = adjacency(d["coords"])
    model = MultiStationModel(Ahat, hybrid=args.hybrid)
    af = baseline_features(d)                       # per-event amp+dist physics features

    tr0, va0, te0 = split_chrono(d["wtime"])
    am, asd = af[tr0].mean(0), af[tr0].std(0) + 1e-6
    afn = ((af - am) / asd).astype(np.float32)

    if args.overfit:
        idx = np.arange(min(16, len(mag)))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X, M, L, A, Y = (torch.tensor(Xn[idx]), torch.tensor(mask[idx]), torch.tensor(logdist[idx]),
                         torch.tensor(afn[idx]), torch.tensor(mag[idx]))
        for _ in range(200):
            model.train(); opt.zero_grad()
            loss = nn.functional.mse_loss(model(X, M, L, A), Y)
            loss.backward(); opt.step()
        print(f"[overfit] final MSE={loss.item():.4f} (should -> ~0, model can learn magnitude)")
        return

    tr, va, te = tr0, va0, te0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Mtr, Ltr, Atr, Ytr = (torch.tensor(Xn[tr]), torch.tensor(mask[tr]),
                               torch.tensor(logdist[tr]), torch.tensor(afn[tr]), torch.tensor(mag[tr]))
    for ep in range(args.epochs):
        model.train()
        for b in torch.randperm(len(tr)).split(32):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr[b], Mtr[b], Ltr[b], Atr[b]), Ytr[b])
            loss.backward(); opt.step()

    model.eval()
    yte = mag[te]
    with torch.no_grad():
        pred = model(torch.tensor(Xn[te]), torch.tensor(mask[te]),
                     torch.tensor(logdist[te]), torch.tensor(afn[te])).numpy()
        m_near = np.zeros_like(mask[te])
        for i, gi in enumerate(te):
            recorded = np.where(mask[gi])[0]
            m_near[i, recorded[np.argmin(d["dist"][gi][recorded])]] = True
        pred_near = model(torch.tensor(Xn[te]), torch.tensor(m_near),
                          torch.tensor(logdist[te]), torch.tensor(afn[te])).numpy()

    def stats(p):
        mae = np.mean(np.abs(p - yte))
        r2 = 1 - np.sum((p - yte) ** 2) / np.sum((yte - yte.mean()) ** 2)
        return mae, r2

    bf = baseline_features(d)
    lr = LinearRegression().fit(bf[tr], mag[tr])
    mae_b, r2_b = stats(lr.predict(bf[te]))
    mae_m, r2_m = stats(pred)
    mae_n, r2_n = stats(pred_near)
    tag = "hybrid: waveform + amp/dist" if args.hybrid else "pure waveform"
    print(f"\n=== MAGNITUDE (test n={len(te)}, mag {yte.min():.1f}..{yte.max():.1f})  [{tag}] ===")
    print(f"  CNN+GNN+Transformer (all stations): MAE={mae_m:.3f}  R2={r2_m:+.3f}")
    print(f"  same model, nearest 1 station only: MAE={mae_n:.3f}  R2={r2_n:+.3f}   (ablation)")
    print(f"  amp+dist network baseline:          MAE={mae_b:.3f}  R2={r2_b:+.3f}")


if __name__ == "__main__":
    main()
