"""Phase 1, Steps 2-3: waveform CNN+Transformer for detection + magnitude, vs classic baselines.

Backbone: 1-D CNN over the trace -> Transformer encoder over the conv-feature sequence.
Two heads: detection (binary) and magnitude (regression, event windows only, with log-amplitude
and log-distance as auxiliary scalars). Chronological split (avoids aftershock leakage).
Baselines it must beat: STA/LTA (detection) and amplitude+distance linear regression (magnitude).

  python scripts/seismic_train.py --overfit     # Step-2 sanity: memorize a tiny batch
  python scripts/seismic_train.py               # full train + eval vs baselines
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import matthews_corrcoef, roc_auc_score  # noqa: E402

NPZ = Path(__file__).resolve().parents[1] / "data" / "processed" / "seismic_phase1.npz"
SR = 100.0


class WaveBackbone(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
        )
        enc = nn.TransformerEncoderLayer(64, nhead=4, dim_feedforward=128,
                                         batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):                      # x: (B, NPTS)
        h = self.conv(x.unsqueeze(1)).transpose(1, 2)   # (B, L, 64)
        h = self.tr(h).mean(dim=1)                       # (B, 64)
        return torch.relu(self.proj(h))


class SeisModel(nn.Module):
    def __init__(self, n_aux=2, dim=64):
        super().__init__()
        self.backbone = WaveBackbone(dim)
        self.det = nn.Linear(dim, 1)
        self.mag = nn.Sequential(nn.Linear(dim + n_aux, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, aux):
        e = self.backbone(x)
        return self.det(e).squeeze(-1), self.mag(torch.cat([e, aux], -1)).squeeze(-1)


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


def prep(d):
    w = d["waves"].astype(np.float32)
    w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-6)   # per-window norm
    logdist = np.where(d["dist"] > 0, np.log10(np.maximum(d["dist"], 1.0)), 0.0)
    aux = np.stack([d["logamp"], logdist], axis=1).astype(np.float32)
    return w, aux


def split_chrono(wtime, fracs=(0.7, 0.15)):
    order = np.argsort(wtime)
    n = len(order); i1 = int(n * fracs[0]); i2 = int(n * (fracs[0] + fracs[1]))
    return order[:i1], order[i1:i2], order[i2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--npz", default=str(NPZ))
    args = ap.parse_args()
    torch.manual_seed(0)
    d = dict(np.load(args.npz, allow_pickle=True))
    w, aux = prep(d)
    ydet, mag = d["ydet"].astype(np.float32), d["mag"].astype(np.float32)

    if args.overfit:                            # Step-2 sanity: memorize 32 samples
        idx = np.arange(32)
        model = SeisModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X, A = torch.tensor(w[idx]), torch.tensor(aux[idx])
        yd, ym = torch.tensor(ydet[idx]), torch.tensor(mag[idx])
        ev = yd == 1
        for e in range(200):
            model.train(); opt.zero_grad()
            pd_, pm = model(X, A)
            loss = nn.functional.binary_cross_entropy_with_logits(pd_, yd)
            if ev.any():
                loss = loss + nn.functional.mse_loss(pm[ev], ym[ev])
            loss.backward(); opt.step()
        acc = ((torch.sigmoid(pd_) > 0.5).float() == yd).float().mean().item()
        print(f"[overfit] final loss={loss.item():.4f}  train det-acc={acc:.3f} "
              f"(should reach ~1.0 -> model can learn)")
        return

    # standardize aux on train
    tr, va, te = split_chrono(d["wtime"])
    amean, astd = aux[tr].mean(0), aux[tr].std(0) + 1e-6
    auxn = (aux - amean) / astd

    model = SeisModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Atr = torch.tensor(w[tr]), torch.tensor(auxn[tr])
    ydtr, ymtr = torch.tensor(ydet[tr]), torch.tensor(mag[tr])
    evtr = ydtr == 1
    bs = 64
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(tr))
        for b in perm.split(bs):
            opt.zero_grad()
            pdd, pmm = model(Xtr[b], Atr[b])
            loss = nn.functional.binary_cross_entropy_with_logits(pdd, ydtr[b])
            m = ydtr[b] == 1
            if m.any():
                loss = loss + nn.functional.mse_loss(pmm[m], ymtr[b][m])
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pdt, pmt = model(torch.tensor(w[te]), torch.tensor(auxn[te]))
        pdt = torch.sigmoid(pdt).numpy(); pmt = pmt.numpy()
        pdv, _ = model(torch.tensor(w[va]), torch.tensor(auxn[va]))
        pdv = torch.sigmoid(pdv).numpy()

    # ---- detection: model vs STA/LTA ----
    yte = ydet[te]
    auc_m = roc_auc_score(yte, pdt)
    thr = max(np.quantile(pdv, np.linspace(0.3, 0.9, 30)),
              key=lambda t: matthews_corrcoef(ydet[va], pdv >= t))
    mcc_m = matthews_corrcoef(yte, pdt >= thr)
    slt = sta_lta_scores(d["waves"])
    auc_s = roc_auc_score(ydet, slt)            # whole-set AUC (threshold-free baseline)
    print(f"\n=== DETECTION (test n={len(te)}, {int(yte.sum())} event) ===")
    print(f"  CNN+Transformer : AUC={auc_m:.3f}  MCC={mcc_m:+.3f}")
    print(f"  STA/LTA baseline: AUC={auc_s:.3f}")

    # ---- magnitude: model vs amplitude+distance regression ----
    evte = (ydet[te] == 1)
    ym = mag[te][evte]
    mae_m = np.mean(np.abs(pmt[evte] - ym))
    evtr_mask = ydet[tr] == 1
    lr = LinearRegression().fit(aux[tr][evtr_mask], mag[tr][evtr_mask])
    base = lr.predict(aux[te][evte])
    mae_b = np.mean(np.abs(base - ym))
    sst = np.sum((ym - ym.mean()) ** 2)
    r2_m = 1 - np.sum((pmt[evte] - ym) ** 2) / sst
    r2_b = 1 - np.sum((base - ym) ** 2) / sst
    print(f"\n=== MAGNITUDE (test events n={int(evte.sum())}) ===")
    print(f"  CNN+Transformer : MAE={mae_m:.3f}  R2={r2_m:+.3f}")
    print(f"  amp+dist baseline: MAE={mae_b:.3f}  R2={r2_b:+.3f}")


if __name__ == "__main__":
    main()
