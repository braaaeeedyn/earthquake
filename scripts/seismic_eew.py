"""Phase 2b: Earthquake Early Warning -- predict future peak ground velocity (PGV) from early P.

Input: first T_EARLY samples of each station's 3C record (5s pre-P + 3s early P-wave).
Target: log10(PGV) = peak horizontal velocity over the FUTURE part of the trace (after T_EARLY),
so the peak shaking is never in the input -> a genuine seconds-ahead prediction.

Per-station node regression: CNN(early P) -> GNN over the network -> Transformer -> per-station PGV.
Must beat the classic EEW baseline (early-P peak amplitude + distance -> PGV). Flags isolate and
then bolster the deep model's edge:
  --no-graph : identity adjacency (kills the GNN) -> shows the network's contribution
  --hybrid   : feed the baseline's amp+dist scalars to each node head -> deep ON TOP of the physics

  python scripts/seismic_eew.py                 # full graph, pure waveform
  python scripts/seismic_eew.py --hybrid        # bolstered
  python scripts/seismic_eew.py --no-graph      # ablation
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402

from seismic_train_multi import GCN, WaveCNN3, adjacency, split_chrono  # noqa: E402

NPZ = Path(__file__).resolve().parents[1] / "data" / "processed" / "seismic_phase2a.npz"
SR = 100.0
T_EARLY = int(8.0 * SR)          # 8 s of input (5 s pre-P + ~3 s early P)


class EEWModel(nn.Module):
    """Per-station PGV regression from early-P, with optional graph + hybrid amp/dist scalars."""

    def __init__(self, Ahat, dim=64, hybrid=False):
        super().__init__()
        self.register_buffer("Ahat", torch.tensor(Ahat))
        self.hybrid = hybrid
        self.cnn = WaveCNN3(dim)
        self.g1 = GCN(dim + 1, dim); self.g2 = GCN(dim, dim)
        enc = nn.TransformerEncoderLayer(dim, 4, 128, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, 1)
        hin = dim + (2 if hybrid else 0)
        self.head = nn.Sequential(nn.Linear(hin, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, mask, logdist, aux=None):
        B, S = x.shape[:2]
        e = self.cnn(x.reshape(B * S, 3, -1)).reshape(B, S, -1)
        feat = torch.cat([e, logdist.unsqueeze(-1)], -1) * mask.unsqueeze(-1).float()
        h = self.g2(self.g1(feat, self.Ahat), self.Ahat)
        h = self.tr(h, src_key_padding_mask=~mask).masked_fill((~mask).unsqueeze(-1), 0.0)
        if self.hybrid:
            h = torch.cat([h, aux], -1)
        return self.head(h).squeeze(-1)          # (B, S) per-station log-PGV


def build_arrays(d):
    X, mask, dist = d["X"].astype(np.float32), d["mask"], d["dist"]
    early = X[..., :T_EARLY]
    future = X[..., T_EARLY:]
    horiz_fut = np.sqrt(future[:, :, 1] ** 2 + future[:, :, 2] ** 2)
    logpgv = np.log10(horiz_fut.max(-1) + 1e-9).astype(np.float32)          # target (N,S)
    early_peak = np.sqrt(early[:, :, 1] ** 2 + early[:, :, 2] ** 2).max(-1)
    log_evpeak = np.log10(early_peak + 1e-9).astype(np.float32)
    logdist = np.where(dist > 0, np.log10(np.maximum(dist, 1.0)), 0.0).astype(np.float32)
    scale = early[mask].std() + 1e-12
    return early / scale, mask, logpgv, log_evpeak, logdist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npz", default=str(NPZ))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = dict(np.load(args.npz, allow_pickle=True))
    early, mask, logpgv, log_evpeak, logdist = build_arrays(d)
    aux = np.stack([log_evpeak, logdist], -1).astype(np.float32)            # (N,S,2)
    Ahat = np.eye(len(d["coords"]), dtype=np.float32) if args.no_graph else adjacency(d["coords"])

    tr, va, te = split_chrono(d["wtime"])
    # standardize hybrid aux on train (recorded stations only)
    am, asd = aux[tr][mask[tr]].mean(0), aux[tr][mask[tr]].std(0) + 1e-6
    auxn = (aux - am) / asd

    model = EEWModel(Ahat, hybrid=args.hybrid)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Mtr, Ltr, Atr, Ytr = (torch.tensor(early[tr]), torch.tensor(mask[tr]),
                               torch.tensor(logdist[tr]), torch.tensor(auxn[tr]),
                               torch.tensor(logpgv[tr]))
    for ep in range(args.epochs):
        model.train()
        for b in torch.randperm(len(tr)).split(32):
            opt.zero_grad()
            out = model(Xtr[b], Mtr[b], Ltr[b], Atr[b])
            m = Mtr[b]
            loss = nn.functional.mse_loss(out[m], Ytr[b][m])
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(early[te]), torch.tensor(mask[te]),
                     torch.tensor(logdist[te]), torch.tensor(auxn[te])).numpy()
    mte = mask[te]
    yte = logpgv[te][mte]; pte = pred[mte]

    # baseline: early-P peak amplitude + distance -> PGV (per recorded station)
    feat_tr = aux[tr][mask[tr]]; feat_te = aux[te][mte]
    lr = LinearRegression().fit(feat_tr, logpgv[tr][mask[tr]])
    pbase = lr.predict(feat_te)

    def stats(p):
        mae = np.mean(np.abs(p - yte))
        r2 = 1 - np.sum((p - yte) ** 2) / np.sum((yte - yte.mean()) ** 2)
        return mae, r2

    mae_m, r2_m = stats(pte); mae_b, r2_b = stats(pbase)
    tag = ("hybrid" if args.hybrid else "pure") + (" / NO-GRAPH" if args.no_graph else "")
    print(f"\n=== EEW: predict future log10(PGV) from first {T_EARLY/SR:.0f}s  [{tag}] ===")
    print(f"  test: {len(te)} events, {mte.sum()} station-predictions, PGV log10 range "
          f"{yte.min():.1f}..{yte.max():.1f}")
    print(f"  CNN+GNN+Transformer : MAE={mae_m:.3f}  R2={r2_m:+.3f}")
    print(f"  early-amp+dist base : MAE={mae_b:.3f}  R2={r2_b:+.3f}")

    # operational view: alert if predicted PGV exceeds a 'notable shaking' threshold
    thr = np.quantile(logpgv[tr][mask[tr]], 0.70)
    ytrue = (yte > thr).astype(int)
    for name, p in (("deep", pte), ("baseline", pbase)):
        yh = (p > thr).astype(int)
        tp = int(((yh == 1) & (ytrue == 1)).sum()); fp = int(((yh == 1) & (ytrue == 0)).sum())
        fn = int(((yh == 0) & (ytrue == 1)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        mcc = matthews_corrcoef(ytrue, yh) if len(set(yh)) > 1 else 0.0
        print(f"  alert[{name:8s}] recall={rec:.2f} precision={prec:.2f} MCC={mcc:+.3f}")


if __name__ == "__main__":
    main()
