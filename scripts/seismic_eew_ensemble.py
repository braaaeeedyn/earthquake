"""Phase 2b bolster: seed-ensemble the EEW hybrid model (averaging predictions cuts variance).

Trains K hybrid (graph) models on different seeds, averages their per-station PGV predictions,
and evaluates the ensemble vs the early-amp+dist baseline -- the honest way to firm up a noisy
small-sample result (no cherry-picked seed).

  python scripts/seismic_eew_ensemble.py --k 5
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

from seismic_eew import EEWModel, build_arrays  # noqa: E402
from seismic_train_multi import adjacency, split_chrono  # noqa: E402

NPZ = Path(__file__).resolve().parents[1] / "data" / "processed" / "seismic_phase2a.npz"


def train_predict(d, early, mask, logpgv, auxn, logdist, Ahat, tr, te, seed, epochs):
    torch.manual_seed(seed)
    model = EEWModel(Ahat, hybrid=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Mtr, Ltr, Atr, Ytr = (torch.tensor(early[tr]), torch.tensor(mask[tr]),
                               torch.tensor(logdist[tr]), torch.tensor(auxn[tr]),
                               torch.tensor(logpgv[tr]))
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(tr)).split(32):
            opt.zero_grad()
            out = model(Xtr[b], Mtr[b], Ltr[b], Atr[b])
            m = Mtr[b]
            nn.functional.mse_loss(out[m], Ytr[b][m]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(early[te]), torch.tensor(mask[te]),
                     torch.tensor(logdist[te]), torch.tensor(auxn[te])).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--npz", default=str(NPZ))
    args = ap.parse_args()
    d = dict(np.load(args.npz, allow_pickle=True))
    early, mask, logpgv, log_evpeak, logdist = build_arrays(d)
    aux = np.stack([log_evpeak, logdist], -1).astype(np.float32)
    Ahat = adjacency(d["coords"])
    tr, va, te = split_chrono(d["wtime"])
    am, asd = aux[tr][mask[tr]].mean(0), aux[tr][mask[tr]].std(0) + 1e-6
    auxn = (aux - am) / asd
    mte = mask[te]; yte = logpgv[te][mte]

    preds = [train_predict(d, early, mask, logpgv, auxn, logdist, Ahat, tr, te, s, args.epochs)
             for s in range(args.k)]
    ens = np.mean(preds, axis=0)[mte]
    singles = np.array([p[mte] for p in preds])

    def stats(p):
        return np.mean(np.abs(p - yte)), 1 - np.sum((p - yte) ** 2) / np.sum((yte - yte.mean()) ** 2)

    lr = LinearRegression().fit(aux[tr][mask[tr]], logpgv[tr][mask[tr]])
    pbase = lr.predict(aux[te][mte])
    mae_e, r2_e = stats(ens); mae_b, r2_b = stats(pbase)
    single_r2 = [stats(s)[1] for s in singles]

    print(f"\n=== EEW ENSEMBLE (k={args.k}, hybrid+graph)  n_test={mte.sum()} station-preds ===")
    print(f"  single models R2 : mean {np.mean(single_r2):.3f} (range {min(single_r2):.3f}..{max(single_r2):.3f})")
    print(f"  ENSEMBLE        R2 : {r2_e:+.3f}   MAE={mae_e:.3f}")
    print(f"  baseline        R2 : {r2_b:+.3f}   MAE={mae_b:.3f}")

    thr = np.quantile(logpgv[tr][mask[tr]], 0.70)
    yt = (yte > thr).astype(int)
    for name, p in (("ensemble", ens), ("baseline", pbase)):
        yh = (p > thr).astype(int)
        tp = int(((yh == 1) & (yt == 1)).sum()); fp = int(((yh == 1) & (yt == 0)).sum())
        fn = int(((yh == 0) & (yt == 1)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0; prec = tp / (tp + fp) if (tp + fp) else 0.0
        print(f"  alert[{name:8s}] recall={rec:.2f} precision={prec:.2f} "
              f"MCC={matthews_corrcoef(yt, yh):+.3f}")


if __name__ == "__main__":
    main()
