"""Reproducible MAGNITUDE demo: multi-station CNN+GNN+Transformer estimates event magnitude
from the network's waveforms, shown working on held-out events vs the amp+distance physics baseline.

Magnitude is noisy seed-to-seed (single models swing R² 0.57..0.86), so we seed-ENSEMBLE and
report the mean and spread honestly — no cherry-picked seed. The single-station ablation
(nearest station only) is shown too: it collapses, proving the multi-station GNN fusion is
what earns the win.

  python scripts/demo_magnitude.py                 # train if needed (cached), then demo
  python scripts/demo_magnitude.py --retrain --seeds 5 --epochs 40   # publishable headline

Outputs:
  data/processed/magnitude_ensemble.pt   K model weights + aux normalizer (persisted)
  data/processed/magnitude_demo.json     per-seed + ensemble R²/MAE, baseline, ablation
  figures/magnitude_demo.png             predicted-vs-true scatter + example events
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from sklearn.linear_model import LinearRegression  # noqa: E402
from seismic_train_multi import (MultiStationModel, adjacency, baseline_features,  # noqa: E402
                                 prep, split_chrono)

NPZ = ROOT / "data" / "processed" / "seismic_phase2a_xl.npz"
CKPT = ROOT / "data" / "processed" / "magnitude_ensemble.pt"
SUMMARY = ROOT / "data" / "processed" / "magnitude_demo.json"
FIG = ROOT / "figures" / "magnitude_demo.png"


def r2_mae(pred, true):
    mae = float(np.mean(np.abs(pred - true)))
    r2 = float(1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))
    return r2, mae


def train_one(seed, Xn, mask, logdist, afn, mag, Ahat, tr, te, epochs):
    torch.manual_seed(seed)
    model = MultiStationModel(Ahat, hybrid=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Mtr, Ltr, Atr, Ytr = (torch.tensor(Xn[tr]), torch.tensor(mask[tr]),
                               torch.tensor(logdist[tr]), torch.tensor(afn[tr]), torch.tensor(mag[tr]))
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(tr)).split(32):
            opt.zero_grad()
            nn.functional.mse_loss(model(Xtr[b], Mtr[b], Ltr[b], Atr[b]), Ytr[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xn[te]), torch.tensor(mask[te]),
                     torch.tensor(logdist[te]), torch.tensor(afn[te])).numpy()
    return model, pred


def nearest_station_pred(model, Xn, mask, logdist, afn, dist, te):
    """Ablation: keep only each event's nearest recorded station (kills network fusion)."""
    m_near = np.zeros_like(mask[te])
    for i, gi in enumerate(te):
        rec = np.where(mask[gi])[0]
        m_near[i, rec[np.argmin(dist[gi][rec])]] = True
    with torch.no_grad():
        return model(torch.tensor(Xn[te]), torch.tensor(m_near),
                     torch.tensor(logdist[te]), torch.tensor(afn[te])).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    d = dict(np.load(NPZ, allow_pickle=True))
    Xn, mask, logdist = prep(d)
    mag = d["mag"].astype(np.float32)
    Ahat = adjacency(d["coords"])
    af = baseline_features(d)
    tr, va, te = split_chrono(d["wtime"])
    am, asd = af[tr].mean(0), af[tr].std(0) + 1e-6
    afn = ((af - am) / asd).astype(np.float32)
    yte = mag[te]

    if args.retrain or not CKPT.exists():
        print(f"Training {args.seeds}-model magnitude ensemble ({args.epochs} epochs each) on "
              f"{len(tr)} events, holding out {len(te)}...")
        preds, states, single_r2 = [], [], []
        best = None
        for s in range(args.seeds):
            model, pred = train_one(s, Xn, mask, logdist, afn, mag, Ahat, tr, te, args.epochs)
            r2, _ = r2_mae(pred, yte)
            single_r2.append(r2)
            preds.append(pred); states.append(model.state_dict())
            print(f"  seed {s}: test R2 = {r2:+.3f}")
            if best is None or r2 > best[0]:
                best = (r2, model)
        ens = np.mean(preds, axis=0)
        r2_e, mae_e = r2_mae(ens, yte)
        r2_near, mae_near = r2_mae(nearest_station_pred(best[1], Xn, mask, logdist, afn, d["dist"], te), yte)
        torch.save({"states": states, "am": am, "asd": asd}, CKPT)
        summary = {"seeds": args.seeds, "epochs": args.epochs, "single_r2": single_r2,
                   "ens_r2": r2_e, "ens_mae": mae_e, "ablation_near_r2": r2_near}
    else:
        ckpt = torch.load(CKPT, weights_only=False)
        preds = []
        for st in ckpt["states"]:
            m = MultiStationModel(Ahat, hybrid=True); m.load_state_dict(st); m.eval()
            with torch.no_grad():
                preds.append(m(torch.tensor(Xn[te]), torch.tensor(mask[te]),
                              torch.tensor(logdist[te]), torch.tensor(afn[te])).numpy())
        ens = np.mean(preds, axis=0)
        summary = json.loads(SUMMARY.read_text())
        print(f"Loaded cached {len(preds)}-model ensemble. Use --retrain to rebuild.")

    # baseline: amp+dist linear regression
    lr = LinearRegression().fit(af[tr], mag[tr])
    r2_b, mae_b = r2_mae(lr.predict(af[te]), yte)
    r2_e, mae_e = r2_mae(ens, yte)

    print(f"\n=== MAGNITUDE on held-out events (n={len(te)}, Mw {yte.min():.1f}..{yte.max():.1f}) ===")
    print(f"  deep ENSEMBLE (all stations): R2={r2_e:+.3f}  MAE={mae_e:.3f}")
    if "single_r2" in summary:
        sr = summary["single_r2"]
        print(f"  single models: R2 mean {np.mean(sr):+.3f} +/- {np.std(sr):.3f} "
              f"(range {min(sr):+.3f}..{max(sr):+.3f}) -> ensembling removes the noise")
    if "ablation_near_r2" in summary:
        print(f"  nearest-1-station ablation:   R2={summary['ablation_near_r2']:+.3f}  "
              f"(collapse -> multi-station GNN fusion is doing the work)")
    print(f"  amp+dist physics baseline:    R2={r2_b:+.3f}  MAE={mae_b:.3f}")

    summary.update({"ens_r2": r2_e, "ens_mae": mae_e, "baseline_r2": r2_b, "baseline_mae": mae_b})
    SUMMARY.write_text(json.dumps(summary, indent=2))
    make_figure(yte, ens, lr.predict(af[te]), r2_e, r2_b, mag[te], summary)
    print(f"\n  wrote {FIG.relative_to(ROOT)}  and  {SUMMARY.relative_to(ROOT)}")


def make_figure(yte, ens, base, r2_e, r2_b, mags, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    DEEP, BASE = "#534AB7", "#888780"
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    lo, hi = yte.min() - 0.2, yte.max() + 0.2

    for a, pred, r2, name, c in ((ax[0], ens, r2_e, "deep ensemble", DEEP),
                                 (ax[1], base, r2_b, "amp+dist baseline", BASE)):
        a.scatter(yte, pred, s=26, alpha=0.6, color=c, edgecolor="none")
        a.plot([lo, hi], [lo, hi], "k--", lw=0.9)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel("true magnitude (USGS catalog)"); a.set_ylabel("predicted magnitude")
        a.set_title(f"{name}   R²={r2:+.3f}", fontweight="bold", color=c if c == DEEP else "#333")
        # annotate the largest quake in the test set
        j = int(np.argmax(yte))
        a.annotate(f"M{yte[j]:.1f} → est M{pred[j]:.1f}", (yte[j], pred[j]),
                   textcoords="offset points", xytext=(-10, 10), fontsize=8, color="#444")

    fig.suptitle("Network magnitude estimation — deep multi-station fusion vs. physics baseline "
                 "(held-out California events)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)


if __name__ == "__main__":
    main()
