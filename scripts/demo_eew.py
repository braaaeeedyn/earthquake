"""Reproducible EARLY-WARNING (EEW) demo: from only the first 8 s of an event, predict the
FUTURE peak ground shaking (PGV) at each station -- before the strong shaking arrives -- and
decide whether to raise a strong-shaking alert. Shown working on held-out events vs the classic
early-amp+distance baseline.

Single EEW models are noisy (R² -0.06..0.79), so we seed-ENSEMBLE (the honest fix) and report
the ensemble. The operational metric is alert RECALL: of the station-locations that really did
shake hard, how many did we warn?

  python scripts/demo_eew.py                 # train if needed (cached), then demo
  python scripts/demo_eew.py --retrain --k 5 --epochs 60   # publishable headline

Outputs:
  data/processed/eew_ensemble.pt   K model weights + aux normalizer (persisted)
  data/processed/eew_demo.json     ensemble R²/MAE + alert recall/precision/MCC vs baseline
  figures/eew_demo.png             predicted-vs-true PGV + alert-quality bars
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
from sklearn.metrics import matthews_corrcoef  # noqa: E402
from seismic_eew import EEWModel, build_arrays  # noqa: E402
from seismic_train_multi import adjacency, split_chrono  # noqa: E402

NPZ = ROOT / "data" / "processed" / "seismic_phase2a_xl.npz"
CKPT = ROOT / "data" / "processed" / "eew_ensemble.pt"
SUMMARY = ROOT / "data" / "processed" / "eew_demo.json"
FIG = ROOT / "figures" / "eew_demo.png"


def train_one(seed, early, mask, logpgv, auxn, logdist, Ahat, tr, te, epochs):
    torch.manual_seed(seed)
    model = EEWModel(Ahat, hybrid=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Mtr, Ltr, Atr, Ytr = (torch.tensor(early[tr]), torch.tensor(mask[tr]),
                               torch.tensor(logdist[tr]), torch.tensor(auxn[tr]), torch.tensor(logpgv[tr]))
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(tr)).split(32):
            opt.zero_grad()
            out = model(Xtr[b], Mtr[b], Ltr[b], Atr[b])
            nn.functional.mse_loss(out[Mtr[b]], Ytr[b][Mtr[b]]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model, model(torch.tensor(early[te]), torch.tensor(mask[te]),
                            torch.tensor(logdist[te]), torch.tensor(auxn[te])).numpy()


def predict_ens(states, Ahat, early, mask, logdist, auxn, idx):
    """Mean prediction of an ensemble (list of state_dicts) over the given indices."""
    preds = []
    for st in states:
        m = EEWModel(Ahat, hybrid=True); m.load_state_dict(st); m.eval()
        with torch.no_grad():
            preds.append(m(torch.tensor(early[idx]), torch.tensor(mask[idx]),
                           torch.tensor(logdist[idx]), torch.tensor(auxn[idx])).numpy())
    return np.mean(preds, axis=0)


def r2_mae(pred, true):
    mae = float(np.mean(np.abs(pred - true)))
    r2 = float(1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))
    return r2, mae


def fbeta(pred, true, damage_thr, decide_thr, beta=2.0):
    """F-beta of the alert (truth = true>damage_thr, alert = pred>decide_thr). beta>1 favors recall."""
    yt = (true > damage_thr).astype(int); yh = (pred > decide_thr).astype(int)
    tp = int(((yh == 1) & (yt == 1)).sum()); fp = int(((yh == 1) & (yt == 0)).sum())
    fn = int(((yh == 0) & (yt == 1)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def tune_decide_thr(pred_va, true_va, damage_thr, beta=2.0):
    """Alert decision threshold that maximizes recall-weighted F-beta on validation.
    The 'strong shaking' definition (damage_thr) is fixed; only the alert trigger is tuned,
    reflecting EEW's asymmetric cost (a missed warning is worse than a false alarm)."""
    yt = (true_va > damage_thr).astype(int)
    if yt.sum() == 0 or yt.sum() == len(yt):
        return float(damage_thr)
    grid = np.quantile(pred_va, np.linspace(0.02, 0.95, 80))
    return float(max(grid, key=lambda t: fbeta(pred_va, true_va, damage_thr, t, beta)))


def alert_stats(pred, true, damage_thr, decide_thr):
    """Alert quality: truth = true>damage_thr, alert = pred>decide_thr (tuned separately)."""
    yt = (true > damage_thr).astype(int); yh = (pred > decide_thr).astype(int)
    tp = int(((yh == 1) & (yt == 1)).sum()); fp = int(((yh == 1) & (yt == 0)).sum())
    fn = int(((yh == 0) & (yt == 1)).sum())
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    mcc = matthews_corrcoef(yt, yh) if len(set(yh)) > 1 else 0.0
    return {"recall": rec, "precision": prec, "mcc": float(mcc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    d = dict(np.load(NPZ, allow_pickle=True))
    early, mask, logpgv, log_evpeak, logdist = build_arrays(d)
    aux = np.stack([log_evpeak, logdist], -1).astype(np.float32)
    Ahat = adjacency(d["coords"])
    tr, va, te = split_chrono(d["wtime"])
    am, asd = aux[tr][mask[tr]].mean(0), aux[tr][mask[tr]].std(0) + 1e-6
    auxn = (aux - am) / asd
    mte = mask[te]; yte = logpgv[te][mte]
    mva = mask[va]; yva = logpgv[va][mva]

    if args.retrain or not CKPT.exists():
        print(f"Training {args.k}-model EEW ensemble ({args.epochs} epochs each) on {len(tr)} "
              f"events, holding out {len(te)}...")
        states, single_r2 = [], []
        for s in range(args.k):
            model, pred = train_one(s, early, mask, logpgv, auxn, logdist, Ahat, tr, te, args.epochs)
            r2, _ = r2_mae(pred[mte], yte)
            single_r2.append(r2); states.append(model.state_dict())
            print(f"  seed {s}: test R2 = {r2:+.3f}")
        torch.save({"states": states, "am": am, "asd": asd}, CKPT)
        summary = {"k": args.k, "epochs": args.epochs, "single_r2": single_r2}
    else:
        states = torch.load(CKPT, weights_only=False)["states"]
        summary = json.loads(SUMMARY.read_text())
        print(f"Loaded cached {len(states)}-model ensemble. Use --retrain to rebuild.")

    # ensemble predictions on test (report) and validation (operating-point tuning)
    ens = predict_ens(states, Ahat, early, mask, logdist, auxn, te)[mte]
    ens_va = predict_ens(states, Ahat, early, mask, logdist, auxn, va)[mva]
    r2_e, mae_e = r2_mae(ens, yte)
    lr = LinearRegression().fit(aux[tr][mask[tr]], logpgv[tr][mask[tr]])
    pbase = lr.predict(aux[te][mte]); pbase_va = lr.predict(aux[va][mva])
    r2_b, mae_b = r2_mae(pbase, yte)

    thr = float(np.quantile(logpgv[tr][mask[tr]], 0.70))     # fixed 'strong shaking' definition
    # alert trigger tuned on validation, recall-weighted (F2) and applied fairly to both models
    t_deep = tune_decide_thr(ens_va, yva, thr)
    t_base = tune_decide_thr(pbase_va, yva, thr)
    a_deep = alert_stats(ens, yte, thr, t_deep)
    a_base = alert_stats(pbase, yte, thr, t_base)
    a_deep_naive = alert_stats(ens, yte, thr, thr)          # untuned point, for transparency

    print(f"\n=== EEW on held-out events: predict future shaking from first 8 s "
          f"({int(mte.sum())} station-locations) ===")
    print(f"  deep ENSEMBLE R2={r2_e:+.3f}  MAE={mae_e:.3f}    baseline R2={r2_b:+.3f}  MAE={mae_b:.3f}")
    if "single_r2" in summary:
        sr = summary["single_r2"]
        print(f"  single models R2 range {min(sr):+.3f}..{max(sr):+.3f} -> ensembling stabilizes it")
    print(f"  strong-shaking ALERT (trigger tuned on validation, recall-weighted F2):")
    print(f"    deep     recall={a_deep['recall']:.2f}  precision={a_deep['precision']:.2f}  "
          f"MCC={a_deep['mcc']:+.3f}   (untuned trigger recall={a_deep_naive['recall']:.2f})")
    print(f"    baseline recall={a_base['recall']:.2f}  precision={a_base['precision']:.2f}  "
          f"MCC={a_base['mcc']:+.3f}")

    summary.update({"ens_r2": r2_e, "ens_mae": mae_e, "baseline_r2": r2_b,
                    "alert_deep": a_deep, "alert_baseline": a_base, "alert_thr": thr,
                    "decide_deep": t_deep, "decide_base": t_base})
    SUMMARY.write_text(json.dumps(summary, indent=2))
    make_figure(yte, ens, pbase, r2_e, r2_b, thr, t_deep, a_deep, a_base)
    print(f"\n  wrote {FIG.relative_to(ROOT)}  and  {SUMMARY.relative_to(ROOT)}")


def make_figure(yte, ens, pbase, r2_e, r2_b, thr, t_deep, a_deep, a_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    DEEP, BASE, GOOD, BAD = "#534AB7", "#888780", "#1f9d6b", "#d9534f"
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))

    # scatter: predicted vs true future PGV, with alert-threshold quadrants
    lo, hi = yte.min() - 0.2, yte.max() + 0.2
    ax[0].scatter(yte, ens, s=18, alpha=0.5, color=DEEP, edgecolor="none")
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.9)
    ax[0].axhline(t_deep, color=BAD, lw=0.8, ls=":"); ax[0].axvline(thr, color=BAD, lw=0.8, ls=":")
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("true future log₁₀(PGV)"); ax[0].set_ylabel("predicted from first 8 s")
    ax[0].set_title(f"EEW shaking prediction   R²={r2_e:+.3f}\n(dotted = strong-shaking alert line)",
                    fontweight="bold", color=DEEP)

    # alert-quality bars
    metrics = ["recall", "precision", "mcc"]
    x = np.arange(len(metrics)); wbar = 0.36
    ax[1].bar(x - wbar/2, [a_deep[m] for m in metrics], wbar, color=DEEP, label="deep ensemble")
    ax[1].bar(x + wbar/2, [a_base[m] for m in metrics], wbar, color=BASE, label="baseline")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["recall", "precision", "MCC"])
    ax[1].set_ylim(0, 1); ax[1].set_title("Strong-shaking alert quality", fontweight="bold")
    ax[1].legend(fontsize=9)
    for i, m in enumerate(metrics):
        ax[1].text(i - wbar/2, a_deep[m] + 0.02, f"{a_deep[m]:.2f}", ha="center", fontsize=8)
        ax[1].text(i + wbar/2, a_base[m] + 0.02, f"{a_base[m]:.2f}", ha="center", fontsize=8)

    fig.suptitle("Earthquake early warning — future shaking predicted from the first 8 seconds "
                 "(held-out California events)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)


if __name__ == "__main__":
    main()
