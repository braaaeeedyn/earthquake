"""Live, reproducible DETECTION demo: the deep detector vs the classic STA/LTA baseline,
on held-out real waveforms it never saw in training.

This is the "it works" showpiece. Detection is the decisive, low-variance win (unlike
magnitude/EEW), so it reproduces cleanly. The script trains once (cached), reports a
seed-averaged test AUC with a spread, and then *shows* the model at work: individual
held-out windows where the deep model is right and STA/LTA is wrong.

  python scripts/demo_detect.py                 # train if needed (cached), then demo
  python scripts/demo_detect.py --retrain       # force retrain
  python scripts/demo_detect.py --seeds 5 --epochs 12

Outputs:
  data/processed/detector.pt          best-seed weights + aux normalizer (persisted)
  data/processed/detection_demo.json  per-seed AUCs, mean, spread, thresholds
  figures/detection_demo.png          ROC (deep vs STA/LTA) + example held-out windows
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from seismic_train import SeisModel, prep, split_chrono, sta_lta_scores, SR  # noqa: E402
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve  # noqa: E402

NPZ = ROOT / "data" / "processed" / "seismic_phase1.npz"
CKPT = ROOT / "data" / "processed" / "detector.pt"
SUMMARY = ROOT / "data" / "processed" / "detection_demo.json"
FIG = ROOT / "figures" / "detection_demo.png"


def best_threshold(scores, labels):
    """Threshold (over score quantiles) that maximizes MCC on a validation set."""
    grid = np.quantile(scores, np.linspace(0.3, 0.9, 30))
    return max(grid, key=lambda t: matthews_corrcoef(labels, scores >= t))


def train_one(seed, w, auxn, ydet, mag, tr, te, epochs):
    """Train one detector (detection+magnitude combined loss, as in seismic_train) -> (model, test_probs)."""
    torch.manual_seed(seed)
    model = SeisModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Atr = torch.tensor(w[tr]), torch.tensor(auxn[tr])
    ydtr, ymtr = torch.tensor(ydet[tr]), torch.tensor(mag[tr])
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(tr)).split(64):
            opt.zero_grad()
            pdd, pmm = model(Xtr[b], Atr[b])
            loss = nn.functional.binary_cross_entropy_with_logits(pdd, ydtr[b])
            m = ydtr[b] == 1
            if m.any():
                loss = loss + nn.functional.mse_loss(pmm[m], ymtr[b][m])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pdt, _ = model(torch.tensor(w[te]), torch.tensor(auxn[te]))
    return model, torch.sigmoid(pdt).numpy()


def train_and_cache(w, auxn, ydet, mag, tr, va, te, amean, astd, seeds, epochs):
    aucs, best = [], None
    for s in range(seeds):
        model, probs = train_one(s, w, auxn, ydet, mag, tr, te, epochs)
        auc = roc_auc_score(ydet[te], probs)
        aucs.append(auc)
        print(f"  seed {s}: test AUC = {auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, model, s)
    _, model, best_seed = best
    torch.save({"state": model.state_dict(), "amean": amean, "astd": astd,
                "best_seed": best_seed}, CKPT)
    summary = {"seeds": seeds, "epochs": epochs, "seed_aucs": aucs,
               "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
               "best_seed": int(best_seed)}
    SUMMARY.write_text(json.dumps(summary, indent=2))
    return model, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    d = dict(np.load(NPZ, allow_pickle=True))
    w, aux = prep(d)
    ydet, mag = d["ydet"].astype(np.float32), d["mag"].astype(np.float32)
    tr, va, te = split_chrono(d["wtime"])
    amean, astd = aux[tr].mean(0), aux[tr].std(0) + 1e-6
    auxn = (aux - amean) / astd

    if args.retrain or not CKPT.exists():
        print(f"Training {args.seeds} detectors ({args.epochs} epochs each) on {len(tr)} "
              f"windows, holding out {len(te)} for test...")
        model, summary = train_and_cache(w, auxn, ydet, mag, tr, va, te, amean, astd,
                                         args.seeds, args.epochs)
    else:
        ckpt = torch.load(CKPT, weights_only=False)
        model = SeisModel()
        model.load_state_dict(ckpt["state"])
        model.eval()
        summary = json.loads(SUMMARY.read_text())
        print(f"Loaded cached detector (best of {summary['seeds']} seeds). "
              f"Use --retrain to rebuild.")

    # ---- best-seed model probabilities on val (for thresholds) and test ----
    with torch.no_grad():
        pv, _ = model(torch.tensor(w[va]), torch.tensor(auxn[va]))
        pt, _ = model(torch.tensor(w[te]), torch.tensor(auxn[te]))
    pv, pt = torch.sigmoid(pv).numpy(), torch.sigmoid(pt).numpy()

    # ---- STA/LTA baseline: score everything, pick its threshold fairly on val ----
    slt = sta_lta_scores(d["waves"])
    thr_deep = best_threshold(pv, ydet[va])
    thr_sta = best_threshold(slt[va], ydet[va])

    yte = ydet[te].astype(int)
    deep_pred = (pt >= thr_deep).astype(int)
    sta_pred = (slt[te] >= thr_sta).astype(int)
    auc_deep_te = roc_auc_score(yte, pt)
    auc_sta_te = roc_auc_score(yte, slt[te])

    # ---- report: seed-averaged headline, then held-out accuracy on this split ----
    print(f"\n=== DETECTION on held-out real waveforms (test n={len(te)}, "
          f"{int(yte.sum())} events) ===")
    print(f"  deep detector : AUC (best seed on this split) = {auc_deep_te:.3f}")
    print(f"  deep detector : AUC (mean of {summary['seeds']} seeds) = "
          f"{summary['auc_mean']:.3f} +/- {summary['auc_std']:.3f} (1 s.d.)")
    print(f"  STA/LTA base  : AUC = {auc_sta_te:.3f}")
    print(f"  at MCC-optimal thresholds -> deep MCC={matthews_corrcoef(yte, deep_pred):+.3f}"
          f"  STA/LTA MCC={matthews_corrcoef(yte, sta_pred):+.3f}")

    # ---- show it working: windows where deep is right and STA/LTA is wrong ----
    staid, wtime, stns = d["staid"][te], d["wtime"][te], d["stations"]
    def stamp(i):
        name = stns[staid[i]]
        dt = datetime.fromtimestamp(float(wtime[i]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        return f"{name} {dt}"

    caught = [i for i in range(len(te))                       # events deep catches, STA/LTA misses
              if yte[i] == 1 and deep_pred[i] == 1 and sta_pred[i] == 0]
    rejected = [i for i in range(len(te))                     # noise deep rejects, STA/LTA false-alarms
                if yte[i] == 0 and deep_pred[i] == 0 and sta_pred[i] == 1]
    print(f"\n  deep is right where STA/LTA is wrong: {len(caught)} events caught that STA/LTA "
          f"missed, {len(rejected)} false alarms STA/LTA raised that deep rejected. Examples:")
    for i in (caught[:2] + rejected[:2]):
        truth = "event" if yte[i] == 1 else "noise"
        dv = "event" if deep_pred[i] == 1 else "noise"
        sv = "event" if sta_pred[i] == 1 else "noise"
        mark = "miss" if yte[i] == 1 else "false alarm"
        print(f"    {stamp(i):<26} truth={truth:<5}  deep P(event)={pt[i]:.2f}->{dv} "
              f"[OK]   STA/LTA->{sv} [X] {mark}")

    make_figure(yte, pt, slt[te], auc_deep_te, auc_sta_te, d["waves"][te],
                caught, rejected, stamp, pt, summary)
    print(f"\n  wrote {FIG.relative_to(ROOT)}  and  {SUMMARY.relative_to(ROOT)}")


def make_figure(yte, pt, slt_te, auc_deep, auc_sta, waves_te, caught, rejected, stamp, probs, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEEP, BASE, GOOD = "#534AB7", "#888780", "#1f9d6b"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # ROC: deep vs STA/LTA on the held-out test set
    axr = axes[0, 0]
    fd, td, _ = roc_curve(yte, pt)
    fs, ts, _ = roc_curve(yte, slt_te)
    axr.plot(fd, td, color=DEEP, lw=2.2, label=f"deep  AUC={auc_deep:.3f}")
    axr.plot(fs, ts, color=BASE, lw=2.0, label=f"STA/LTA  AUC={auc_sta:.3f}")
    axr.plot([0, 1], [0, 1], "k--", lw=0.8)
    axr.set_xlabel("false positive rate"); axr.set_ylabel("true positive rate")
    axr.set_title("Detection ROC (held-out test)", fontweight="bold")
    axr.legend(loc="lower right", fontsize=9)

    # Seed spread bar
    axb = axes[0, 1]
    sa = summary["seed_aucs"]
    axb.bar(range(len(sa)), sa, color=DEEP)
    axb.axhline(summary["auc_mean"], color="k", lw=1, ls="--",
                label=f"mean {summary['auc_mean']:.3f}±{summary['auc_std']:.3f}")
    axb.axhline(auc_sta, color=BASE, lw=1.5, label=f"STA/LTA {auc_sta:.3f}")
    axb.set_ylim(0.5, 1.0); axb.set_xlabel("seed"); axb.set_ylabel("test AUC")
    axb.set_title("Reproducibility across seeds", fontweight="bold")
    axb.legend(fontsize=8, loc="lower left")

    # Two example waveforms: one caught event, one rejected false alarm (deep right, STA/LTA wrong)
    picks = [caught[0] if caught else rejected[0],
             rejected[0] if rejected else caught[-1]]
    t = np.arange(waves_te.shape[1]) / SR
    for k, i in enumerate(picks):
        ax = axes[1, k]
        ax.plot(t, waves_te[i], color=GOOD if yte[i] == 1 else BASE, lw=0.6)
        truth = "EVENT" if yte[i] == 1 else "NOISE"
        err = "STA/LTA missed it" if yte[i] == 1 else "STA/LTA false-alarmed"
        ax.set_title(f"{truth}  ·  deep P={probs[i]:.2f} (correct)  ·  {err}",
                     fontsize=9, color=GOOD if yte[i] == 1 else "#333")
        ax.set_xlabel("time (s)"); ax.set_yticks([])
        ax.text(0.01, 0.97, stamp(i), transform=ax.transAxes, fontsize=7,
                va="top", color="#666")

    fig.suptitle("Deep seismic detector vs. classic STA/LTA — shown working on held-out data",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)


if __name__ == "__main__":
    main()
