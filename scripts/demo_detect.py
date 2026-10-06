"""Train + evaluate the v2 DETECTOR on the live network's data (data/processed/v2/detection.npz).

What changed vs v1 (HOW_IT_WORKS.md section 4.1):
  - onset-position augmentation: P is placed anywhere 1-25 s into the 30 s window, so the detector
    is time-invariant like the live sliding window (v1 only ever saw P at exactly 5 s);
  - negatives = random-time noise (all hours, screened against ANY M>=1 quake) + HARD negatives
    (the live daemon's own Sep-2026 false declarations); balanced batches, hard ones oversampled;
  - seeds are selected on VALIDATION AUC (v1 picked the best seed on the test set);
  - besides AUC/MCC vs STA/LTA on the chronological test split, it reports the OPERATIONAL number:
    per-window false-positive rate on held-out live noise (Oct 2-5 2026 hard negatives).

Splits: events/noise chronological 70/15/15 by time (locked rule). Hard negatives by date:
train Sep 22-28, val Sep 29-Oct 1, test >= Oct 2 2026 (the old daemon log starts Sep 22; all after the event data, which ends Aug 2026).

  python scripts/demo_detect.py --retrain --seeds 5      # train (GPU if available) + evaluate
  python scripts/demo_detect.py                          # evaluate the saved checkpoint
  python scripts/demo_detect.py --out C.pt --compare data/processed/detector.pt   # paired vs the champion

Every headline number carries a 95% bootstrap CI, clustered by event (all station windows of one quake,
or of one noise time, resample together). With MLFLOW_TRACKING_URI set the run is logged (scripts/tracking.py).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve  # noqa: E402

import tracking  # noqa: E402
from eq import network, stats  # noqa: E402
from eq.models import DetectorNet, sta_lta_scores  # noqa: E402
from eq.pipeline import SR, Config, det_prep  # noqa: E402

NPZ = ROOT / "data" / "processed" / "v2" / "detection.npz"
CKPT = ROOT / "data" / "processed" / "detector.pt"
SUMMARY = ROOT / "data" / "processed" / "detection_demo.json"
FIG = ROOT / "figures" / "detection_demo.png"
NPTS = 3000
HARD_VAL = datetime(2026, 9, 29, tzinfo=timezone.utc).timestamp()
HARD_TEST = datetime(2026, 10, 2, tzinfo=timezone.utc).timestamp()


def split_chrono(t, fr=(0.7, 0.15)):
    o = np.argsort(t, kind="stable")
    n = len(o)
    return o[:int(n * fr[0])], o[int(n * fr[0]):int(n * (fr[0] + fr[1]))], o[int(n * (fr[0] + fr[1])):]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                   # noqa: BLE001
        return "unknown"


def load():
    d = np.load(NPZ)
    pos, noise, hard = d["pos"], d["noise"], d["hard"]
    # one chronological split over ALL event + noise windows (no window's future leaks into training)
    t_all = np.concatenate([d["pos_time"], d["noise_time"]])
    tr, va, te = split_chrono(t_all)
    npos = len(pos)
    sp = {k: (idx[idx < npos], idx[idx >= npos] - npos) for k, idx in (("tr", tr), ("va", va), ("te", te))}
    ht = d["hard_time"]
    hs = {"tr": np.flatnonzero(ht < HARD_VAL), "va": np.flatnonzero((ht >= HARD_VAL) & (ht < HARD_TEST)),
          "te": np.flatnonzero(ht >= HARD_TEST)}
    return d, pos, noise, hard, sp, hs, int(d["p_index"])


def crops(arr, idx, starts):
    """(len(idx), NPTS) detector inputs from arr[idx] at `starts` (det_prep: same as live)."""
    if len(idx) == 0:
        return np.zeros((0, NPTS), np.float32)
    return det_prep(np.stack([arr[i, s:s + NPTS] for i, s in zip(idx, starts)]))


def pos_starts(n, p_index, rng, fixed=None):
    u = rng.uniform(1.0, 25.0, n) if fixed is None else np.full(n, fixed)
    return (p_index - u * SR).astype(int)


def neg_starts(arr, n, rng, fixed=False):
    L = arr.shape[1]
    return np.zeros(n, int) if fixed else rng.integers(0, L - NPTS + 1, n)


def clusters(d, sp, split):
    """Bootstrap cluster per evaluation row: the event's origin time (positives) / the noise time."""
    pi, ni = sp[split]
    return np.r_[d["pos_time"][pi], -d["noise_time"][ni]]


def load_ckpt(path, device):
    ck = torch.load(path, weights_only=False, map_location=device)
    model = DetectorNet().to(device)
    model.load_state_dict(ck["state"])
    return model, ck["thr"]


def eval_sets(pos, noise, hard, sp, hs, p_index, split, onset=5.0):
    """Deterministic evaluation windows: positives with P at `onset` s, noise/hard from their start."""
    rng = np.random.default_rng(0)
    pi, ni = sp[split]
    X = np.concatenate([crops(pos, pi, pos_starts(len(pi), p_index, rng, onset)),
                        crops(noise, ni, neg_starts(noise, len(ni), rng, True))])
    y = np.r_[np.ones(len(pi)), np.zeros(len(ni))]
    H = crops(hard, hs[split], neg_starts(hard, len(hs[split]), rng, True)) if len(hs[split]) else np.zeros((0, NPTS))
    return X, y, H


def predict(model, X, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(torch.sigmoid(model(torch.tensor(X[i:i + 1024], dtype=torch.float32, device=device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_one(seed, pos, noise, hard, sp, hs, p_index, epochs, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DetectorNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ptr, ntr, htr = sp["tr"][0], sp["tr"][1], hs["tr"]
    bs = 256
    for _ in range(epochs):
        model.train()
        order = rng.permutation(ptr)
        for b in range(0, len(order), bs // 2):
            pi = order[b:b + bs // 2]
            n_h = len(pi) // 3 if len(htr) else 0                       # ~1/3 of negatives are hard
            ni = rng.choice(ntr, len(pi) - n_h)
            hi = rng.choice(htr, n_h) if n_h else np.zeros(0, int)
            X = np.concatenate([crops(pos, pi, pos_starts(len(pi), p_index, rng)),
                                crops(noise, ni, neg_starts(noise, len(ni), rng)),
                                crops(hard, hi, neg_starts(hard, len(hi), rng)) if n_h else np.zeros((0, NPTS))])
            X *= rng.choice([-1.0, 1.0], (len(X), 1)).astype(np.float32)   # polarity augmentation
            y = np.r_[np.ones(len(pi)), np.zeros(len(ni) + len(hi))].astype(np.float32)
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(torch.tensor(X, dtype=torch.float32, device=device)), torch.tensor(y, device=device))
            loss.backward()
            opt.step()
        sched.step()
    return model


def best_threshold(scores, labels):
    grid = np.quantile(scores, np.linspace(0.3, 0.98, 60))
    return float(max(grid, key=lambda t: matthews_corrcoef(labels, scores >= t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--out", default=str(CKPT))
    ap.add_argument("--compare", help="checkpoint to compare against, paired on the same test set (the champion)")
    args = ap.parse_args()
    summary_fp = SUMMARY if Path(args.out) == CKPT else Path(args.out).with_suffix(".json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {"arch": "cnn-transformer", "lr": 1e-3, "weight_decay": 1e-4, "epochs": args.epochs, "batch": 256,
              "seeds": args.seeds, "retrain": args.retrain}
    with tracking.run("detect", "train" if args.retrain else "evaluate", params):
        run(args, summary_fp, device)


def run(args, summary_fp, device):
    d, pos, noise, hard, sp, hs, p_index = load()
    print(f"windows: pos {len(pos)}  noise {len(noise)}  hard {len(hard)}  | train pos/noise/hard "
          f"{len(sp['tr'][0])}/{len(sp['tr'][1])}/{len(hs['tr'])}  device {device}")
    Xva, yva, Hva = eval_sets(pos, noise, hard, sp, hs, p_index, "va")
    Xte, yte, Hte = eval_sets(pos, noise, hard, sp, hs, p_index, "te")

    if args.retrain or not Path(args.out).exists():
        best, seed_val, seed_test = None, [], []
        for s in range(args.seeds):
            m = train_one(s, pos, noise, hard, sp, hs, p_index, args.epochs, device)
            av = roc_auc_score(yva, predict(m, Xva, device))
            at = roc_auc_score(yte, predict(m, Xte, device))
            seed_val.append(av); seed_test.append(at)
            print(f"  seed {s}: val AUC {av:.4f}   test AUC {at:.4f}", flush=True)
            if best is None or av > best[0]:                             # select on VALIDATION
                best = (av, m, s)
        _, model, bseed = best
        pv = predict(model, np.concatenate([Xva, Hva]), device)
        thr = best_threshold(pv, np.r_[yva, np.zeros(len(Hva))])
        torch.save({"state": model.state_dict(), "thr": thr, "best_seed": bseed, "stations": network.CODES,
                    "npts": NPTS, "trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "seeds": args.seeds, "git": git_commit(),
                    "dataset": json.loads((NPZ.parent / "dataset_meta.json").read_text())},
                   args.out)
        summary = {"seeds": args.seeds, "epochs": args.epochs, "seed_val_auc": seed_val, "seed_test_auc": seed_test,
                   "auc_mean": float(np.mean(seed_test)), "auc_std": float(np.std(seed_test)), "best_seed": bseed}
    else:
        summary = json.loads(summary_fp.read_text()) if summary_fp.exists() else {"auc_mean": float("nan"),
                                                                                    "auc_std": float("nan")}
    model, thr = load_ckpt(args.out, device)

    pt = predict(model, Xte, device)
    auc = roc_auc_score(yte, pt)
    mcc = matthews_corrcoef(yte, pt >= thr)
    slt = sta_lta_scores(Xte)
    sva = sta_lta_scores(Xva)
    auc_s = roc_auc_score(yte, slt)
    mcc_s = matthews_corrcoef(yte, slt >= best_threshold(sva, yva))
    # time invariance: AUC with P at other positions in the window
    inv = {}
    for onset in (2.0, 12.0, 22.0):
        Xo, yo, _ = eval_sets(pos, noise, hard, sp, hs, p_index, "te", onset)
        inv[onset] = float(roc_auc_score(yo, predict(model, Xo, device)))
    # operational: per-window false-positive rate on held-out LIVE noise and on test noise, at the checkpoint's
    # MCC threshold AND at the live trigger threshold (pipeline_config.json det_thresh -- what the daemon uses)
    trig = Config.load().det_thresh
    ph = predict(model, Hte, device) if len(Hte) else np.zeros(0)
    fpr_live = float(np.mean(ph >= thr)) if len(Hte) else float("nan")
    fpr_noise = float(np.mean(pt[yte == 0] >= thr))
    at_trig = {"trigger": trig, "fpr_test_noise": float(np.mean(pt[yte == 0] >= trig)),
               "fpr_live_noise": float(np.mean(ph >= trig)) if len(Hte) else None,
               "event_recall": float(np.mean(pt[yte == 1] >= trig))}
    # 95% CIs (event-clustered bootstrap) and the paired deep-vs-baseline difference
    cl = clusters(d, sp, "te")
    mcc_at = lambda t: (lambda y, p: matthews_corrcoef(y, p >= t))  # noqa: E731
    _, *auc_ci = stats.bootstrap_ci(roc_auc_score, yte, pt, cl)
    _, *mcc_ci = stats.bootstrap_ci(mcc_at(thr), yte, pt, cl)
    _, *sta_ci = stats.bootstrap_ci(roc_auc_score, yte, slt, cl)
    d_base = stats.paired_bootstrap(roc_auc_score, yte, pt, slt, cl)[:3]
    print(f"\n=== DETECTION, chronological test (n={len(yte)}, {int(yte.sum())} event windows) ===")
    print(f"  deep detector : AUC {auc:.4f}  MCC {mcc:+.3f}   (seed mean AUC {summary['auc_mean']:.4f} "
          f"+/- {summary['auc_std']:.4f})")
    print(f"                  95% CI AUC {auc_ci[0]:.4f}-{auc_ci[1]:.4f}  MCC {mcc_ci[0]:+.3f}-{mcc_ci[1]:+.3f}")
    print(f"  STA/LTA       : AUC {auc_s:.4f}  MCC {mcc_s:+.3f}   (AUC CI {sta_ci[0]:.4f}-{sta_ci[1]:.4f})")
    print(f"  deep - STA/LTA: dAUC {d_base[0]:+.4f}  95% CI {d_base[1]:+.4f}..{d_base[2]:+.4f}")
    print("  time-invariance (AUC with P at 2/12/22 s): " + ", ".join(f"{v:.4f}" for v in inv.values()))
    print(f"  per-window FPR at thr={thr:.3f}: test noise {fpr_noise:.4f}   held-out LIVE noise "
          f"(Oct 2-5, n={len(Hte)}) {fpr_live:.4f}")
    print(f"  per-window FPR at the LIVE trigger {trig:g}: test noise {at_trig['fpr_test_noise']:.4f}   live noise "
          f"{at_trig['fpr_live_noise']}   event-window recall {at_trig['event_recall']:.4f}")
    summary.update({"test_auc": auc, "test_mcc": mcc, "sta_lta_auc": auc_s, "sta_lta_mcc": mcc_s, "thr": thr,
                    "n_test": int(len(yte)), "n_test_events": int(yte.sum()), "auc_by_onset": inv,
                    "fpr_test_noise": fpr_noise, "fpr_live_noise": fpr_live, "n_live_noise": int(len(Hte)),
                    "fpr_test_noise_at_trigger": at_trig["fpr_test_noise"],
                    "fpr_live_noise_at_trigger": at_trig["fpr_live_noise"], "recall_at_trigger": at_trig["event_recall"],
                    "trigger": trig,
                    "test_auc_ci": auc_ci, "test_mcc_ci": mcc_ci, "sta_lta_auc_ci": sta_ci, "d_auc_vs_sta_lta": d_base,
                    "test_start": float(np.min(d["pos_time"][sp["te"][0]])),
                    "dataset_version": ck_dataset_version(args.out, device)})
    if args.compare:                                     # paired vs another checkpoint (the gate's input)
        other, thr_o = load_ckpt(args.compare, device)
        po = predict(other, Xte, device)
        summary["vs"] = {"ckpt": str(args.compare),
                         "d_auc": stats.paired_bootstrap(roc_auc_score, yte, pt, po, cl)[:3],
                         # each model at its own validation threshold
                         "d_mcc": stats.paired_bootstrap(matthews_corrcoef, yte, pt >= thr, po >= thr_o, cl)[:3],
                         "fpr_live_noise_other": float(np.mean(predict(other, Hte, device) >= thr_o)) if len(Hte) else None,
                         # both at the live trigger threshold: the false triggers the daemon would actually see
                         "fpr_test_noise_at_trigger_other": float(np.mean(po[yte == 0] >= trig)),
                         "fpr_live_noise_at_trigger_other": float(np.mean(predict(other, Hte, device) >= trig))
                         if len(Hte) else None}
        print(f"  vs {args.compare}: dAUC {summary['vs']['d_auc'][0]:+.5f} "
              f"(CI {summary['vs']['d_auc'][1]:+.5f}..{summary['vs']['d_auc'][2]:+.5f})  dMCC {summary['vs']['d_mcc'][0]:+.3f}")
    summary["mlflow_run_id"] = tracking.active_run_id()
    summary_fp.write_text(json.dumps(summary, indent=2))
    tracking.log_metrics(summary)
    tracking.log_files([args.out, summary_fp])
    if summary_fp == SUMMARY:
        make_figure(yte, pt, slt, auc, auc_s, summary)
    print(f"  wrote {summary_fp}")


def ck_dataset_version(path, device):
    return torch.load(path, weights_only=False, map_location=device).get("dataset", {}).get("version", "v2-2026-09-01")


def make_figure(yte, pt, slt, auc_deep, auc_sta, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    DEEP, BASE = "#111111", "#888780"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    fd, td, _ = roc_curve(yte, pt)
    fs, ts, _ = roc_curve(yte, slt)
    ax[0].plot(fd, td, color=DEEP, lw=2.2, label=f"deep  AUC={auc_deep:.3f}")
    ax[0].plot(fs, ts, color=BASE, lw=2.0, label=f"STA/LTA  AUC={auc_sta:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("Detection ROC (chronological test)", fontweight="bold")
    ax[0].legend(loc="lower right", fontsize=9)
    sa = summary["seed_test_auc"] if "seed_test_auc" in summary else []
    ax[1].bar(range(len(sa)), sa, color=DEEP)
    ax[1].axhline(auc_sta, color=BASE, lw=1.5, label=f"STA/LTA {auc_sta:.3f}")
    ax[1].set_ylim(0.5, 1.0); ax[1].set_xlabel("seed"); ax[1].set_ylabel("test AUC")
    ax[1].set_title("Reproducibility across seeds", fontweight="bold"); ax[1].legend(fontsize=8)
    fig.suptitle("Deep detector vs STA/LTA on the live network (held-out, 2000-2026 data)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)


if __name__ == "__main__":
    main()
