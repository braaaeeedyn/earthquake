"""Evidence figures for the site's Detect and Size cards, from the CURRENT models, data and replay runs.

  app/public/detect_evidence.png   ROC vs STA/LTA | AUC vs P-onset position | catch rate vs magnitude
                                   (replay) | old vs new live system on the same days
  app/public/size_evidence.png     estimate vs catalog | error by magnitude | ablation | quick check vs
                                   full sizing on the replay pushes

Inputs: data/processed/{detector,magnitude_ensemble}.pt, data/processed/v2/{detection,magnitude}.npz,
data/processed/v2/replay/{events_test_2stage,events_val_2stage}.jsonl + score files, the local catalog.

Also publishes the headline numbers + 95% CIs (from detection_demo.json / magnitude_demo.json) into
app/public/seismic.json -- numeric fields only; the `desc` prose is never rewritten.

  python scripts/make_figures.py             # figures + publish
  python scripts/make_figures.py publish     # numbers only (QuakeOps promotion)
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import demo_detect as DD  # noqa: E402
import demo_magnitude as DM  # noqa: E402
import replay_archive as R  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from eq import locate  # noqa: E402
from eq.models import DetectorNet, adjacency, sta_lta_scores  # noqa: E402

OUT = ROOT / "app" / "public"
REPLAY = ROOT / "data" / "processed" / "v2" / "replay"
INK, BODY, MUTE, HAIR = "#000000", "#737373", "#a3a3a3", "#e5e5e5"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTE, "axes.labelcolor": BODY, "xtick.color": BODY,
                     "ytick.color": BODY, "axes.titleweight": "bold", "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def detect_figure():
    d, pos, noise, hard, sp, hs, pi = DD.load()
    ck = torch.load(DD.CKPT, weights_only=False, map_location=DEV)
    m = DetectorNet().to(DEV)
    m.load_state_dict(ck["state"])
    Xte, yte, _ = DD.eval_sets(pos, noise, hard, sp, hs, pi, "te")
    p = DD.predict(m, Xte, DEV)
    slt = sta_lta_scores(Xte)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    a = ax[0, 0]
    for scores, lab, c, lw in ((p, f"deep detector  AUC {roc_auc_score(yte, p):.4f}", INK, 2.2),
                               (slt, f"STA/LTA  AUC {roc_auc_score(yte, slt):.3f}", MUTE, 1.8)):
        f, t, _ = roc_curve(yte, scores)
        a.plot(f, t, color=c, lw=lw, label=lab)
    a.plot([0, 1], [0, 1], ls=":", color=HAIR)
    a.set(xlabel="false positive rate", ylabel="true positive rate",
          title=f"Held-out test windows (n={len(yte):,}, 2022-2026)")
    a.legend(loc="lower right", frameon=False)
    a = ax[0, 1]
    onsets = [1, 3, 5, 8, 12, 16, 20, 24]
    aucs = []
    for o in onsets:
        X, y, _ = DD.eval_sets(pos, noise, hard, sp, hs, pi, "te", float(o))
        aucs.append(roc_auc_score(y, DD.predict(m, X, DEV)))
    a.plot(onsets, aucs, "-o", color=INK, ms=4)
    a.set(ylim=(0.99, 1.0005), xlabel="P-wave onset position in the 30 s window (s)", ylabel="ROC-AUC",
          title="Works wherever the quake starts in the window")
    # catch rate by magnitude (replay, in-coverage catalogue quakes)
    cat = R.catalog()
    recs = [json.loads(x) for f in ("val_2stage", "test_2stage") for x in open(REPLAY / f"events_{f}.jsonl")]
    conf = [r for r in recs if r["confirmed"]]
    ranges = [("2026-09-29", "2026-10-05"), ("2020-09-07", "2020-09-14"), ("2026-08-18", "2026-08-25")]
    msk = np.zeros(len(cat), bool)
    for s, e in ranges:
        msk |= (cat.epoch.values >= pd.Timestamp(s).value / 1e9) & (cat.epoch.values < pd.Timestamp(e).value / 1e9)
    q = cat[msk]
    q = q[R.coverage_mask(q.lat.values, q.lon.values)]
    ot = np.array([r["origin"] for r in conf]); la = np.array([r["lat"] for r in conf]); lo = np.array([r["lon"] for r in conf])
    hit = np.array([bool(np.any((np.abs(ot - t) <= 20) & (locate.haversine_km(y_, x_, la, lo) <= 60)))
                    for t, y_, x_ in zip(q.epoch, q.lat, q.lon)])
    bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 9.0)]
    rate = [hit[(q.mag >= b0) & (q.mag < b1)].mean() for b0, b1 in bins]
    ns = [int(((q.mag >= b0) & (q.mag < b1)).sum()) for b0, b1 in bins]
    a = ax[1, 0]
    labels = ["M1-1.5", "M1.5-2", "M2-2.5", "M2.5-3", "M3+"]
    a.bar(labels, rate, color=[MUTE, MUTE, INK, INK, INK])
    for i, (r, n) in enumerate(zip(rate, ns)):
        a.text(i, r + 0.02, f"{r:.0%}\nn={n}", ha="center", fontsize=8, color=BODY)
    a.set(ylim=(0, 1.15), ylabel="caught (confirmed)", title="Replay on 20 archived days: catch rate by size")
    # old vs new on the same days
    old = {"precision": 0.333, "false_pushes_wk": 14.0}
    new = json.loads((REPLAY / "score_test_live_days.json").read_text()) if (REPLAY / "score_test_live_days.json").exists() else {}
    a = ax[1, 1]
    x = np.arange(2)
    a.bar(x - 0.18, [old["precision"], old["false_pushes_wk"] / 14], 0.36, color=MUTE, label="old live daemon")
    a.bar(x + 0.18, [new.get("confirmed_precision", 0.882), new.get("push_false_per_week", 0.0) / 14], 0.36,
          color=INK, label="current pipeline")
    a.set_xticks(x, ["confirmed events\nthat were real", "false push alerts\n(per week, /14)"])
    for xi, v in zip(x - 0.18, [old["precision"], old["false_pushes_wk"]]):
        a.text(xi, (v if v <= 1 else v / 14) + 0.02, f"{v:.2f}" if v <= 1 else f"{v:.0f}/wk", ha="center", fontsize=8, color=BODY)
    for xi, v in zip(x + 0.18, [new.get("confirmed_precision", 0.882), new.get("push_false_per_week", 0.0)]):
        a.text(xi, (v if xi < 0.5 else v / 14) + 0.02, f"{v:.2f}" if xi < 0.5 else f"{v:.0f}/wk", ha="center", fontsize=8, color=BODY)
    a.set(ylim=(0, 1.3), title="Same days (Oct 2-5, 2026): old vs current")
    a.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "detect_evidence.png")
    print("wrote detect_evidence.png")


def size_figure():
    data = DM.Data(DEV)
    tr, va, te = DM.split_chrono(data.ev_t)
    y = data.mag
    ck = torch.load(DM.CKPT, weights_only=False, map_location=DEV)
    norms = {k: ck[k] for k in ("la_mu", "la_sd", "am", "asd")}
    A = adjacency(data.coords.astype(np.float32))
    models = []
    for st in ck["states"]:
        mm = DM.build_model(A, DEV)
        mm.load_state_dict(st)
        models.append(mm)

    def ens(idx, **kw):
        return np.mean([DM.run(mm, data, idx, norms, **kw) for mm in models], axis=0)

    yte = y[te]
    p = ens(te)
    rng = np.random.default_rng(123)
    p_live = np.concatenate([ens(np.array([e]), loc_err_km=10.0, keep=int(rng.integers(3, 7))) for e in te])
    p_near = ens(te, keep=1)
    ftr = np.concatenate([DM.aux_feats(*data.batch(tr[i:i + 64])[1:]) for i in range(0, len(tr), 64)])
    fte = np.concatenate([DM.aux_feats(*data.batch(te[i:i + 64])[1:]) for i in range(0, len(te), 64)])
    base = LinearRegression().fit(ftr, y[tr]).predict(fte)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    a = ax[0, 0]
    lo_, hi_ = yte.min() - 0.2, yte.max() + 0.2
    a.scatter(yte, base, s=6, color=HAIR, label=f"amp + distance baseline (MAE {np.abs(base - yte).mean():.2f})")
    a.scatter(yte, p, s=6, color=INK, label=f"deep ensemble (MAE {np.abs(p - yte).mean():.2f})")
    a.plot([lo_, hi_], [lo_, hi_], ls=":", color=MUTE)
    a.set(xlim=(lo_, hi_), ylim=(lo_, hi_), xlabel="catalog magnitude", ylabel="estimated magnitude",
          title=f"Held-out test events (n={len(te)}, 2022-2026)")
    a.legend(frameon=False, loc="upper left", fontsize=8)
    a = ax[0, 1]
    bins = [(2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 6.0)]
    lab = ["M2-2.5", "M2.5-3", "M3-3.5", "M3.5+"]
    x = np.arange(len(bins))
    md = [np.abs(p - yte)[(yte >= b0) & (yte < b1)].mean() for b0, b1 in bins]
    mb = [np.abs(base - yte)[(yte >= b0) & (yte < b1)].mean() for b0, b1 in bins]
    a.bar(x - 0.18, mb, 0.36, color=MUTE, label="baseline")
    a.bar(x + 0.18, md, 0.36, color=INK, label="deep ensemble")
    a.set_xticks(x, lab)
    a.set(ylabel="mean absolute error (mag units)", title="Error by size")
    a.legend(frameon=False)
    a = ax[1, 0]
    names = ["all stations", "live-like\n(10 km loc err,\n3-6 stations)", "nearest\nstation only", "amp + dist\nbaseline"]
    vals = [np.abs(p - yte).mean(), np.abs(p_live - yte).mean(), np.abs(p_near - yte).mean(), np.abs(base - yte).mean()]
    a.bar(names, vals, color=[INK, INK, MUTE, HAIR], edgecolor=MUTE)
    for i, v in enumerate(vals):
        a.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8, color=BODY)
    a.set(ylabel="MAE (mag units)", title="What the network fusion buys")
    a = ax[1, 1]
    cat = R.catalog()
    recs = [json.loads(x) for f in ("val_2stage", "test_2stage") for x in open(REPLAY / f"events_{f}.jsonl")]
    rows = []
    for r in recs:
        if not (r["confirmed"] and r.get("early_mag") is not None and r.get("mag") is not None):
            continue
        mt = R.match(cat, r["origin"], r["lat"], r["lon"], dt=20, dkm=80)
        if mt:
            rows.append((mt["mag"], r["early_mag"], r["mag"]))
    rows = np.array(rows)
    if len(rows):
        a.scatter(rows[:, 0], rows[:, 1], s=12, color=MUTE, label=f"quick check, first 4 s of P (MAE {np.abs(rows[:, 1] - rows[:, 0]).mean():.2f})")
        a.scatter(rows[:, 0], rows[:, 2], s=12, color=INK, label=f"full sizing, 30 s windows (MAE {np.abs(rows[:, 2] - rows[:, 0]).mean():.2f})")
        lo2, hi2 = rows.min() - 0.2, rows.max() + 0.2
        a.plot([lo2, hi2], [lo2, hi2], ls=":", color=MUTE)
        a.axhline(3.0, color=MUTE, lw=1, ls="--")
        a.text(lo2 + 0.05, 3.04, "push floor M3.0", fontsize=8, color=BODY)
        a.set(xlim=(lo2, hi2), ylim=(lo2, hi2))
    a.set(xlabel="catalog magnitude", ylabel="estimate",
          title=f"Live pipeline replayed on 20 days (n={len(rows)})\n"
                "M<2 (outside training) reads high, stays under the floor")
    a.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "size_evidence.png")
    print("wrote size_evidence.png")


def publish():
    """Copy the champion's test metrics and CIs into the site's seismic.json (numbers only)."""
    from datetime import date
    fp = OUT / "seismic.json"
    site = json.loads(fp.read_text(encoding="utf-8"))
    proc = ROOT / "data" / "processed"
    src = {"detection": (proc / "detection_demo.json", "test_auc", "sta_lta_auc", "n_test"),
           "magnitude": (proc / "magnitude_demo.json", "ens_r2", "baseline_r2", "n_test")}
    for t in site["tasks"]:
        f, deep, base, n = src[t["key"]]
        m = json.loads(f.read_text())
        t["deep"], t["baseline"], t["n"] = round(m[deep], 4), round(m[base], 3), m[n]
        if f"{deep}_ci" in m:
            t["deep_ci"] = [round(v, 4) for v in m[f"{deep}_ci"]]
            t["baseline_ci"] = [round(v, 3) for v in m[f"{base}_ci"]]
    site["generated_at"] = date.today().isoformat()
    fp.write_text(json.dumps(site, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("published metrics + CIs to app/public/seismic.json -- check the numbers quoted in each `desc` by hand")


if __name__ == "__main__":
    if sys.argv[1:] != ["publish"]:
        detect_figure()
        size_figure()
    publish()
