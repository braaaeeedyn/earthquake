"""Train + evaluate the v2 multi-station MAGNITUDE ensemble (data/processed/v2/magnitude.npz).

Training geometry == live geometry (see URGENT_PLAN.md section 2):
  - every station window starts 5 s before its PICKED P (same picker as live), 30 s long;
  - input representation: each station's 3-C window is normalized to unit peak (the CNN reads SHAPE)
    and its standardized log10 peak velocity is a graph-node feature next to log-distance (the graph
    reads SIZE). A single global amplitude scale (v1) left near-field M5-7 records thousands of times
    larger than distant M3s and trained unstably on the larger v2 data;
  - augmentation reproduces live uncertainty: epicentre jittered by a locator-like error (distances
    recomputed), +-0.5 s pick jitter, and station subsets (only the nearest k, or random dropout),
    because live sizes with whichever stations have delivered P+25 s;
  - every normalizer and the station list are stored IN the checkpoint (no hand-kept SCALE constant).

Reports (chronological 70/15/15 by event time): ensemble R^2/MAE vs the amp+distance baseline, the
nearest-1-station ablation, and a LIVE-LIKE score (10 km location error, nearest 3-6 stations).

  python scripts/demo_magnitude.py --retrain --seeds 5 --epochs 40
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
from sklearn.linear_model import LinearRegression  # noqa: E402
from seismic_train_multi import MultiStationModel, adjacency  # noqa: E402

from eq import locate, network  # noqa: E402

NPZ = ROOT / "data" / "processed" / "v2" / "magnitude.npz"
CKPT = ROOT / "data" / "processed" / "magnitude_ensemble.pt"
SUMMARY = ROOT / "data" / "processed" / "magnitude_demo.json"
FIG = ROOT / "figures" / "magnitude_demo.png"
NPTS, OFF = 3000, 100                          # stored rows are P-6 s .. P+26 s; P-5 s is offset 100


def split_chrono(t, fr=(0.7, 0.15)):
    o = np.argsort(t, kind="stable")
    n = len(o)
    return o[:int(n * fr[0])], o[int(n * fr[0]):int(n * (fr[0] + fr[1]))], o[int(n * (fr[0] + fr[1])):]


def build_model(Ahat, device):
    return MultiStationModel(Ahat, hybrid=True, amp_feature=True).to(device)


class Data:
    def __init__(self, device):
        d = np.load(NPZ)
        self.mag = d["mag"].astype(np.float32)
        self.ev_t, self.ev_lat, self.ev_lon = d["ev_time"], d["ev_lat"], d["ev_lon"]
        self.row_ev, self.row_sta = d["row_ev"], d["row_sta"]
        assert list(d["stations"]) == network.CODES, "dataset built on a different network"
        self.coords = network.COORDS
        self.S = len(network.CODES)
        self.rows_of = [np.flatnonzero(self.row_ev == e) for e in range(len(self.mag))]
        x = d["x"].astype(np.float32) * d["s"][..., None]                # (R, 3, 3200) m/s
        peak = np.abs(x[:, :, OFF:OFF + NPTS]).max(axis=(1, 2)) + 1e-12
        self.logpeak = np.log10(peak).astype(np.float32)                  # per-row log10 peak velocity
        self.x = torch.tensor(x / peak[:, None, None], dtype=torch.float32, device=device)   # unit peak
        self.device = device

    def batch(self, evs, rng=None, loc_err_km=0.0, keep=None, jitter=0):
        """Dense (B,S,3,NPTS) unit-peak windows, mask, distances (from a possibly-jittered epicentre)
        and per-station log10 peak velocity."""
        B = len(evs)
        X = torch.zeros((B, self.S, 3, NPTS), device=self.device)
        M = np.zeros((B, self.S), bool)
        D = np.zeros((B, self.S), np.float32)
        LA = np.zeros((B, self.S), np.float32)
        for b, e in enumerate(evs):
            lat, lon = self.ev_lat[e], self.ev_lon[e]
            if loc_err_km > 0:
                g = rng if rng is not None else np.random.default_rng(int(e))
                dy, dx = g.normal(0, loc_err_km, 2)
                lat, lon = lat + dy / 111.0, lon + dx / (111.0 * np.cos(np.radians(lat)))
            D[b] = locate.haversine_km(lat, lon, self.coords[:, 0], self.coords[:, 1])
            rows = self.rows_of[e]
            stas = self.row_sta[rows]
            sel = np.ones(len(rows), bool)
            if keep == "train" and rng is not None and len(rows) > 3:
                if rng.random() < 0.5:                                    # nearest k only (live sizing)
                    k = rng.integers(3, len(rows) + 1)
                    sel[:] = False
                    sel[np.argsort(D[b][stas])[:k]] = True
                else:                                                     # random dropout, keep >= 3
                    sel = rng.random(len(rows)) > 0.2
                    if sel.sum() < 3:
                        sel[rng.permutation(len(rows))[:3]] = True
            elif isinstance(keep, int):
                sel[:] = False
                sel[np.argsort(D[b][stas])[:keep]] = True
            for r, s in zip(rows[sel], stas[sel]):
                o = OFF + (int(rng.integers(-jitter, jitter + 1)) if (jitter and rng is not None) else 0)
                X[b, s] = self.x[r, :, o:o + NPTS]
                M[b, s] = True
                LA[b, s] = self.logpeak[r]
        return X, M, D, LA


def aux_feats(M, D, LA):
    """Network amp+dist features [mean/max log peak velocity, mean/min log distance] per event."""
    ld = np.log10(np.maximum(D, 1.0))
    return np.asarray([[LA[b][m].mean(), LA[b][m].max(), ld[b][m].mean(), ld[b][m].min()]
                       for b, m in enumerate(M)], np.float32)


def forward(model, X, M, D, LA, norms, device):
    A = (aux_feats(M, D, LA) - norms["am"]) / norms["asd"]
    logd = np.where(D > 0, np.log10(np.maximum(D, 1.0)), 0.0).astype(np.float32)
    las = ((LA - norms["la_mu"]) / norms["la_sd"] * M).astype(np.float32)
    t = lambda a: torch.tensor(a, device=device)  # noqa: E731
    return model(X, t(M), t(logd), t(A.astype(np.float32)), t(las))


def run(model, data, evs, norms, **kw):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(evs), 64):
            out.append(forward(model, *data.batch(evs[i:i + 64], **kw), norms, data.device).cpu().numpy())
    return np.concatenate(out)


def r2_mae(p, y):
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)), float(np.mean(np.abs(p - y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--out", default=str(CKPT))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Data(device)
    tr, va, te = split_chrono(data.ev_t)
    y = data.mag
    print(f"events {len(y)} (train {len(tr)} / val {len(va)} / test {len(te)}), rows {len(data.row_ev)}, "
          f"M {y.min():.1f}..{y.max():.1f}, device {device}")
    Ahat = adjacency(data.coords.astype(np.float32))

    if args.retrain or not Path(args.out).exists():
        tr_rows = np.isin(data.row_ev, tr)
        norms = {"la_mu": float(data.logpeak[tr_rows].mean()), "la_sd": float(data.logpeak[tr_rows].std() + 1e-6)}
        feats = np.concatenate([aux_feats(*data.batch(tr[i:i + 64])[1:]) for i in range(0, len(tr), 64)])
        norms["am"], norms["asd"] = feats.mean(0), feats.std(0) + 1e-6
        states, val_r2 = [], []
        for s in range(args.seeds):
            torch.manual_seed(s)
            rng = np.random.default_rng(s)
            model = build_model(Ahat, device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
            for _ in range(args.epochs):
                model.train()
                for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 32)):
                    batch = data.batch(b, rng=rng, loc_err_km=8.0, keep="train", jitter=50)
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(forward(model, *batch, norms, device),
                                                  torch.tensor(y[b], device=device))
                    loss.backward()
                    opt.step()
                sched.step()
            val_r2.append(r2_mae(run(model, data, va, norms), y[va])[0])
            print(f"  seed {s}: val R2 {val_r2[-1]:+.3f}", flush=True)
            states.append({k: v.cpu() for k, v in model.state_dict().items()})
        torch.save({"states": states, **norms, "input": "unit-peak+logamp", "stations": network.CODES,
                    "coords": data.coords, "trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                          text=True).stdout.strip(),
                    "dataset": json.loads((NPZ.parent / "dataset_meta.json").read_text())}, args.out)
        summary = {"seeds": args.seeds, "epochs": args.epochs, "seed_val_r2": val_r2}
    else:
        summary = json.loads(SUMMARY.read_text())

    ck = torch.load(args.out, weights_only=False, map_location=device)
    norms = {k: ck[k] for k in ("la_mu", "la_sd", "am", "asd")}
    models = []
    for st in ck["states"]:
        m = build_model(Ahat, device)
        m.load_state_dict(st)
        models.append(m)

    def ens(evs, **kw):
        return np.mean([run(m, data, evs, norms, **kw) for m in models], axis=0)

    yte = y[te]
    p_all = ens(te)
    p_near = ens(te, keep=1)
    rng_live = np.random.default_rng(123)
    p_live = np.concatenate([ens(np.array([e]), loc_err_km=10.0, keep=int(rng_live.integers(3, 7)))
                             for e in te])
    singles = [r2_mae(run(m, data, te, norms), yte)[0] for m in models]
    # amp+distance baseline (true distances, all stations), fit on train
    ftr = np.concatenate([aux_feats(*data.batch(tr[i:i + 64])[1:]) for i in range(0, len(tr), 64)])
    fte = np.concatenate([aux_feats(*data.batch(te[i:i + 64])[1:]) for i in range(0, len(te), 64)])
    base = LinearRegression().fit(ftr, y[tr]).predict(fte)
    r = {k: r2_mae(p, yte) for k, p in (("ensemble", p_all), ("near1", p_near), ("live_like", p_live),
                                          ("baseline", base))}
    print(f"\n=== MAGNITUDE, chronological test (n={len(te)} events, M {yte.min():.1f}..{yte.max():.1f}) ===")
    print(f"  deep ensemble (all stations)       R2 {r['ensemble'][0]:+.3f}  MAE {r['ensemble'][1]:.3f}")
    print(f"  single models                      R2 {np.mean(singles):+.3f} +/- {np.std(singles):.3f}")
    print(f"  LIVE-LIKE (10 km loc err, 3-6 sta) R2 {r['live_like'][0]:+.3f}  MAE {r['live_like'][1]:.3f}"
          f"  bias {np.mean(p_live - yte):+.2f}")
    print(f"  nearest-1-station ablation         R2 {r['near1'][0]:+.3f}  MAE {r['near1'][1]:.3f}")
    print(f"  amp+dist baseline                  R2 {r['baseline'][0]:+.3f}  MAE {r['baseline'][1]:.3f}")
    summary.update({"n_test": int(len(te)), "ens_r2": r["ensemble"][0], "ens_mae": r["ensemble"][1],
                    "single_r2": singles, "live_like_r2": r["live_like"][0], "live_like_mae": r["live_like"][1],
                    "live_like_bias": float(np.mean(p_live - yte)), "ablation_near_r2": r["near1"][0],
                    "baseline_r2": r["baseline"][0], "baseline_mae": r["baseline"][1],
                    "mag_range": [float(yte.min()), float(yte.max())]})
    SUMMARY.write_text(json.dumps(summary, indent=2))
    make_figure(yte, p_all, base, r["ensemble"][0], r["baseline"][0])
    print(f"  wrote {FIG.relative_to(ROOT)} and {SUMMARY.relative_to(ROOT)}")


def make_figure(yte, ens, base, r2_e, r2_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    DEEP, BASE = "#111111", "#888780"
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    lo, hi = yte.min() - 0.2, yte.max() + 0.2
    for a, pred, r2, name, c in ((ax[0], ens, r2_e, "deep ensemble", DEEP),
                                 (ax[1], base, r2_b, "amp+dist baseline", BASE)):
        a.scatter(yte, pred, s=14, alpha=0.5, color=c, edgecolor="none")
        a.plot([lo, hi], [lo, hi], "k--", lw=0.9)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel("catalog magnitude"); a.set_ylabel("predicted magnitude")
        a.set_title(f"{name}   R²={r2:+.3f}", fontweight="bold")
    fig.suptitle("Network magnitude on the live network (held-out SoCal events, 2000-2026 data)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)


if __name__ == "__main__":
    main()
