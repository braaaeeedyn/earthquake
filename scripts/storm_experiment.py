"""Lever 3 (separate test): geomagnetic-storm masking via real Kp.

Hypothesis to check, not assume: common-mode subtraction already removes the planetary field
(mostly the storm signal), so restricting to quiet-time windows may add little on top. For the
best config (scalar common-mode, standard fusion), compare the full sample set against the
quiet-only subset (input windows with few high-Kp days), reporting n for both.

  python scripts/storm_experiment.py --seeds 3 --epochs 20
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eq.config import Config  # noqa: E402
from eq.data.real import FULL_START, build_region_inputs, region_config  # noqa: E402
from eq.features import BAND_MINUTE, FEATURE_NAMES, assemble_feature_samples, feature_grid  # noqa: E402
from eq.models import (  # noqa: E402
    FusedClassifier, SpatialCNN, StationGNN, TemporalTransformer, adjacency_tensor, pos_weight,
)
import grid_experiment as ge  # noqa: E402
from methodology_experiments import common_mode_residual  # noqa: E402
from vector_experiment import evaluate_full  # noqa: E402

CELLS = [("california", 500.0, 4.5), ("japan", 500.0, 5.0)]
KP_STORM = 4.0          # daily max Kp >= this = geomagnetically active/disturbed day
QUIET_MAX_FRAC = 0.25   # keep windows with < this fraction of disturbed days
WINDOW = 27
_KP_CACHE = Path(__file__).resolve().parents[1] / "data" / "raw" / "kp_daily.csv"


def fetch_kp_daily(start: str, n_days: int) -> np.ndarray:
    """Daily max Kp aligned to day index 0..n_days-1 (NaN where missing); cached to CSV."""
    if _KP_CACHE.exists():
        s = pd.read_csv(_KP_CACHE, parse_dates=["date"]).set_index("date")["kp_max"]
    else:
        end = (pd.Timestamp(start) + pd.Timedelta(days=n_days)).strftime("%Y-%m-%d")
        url = (f"https://kp.gfz-potsdam.de/kpdata?startdate={start}&enddate={end}&format=kp1")
        with urllib.request.urlopen(url, timeout=120) as r:
            txt = r.read().decode("utf-8", "replace")
        recs = {}
        for ln in txt.splitlines():
            f = ln.split()
            if len(f) < 15 or not f[0].isdigit():
                continue
            date = pd.Timestamp(f"{f[0]}-{f[1]}-{f[2]}")
            recs[date] = max(float(v) for v in f[7:15])
        s = pd.Series(recs, name="kp_max").sort_index()
        _KP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        s.rename_axis("date").reset_index().to_csv(_KP_CACHE, index=False)
    base = pd.Timestamp(start)
    idx = pd.date_range(base, periods=n_days, freq="D")
    return s.reindex(idx).to_numpy(dtype=float)


def disturbed_fraction(anchors: np.ndarray, kp_daily: np.ndarray) -> np.ndarray:
    """Fraction of each anchor's 27-day input window that is geomagnetically disturbed."""
    storm = (kp_daily >= KP_STORM).astype(float)          # NaN>=x -> 0 (treated as quiet)
    csum = np.concatenate([[0.0], np.cumsum(storm)])
    return np.array([(csum[t + 1] - csum[t - WINDOW + 1]) / WINDOW for t in anchors])


def run_cell(region, prox, mag, seeds, epochs, cache):
    base = Config()
    if region not in cache:
        readings, catalog, n_days, stations = build_region_inputs(region, min_magnitude=4.5)
        res = common_mode_residual(readings)
        cfg0 = region_config(base, stations, 300.0, 5.0, FULL_START, n_days)
        grids = feature_grid(res, cfg0, BAND_MINUTE, samples_per_hour=60)
        kp = fetch_kp_daily(FULL_START, n_days)
        cache[region] = (grids, catalog, n_days, stations, adjacency_tensor(cfg0), kp)
    grids, catalog, n_days, stations, adj, kp = cache[region]
    n_st, n_f = len(stations), len(FEATURE_NAMES)
    cfg = region_config(base, stations, prox, mag, FULL_START, n_days)
    X, y, anchors = assemble_feature_samples(grids, catalog, cfg)
    frac = disturbed_fraction(anchors, kp)
    quiet = frac < QUIET_MAX_FRAC

    make = lambda: FusedClassifier([
        SpatialCNN(n_st, n_f), StationGNN(n_st, adj, n_f), TemporalTransformer(n_st, n_f)])
    rows = []
    for label, mask in (("all", np.ones(len(y), bool)), ("quiet", quiet)):
        Xs, ys, ans = X[mask], y[mask], anchors[mask]
        prepared = ge._loaders_for_cell(Xs, ys, ans, cfg)
        if prepared is None:
            print(f"  {region} M{mag} {prox:.0f}km [{label}] empty split (n={mask.sum()})", flush=True)
            continue
        loaders, y_tr = prepared
        pw = pos_weight(torch.tensor(y_tr, dtype=torch.float32))
        m = evaluate_full(make, loaders, seeds, pw, epochs)
        row = {"region": region, "magnitude": mag, "proximity_km": prox, "subset": label,
               "n_samples": int(mask.sum()), "n_pos": int(ys.sum()),
               "base_rate": m["base_rate"], "majority_acc": m["majority_acc"],
               **{k: {"mean": m[k][0], "ci": m[k][1]} for k in
                  ("roc", "precision", "recall", "f1", "accuracy", "bal_acc")}}
        rows.append(row)
        print(f"  {region} M{mag} {prox:.0f}km [{label:>5}] n={row['n_samples']} pos={row['n_pos']} "
              f"ROC={m['roc'][0]:.3f}±{m['roc'][1]:.3f} bal_acc={m['bal_acc'][0]:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "processed" / "storm_results.json"))
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))

    results, cache = [], {}
    for region, prox, mag in CELLS:
        print(f"\n=== {region} M{mag} {prox:.0f}km — storm masking (Kp>= {KP_STORM}) ===", flush=True)
        results.extend(run_cell(region, prox, mag, seeds, args.epochs, cache))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"seeds": args.seeds, "epochs": args.epochs,
                                          "kp_storm": KP_STORM, "quiet_max_frac": QUIET_MAX_FRAC,
                                          "results": results}, indent=2))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
