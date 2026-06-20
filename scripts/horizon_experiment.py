"""Shorter forecast horizon: does a 1-day "quake tomorrow?" question carry more signal than 7-day?

Sweeps horizon 1/3/7 days on the best config (scalar common-mode, standard fusion) for the
well-sampled cells. Concentrating the label near the event (pro) competes with ~7x fewer
positives (con); Japan is over-saturated at 7d so a short horizon should de-saturate it.
Reports base rate, positives, ROC-AUC, balanced accuracy, and Brier skill vs climatology.

  python scripts/horizon_experiment.py --seeds 3 --epochs 20
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

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

CELLS = [("japan", 500.0, 5.0), ("japan", 300.0, 5.0), ("california", 500.0, 4.5)]
HORIZONS = (1, 3, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "processed" / "horizon_results.json"))
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    base = Config()

    results, cache = [], {}
    for region, prox, mag in CELLS:
        if region not in cache:
            readings, catalog, n_days, stations = build_region_inputs(region, min_magnitude=4.5)
            res = common_mode_residual(readings)
            cfg0 = region_config(base, stations, 300.0, 5.0, FULL_START, n_days)
            grids = feature_grid(res, cfg0, BAND_MINUTE, samples_per_hour=60)
            cache[region] = (grids, catalog, n_days, stations, adjacency_tensor(cfg0))
        grids, catalog, n_days, stations, adj = cache[region]
        n_st, n_f = len(stations), len(FEATURE_NAMES)
        make = lambda: FusedClassifier([
            SpatialCNN(n_st, n_f), StationGNN(n_st, adj, n_f), TemporalTransformer(n_st, n_f)])

        print(f"\n=== {region} M{mag} {prox:.0f}km — horizon sweep (scalar common-mode) ===", flush=True)
        for h in HORIZONS:
            cfg = region_config(base, stations, prox, mag, FULL_START, n_days)
            cfg = replace(cfg, labeling=replace(cfg.labeling, horizon_days=h))
            X, y, anchors = assemble_feature_samples(grids, catalog, cfg)
            prepared = ge._loaders_for_cell(X, y, anchors, cfg)
            if prepared is None:
                print(f"  h={h}d: empty split", flush=True)
                continue
            loaders, y_tr = prepared
            pw = pos_weight(torch.tensor(y_tr, dtype=torch.float32))
            m = evaluate_full(make, loaders, seeds, pw, args.epochs)
            row = {"region": region, "magnitude": mag, "proximity_km": prox, "horizon": h,
                   "n_samples": int(len(y)), "n_pos": int(y.sum()),
                   "base_rate": m["base_rate"], "majority_acc": m["majority_acc"],
                   **{k: {"mean": m[k][0], "ci": m[k][1]} for k in
                      ("roc", "bal_acc", "brier_ss", "recall", "precision", "f1")}}
            results.append(row)
            print(f"  h={h}d  base={row['base_rate']:.3f} n+={row['n_pos']}  "
                  f"ROC={m['roc'][0]:.3f}±{m['roc'][1]:.3f}  bal_acc={m['bal_acc'][0]:.3f}  "
                  f"BrierSS={m['brier_ss'][0]:+.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"seeds": args.seeds, "epochs": args.epochs,
                                          "results": results}, indent=2))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
