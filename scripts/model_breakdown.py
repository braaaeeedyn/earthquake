"""Per-backbone breakdown on common-mode features: is one of CNN/GNN/Transformer a weak link?

For each trustworthy cell (large positive count), build the common-mode-residual features once,
then train+evaluate each sub-model standalone (Classifier over one backbone) and the fusion,
all on identical data/seeds. ROC-AUC per model exposes which backbone carries (or drags) the signal.

  python scripts/model_breakdown.py --seeds 3 --epochs 20
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eq.config import Config  # noqa: E402
from eq.data.real import build_region_inputs, region_config  # noqa: E402
from eq.features import BAND_MINUTE, FEATURE_NAMES, assemble_feature_samples, feature_grid  # noqa: E402
from eq.models import (  # noqa: E402
    Classifier, FusedClassifier, SpatialCNN, StationGNN, TemporalTransformer,
    adjacency_tensor, pos_weight,
)
from eq.models.evaluate import repeated_auc  # noqa: E402
import grid_experiment as ge  # noqa: E402
from methodology_experiments import common_mode_residual  # noqa: E402

# (region, proximity_km, magnitude) — only well-sampled cells (n+ >= ~100).
CELLS = [
    ("japan", 500.0, 5.0),
    ("japan", 300.0, 5.0),
    ("california", 500.0, 4.5),
]


def model_makers(n_st, n_f, adj):
    return {
        "CNN":         lambda: Classifier(SpatialCNN(n_st, n_f)),
        "GNN":         lambda: Classifier(StationGNN(n_st, adj, n_f)),
        "Transformer": lambda: Classifier(TemporalTransformer(n_st, n_f)),
        "Fused":       lambda: FusedClassifier([
            SpatialCNN(n_st, n_f), StationGNN(n_st, adj, n_f), TemporalTransformer(n_st, n_f)]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "processed" / "model_breakdown.json"))
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    base = Config()

    results = []
    cache = {}
    for region, prox, mag in CELLS:
        if region not in cache:
            readings, catalog, n_days, stations = build_region_inputs(
                region, min_magnitude=4.5, samples="minute")
            res = common_mode_residual(readings)
            cfg0 = region_config(base, stations, 300.0, 5.0, ge.FULL_START, n_days)
            grids = feature_grid(res, cfg0, BAND_MINUTE, samples_per_hour=60)
            cache[region] = (grids, catalog, n_days, stations, adjacency_tensor(cfg0))
        grids, catalog, n_days, stations, adj = cache[region]
        n_st, n_f = len(stations), len(FEATURE_NAMES)

        cfg = region_config(base, stations, prox, mag, ge.FULL_START, n_days)
        X, y, anchors = assemble_feature_samples(grids, catalog, cfg)
        loaders, y_tr = ge._loaders_for_cell(X, y, anchors, cfg)
        pw = pos_weight(torch.tensor(y_tr, dtype=torch.float32))

        print(f"\n=== {region} M{mag} {prox:.0f}km (common-mode features) ===", flush=True)
        for name, make in model_makers(n_st, n_f, adj).items():
            agg = repeated_auc(make, loaders, seeds=seeds, pos_weight=pw, epochs=args.epochs)
            row = {"region": region, "magnitude": mag, "proximity_km": prox, "model": name,
                   "n_test": agg["n_test"], "positives": agg["positives"],
                   "base_rate": agg["base_rate"],
                   "roc_auc": agg["roc_auc"]["mean"], "roc_ci": agg["roc_auc"]["ci"],
                   "pr_over_base": agg["pr_auc_over_base"]["mean"]}
            results.append(row)
            print(f"  {name:<12} ROC={row['roc_auc']:.3f}±{row['roc_ci']:.3f}  "
                  f"PR/base={row['pr_over_base']:.2f}  (n+={row['positives']})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"seeds": args.seeds, "epochs": args.epochs, "results": results}, indent=2))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
