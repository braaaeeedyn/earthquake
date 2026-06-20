"""Two-region precursor test: California vs Japan, identical methodology.

For each region, computes ULF-band power + |dF/dt| features once, then evaluates a fixed
grid of proximity radii x magnitude cutoffs. Every cell is reported (including nulls) with
ROC-AUC and PR-AUC (mean +/- 95% CI across seeds), PR-AUC vs the positive base rate as the
no-skill reference, and the base rate itself. Nothing is tuned to maximize any metric.

  python scripts/grid_experiment.py                         # both regions, minute, full record
  python scripts/grid_experiment.py --region california --start 2021-01-01 --days 1095 \
      --seeds 2 --epochs 15                                 # quick validation on cached data
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eq.config import Config  # noqa: E402
from eq.data.real import FULL_DAYS, FULL_START, build_region_inputs, region_config  # noqa: E402
from eq.features import (  # noqa: E402
    BAND_MINUTE, BAND_SECOND, FEATURE_NAMES, apply_feature_normalizer,
    assemble_feature_samples, feature_grid, fit_feature_normalizer,
)
from eq.pipeline import chronological_split  # noqa: E402
from eq.models import (  # noqa: E402
    FusedClassifier, SpatialCNN, StationGNN, TemporalTransformer,
    adjacency_tensor, make_loaders, pos_weight,
)
from eq.models.evaluate import repeated_auc  # noqa: E402

GRID_PROX = (300.0, 500.0, 700.0, 1000.0)
GRID_MAG = (4.5, 5.0)


def _loaders_for_cell(X, y, anchors, cfg):
    tr, va, te = chronological_split(anchors, cfg)
    if not (len(tr) and len(va) and len(te)):
        return None
    stats = fit_feature_normalizer(X[tr])
    tensors = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        xn = apply_feature_normalizer(X[idx], stats)
        tensors[name] = (torch.tensor(xn, dtype=torch.float32),
                         torch.tensor(y[idx], dtype=torch.float32))
    return make_loaders(tensors, batch_size=64), y[tr]


def run_region(region, start, n_days, samples, seeds, epochs):
    band = BAND_SECOND if samples == "second" else BAND_MINUTE
    sph = 3600 if samples == "second" else 60
    base = Config()
    readings, catalog, n_days, stations = build_region_inputs(
        region, start=start, n_days=n_days, min_magnitude=4.5, samples=samples)
    feat_cfg = region_config(base, stations, GRID_PROX[0], GRID_MAG[0], start, n_days)
    grids = feature_grid(readings, feat_cfg, band, samples_per_hour=sph)
    adj = adjacency_tensor(feat_cfg)
    n_st, n_f = len(stations), len(FEATURE_NAMES)

    cells = []
    for mag in GRID_MAG:
        for prox in GRID_PROX:
            cfg = region_config(base, stations, prox, mag, start, n_days)
            X, y, anchors = assemble_feature_samples(grids, catalog, cfg)
            prepared = _loaders_for_cell(X, y, anchors, cfg)
            cell = {"region": region, "magnitude": mag, "proximity_km": prox}
            if prepared is None:
                cells.append({**cell, "status": "empty split"})
                continue
            loaders, y_tr = prepared
            pw = pos_weight(torch.tensor(y_tr, dtype=torch.float32))
            make = lambda: FusedClassifier([
                SpatialCNN(n_st, n_f), StationGNN(n_st, adj, n_f), TemporalTransformer(n_st, n_f)
            ])
            agg = repeated_auc(make, loaders, seeds=seeds, pos_weight=pw, epochs=epochs)
            if agg["positives"] == 0:
                cells.append({**cell, "status": "no positives in test",
                              "n_test": agg["n_test"], "base_rate": 0.0})
                continue
            cells.append({
                **cell, "status": "ok",
                "n_test": agg["n_test"], "positives": agg["positives"],
                "base_rate": agg["base_rate"],
                "roc_auc": agg["roc_auc"]["mean"], "roc_ci": agg["roc_auc"]["ci"],
                "pr_auc": agg["pr_auc"]["mean"], "pr_ci": agg["pr_auc"]["ci"],
                "pr_over_base": agg["pr_auc_over_base"]["mean"],
                "pr_curve": _pr_curve(agg["y_test"], agg["mean_prob"]),
            })
            c = cells[-1]
            print(f"  [{region} M{mag} {prox:.0f}km] base={c['base_rate']:.3f} "
                  f"ROC-AUC={c['roc_auc']:.3f}±{c['roc_ci']:.3f} "
                  f"PR-AUC={c['pr_auc']:.3f}±{c['pr_ci']:.3f} "
                  f"PR/base={c['pr_over_base']:.2f}", flush=True)
    return cells


def _pr_curve(y, prob, n=60):
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y, prob)
    idx = np.linspace(0, len(p) - 1, min(n, len(p))).astype(int)
    return {"precision": p[idx].tolist(), "recall": r[idx].tolist()}


def _fmt(cell):
    if cell.get("status") != "ok":
        return f"{cell.get('status', '-'):>22}"
    return (f"ROC {cell['roc_auc']:.3f}±{cell['roc_ci']:.3f}  "
            f"PR {cell['pr_auc']:.3f}/{cell['base_rate']:.3f}={cell['pr_over_base']:.2f}")


def print_comparison(by_region):
    regions = list(by_region)
    print("\n" + "=" * 100)
    print("PRECURSOR SIGNAL: ROC-AUC and PR-AUC(/base rate) — no skill = ROC 0.5, PR/base 1.0")
    print("=" * 100)
    header = f"{'M':>4} {'prox':>6}  " + "  ".join(f"{r.upper():^40}" for r in regions)
    print(header)
    print("-" * len(header))
    lookup = {r: {(c["magnitude"], c["proximity_km"]): c for c in by_region[r]} for r in regions}
    for mag in GRID_MAG:
        for prox in GRID_PROX:
            row = f"{mag:>4} {prox:>6.0f}  "
            row += "  ".join(f"{_fmt(lookup[r].get((mag, prox), {})):^40}" for r in regions)
            print(row)
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["california", "japan", "both"], default="both")
    ap.add_argument("--samples", choices=["minute", "second"], default="minute")
    ap.add_argument("--start", default=FULL_START)
    ap.add_argument("--days", type=int, default=FULL_DAYS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "grid_results.json"))
    args = ap.parse_args()

    regions = ["california", "japan"] if args.region == "both" else [args.region]
    seeds = tuple(range(args.seeds))
    by_region = {}
    for region in regions:
        print(f"\n=== {region} ({args.samples}, {args.days}d from {args.start}) ===", flush=True)
        by_region[region] = run_region(region, args.start, args.days, args.samples, seeds, args.epochs)

    print_comparison(by_region)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"samples": args.samples, "start": args.start, "days": args.days,
         "seeds": args.seeds, "epochs": args.epochs, "regions": by_region}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
