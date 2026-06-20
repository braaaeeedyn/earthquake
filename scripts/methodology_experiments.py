"""Try each proposed methodology change individually, vs a matched baseline.

Each variant is compared against a baseline run at the SAME seeds/epochs/cells, so any
difference is the change itself and not a budget difference. This is exploratory (fewer
seeds/epochs than scripts/grid_experiment.py), so treat ROC-AUC moves as suggestive, not
confirmatory — small positive counts make single cells noisy by construction.

Variants:
  commonmode : subtract the cross-station common-mode (planetary field) before features,
               leaving the local anomaly. Run raw vs residual on identical cells.
  closeprox  : tighter proximity radii (50-300 km) anchored on the station cluster.
  secondres  : 1-second ULF band vs 1-minute, Japan 2018 (only second-data we have cached).
  bigshort   : bigger quakes (M5.5) and a shorter 3-day horizon vs M5.0 / 7-day.

  python scripts/methodology_experiments.py --seeds 3 --epochs 20
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eq.config import Config  # noqa: E402
from eq.data.real import build_region_inputs, region_config  # noqa: E402
from eq.features import (  # noqa: E402
    BAND_MINUTE, BAND_SECOND, FEATURE_NAMES, assemble_feature_samples, feature_grid,
)
from eq.models import (  # noqa: E402
    FusedClassifier, SpatialCNN, StationGNN, TemporalTransformer,
    adjacency_tensor, pos_weight,
)
from eq.models.evaluate import repeated_auc  # noqa: E402
import grid_experiment as ge  # noqa: E402  (reuse _loaders_for_cell)


def common_mode_residual(readings: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Remove the globally-coherent field: per station, subtract its baseline, then the
    cross-station mean deviation (the planetary common-mode). Leaves the local residual.

    ULF power and |dF/dt| are offset-invariant, so removing the per-station baseline is
    harmless; removing the common-mode is the point.
    """
    df = pd.DataFrame(readings).astype("float32")
    dev = df.sub(df.median())
    common = dev.mean(axis=1, skipna=True)
    resid = dev.sub(common, axis=0)
    return {c: resid[c].dropna() for c in df.columns}


def run_cell(region, grids, catalog, adj, stations, start, n_days, prox, mag, horizon,
             seeds, epochs, tag):
    """Train+evaluate the fused model on one labeling cell; return a result dict."""
    base = Config()
    cfg = region_config(base, stations, prox, mag, start, n_days)
    if horizon is not None:
        cfg = replace(cfg, labeling=replace(cfg.labeling, horizon_days=horizon))
    n_st, n_f = len(stations), len(FEATURE_NAMES)
    cell = {"variant": tag, "region": region, "magnitude": mag, "proximity_km": prox,
            "horizon": cfg.labeling.horizon_days}
    try:
        X, y, anchors = assemble_feature_samples(grids, catalog, cfg)
    except ValueError as e:
        return {**cell, "status": f"no samples: {e}"}
    prepared = ge._loaders_for_cell(X, y, anchors, cfg)
    if prepared is None:
        return {**cell, "status": "empty split"}
    loaders, y_tr = prepared
    pw = pos_weight(torch.tensor(y_tr, dtype=torch.float32))
    make = lambda: FusedClassifier([
        SpatialCNN(n_st, n_f), StationGNN(n_st, adj, n_f), TemporalTransformer(n_st, n_f)])
    agg = repeated_auc(make, loaders, seeds=tuple(range(seeds)), pos_weight=pw, epochs=epochs)
    if agg["positives"] == 0:
        return {**cell, "status": "no positives in test", "n_test": agg["n_test"]}
    out = {**cell, "status": "ok", "n_test": agg["n_test"], "positives": agg["positives"],
           "base_rate": agg["base_rate"],
           "roc_auc": agg["roc_auc"]["mean"], "roc_ci": agg["roc_auc"]["ci"],
           "pr_auc": agg["pr_auc"]["mean"], "pr_over_base": agg["pr_auc_over_base"]["mean"]}
    print(f"  [{tag}/{region} M{mag} {prox:.0f}km h{out['horizon']}] "
          f"base={out['base_rate']:.3f} n+={out['positives']} "
          f"ROC={out['roc_auc']:.3f}±{out['roc_ci']:.3f} PR/base={out['pr_over_base']:.2f}",
          flush=True)
    return out


def _grids_adj(region, readings, stations, start, n_days, band, sph):
    base = Config()
    cfg0 = region_config(base, stations, 300.0, 5.0, start, n_days)
    grids = feature_grid(readings, cfg0, band, samples_per_hour=sph)
    return grids, adjacency_tensor(cfg0)


def variant_commonmode(seeds, epochs):
    """Raw vs common-mode-residual features on identical cells, both regions."""
    cells = []
    plan = {
        "california": [(150.0, 4.5), (300.0, 4.5), (500.0, 4.5), (300.0, 5.0)],
        "japan": [(300.0, 5.0), (500.0, 5.0)],
    }
    for region, grid in plan.items():
        readings, catalog, n_days, stations = build_region_inputs(
            region, min_magnitude=4.5, samples="minute")
        raw_grids, adj = _grids_adj(region, readings, stations,
                                    ge.FULL_START, n_days, BAND_MINUTE, 60)
        res_grids, _ = _grids_adj(region, common_mode_residual(readings), stations,
                                  ge.FULL_START, n_days, BAND_MINUTE, 60)
        for prox, mag in grid:
            cells.append(run_cell(region, raw_grids, catalog, adj, stations, ge.FULL_START,
                                  n_days, prox, mag, None, seeds, epochs, "raw"))
            cells.append(run_cell(region, res_grids, catalog, adj, stations, ge.FULL_START,
                                  n_days, prox, mag, None, seeds, epochs, "commonmode"))
    return cells


def variant_closeprox(seeds, epochs):
    """Tighter proximity radii on California (standard features)."""
    readings, catalog, n_days, stations = build_region_inputs(
        "california", min_magnitude=4.5, samples="minute")
    grids, adj = _grids_adj("california", readings, stations,
                            ge.FULL_START, n_days, BAND_MINUTE, 60)
    cells = []
    for mag, proxes in ((4.5, (50.0, 100.0, 150.0, 200.0, 300.0)), (5.0, (100.0, 150.0, 300.0))):
        for prox in proxes:
            cells.append(run_cell("california", grids, catalog, adj, stations, ge.FULL_START,
                                  n_days, prox, mag, None, seeds, epochs, "closeprox"))
    return cells


def variant_secondres(seeds, epochs):
    """1-second ULF band vs 1-minute, Japan 2018 (the only second-data cached)."""
    start, n_days = "2018-01-01", 365
    cells = []
    for tag, samples, band, sph in (("minute2018", "minute", BAND_MINUTE, 60),
                                    ("second2018", "second", BAND_SECOND, 3600)):
        readings, catalog, nd, stations = build_region_inputs(
            "japan", start=start, n_days=n_days, min_magnitude=4.5, samples=samples)
        grids, adj = _grids_adj("japan", readings, stations, start, nd, band, sph)
        for prox in (300.0, 500.0):
            cells.append(run_cell("japan", grids, catalog, adj, stations, start, nd,
                                  prox, 5.0, None, seeds, epochs, tag))
    return cells


def variant_bigshort(seeds, epochs):
    """Bigger quakes (M5.5) and shorter horizon (3d) vs M5.0 / 7d, California."""
    readings, catalog, n_days, stations = build_region_inputs(
        "california", min_magnitude=4.5, samples="minute")
    grids, adj = _grids_adj("california", readings, stations,
                            ge.FULL_START, n_days, BAND_MINUTE, 60)
    cells = []
    specs = [(300.0, 5.0, 7, "base_m5h7"), (500.0, 5.0, 7, "base_m5h7"),
             (300.0, 5.0, 3, "short_h3"), (300.0, 5.5, 3, "big_m55_h3"),
             (500.0, 5.5, 3, "big_m55_h3")]
    for prox, mag, hor, tag in specs:
        cells.append(run_cell("california", grids, catalog, adj, stations, ge.FULL_START,
                              n_days, prox, mag, hor, seeds, epochs, tag))
    return cells


VARIANTS = {
    "commonmode": variant_commonmode,
    "closeprox": variant_closeprox,
    "secondres": variant_secondres,
    "bigshort": variant_bigshort,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "processed" / "methodology_results.json"))
    args = ap.parse_args()

    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    results = {}
    for name in names:
        print(f"\n=== variant: {name} (seeds={args.seeds}, epochs={args.epochs}) ===", flush=True)
        results[name] = VARIANTS[name](args.seeds, args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"seeds": args.seeds, "epochs": args.epochs, "variants": results}, indent=2))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
