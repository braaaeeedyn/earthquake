"""Phase 3+5 real-data run: INTERMAGNET + USGS, proximity sweep, repeated-seed CIs.

Fetches real geomagnetic data for the western-US cluster and the USGS catalog (cached),
then for each proximity radius {500, 700, 1000} km builds the labeled dataset and evaluates
the fused model across several seeds, reporting mean ± 95% CI vs the 60% baseline. This is
the honest, generalization-facing counterpart to scripts/train_models.py (synthetic).
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eq.config import Config  # noqa: E402
from eq.data.real import build_real_inputs, real_config  # noqa: E402
from eq.pipeline import (  # noqa: E402
    apply_normalizer, assemble_samples, chronological_split, fit_normalizer,
    hourly_to_daily, minute_to_hourly,
)
from eq.models import (  # noqa: E402
    FusedClassifier, SpatialCNN, StationGNN, TemporalTransformer,
    adjacency_tensor, make_loaders, pos_weight,
)
from eq.models.evaluate import format_row, repeated_evaluate  # noqa: E402

BASELINE_ACC = 0.60
PROXIMITIES = (500.0, 700.0, 1000.0)
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 60


def _splits_to_loaders(x, y, anchors, cfg):
    tr, va, te = chronological_split(anchors, cfg)
    stats = fit_normalizer(x[tr])
    data = {
        "train": (apply_normalizer(x[tr], stats), y[tr]),
        "val": (apply_normalizer(x[va], stats), y[va]),
        "test": (apply_normalizer(x[te], stats), y[te]),
    }
    tensors = {k: (torch.tensor(v[0], dtype=torch.float32),
                   torch.tensor(v[1], dtype=torch.float32)) for k, v in data.items()}
    return make_loaders(tensors, batch_size=32), y[tr], y[te]


def main() -> None:
    base = Config()
    print("Fetching real data (INTERMAGNET + USGS; cached after first run)...")
    readings, catalog, n_days, stations = build_real_inputs()
    print(f"Stations: {', '.join(f'{s.code}({s.lat:.1f},{s.lon:.1f})' for s in stations)}")
    print(f"Catalog: {len(catalog)} events M>={base.labeling.magnitude_threshold} "
          f"in region over {n_days} days\n")

    # Preprocess once (shared across proximity radii); only labeling depends on proximity.
    ref_cfg = real_config(base, stations, PROXIMITIES[0], n_days=n_days)
    hourly = minute_to_hourly(readings, ref_cfg.preprocess.min_minute_fraction)
    daily = hourly_to_daily(hourly, ref_cfg.start_date, n_days, stations)
    adj = adjacency_tensor(ref_cfg)
    n_st = len(stations)

    print(f"Baseline to beat: {BASELINE_ACC:.0%} accuracy (Liu et al. 2022)")
    print(f"Fused model, {len(SEEDS)} seeds, mean ± 95% CI on held-out test:\n")

    for prox in PROXIMITIES:
        cfg = real_config(base, stations, prox, n_days=n_days)
        x, y, anchors = assemble_samples(daily, stations, catalog, cfg)
        loaders, y_tr, y_te = _splits_to_loaders(x, y, anchors, cfg)
        pw = pos_weight(torch.tensor(y_tr, dtype=torch.float32))

        make = lambda: FusedClassifier([
            SpatialCNN(n_st), StationGNN(n_st, adj), TemporalTransformer(n_st)
        ])
        agg = repeated_evaluate(make, loaders, seeds=SEEDS, pos_weight=pw, epochs=EPOCHS)

        beats = agg["accuracy"]["mean"] > BASELINE_ACC
        print(f"proximity={prox:.0f} km  (test n={agg['n_test']}, pos={agg['positives']}, "
              f"pos_rate={agg['positives']/agg['n_test']:.2f})")
        print(format_row("Fused", agg))
        print(f"    accuracy {'BEATS' if beats else 'does NOT beat'} baseline; "
              f"acc range [{agg['accuracy']['min']:.3f}, {agg['accuracy']['max']:.3f}] across seeds\n")


if __name__ == "__main__":
    main()
