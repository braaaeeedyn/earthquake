"""Feature tests: ULF-band power detects in-band power; sample assembly shape (no network)."""
import numpy as np

from eq.config import Config
from eq.features import (
    FEATURE_NAMES, BAND_MINUTE, _hourly_features, assemble_feature_samples, feature_grid,
)
from eq.synthetic import generate_synthetic


def test_ulf_power_responds_to_in_band_signal():
    sph, t = 60, np.arange(60) * 60.0  # 60 one-minute samples, seconds
    in_band = 3.0e-3   # within BAND_MINUTE (5e-4 .. 8e-3)
    out_band = 5.0e-5  # below the band
    rows = np.stack([
        np.sin(2 * np.pi * in_band * t),
        np.sin(2 * np.pi * out_band * t),
    ])
    feats = _hourly_features(rows, sph, BAND_MINUTE, min_fraction=0.5)
    assert feats[0, 0] > feats[1, 0]  # in-band hour has more ULF power than out-of-band


def test_missing_hour_is_nan():
    rows = np.full((1, 60), np.nan)
    feats = _hourly_features(rows, 60, BAND_MINUTE, min_fraction=0.5)
    assert np.isnan(feats[0]).all()


def test_feature_sample_shape():
    cfg = Config()
    from dataclasses import replace
    cfg = replace(cfg, n_days=300, stations=cfg.stations[:3])
    readings, catalog, n_days = generate_synthetic(cfg, seed=0)
    grids = feature_grid(readings, cfg, BAND_MINUTE, samples_per_hour=60)
    x, y, anchors = assemble_feature_samples(grids, catalog, cfg)
    assert x.shape[1:] == (3, len(FEATURE_NAMES), 24, 27)  # (S, F, 24, 27)
    assert x.shape[0] == len(y) == len(anchors)
    assert set(np.unique(y)).issubset({0, 1})
