"""Phase 1 pipeline tests: shapes, determinism, leakage-safety, labeling, normalization."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from eq.config import Config
from eq import pipeline as P


def tiny_cfg() -> Config:
    """Smaller/faster config that still survives the 34-day embargo on all splits."""
    base = Config()
    return replace(base, n_days=300, stations=base.stations[:3])


# --- Step 1: minute -> hourly ------------------------------------------------
def test_hourly_average_is_correct():
    idx = pd.date_range("2020-01-01", periods=120, freq="1min")
    vals = np.concatenate([np.full(60, 10.0), np.full(60, 20.0)])
    hourly = P.minute_to_hourly({"S1": pd.Series(vals, index=idx)}, 0.5).sort_values("time")
    assert np.isclose(hourly["F"].iloc[0], 10.0)
    assert np.isclose(hourly["F"].iloc[1], 20.0)


def test_sparse_hour_is_marked_missing():
    idx = pd.date_range("2020-01-01", periods=20, freq="1min")  # only 20 of 60 minutes
    hourly = P.minute_to_hourly({"S1": pd.Series(np.ones(20), index=idx)}, 0.5)
    assert np.isnan(hourly["F"].iloc[0])


# --- Step 4: assembled sample shapes + determinism ---------------------------
def test_dataset_shapes():
    cfg = tiny_cfg()
    data = P.build_dataset(cfg, seed=0)
    x_tr, y_tr, a_tr = data["train"]
    s = len(cfg.stations)
    assert x_tr.shape[1:] == (s, 24, 27)
    assert x_tr.shape[0] == y_tr.shape[0] == a_tr.shape[0]
    assert set(np.unique(y_tr)).issubset({0, 1})


def test_determinism():
    cfg = tiny_cfg()
    d1 = P.build_dataset(cfg, seed=0)
    d2 = P.build_dataset(cfg, seed=0)
    np.testing.assert_array_equal(d1["train"][0], d2["train"][0])
    np.testing.assert_array_equal(d1["train"][1], d2["train"][1])
    np.testing.assert_array_equal(d1["test"][2], d2["test"][2])


# --- Step 5: chronological splits, no leakage --------------------------------
def test_chronological_split_no_leakage():
    cfg = tiny_cfg()
    data = P.build_dataset(cfg, seed=1)
    _, _, a_tr = data["train"]
    _, _, a_va = data["val"]
    _, _, a_te = data["test"]
    w, h = cfg.labeling.window_days, cfg.labeling.horizon_days

    # Splits are disjoint in anchor day...
    assert set(a_tr).isdisjoint(a_va)
    assert set(a_va).isdisjoint(a_te)
    assert set(a_tr).isdisjoint(a_te)
    # ...and temporally ordered with no overlapping coverage (label end < next input start).
    if len(a_tr) and len(a_va):
        assert a_tr.max() + h < a_va.min() - (w - 1)
    if len(a_va) and len(a_te):
        assert a_va.max() + h < a_te.min() - (w - 1)
    # All three splits are non-empty under tiny_cfg.
    assert len(a_tr) > 0 and len(a_va) > 0 and len(a_te) > 0


# --- Step 3: labeling rules filter magnitude + proximity ---------------------
def test_labeling_filters_magnitude_and_proximity():
    cfg = Config()
    base = pd.Timestamp(cfg.start_date)
    st = cfg.stations[0]
    catalog = pd.DataFrame([
        {"time": base + pd.Timedelta(days=10), "lat": st.lat, "lon": st.lon, "mag": 5.5},        # qualifies
        {"time": base + pd.Timedelta(days=20), "lat": st.lat, "lon": st.lon, "mag": 3.0},        # too weak
        {"time": base + pd.Timedelta(days=30), "lat": st.lat + 20, "lon": st.lon + 40, "mag": 6.0},  # too far
    ])
    days = P.positive_event_days(catalog, cfg.stations, cfg)
    assert 10 in days
    assert 20 not in days
    assert 30 not in days


# --- Step 6: normalization uses train statistics -----------------------------
def test_train_is_standardized():
    cfg = tiny_cfg()
    data = P.build_dataset(cfg, seed=2)
    x_tr = data["train"][0]
    assert abs(np.mean(x_tr)) < 0.5   # roughly zero-mean after standardization
    assert np.all(np.isfinite(x_tr))  # NaNs filled


def test_positive_class_present():
    """The synthetic fixture must contain both classes, or downstream models can't learn."""
    data = P.build_dataset(tiny_cfg(), seed=0)
    y_all = np.concatenate([data["train"][1], data["val"][1], data["test"][1]])
    assert y_all.sum() > 0
    assert y_all.sum() < len(y_all)
