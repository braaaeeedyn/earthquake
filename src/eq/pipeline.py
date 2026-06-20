"""Deterministic data pipeline (MVP §3.2).

raw minute readings -> hourly means -> daily 24-vectors -> 24x27 matrices per station
-> binary labels (earthquake within horizon) -> chronological train/val/test splits.

Everything here is deterministic given a seed: same inputs -> identical outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, Station
from .graph import haversine_km


# --------------------------------------------------------------------------- #
# Step 1: minute -> hourly                                                     #
# --------------------------------------------------------------------------- #
def minute_to_hourly(readings: dict[str, pd.Series], min_minute_fraction: float) -> pd.DataFrame:
    """Average minute readings into hourly means.

    An hour is marked NaN if fewer than ``min_minute_fraction`` of its 60 minutes
    are present (gap handling). Returns long frame [time, station, F].
    """
    out = []
    threshold = 60 * min_minute_fraction
    for station, series in readings.items():
        resampler = series.resample("1h")
        mean = resampler.mean()
        count = resampler.count()
        mean = mean.where(count >= threshold, np.nan)
        h = mean.rename("F").reset_index()
        h.columns = ["time", "F"]
        h["station"] = station
        out.append(h)
    return pd.concat(out, ignore_index=True)[["time", "station", "F"]]


# --------------------------------------------------------------------------- #
# Step 2: hourly -> daily 24-vectors                                          #
# --------------------------------------------------------------------------- #
def hourly_to_daily(hourly: pd.DataFrame, start_date, n_days: int,
                    stations: tuple[Station, ...]) -> dict[str, np.ndarray]:
    """Per station, a (n_days, 24) array of hourly values (NaN where missing)."""
    h = hourly.copy()
    h["time"] = pd.to_datetime(h["time"])
    base = pd.Timestamp(start_date)
    h["day"] = (h["time"].dt.normalize() - base).dt.days
    h["hour"] = h["time"].dt.hour

    daily: dict[str, np.ndarray] = {}
    for st in stations:
        arr = np.full((n_days, 24), np.nan)
        sub = h[h["station"] == st.code]
        day = sub["day"].to_numpy()
        hour = sub["hour"].to_numpy()
        val = sub["F"].to_numpy()
        valid = (day >= 0) & (day < n_days)
        arr[day[valid], hour[valid]] = val[valid]
        daily[st.code] = arr
    return daily


# --------------------------------------------------------------------------- #
# Step 3: labeling                                                            #
# --------------------------------------------------------------------------- #
def positive_event_days(catalog: pd.DataFrame, stations: tuple[Station, ...], cfg: Config) -> set[int]:
    """Day indices on which a *qualifying* earthquake occurs.

    Qualifying = magnitude >= threshold AND within proximity_km of any station.
    """
    base = pd.Timestamp(cfg.start_date)
    days: set[int] = set()
    for _, ev in catalog.iterrows():
        if ev["mag"] < cfg.labeling.magnitude_threshold:
            continue
        near = any(
            haversine_km(ev["lat"], ev["lon"], s.lat, s.lon) <= cfg.labeling.proximity_km
            for s in stations
        )
        if near:
            days.add(int((pd.Timestamp(ev["time"]).normalize() - base).days))
    return days


# --------------------------------------------------------------------------- #
# Step 4: assemble 24x27 samples + labels                                     #
# --------------------------------------------------------------------------- #
def assemble_samples(daily: dict[str, np.ndarray], stations: tuple[Station, ...],
                     catalog: pd.DataFrame, cfg: Config):
    """Build samples X (N, S, 24, 27), labels y (N,), and anchor day indices (N,).

    Anchor day ``t`` uses window days [t-26 .. t] as the matrix and is labeled 1 if a
    qualifying earthquake occurs in horizon days [t+1 .. t+7].
    """
    lab = cfg.labeling
    w, h, hpd = lab.window_days, lab.horizon_days, lab.hours_per_day
    n_days = next(iter(daily.values())).shape[0]
    codes = [s.code for s in stations]
    quake_days = positive_event_days(catalog, stations, cfg)

    x_list, y_list, anchors = [], [], []
    for t in range(w - 1, n_days - h):
        cols = list(range(t - w + 1, t + 1))                 # 27 consecutive days
        mat = np.stack([daily[c][cols].T for c in codes])    # (S, 24, 27)
        if np.isnan(mat).mean() > cfg.preprocess.max_sample_missing_fraction:
            continue                                          # too sparse -> drop
        label = int(any((t + 1) <= d <= (t + h) for d in quake_days))
        x_list.append(mat)
        y_list.append(label)
        anchors.append(t)

    if not x_list:
        raise ValueError("No samples assembled; check data coverage / missingness threshold.")
    assert mat.shape == (len(codes), hpd, w)
    return np.stack(x_list), np.asarray(y_list, dtype=np.int64), np.asarray(anchors, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Step 5: chronological splits with leakage embargo                           #
# --------------------------------------------------------------------------- #
def chronological_split(anchors: np.ndarray, cfg: Config):
    """Split sample indices into train/val/test by anchor day (chronological).

    A ``embargo_days`` gap is enforced between splits so that no train label window
    overlaps a val/test input window (and likewise val->test). Returns index arrays
    into the original sample order.
    """
    order = np.argsort(anchors, kind="stable")
    n = len(order)
    i1 = int(n * cfg.split.train_frac)
    i2 = int(n * (cfg.split.train_frac + cfg.split.val_frac))
    train_idx, val_idx, test_idx = order[:i1], order[i1:i2], order[i2:]
    emb = cfg.split.embargo_days

    if len(train_idx) and len(val_idx):
        val_idx = val_idx[anchors[val_idx] > anchors[train_idx].max() + emb]
    prev = val_idx if len(val_idx) else train_idx
    if len(prev) and len(test_idx):
        test_idx = test_idx[anchors[test_idx] > anchors[prev].max() + emb]
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# Step 6: normalization (fit on TRAIN only)                                    #
# --------------------------------------------------------------------------- #
def fit_normalizer(x_train: np.ndarray):
    """Per-station mean/std over all valid training cells (leakage-safe)."""
    s = x_train.shape[1]
    mean = np.zeros(s)
    std = np.ones(s)
    for i in range(s):
        vals = x_train[:, i, :, :]
        mean[i] = np.nanmean(vals) if vals.size else 0.0
        sd = np.nanstd(vals) if vals.size else 1.0
        std[i] = sd if sd > 1e-8 else 1.0
    return mean, std


def apply_normalizer(x: np.ndarray, stats, fill: float = 0.0) -> np.ndarray:
    """Standardize per station and fill remaining NaNs with ``fill`` (the mean)."""
    mean, std = stats
    xn = x.astype(float).copy()
    for i in range(x.shape[1]):
        xn[:, i] = (xn[:, i] - mean[i]) / std[i]
    return np.nan_to_num(xn, nan=fill)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def build_dataset_from(readings: dict, catalog: pd.DataFrame, n_days: int, cfg: Config) -> dict:
    """Run the deterministic pipeline on any source of readings+catalog (synthetic or real).

    Returns normalized train/val/test splits + metadata. Source-agnostic: synthetic and real
    data go through exactly these steps.
    """
    hourly = minute_to_hourly(readings, cfg.preprocess.min_minute_fraction)
    daily = hourly_to_daily(hourly, cfg.start_date, n_days, cfg.stations)
    x, y, anchors = assemble_samples(daily, cfg.stations, catalog, cfg)
    tr, va, te = chronological_split(anchors, cfg)

    stats = fit_normalizer(x[tr])
    return {
        "train": (apply_normalizer(x[tr], stats), y[tr], anchors[tr]),
        "val": (apply_normalizer(x[va], stats), y[va], anchors[va]),
        "test": (apply_normalizer(x[te], stats), y[te], anchors[te]),
        "stats": stats,
        "catalog": catalog,
        "n_days": n_days,
    }


def build_dataset(cfg: Config, seed: int = 0) -> dict:
    """Full synthetic-data run: returns normalized train/val/test splits + metadata."""
    from .synthetic import generate_synthetic

    readings, catalog, n_days = generate_synthetic(cfg, seed)
    return build_dataset_from(readings, catalog, n_days, cfg)


def build_dataset_real(cfg: Config, proximity_km: float | None = None) -> dict:
    """Full real-data run (INTERMAGNET + USGS) behind the same interface as build_dataset."""
    from .data.real import build_real_inputs, real_config

    readings, catalog, n_days, stations = build_real_inputs()
    prox = proximity_km if proximity_km is not None else cfg.labeling.proximity_km
    rcfg = real_config(cfg, stations, prox, n_days=n_days)
    return build_dataset_from(readings, catalog, n_days, rcfg)
