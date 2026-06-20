"""Precursor-oriented features: ULF-band power + field derivative (MVP precursor test).

The geomagnetic precursor literature locates any hypothesized signal in ULF-band power and
in short-term field variability, not in the raw field level. For each station-hour we
compute, from that hour's high-rate samples:

  - ULF-band power: summed FFT power in a frequency band (band is resolution-limited —
    1-minute data reaches only ~0.008 Hz; 1-second data reaches 0.5 Hz).
  - |dF/dt|: mean absolute first difference (short-term variability).

This yields F=2 feature channels per station, keeping the locked 24x27 (hours x days) grid:
samples are (S, F, 24, 27). The precursor question is whether these features carry signal
about an earthquake in the next 7 days.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import Config
from .pipeline import positive_event_days

# Resolution-limited ULF bands (Hz). Lower bound avoids the DC/near-DC bin.
BAND_MINUTE = (5e-4, 8.0e-3)   # 1-min data: Nyquist 1/120 Hz
BAND_SECOND = (1e-3, 0.5)      # 1-sec data: Nyquist 0.5 Hz
FEATURE_NAMES = ("ulf_power", "abs_dFdt")

# Vector path: per-component ULF power + the Z/H polarization ratio (the canonical ULF
# precursor metric), instead of collapsing to the scalar total field.
POLAR_FEATURE_NAMES = ("ulf_H", "ulf_Z", "polar_ZH", "abs_dFdt")


def _hour_matrix(series: pd.Series, base: pd.Timestamp, n_days: int, sph: int) -> np.ndarray:
    """Reshape a station's series onto a dense (n_days*24, sph) grid (NaN where missing)."""
    full = pd.date_range(base, periods=n_days * 24 * sph, freq=pd.Timedelta(seconds=3600 // sph))
    vals = series.reindex(full).to_numpy(dtype=float)
    return vals.reshape(n_days * 24, sph)


def _hourly_features(rows: np.ndarray, sph: int, band: tuple[float, float],
                     min_fraction: float) -> np.ndarray:
    """(n_hours, sph) -> (n_hours, 2): [ULF power, mean |dF/dt|]; NaN for sparse hours."""
    valid = np.isfinite(rows).sum(axis=1)
    enough = valid >= min_fraction * sph

    with warnings.catch_warnings():  # all-NaN hours -> NaN, handled by `enough` mask below
        warnings.simplefilter("ignore", RuntimeWarning)
        detr = rows - np.nanmean(rows, axis=1, keepdims=True)
        filled = np.nan_to_num(detr, nan=0.0)
        spectrum = np.fft.rfft(filled, axis=1)
        power = (np.abs(spectrum) ** 2) / sph
        freqs = np.fft.rfftfreq(sph, d=3600.0 / sph)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        ulf = power[:, mask].sum(axis=1)

        diffs = np.abs(np.diff(rows, axis=1))
        dfdt = np.nanmean(diffs, axis=1)

    feats = np.stack([ulf, dfdt], axis=1)
    feats[~enough] = np.nan
    return feats


def _hour_band_power(rows: np.ndarray, sph: int, band: tuple[float, float],
                     min_fraction: float) -> np.ndarray:
    """(n_hours, sph) -> (n_hours,) ULF-band power; NaN for sparse hours."""
    enough = np.isfinite(rows).sum(axis=1) >= min_fraction * sph
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        detr = rows - np.nanmean(rows, axis=1, keepdims=True)
        spectrum = np.fft.rfft(np.nan_to_num(detr, nan=0.0), axis=1)
        power = (np.abs(spectrum) ** 2) / sph
        freqs = np.fft.rfftfreq(sph, d=3600.0 / sph)
        ulf = power[:, (freqs >= band[0]) & (freqs <= band[1])].sum(axis=1)
    ulf[~enough] = np.nan
    return ulf


def _hour_dfdt(rows: np.ndarray, sph: int, min_fraction: float) -> np.ndarray:
    """(n_hours, sph) -> (n_hours,) mean |first difference|; NaN for sparse hours."""
    enough = np.isfinite(rows).sum(axis=1) >= min_fraction * sph
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dfdt = np.nanmean(np.abs(np.diff(rows, axis=1)), axis=1)
    dfdt[~enough] = np.nan
    return dfdt


def vector_feature_grid(components: dict[str, pd.DataFrame], cfg: Config,
                        band: tuple[float, float], samples_per_hour: int = 60,
                        eps: float = 1e-6) -> dict[str, np.ndarray]:
    """Per station, a (n_days, 24, 4) array of [ULF(H), ULF(Z), log Z/H ratio, |dF/dt|].

    ``components[code]`` is a DataFrame with H (horizontal), Z (vertical), F (total) series.
    The log polarization ratio log(P_Z/P_H) is the canonical ULF-precursor channel.
    """
    base = pd.Timestamp(cfg.start_date)
    mf = cfg.preprocess.min_minute_fraction
    sph = samples_per_hour
    grids: dict[str, np.ndarray] = {}
    for st in cfg.stations:
        df = components[st.code]
        rows = {k: _hour_matrix(df[k], base, cfg.n_days, sph) for k in ("H", "Z", "F")}
        ulf_h = _hour_band_power(rows["H"], sph, band, mf)
        ulf_z = _hour_band_power(rows["Z"], sph, band, mf)
        polar = np.log((ulf_z + eps) / (ulf_h + eps))
        dfdt = _hour_dfdt(rows["F"], sph, mf)
        feats = np.stack([ulf_h, ulf_z, polar, dfdt], axis=1)
        grids[st.code] = feats.reshape(cfg.n_days, 24, len(POLAR_FEATURE_NAMES))
    return grids


def feature_grid(readings: dict[str, pd.Series], cfg: Config, band: tuple[float, float],
                 samples_per_hour: int = 60) -> dict[str, np.ndarray]:
    """Per station, a (n_days, 24, 2) array of [ULF power, |dF/dt|]."""
    base = pd.Timestamp(cfg.start_date)
    min_frac = cfg.preprocess.min_minute_fraction
    grids: dict[str, np.ndarray] = {}
    for st in cfg.stations:
        rows = _hour_matrix(readings[st.code], base, cfg.n_days, samples_per_hour)
        feats = _hourly_features(rows, samples_per_hour, band, min_frac)
        grids[st.code] = feats.reshape(cfg.n_days, 24, len(FEATURE_NAMES))
    return grids


def assemble_feature_samples(grids: dict[str, np.ndarray], catalog: pd.DataFrame, cfg: Config):
    """Build X (N, S, F, 24, 27), labels y (N,), anchors (N,) from feature grids.

    Same windowing/labeling/missingness rules as the raw pipeline, with an added feature axis.
    """
    lab = cfg.labeling
    w, h = lab.window_days, lab.horizon_days
    codes = [s.code for s in cfg.stations]
    f = next(iter(grids.values())).shape[2]   # channel count (2 scalar, 4 vector)
    n_days = next(iter(grids.values())).shape[0]
    quake_days = positive_event_days(catalog, cfg.stations, cfg)

    x_list, y_list, anchors = [], [], []
    for t in range(w - 1, n_days - h):
        cols = list(range(t - w + 1, t + 1))                       # 27 consecutive days
        # per station: (24, 27, F) -> (F, 24, 27); stack stations -> (S, F, 24, 27)
        mat = np.stack([grids[c][cols].transpose(2, 1, 0) for c in codes])
        if np.isnan(mat).mean() > cfg.preprocess.max_sample_missing_fraction:
            continue
        label = int(any((t + 1) <= d <= (t + h) for d in quake_days))
        x_list.append(mat)
        y_list.append(label)
        anchors.append(t)

    if not x_list:
        raise ValueError("No feature samples assembled; check coverage / missingness threshold.")
    assert x_list[0].shape == (len(codes), f, lab.hours_per_day, w)
    return np.stack(x_list), np.asarray(y_list, dtype=np.int64), np.asarray(anchors, dtype=np.int64)


def fit_feature_normalizer(x_train: np.ndarray):
    """Per (station, feature) channel mean/std over valid training cells (leakage-safe)."""
    s, f = x_train.shape[1], x_train.shape[2]
    mean = np.zeros((s, f))
    std = np.ones((s, f))
    for i in range(s):
        for j in range(f):
            vals = x_train[:, i, j]
            mean[i, j] = np.nanmean(vals) if np.isfinite(vals).any() else 0.0
            sd = np.nanstd(vals) if np.isfinite(vals).any() else 1.0
            std[i, j] = sd if sd > 1e-8 else 1.0
    return mean, std


def apply_feature_normalizer(x: np.ndarray, stats, fill: float = 0.0) -> np.ndarray:
    mean, std = stats
    xn = x.astype(float).copy()
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            xn[:, i, j] = (xn[:, i, j] - mean[i, j]) / std[i, j]
    return np.nan_to_num(xn, nan=fill)
