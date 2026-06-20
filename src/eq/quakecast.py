"""Catalog-based probabilistic earthquake forecasting (region-aggregated).

Predict P(>=1 M>=TARGET quake somewhere in the region within the next HORIZON days) from the
history of past earthquakes alone -- the data that actually carries signal (aftershock
clustering, Omori decay, Gutenberg-Richter), unlike the (paused) geomagnetic approach.

Design split that matters:
  - INPUT catalog: complete to a low magnitude Mc (~2.5) -> drives the clustering features.
  - TARGET events: M >= target_mag -> what we forecast.
All features at issue-day t use only events strictly BEFORE t (no leakage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data.usgs import fetch_catalog

# California seismicity box (lat_min, lat_max, lon_min, lon_max) -- the actual region,
# not the geomagnetic-station bbox the precursor work used.
CA_BBOX = (32.0, 42.5, -125.0, -114.0)
_CACHE = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_catalog(start: str, end: str, min_mag: float = 2.5, bbox=CA_BBOX,
                 region: str = "california_seis", cache_root: Path = _CACHE) -> pd.DataFrame:
    """Fetch a region catalog, chunked by year (USGS FDSN caps at 20k events/request)."""
    cache = cache_root / f"usgs_{region}_m{min_mag:g}.csv"
    if cache.exists():
        raw = pd.read_csv(cache, parse_dates=["time"])
        return raw.sort_values("time").reset_index(drop=True)
    years = pd.date_range(start, end, freq="YS").union(
        [pd.Timestamp(start), pd.Timestamp(end)])
    parts = []
    for a, b in zip(years[:-1], years[1:]):
        parts.append(fetch_catalog(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"),
                                   min_mag, bbox, cache_path=None))
    cat = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["time", "lat", "lon", "mag"])
    cat = cat.sort_values("time").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(cache, index=False)
    return cat


def _event_days(catalog: pd.DataFrame, start: str) -> np.ndarray:
    base = pd.Timestamp(start)
    return ((catalog["time"].dt.normalize() - base).dt.days).to_numpy()


def daily_targets(catalog: pd.DataFrame, start: str, n_days: int, target_mag: float,
                  horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """y[t] = 1 if >=1 event M>=target_mag occurs in days [t+1, t+horizon]; also the
    per-day target-event indicator (for base-rate/diagnostics)."""
    big = catalog[catalog["mag"] >= target_mag]
    days = _event_days(big, start)
    pos_day = np.zeros(n_days, dtype=bool)
    d = days[(days >= 0) & (days < n_days)]
    pos_day[d] = True
    csum = np.concatenate([[0], np.cumsum(pos_day.astype(int))])
    y = np.array([(csum[min(t + horizon, n_days)] - csum[min(t + 1, n_days)]) > 0
                  for t in range(n_days)], dtype=int)
    return y, pos_day


def daily_features(catalog: pd.DataFrame, start: str, n_days: int, target_mag: float,
                   mc: float, lags=(1, 7, 30, 90, 365)) -> pd.DataFrame:
    """Per issue-day t, clustering features from events strictly before t (no leakage)."""
    base = pd.Timestamp(start)
    days = _event_days(catalog, start)
    mags = catalog["mag"].to_numpy()
    keep = (days >= 0) & (days < n_days) & (mags >= mc)
    days, mags = days[keep], mags[keep]
    order = np.argsort(days, kind="stable")
    days, mags = days[order], mags[order]
    moment = 10 ** (1.5 * mags + 9.1)                 # seismic moment (N*m)
    is_target = mags >= target_mag

    t = np.arange(n_days)
    # events strictly before day t: index via searchsorted on sorted day array
    lo0 = np.searchsorted(days, t, side="left")        # count of events with day < t
    feats = {"day": t, "date": base + pd.to_timedelta(t, unit="D")}
    for k in lags:
        lo = np.searchsorted(days, t - k, side="left")
        feats[f"cnt_{k}"] = lo0 - lo
    # recent seismic-moment release (log), max magnitude, b-value over 365d
    mcsum = np.concatenate([[0.0], np.cumsum(moment)])
    lo30 = np.searchsorted(days, t - 30, side="left")
    feats["logmoment_30"] = np.log10(np.maximum(mcsum[lo0] - mcsum[lo30], 1.0))
    # days since last event (any Mc) and last target event
    last_any = np.full(n_days, np.nan)
    last_tgt = np.full(n_days, np.nan)
    last_a = last_t = -10**9
    j = 0
    for ti in range(n_days):
        while j < len(days) and days[j] < ti:
            last_a = days[j]
            if is_target[j]:
                last_t = days[j]
            j += 1
        last_any[ti] = ti - last_a
        last_tgt[ti] = ti - last_t
    feats["days_since_any"] = np.clip(last_any, 0, 3650)
    feats["days_since_target"] = np.clip(last_tgt, 0, 3650)
    # max magnitude in last 30 days
    maxmag = np.zeros(n_days)
    for i, ti in enumerate(t):
        lo, hi = lo30[i], lo0[i]
        maxmag[i] = mags[lo:hi].max() if hi > lo else 0.0
    feats["maxmag_30"] = maxmag
    # Gutenberg-Richter b-value over last 365d (MLE: b = log10(e)/(meanM - Mc))
    lo365 = np.searchsorted(days, t - 365, side="left")
    bval = np.full(n_days, np.nan)
    for i in range(n_days):
        m = mags[lo365[i]:lo0[i]]
        if len(m) >= 20:
            bval[i] = np.log10(np.e) / max(m.mean() - mc, 1e-6)
    feats["bvalue_365"] = pd.Series(bval).ffill().fillna(1.0).to_numpy()
    return pd.DataFrame(feats)
