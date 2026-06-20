"""Synthetic INTERMAGNET-style data + earthquake catalog (dev/test fixture).

This is NOT real data. It exists so the full chain (pipeline -> models -> eval -> app)
can be built and tested before real INTERMAGNET access is wired in (Phase 3). It is
deliberately constructed with a *learnable* precursor signal: a magnetic anomaly ramps
up in the trailing window before each positive anchor day, and a qualifying earthquake
is placed in that anchor's 7-day horizon. Real data replaces this behind the same
pipeline interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def generate_synthetic(cfg: Config, seed: int = 0):
    """Return ``(readings, catalog, n_days)``.

    readings : dict[str, pd.Series]  station code -> minute-resolution field (nT), with gaps
    catalog  : pd.DataFrame          columns [time, lat, lon, mag]
    n_days   : int
    """
    rng = np.random.default_rng(seed)
    n_days = cfg.n_days
    base = pd.Timestamp(cfg.start_date)
    minutes_per_day = 24 * 60
    total = n_days * minutes_per_day

    minute_index = pd.date_range(base, periods=total, freq="1min")
    t = np.arange(total)
    diurnal = 20.0 * np.sin(2.0 * np.pi * (t % minutes_per_day) / minutes_per_day)

    lab = cfg.labeling
    # Pick positive anchor days that have a full trailing window and a full horizon.
    candidate = np.where(rng.random(n_days) < 0.18)[0]
    pos_days = candidate[(candidate >= lab.window_days) & (candidate < n_days - lab.horizon_days)]

    lats = np.array([s.lat for s in cfg.stations])
    lons = np.array([s.lon for s in cfg.stations])
    clat, clon = float(lats.mean()), float(lons.mean())

    # --- Earthquake catalog ---------------------------------------------------
    quakes: list[dict] = []
    for d in pos_days:
        k = int(rng.integers(1, lab.horizon_days + 1))  # 1..horizon days after the anchor
        quakes.append({
            "time": base + pd.Timedelta(days=int(d + k)) + pd.Timedelta(hours=int(rng.integers(0, 24))),
            "lat": clat + rng.normal(0.0, 0.3),
            "lon": clon + rng.normal(0.0, 0.3),
            "mag": float(lab.magnitude_threshold + rng.uniform(0.0, 1.5)),
        })
    # A few non-qualifying events (too weak, or too far) so labeling filters get exercised.
    for _ in range(max(3, len(pos_days) // 4)):
        d = int(rng.integers(0, n_days))
        if rng.random() < 0.5:
            quakes.append({"time": base + pd.Timedelta(days=d), "lat": clat, "lon": clon,
                           "mag": float(rng.uniform(2.5, lab.magnitude_threshold - 0.2))})
        else:
            quakes.append({"time": base + pd.Timedelta(days=d), "lat": clat + 15.0, "lon": clon + 25.0,
                           "mag": float(lab.magnitude_threshold + 1.0)})
    catalog = pd.DataFrame(quakes, columns=["time", "lat", "lon", "mag"]).sort_values("time").reset_index(drop=True)

    # --- Precursor anomaly (ramps up over the last 7 days before each positive anchor) ---
    anomaly = np.zeros(total)
    for d in pos_days:
        a0 = max(0, d - 6) * minutes_per_day
        a1 = (d + 1) * minutes_per_day
        anomaly[a0:a1] = np.maximum(anomaly[a0:a1], np.linspace(0.0, 1.0, a1 - a0))

    # --- Per-station minute series -------------------------------------------
    readings: dict[str, pd.Series] = {}
    for si, st in enumerate(cfg.stations):
        baseline = 50000.0 + 100.0 * si
        noise = rng.normal(0.0, 5.0, total)
        gain = 30.0 * (1.0 - 0.1 * si)  # stations respond with slightly different amplitudes
        field = baseline + diurnal + noise + gain * anomaly
        s = pd.Series(field, index=minute_index)
        # Inject missing data (~1% random minute dropouts) to exercise gap handling.
        drop = rng.integers(0, total, int(0.01 * total))
        s.iloc[drop] = np.nan
        readings[st.code] = s

    return readings, catalog, n_days
