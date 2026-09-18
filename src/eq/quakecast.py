"""Region earthquake catalog loading (USGS FDSN).

Fetches a California-region catalog, cached to CSV. Used by the seismic build pipeline
to label waveform windows against nearby catalog events.
"""
from __future__ import annotations

from pathlib import Path

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
