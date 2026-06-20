"""USGS earthquake catalog via the FDSN event web service (public, no auth).

Returns the same ``[time, lat, lon, mag]`` frame the synthetic generator produces, so the
labeling step (`positive_event_days`) is identical for real and synthetic data.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_catalog(start: str, end: str, min_magnitude: float, bbox: tuple[float, float, float, float],
                  cache_path: Path | None = None) -> pd.DataFrame:
    """Earthquakes in ``bbox`` (lat_min, lat_max, lon_min, lon_max) over [start, end).

    Cached to ``cache_path`` (CSV) if given. Returns columns [time, lat, lon, mag].
    """
    if cache_path is not None and Path(cache_path).exists():
        raw = pd.read_csv(cache_path, encoding="utf-8")
    else:
        lat_min, lat_max, lon_min, lon_max = bbox
        params = {
            "format": "csv", "starttime": start, "endtime": end,
            "minmagnitude": min_magnitude,
            "minlatitude": lat_min, "maxlatitude": lat_max,
            "minlongitude": lon_min, "maxlongitude": lon_max,
        }
        url = f"{FDSN_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode("utf-8")
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(text, encoding="utf-8")
        import io
        raw = pd.read_csv(io.StringIO(text))

    out = pd.DataFrame({
        "time": pd.to_datetime(raw["time"]).dt.tz_localize(None),
        "lat": raw["latitude"].astype(float),
        "lon": raw["longitude"].astype(float),
        "mag": raw["mag"].astype(float),
    })
    return out.sort_values("time").reset_index(drop=True)
