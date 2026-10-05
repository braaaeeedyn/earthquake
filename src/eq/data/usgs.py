"""USGS earthquake catalog via the FDSN event web service (public, no auth)."""
from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
ROW_CAP = 20000          # the service refuses/limits larger result sets


def fetch_catalog(start: str, end: str, min_magnitude: float, bbox: tuple[float, float, float, float],
                  cache_path: Path | None = None) -> pd.DataFrame:
    """Earthquakes in ``bbox`` (lat_min, lat_max, lon_min, lon_max) over [start, end).

    Cached to ``cache_path`` (CSV) if given. Returns columns [id, time, lat, lon, depth, mag].
    """
    if cache_path is not None and Path(cache_path).exists():
        raw = pd.read_csv(cache_path, encoding="utf-8")
    else:
        lat_min, lat_max, lon_min, lon_max = bbox
        params = {
            "format": "csv", "starttime": start, "endtime": end,
            "minmagnitude": min_magnitude, "orderby": "time-asc",
            "minlatitude": lat_min, "maxlatitude": lat_max,
            "minlongitude": lon_min, "maxlongitude": lon_max,
        }
        url = f"{FDSN_URL}?{urllib.parse.urlencode(params)}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    text = r.read().decode("utf-8")
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(text, encoding="utf-8")
        raw = pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame(
            columns=["id", "time", "latitude", "longitude", "depth", "mag"])

    out = pd.DataFrame({
        "id": raw["id"].astype(str),
        "time": pd.to_datetime(raw["time"], format="ISO8601").dt.tz_localize(None),
        "lat": raw["latitude"].astype(float),
        "lon": raw["longitude"].astype(float),
        "depth": raw["depth"].astype(float),
        "mag": raw["mag"].astype(float),
    })
    return out.sort_values("time").reset_index(drop=True)
