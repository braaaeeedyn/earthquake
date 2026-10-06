"""USGS earthquake catalogue for Southern California (FDSN event service, public, no auth).

load_catalog() fetches [start, end) in monthly chunks (an interval over the service's 20k-row cap is split
in half and retried) and caches it under data/raw/ with the date range in the file name. Used by the dataset
builder (event selection, noise screening), the replay harness (scoring) and select_network.py.
"""
from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

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


# Southern-California working box (lat_min, lat_max, lon_min, lon_max), padded beyond the network
# so noise windows can be screened against nearby out-of-region quakes too.
SOCAL_BBOX = (31.5, 37.5, -122.0, -113.5)
_CACHE = Path(__file__).resolve().parents[2] / "data" / "raw"


def _fetch_split(a: pd.Timestamp, b: pd.Timestamp, min_mag, bbox):
    """Fetch [a, b); if the result exceeds the service's row cap (it answers HTTP 400, or returns a
    capped set), split the interval in half and recurse (e.g. Ridgecrest, July 2019: >20k M1+)."""
    try:
        df = fetch_catalog(a.strftime("%Y-%m-%dT%H:%M:%S"), b.strftime("%Y-%m-%dT%H:%M:%S"), min_mag, bbox)
    except HTTPError as e:
        if e.code != 400 or (b - a) <= pd.Timedelta(hours=2):
            raise
        mid = a + (b - a) / 2
        return pd.concat([_fetch_split(a, mid, min_mag, bbox), _fetch_split(mid, b, min_mag, bbox)])
    if len(df) >= ROW_CAP and (b - a) > pd.Timedelta(hours=2):
        mid = a + (b - a) / 2
        return pd.concat([_fetch_split(a, mid, min_mag, bbox), _fetch_split(mid, b, min_mag, bbox)])
    return df


def load_catalog(start: str, end: str, min_mag: float = 1.0, bbox=SOCAL_BBOX,
                 region: str = "socal", cache_root: Path = _CACHE) -> pd.DataFrame:
    """Catalog [start, end) in monthly chunks. The cache filename carries the date range, so a
    different range can never silently reuse an old cache."""
    cache = cache_root / f"usgs_{region}_m{min_mag:g}_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    edges = pd.date_range(start, end, freq="MS").union([pd.Timestamp(start), pd.Timestamp(end)])
    parts = []
    part_dir = cache_root / f"usgs_{region}_m{min_mag:g}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    for a, b in zip(edges[:-1], edges[1:]):
        fp = part_dir / f"{a:%Y-%m-%d}_{b:%Y-%m-%d}.csv"
        if fp.exists():
            parts.append(pd.read_csv(fp, parse_dates=["time"]))
            continue
        parts.append(_fetch_split(a, b, min_mag, bbox))
        parts[-1].to_csv(fp, index=False)
        print(f"  catalog {a:%Y-%m}: {len(parts[-1])} events", flush=True)
    cat = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["id"])
    cat = cat.sort_values("time").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(cache, index=False)
    return cat
