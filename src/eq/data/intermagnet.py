"""INTERMAGNET geomagnetic data via the BGS GIN web service, in IAGA-2002 format.

We fetch minute-resolution observatory data and reduce it to a single scalar total-field
series F per station (the synthetic pipeline's unit of input). F is computed from the
reported components — sqrt(X²+Y²+Z²), equivalently sqrt(H²+Z²) — so stations reporting
either orientation are handled. Chunks are cached on disk so a 3-year pull is resumable.
"""
from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

GIN_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
_SENTINEL = 88888.0  # IAGA-2002 uses 88888/99999 for missing values; real fields are ~20k-55k nT


def _total_field(vals: pd.DataFrame) -> pd.Series:
    """Scalar total field F (nT) from component columns; NaN if any needed component missing."""
    cols = set(vals.columns)
    if {"X", "Y", "Z"} <= cols:
        sq = (vals[["X", "Y", "Z"]] ** 2).sum(axis=1, skipna=False)
        return np.sqrt(sq)
    if {"H", "Z"} <= cols:
        sq = (vals[["H", "Z"]] ** 2).sum(axis=1, skipna=False)
        return np.sqrt(sq)
    if "F" in cols:
        return vals["F"]
    raise ValueError(f"Cannot derive total field from components {sorted(cols)}")


def parse_iaga2002(text: str) -> tuple[dict, pd.Series]:
    """Parse one IAGA-2002 file into ``(meta, F_series)``.

    meta has lat, lon (-180..180), code. F_series is minute-indexed total field (nT) with
    NaN at missing samples.
    """
    lines = text.splitlines()
    meta: dict = {}
    comp: list[str] = []
    data_start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("Geodetic Latitude"):
            meta["lat"] = float(s.split()[2])
        elif s.startswith("Geodetic Longitude"):
            lon = float(s.split()[2])
            meta["lon"] = lon - 360.0 if lon > 180.0 else lon
        elif s.startswith("IAGA Code"):
            meta["code"] = s.split()[2]
        elif s.startswith("DATE"):
            comp = [c[-1] for c in s.replace("|", "").split()[3:]]
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("No IAGA-2002 data header (DATE ...) found")

    body = "\n".join(lines[data_start:]).strip()
    if not body:
        return meta, pd.Series(dtype=float, name="F")
    df = pd.read_csv(io.StringIO(body), sep=r"\s+", header=None)
    ts = pd.to_datetime(df[0] + " " + df[1], format="%Y-%m-%d %H:%M:%S.%f")
    vals = df.iloc[:, 3:3 + len(comp)].copy()
    vals.columns = comp
    vals = vals.mask(vals >= _SENTINEL)
    f = _total_field(vals)
    return meta, pd.Series(f.to_numpy(), index=ts, name="F")


def _components_HZF(vals: pd.DataFrame) -> pd.DataFrame:
    """Horizontal H, vertical Z, total F (nT) from component columns; NaN-preserving."""
    cols = set(vals.columns)
    if {"X", "Y", "Z"} <= cols:
        h = np.sqrt((vals[["X", "Y"]] ** 2).sum(axis=1, skipna=False))
        z = vals["Z"]
    elif {"H", "Z"} <= cols:
        h, z = vals["H"], vals["Z"]
    else:
        raise ValueError(f"Cannot derive H/Z from components {sorted(cols)}")
    f = np.sqrt(h ** 2 + z ** 2)
    return pd.DataFrame({"H": h, "Z": z, "F": f})


def parse_iaga2002_components(text: str) -> tuple[dict, pd.DataFrame]:
    """Parse one IAGA-2002 file into ``(meta, frame)`` with H, Z, F columns (minute-indexed)."""
    lines = text.splitlines()
    meta: dict = {}
    comp: list[str] = []
    data_start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("Geodetic Latitude"):
            meta["lat"] = float(s.split()[2])
        elif s.startswith("Geodetic Longitude"):
            lon = float(s.split()[2])
            meta["lon"] = lon - 360.0 if lon > 180.0 else lon
        elif s.startswith("IAGA Code"):
            meta["code"] = s.split()[2]
        elif s.startswith("DATE"):
            comp = [c[-1] for c in s.replace("|", "").split()[3:]]
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("No IAGA-2002 data header (DATE ...) found")

    body = "\n".join(lines[data_start:]).strip()
    if not body:
        return meta, pd.DataFrame(columns=["H", "Z", "F"])
    df = pd.read_csv(io.StringIO(body), sep=r"\s+", header=None)
    ts = pd.to_datetime(df[0] + " " + df[1], format="%Y-%m-%d %H:%M:%S.%f")
    vals = df.iloc[:, 3:3 + len(comp)].copy()
    vals.columns = comp
    vals = vals.mask(vals >= _SENTINEL)
    out = _components_HZF(vals)
    out.index = ts
    return meta, out


def fetch_observatory_components(code: str, start: str, n_days: int, cache_dir: Path,
                                 samples: str = "minute", chunk_days: int | None = None
                                 ) -> tuple[dict, pd.DataFrame]:
    """Like ``fetch_observatory`` but returns an (H, Z, F) frame (reads the same cache)."""
    if chunk_days is None:
        chunk_days = 1 if samples == "second" else 30
    cdir = Path(cache_dir) / code
    cdir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start)
    parts: list[pd.DataFrame] = []
    meta: dict = {}
    day = 0
    while day < n_days:
        dur = min(chunk_days, n_days - day)
        cdate = (start_ts + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        fp = cdir / f"{cdate}_{dur}.iaga"
        text = fp.read_text() if fp.exists() else _fetch_chunk(code, cdate, dur, samples)
        if not fp.exists():
            fp.write_text(text)
        m, frame = parse_iaga2002_components(text)
        meta = meta or m
        if len(frame):
            parts.append(frame)
        day += dur
    if not parts:
        raise ValueError(f"No data returned for {code} from {start} ({n_days}d)")
    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return meta, full


def _fetch_chunk(code: str, start_date: str, duration: int, samples: str = "minute",
                 retries: int = 4) -> str:
    """GET one IAGA-2002 chunk, retrying transient drops (the GIN server resets often)."""
    url = (f"{GIN_URL}?Request=GetData&format=IAGA2002&observatoryIagaCode={code}"
           f"&samplesPerDay={samples}&publicationState=adj-or-rep"
           f"&dataStartDate={start_date}&dataDuration={duration}")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # connection reset / timeout / transient HTTP
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {code} {start_date} (+{duration}d): {last_err}")


def fetch_observatory(code: str, start: str, n_days: int, cache_dir: Path,
                      samples: str = "minute", chunk_days: int | None = None
                      ) -> tuple[dict, pd.Series]:
    """Fetch ``n_days`` of total-field data for one observatory (cached per chunk).

    ``samples`` is "minute" or "second". 1-second chunks are huge, so they are fetched one
    day at a time; minute data uses 30-day chunks. Empty chunks (no data for that window)
    are skipped so partial-coverage records still load.
    """
    if chunk_days is None:
        chunk_days = 1 if samples == "second" else 30
    cdir = Path(cache_dir) / code
    cdir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start)
    parts: list[pd.Series] = []
    meta: dict = {}
    day = 0
    while day < n_days:
        dur = min(chunk_days, n_days - day)
        cdate = (start_ts + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        fp = cdir / f"{cdate}_{dur}.iaga"
        text = fp.read_text() if fp.exists() else _fetch_chunk(code, cdate, dur, samples)
        if not fp.exists():
            fp.write_text(text)
        m, s = parse_iaga2002(text)
        meta = meta or m
        if len(s):
            parts.append(s)
        day += dur
    if not parts:
        raise ValueError(f"No data returned for {code} from {start} ({n_days}d)")
    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return meta, full
