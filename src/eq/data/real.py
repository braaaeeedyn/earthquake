"""Assemble real pipeline inputs by region: INTERMAGNET stations + USGS catalog.

Two regions, identical handling so they are directly comparable:
  california : FRN, TUC, BOU, NEW   (region of interest)
  japan      : KAK, MMB, KNY        (power check — dense coverage, high seismicity)

Real observatories are sparse and far apart, so configs RELAX the locked SoCal-cluster
rules (proximity and graph distances). Station coordinates are read from IAGA-2002 headers.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from ..config import Config, GraphRules, Station
from .intermagnet import fetch_observatory, fetch_observatory_components
from .usgs import fetch_catalog

REGIONS: dict[str, tuple[str, ...]] = {
    "california": ("FRN", "TUC", "BOU", "NEW"),
    "japan": ("KAK", "MMB", "KNY"),
}

# Backward-compat (3-yr California minute path used by pipeline.build_dataset_real).
REAL_STATIONS = REGIONS["california"]
REAL_START = "2021-01-01"
REAL_DAYS = 1095

# Full-record experiment window (~14 years).
FULL_START = "2010-01-01"
FULL_DAYS = 5110

_CACHE = Path(__file__).resolve().parents[3] / "data" / "raw"
_BBOX_MARGIN_DEG = 10.0  # ~1100 km — covers quakes within the widest proximity radius of any station


def build_region_inputs(region: str, start: str = FULL_START, n_days: int = FULL_DAYS,
                        min_magnitude: float = 4.5, cache_root: Path = _CACHE,
                        samples: str = "minute"):
    """Return ``(readings, catalog, n_days, stations)`` for one region.

    Catalog is fetched at the LOWEST grid magnitude (4.5) so both M4.5 and M5.0 labels
    derive from a single download. Mirrors the synthetic generator's output shapes.
    """
    codes = REGIONS[region]
    sub = "intermagnet_sec" if samples == "second" else "intermagnet"
    readings: dict[str, pd.Series] = {}
    stations: list[Station] = []
    for code in codes:
        meta, series = fetch_observatory(code, start, n_days, cache_root / sub, samples=samples)
        readings[code] = series
        stations.append(Station(code, meta["lat"], meta["lon"]))
    stations = tuple(stations)

    lats = [s.lat for s in stations]
    lons = [s.lon for s in stations]
    bbox = (min(lats) - _BBOX_MARGIN_DEG, max(lats) + _BBOX_MARGIN_DEG,
            min(lons) - _BBOX_MARGIN_DEG, max(lons) + _BBOX_MARGIN_DEG)
    end = (pd.Timestamp(start) + pd.Timedelta(days=n_days)).strftime("%Y-%m-%d")
    catalog = fetch_catalog(start, end, min_magnitude, bbox,
                            cache_path=cache_root / f"usgs_{region}.csv")
    return readings, catalog, n_days, stations


def _region_catalog(stations, start, n_days, min_magnitude, cache_root, region):
    lats = [s.lat for s in stations]
    lons = [s.lon for s in stations]
    bbox = (min(lats) - _BBOX_MARGIN_DEG, max(lats) + _BBOX_MARGIN_DEG,
            min(lons) - _BBOX_MARGIN_DEG, max(lons) + _BBOX_MARGIN_DEG)
    end = (pd.Timestamp(start) + pd.Timedelta(days=n_days)).strftime("%Y-%m-%d")
    return fetch_catalog(start, end, min_magnitude, bbox,
                         cache_path=cache_root / f"usgs_{region}.csv")


def build_region_components(region: str, start: str = FULL_START, n_days: int = FULL_DAYS,
                            min_magnitude: float = 4.5, cache_root: Path = _CACHE,
                            samples: str = "minute"):
    """Like ``build_region_inputs`` but each station's readings is an (H, Z, F) frame,
    enabling per-component ULF power and the Z/H polarization ratio."""
    codes = REGIONS[region]
    sub = "intermagnet_sec" if samples == "second" else "intermagnet"
    components: dict[str, pd.DataFrame] = {}
    stations: list[Station] = []
    for code in codes:
        meta, frame = fetch_observatory_components(code, start, n_days, cache_root / sub,
                                                   samples=samples)
        components[code] = frame
        stations.append(Station(code, meta["lat"], meta["lon"]))
    stations = tuple(stations)
    catalog = _region_catalog(stations, start, n_days, min_magnitude, cache_root, region)
    return components, catalog, n_days, stations


def region_config(base: Config, stations, proximity_km: float, magnitude: float,
                  start: str, n_days: int) -> Config:
    """Config for a real region: real stations, given proximity + magnitude, widened graph
    so far-apart observatories stay connected for the GNN."""
    return replace(
        base,
        stations=stations,
        start_date=start,
        n_days=n_days,
        labeling=replace(base.labeling, proximity_km=proximity_km, magnitude_threshold=magnitude),
        graph=GraphRules(connect_km=2500.0, sigma_km=1200.0),
    )


def build_real_inputs(codes=REAL_STATIONS, start=REAL_START, n_days=REAL_DAYS,
                      min_magnitude: float = 5.0, cache_root: Path = _CACHE):
    """Backward-compat 3-yr California minute path (pipeline.build_dataset_real)."""
    readings, catalog, n_days, stations = build_region_inputs(
        "california", start=start, n_days=n_days, min_magnitude=min_magnitude,
        cache_root=cache_root)
    return readings, catalog, n_days, stations


def real_config(base: Config, stations, proximity_km: float, start=REAL_START,
                n_days=REAL_DAYS) -> Config:
    """Backward-compat config for the 3-yr California path."""
    return region_config(base, stations, proximity_km, base.labeling.magnitude_threshold,
                         start, n_days)
