"""Real-data acquisition (Phase 3): USGS earthquake catalog + INTERMAGNET geomagnetic data.

Both sources are normalized to the SAME shapes the synthetic fixture produces, so the
pipeline (`build_dataset_from`) and the models are unchanged whether data is real or fake.
"""
from .usgs import fetch_catalog
from .intermagnet import fetch_observatory, parse_iaga2002
from .real import build_real_inputs, REAL_STATIONS, REAL_START, REAL_DAYS, real_config

__all__ = [
    "fetch_catalog",
    "fetch_observatory",
    "parse_iaga2002",
    "build_real_inputs",
    "real_config",
    "REAL_STATIONS",
    "REAL_START",
    "REAL_DAYS",
]
