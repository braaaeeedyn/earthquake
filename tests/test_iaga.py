"""IAGA-2002 parser tests (no network): header coords + total-field from components."""
import math

import numpy as np

from eq.data.intermagnet import parse_iaga2002

SAMPLE = """ Format                 IAGA-2002                                    |
 IAGA Code              TST                                          |
 Geodetic Latitude      37.091                                       |
 Geodetic Longitude     240.279                                      |
 Reported               XYZG                                         |
DATE       TIME         DOY     TSTX      TSTY      TSTZ      TSTG   |
2021-01-01 00:00:00.000 001     20000.00      0.00  30000.00  99999.00
2021-01-01 00:01:00.000 001     99999.00      0.00  30000.00  99999.00
"""


def test_parses_header_and_converts_longitude():
    meta, _ = parse_iaga2002(SAMPLE)
    assert meta["code"] == "TST"
    assert math.isclose(meta["lat"], 37.091, abs_tol=1e-3)
    assert math.isclose(meta["lon"], 240.279 - 360.0, abs_tol=1e-3)  # 0..360 -> -180..180


def test_total_field_and_missing_values():
    _, f = parse_iaga2002(SAMPLE)
    assert len(f) == 2
    assert math.isclose(f.iloc[0], math.sqrt(20000**2 + 30000**2), rel_tol=1e-6)
    assert np.isnan(f.iloc[1])  # missing X component -> NaN total field
