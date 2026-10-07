"""Estimated shaking intensity (Modified Mercalli, MMI) at a place, for a located and sized quake.

  log10(PGV m/s) = a + b*M + c*log10(R) + d*R + e*log10(Vs30 / Vs30_ref) + event_term

- a..e: fitted on our own stations' recordings (scripts/calibrate_shaking.py -> data/processed/shaking_calibration.json);
  e is the site term (soft ground = low Vs30 shakes harder), Vs30 from the USGS map (app/public/vs30_socal.json).
- event_term: how much harder/softer THIS quake actually shook our stations than the equation predicts (the
  mean log residual over the sized stations, ShakeMap-style bias correction), computed at sizing time.
- PGV -> MMI: Worden et al. (2012) bilinear relation, valid down to weak shaking, plus mmi_offset: people's reports
  (USGS Did You Feel It?) run higher than instrumental intensity; fitted by scripts/validate_mmi.py.

Used by the live daemon (push wording, event term) and, as the same equations in app/src/shaking.ts, by the app to
estimate shaking at the user's home ON THE DEVICE (the location never leaves the phone). The equations and
coefficients are served at /api/shaking-model so both sides stay identical.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CAL_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "shaking_calibration.json"
WORDEN = {"lo": (3.78, 1.47), "hi": (2.89, 3.16), "brk": 0.53}     # MMI = a + b*log10(PGV cm/s), split at brk
EVENT_TERM_CLIP = 0.5                                              # |term| <= 0.5 log10 (factor ~3)
LEVELS = [                                                        # rounded MMI -> (label, what it feels like)
    (1, "Not felt", "not felt"), (2, "Not felt", "too weak to feel"), (3, "Weak", "felt by a few people at rest indoors"),
    (4, "Light", "felt indoors; dishes and windows rattle"), (5, "Moderate", "felt by nearly everyone; unstable objects fall"),
    (6, "Strong", "felt by all; slight damage possible"), (7, "Very strong", "hard to stand; moderate damage"),
    (8, "Severe", "heavy damage to weak buildings"), (9, "Violent", "considerable damage"), (10, "Extreme", "severe damage"),
]
_DEFAULT = {"gmpe": {"a": -5.697, "b": 0.981, "c": -0.868, "d": -0.00456}, "site": {"vs30_ref": 631.0, "e": -0.406},
            "station_vs30": {}}


def load():
    try:
        cal = json.loads(CAL_PATH.read_text())
        return cal if "site" in cal else _DEFAULT                 # pre-v2 file (no site term): use the v2 fit
    except (OSError, ValueError):
        return _DEFAULT


CAL = load()


def estimate_pgv(mag, dist_km, vs30=None, event_term=0.0):
    """Predicted peak ground velocity (m/s) at epicentral distance dist_km (site term only if vs30 is given)."""
    g, s = CAL["gmpe"], CAL["site"]
    r = max(float(dist_km), 1.0)
    log_pgv = g["a"] + g["b"] * mag + g["c"] * math.log10(r) + g["d"] * r + event_term
    if vs30:
        log_pgv += s["e"] * math.log10(float(vs30) / s["vs30_ref"])
    return 10 ** log_pgv


def pgv_to_mmi(pgv_ms):
    y = math.log10(max(pgv_ms, 1e-9) * 100.0)
    a, b = WORDEN["hi"] if y > WORDEN["brk"] else WORDEN["lo"]
    return max(1.0, min(10.0, a + b * y))


def estimate_mmi(mag, dist_km, vs30=None, event_term=0.0, offset=None):
    """MMI people would report: instrumental MMI + the DYFI-calibrated offset (offset=0.0 gives the raw value)."""
    off = CAL.get("mmi_offset", 0.0) if offset is None else offset
    return max(1.0, min(10.0, pgv_to_mmi(estimate_pgv(mag, dist_km, vs30, event_term)) + off))


def describe(mmi):
    """(label, plain description) for an MMI value."""
    _, label, desc = LEVELS[max(1, min(10, math.floor(mmi + 0.5))) - 1]   # round half up, like the app (Math.round)
    return label, desc


def event_term(mag, dist_km, pgv_obs, station_codes):
    """Mean log10(observed / predicted PGV) over the sized stations, clipped. 0.0 without usable data."""
    vs = CAL.get("station_vs30", {})
    res = [math.log10(p) - math.log10(estimate_pgv(mag, r, vs.get(c)))
           for r, p, c in zip(dist_km, pgv_obs, station_codes) if p > 0 and r > 0]
    if not res:
        return 0.0
    return float(np.clip(np.mean(res), -EVENT_TERM_CLIP, EVENT_TERM_CLIP))


def model_for_clients():
    """The equations' coefficients for /api/shaking-model (the app computes the home estimate itself)."""
    return {"gmpe": CAL["gmpe"], "site": CAL["site"], "worden": WORDEN, "event_term_clip": EVENT_TERM_CLIP,
            "mmi_offset": CAL.get("mmi_offset", 0.0),
            "levels": [{"mmi": m, "label": lab, "desc": d} for m, lab, d in LEVELS],
            "residual_sd_log10": CAL.get("residual_sd_log10_test"), "validation": CAL.get("validation")}
