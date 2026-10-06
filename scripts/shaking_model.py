"""Estimated shaking at a subscriber's location -- the wording of the confirmed push.

LIVE USE: live_watch.push_message calls estimate_mmi(mag, distance) + describe() to say e.g. "Weak shaking
possible near you". WHETHER to push is decided elsewhere (pipeline.push_eligible: confirmed, M >= 3.0;
live_watch.alert_devices: a followed sensor within 150 km). There is no waveform stream at the user's
location, so shaking is estimated from magnitude and distance with models CALIBRATED on the training data
(scripts/calibrate_shaking.py -> data/processed/shaking_calibration.json).

The 2-of-3 vote below (alert_votes / alert_level) is the calibration's diagnostic -- calibrate_shaking.py
prints its decision table -- not the live push gate. Its three criteria, with (1) required:

  (1) PGV amplitude   — a ground-motion model fit to the network's recorded peak velocities
                        predicts shaking >= a "notable" floor (a data percentile).
  (2) Intensity (MMI) — that PGV mapped to Modified Mercalli intensity (Worden et al. 2012) >= "felt" (3).
  (3) Felt-distance   — the user is within the distance at which events of this magnitude are
                        actually felt in the network, capped at the data's ~200 km range.

(1)&(2) are the two standard shaking metrics (correlated); (3) is an independent geographic
bound, so the vote can't fire on distance alone or on one marginal amplitude reading. Beyond the
trained ~200 km regime nothing alerts (no data to justify it).

Intensity (MMI) uses the standard Worden (2012) GMICE; approximate for display, not a regional model.
"""
import json
import math
from pathlib import Path

_CAL_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "shaking_calibration.json"

# Fallback if the calibration file is absent — the values fit by calibrate_shaking.py on
# seismic_phase2a_xl.npz, so the module works standalone (e.g. a fresh host before recalibration).
_DEFAULT = {
    "gmpe": {"a": -5.063, "b": 0.923, "c": -0.975, "d": -0.00350},
    "pgv_floor": 0.0006,                 # m/s (~0.06 cm/s, P70 of recorded PGV)
    "alert_mmi": 3.0,
    "envelope": {"p": 107.0, "q": 12.0}, # D_felt(M) = p + q*M km
    "r_data_max": 200.0,
}


def _load():
    try:
        return json.loads(_CAL_PATH.read_text())
    except (OSError, ValueError):
        return _DEFAULT


CAL = _load()


def reload_calibration():
    """Re-read the JSON (after a recalibration run) without restarting the process."""
    global CAL
    CAL = _load()
    return CAL


# ---- shaking estimates -------------------------------------------------------

def estimate_pgv(mag, dist_km):
    """Predicted peak ground velocity (m/s) at `dist_km` from an M`mag` event (data-fit GMPE)."""
    r = max(float(dist_km), 1.0)
    g = CAL["gmpe"]
    return 10 ** (g["a"] + g["b"] * mag + g["c"] * math.log10(r) + g["d"] * r)


# Worden et al. (2012) bilinear GMICE for PGV (cm/s): MMI = a + b*log10(PGV). Valid to low
# intensity (~MMI 2), unlike Wald (1999)'s single line (valid MMI >= 5), which underpredicts weak
# shaking badly. A published relation, not a per-dataset fit, so it lives here as a constant.
WORDEN_PGV = {"lo": (3.78, 1.47), "hi": (2.89, 3.16), "brk": 0.53}  # (a, b); switch at log10(PGV)=brk


def pgv_to_mmi(pgv_ms):
    """Modified Mercalli intensity from PGV (m/s) via the Worden et al. (2012) bilinear GMICE
    (PGV in cm/s). Clamped 1..10."""
    y = math.log10(max(pgv_ms, 1e-9) * 100.0)                 # log10(PGV in cm/s)
    a, b = WORDEN_PGV["hi"] if y > WORDEN_PGV["brk"] else WORDEN_PGV["lo"]
    return max(1.0, min(10.0, a + b * y))


def estimate_mmi(mag, dist_km):
    """Approximate MMI at the user: predicted PGV mapped to intensity. For display + the (2) vote."""
    return pgv_to_mmi(estimate_pgv(mag, dist_km))


def felt_distance(mag):
    """D_felt(mag): the distance events of this magnitude are felt in the network, from the data,
    never beyond the data's recorded range."""
    e = CAL["envelope"]
    return min(CAL["r_data_max"], max(5.0, e["p"] + e["q"] * mag))


# rounded MMI -> (label, what it feels like); index = level - 2
_LEVELS = [
    ("Not felt", "too weak to feel"),                                 # 2
    ("Weak", "barely felt indoors"),                                  # 3
    ("Light", "felt indoors, dishes and windows rattle"),             # 4
    ("Moderate", "felt by nearly everyone, unstable objects fall"),   # 5
    ("Strong", "felt by all, slight damage possible"),                # 6
    ("Very strong", "hard to stand, moderate damage"),                # 7
    ("Severe", "potentially damaging shaking"),                       # 8
]


def describe(mmi):
    """(label, plain description) for an estimated MMI."""
    level = max(2, min(8, round(mmi)))
    return _LEVELS[level - 2]


# ---- the graded alert decision (vote count -> level) ------------------------
# Criterion (1) -- predicted PGV >= the data's notable-shaking floor -- is REQUIRED for ANY alert,
# so nothing imperceptible ever pushes. Above that floor: (1) alone -> POTENTIAL earthquake
# warning (tentative); (1) plus (2) and/or (3) -> EARTHQUAKE WARNING (corroborated).
LEVEL_NONE, LEVEL_POTENTIAL, LEVEL_WARNING = 0, 1, 2


def alert_votes(mag, dist_km):
    """The three criteria as booleans (c1 amplitude, c2 intensity, c3 felt-distance)."""
    pgv = estimate_pgv(mag, dist_km)
    c1 = pgv >= CAL["pgv_floor"]
    c2 = pgv_to_mmi(pgv) >= CAL["alert_mmi"]
    c3 = float(dist_km) <= felt_distance(mag)
    return bool(c1), bool(c2), bool(c3)


def alert_level(mag, dist_km):
    """0 = no alert, 1 = potential warning, 2 = warning (>= 2 of 3 criteria).

    Hard floor: criterion (1) -- predicted PGV >= the data's "notable shaking" amplitude floor --
    is REQUIRED for any alert. This kills the distance-only nuisance case (an imperceptible quake,
    e.g. an M2.4 tens of km away, that the felt-distance bound (3) alone would otherwise flag).
    (Criterion (2), MMI >= felt, implies (1), so requiring (1) is the minimal correct gate.)
    """
    c1, c2, c3 = alert_votes(mag, dist_km)
    if not c1:                          # below the notable-amplitude floor -> no notification
        return LEVEL_NONE
    votes = c1 + c2 + c3
    return LEVEL_WARNING if votes >= 2 else LEVEL_POTENTIAL
