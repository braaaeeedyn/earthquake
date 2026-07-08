"""Estimated shaking at a subscriber's location, and the alert decision built on it.

The project's early-warning model predicts how hard the ground will shake (peak ground
velocity). The live alert product has no real-time waveform stream, so at alert time we
estimate the shaking a subscriber will feel from the reported magnitude and their distance to
the epicentre, using a compact magnitude/distance intensity relation, and DECIDE to alert when
that estimated intensity crosses a threshold -- so the alert fires on predicted shaking at the
user, not merely on being inside a fixed radius. Intensity is reported on the Modified Mercalli
(MMI) scale (an approximation here, not a calibrated regional model).
"""
import math

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


def estimate_mmi(mag, dist_km):
    """Approximate Modified Mercalli intensity at `dist_km` from an M`mag` event.

    Intensity grows with magnitude and decays with log-distance. Clamped to 1..10. This is a
    deliberately simple, transparent approximation, not a calibrated regional IPE.
    """
    r = max(dist_km, 4.0)  # avoid a log blow-up right on top of the epicentre
    mmi = 1.5 * mag - 1.15 * math.log10(r) - 0.5
    return max(1.0, min(10.0, mmi))


def describe(mmi):
    """(label, plain description) for an estimated MMI."""
    level = max(2, min(8, round(mmi)))
    return _LEVELS[level - 2]


def should_alert(mag, dist_km, threshold_mmi):
    """Alert when the estimated shaking at the location reaches the threshold intensity."""
    return estimate_mmi(mag, dist_km) >= threshold_mmi
