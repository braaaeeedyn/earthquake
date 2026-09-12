"""Calibrate the alert-gate criteria on the real training data (seismic_phase2a_xl.npz).

The live alert product decides, for a detected+sized event, whether to notify a subscriber at a
given distance. To be safe AND accurate we use a 2-of-3 vote across three criteria, each grounded
in the SAME data the magnitude model was trained on:

  (1) PGV attenuation  — a ground-motion model log10(PGV)=a+b*M+c*log10(R)+d*R fit to the
      per-station peak velocities the network actually recorded. Predicts shaking amplitude.
  (2) Intensity (MMI)  — that predicted PGV mapped to Modified Mercalli intensity via Worden (2012).
      Predicts the human-felt intensity, with its own threshold.
  (3) Felt-distance envelope — the maximum distance events of a given magnitude are actually
      recorded/felt in this network, capped at the data's distance range (~200 km). A geographic
      guard against extrapolating the amplitude model far outside the data (the 836-km bug).

An event alerts a subscriber only when >= 2 of the 3 agree. (1)&(2) are the two standard shaking
metrics (amplitude / intensity) so they correlate; (3) is an independent geographic bound, so the
vote can't fire on the distance guard alone, nor on a single marginal amplitude reading.

Writes data/processed/shaking_calibration.json (loaded by scripts/shaking_model.py).
Run:  python scripts/calibrate_shaking.py            # fit, print a decision table, save
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "seismic_phase2a_xl.npz"
OUT = ROOT / "data" / "processed" / "shaking_calibration.json"

# Worden et al. (2012) bilinear GMICE for PGV (cm/s): MMI = a + b*log10(PGV), switching at
# log10(PGV)=0.53. Valid to low intensity (~MMI 2), unlike Wald (1999) (valid MMI >= 5), which
# underpredicts weak shaking. Published relation, not fit to this dataset.
WORDEN_LO, WORDEN_HI, WORDEN_BRK = (3.78, 1.47), (2.89, 3.16), 0.53
ALERT_MMI = 3.0                 # "felt" — the intensity threshold criterion (2) must clear
FELT_PCTILE = 70                # criterion (1)/(3): "notable" shaking = this pctile of recorded PGV
ENV_PCTILE = 90                 # criterion (3): felt-distance envelope = this pctile of felt distances
R_DATA_MAX = 200.0              # data's max recorded distance — never claim "felt" beyond it


def load_triples():
    """(M, R_km, PGV_m/s) for every recorded station across all events."""
    d = dict(np.load(DATA, allow_pickle=True))
    X, mask, dist, mag = d["X"], d["mask"].astype(bool), d["dist"], d["mag"]
    pgv = np.abs(X).max(axis=(2, 3))                      # (events, stations) peak |vel| over 3C x time
    M = np.repeat(mag[:, None], mask.shape[1], axis=1)[mask]
    R = dist[mask]
    P = pgv[mask]
    keep = (R > 0) & (P > 0)
    return M[keep], R[keep], P[keep]


def fit_gmpe(M, R, P):
    """log10(PGV_m/s) = a + b*M + c*log10(R) + d*R  (least squares)."""
    A = np.column_stack([np.ones_like(M), M, np.log10(R), R])
    coef, *_ = np.linalg.lstsq(A, np.log10(P), rcond=None)
    pred = A @ coef
    ss_res = np.sum((np.log10(P) - pred) ** 2)
    ss_tot = np.sum((np.log10(P) - np.log10(P).mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return dict(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), d=float(coef[3])), float(r2)


def pgv_to_mmi(pgv_ms):
    y = np.log10(max(pgv_ms, 1e-9) * 100.0)
    a, b = WORDEN_HI if y > WORDEN_BRK else WORDEN_LO
    return float(max(1.0, min(10.0, a + b * y)))


def felt_envelope(M, R, P, pgv_floor):
    """D_felt(mag) ~ p + q*mag: the ENV_PCTILE distance among stations that recorded >= pgv_floor,
    per magnitude bin, fit linearly. Bounds where the network actually feels each size of event."""
    felt = P >= pgv_floor
    bins = np.arange(3.5, 7.5, 0.5)
    xs, ys = [], []
    for lo in bins:
        sel = felt & (M >= lo) & (M < lo + 0.5)
        if sel.sum() >= 5:
            xs.append(lo + 0.25)
            ys.append(np.percentile(R[sel], ENV_PCTILE))
    if len(xs) < 2:
        return dict(p=float(R_DATA_MAX), q=0.0)          # fallback: flat cap
    q, p = np.polyfit(xs, ys, 1)                          # y = q*x + p
    return dict(p=float(p), q=float(q))


def main():
    M, R, P = load_triples()
    gmpe, r2 = fit_gmpe(M, R, P)
    pgv_floor = float(np.percentile(P, FELT_PCTILE))
    cal = dict(gmpe=gmpe, pgv_floor=pgv_floor, alert_mmi=ALERT_MMI,
               gmice="worden2012_pgv", r_data_max=R_DATA_MAX)
    cal["envelope"] = felt_envelope(M, R, P, pgv_floor)

    print(f"GMPE fit  log10(PGV_m/s) = {gmpe['a']:.3f} + {gmpe['b']:.3f}*M "
          f"+ {gmpe['c']:.3f}*log10(R) + {gmpe['d']:.5f}*R   (R^2={r2:.2f}, n={len(M)})")
    print(f"pgv_floor (P{FELT_PCTILE} of recorded PGV) = {pgv_floor*100:.3f} cm/s"
          f"  (MMI {pgv_to_mmi(pgv_floor):.1f});  alert MMI >= {ALERT_MMI}")
    print(f"felt envelope  D_felt(M) = {cal['envelope']['p']:.0f} + {cal['envelope']['q']:.0f}*M km"
          f"  (capped at {R_DATA_MAX:.0f} km)")

    OUT.write_text(json.dumps(cal, indent=2))
    print(f"saved -> {OUT.relative_to(ROOT)}\n")

    # Print the decision table using the REAL runtime gate (shaking_model reads the JSON we saved).
    import shaking_model
    shaking_model.reload_calibration()
    lvl = {0: "-", 1: "POTENTIAL", 2: "WARNING"}
    print(f"{'M':>4} {'R(km)':>6} | {'PGV cm/s':>9} {'MMI':>4} {'Dfelt':>6} | c1 c2 c3 | alert")
    print("-" * 62)
    for mag in (4.0, 5.0, 5.5, 6.0, 6.5, 7.0):
        for r in (10, 30, 60, 100, 150, 200, 300, 836):
            c1, c2, c3 = shaking_model.alert_votes(mag, r)
            level = shaking_model.alert_level(mag, r)
            pgv = shaking_model.estimate_pgv(mag, r)
            print(f"{mag:>4} {r:>6} | {pgv*100:>9.3f} {shaking_model.pgv_to_mmi(pgv):>4.1f} "
                  f"{shaking_model.felt_distance(mag):>6.0f} | {int(c1):>2} {int(c2):>2} {int(c3):>2} "
                  f"| {lvl[level]}")
        print()


if __name__ == "__main__":
    main()
