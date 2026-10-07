"""Fit the shaking model (ground-motion equation + site term) on the v2 magnitude dataset.

Every station record of data/processed/v2/magnitude.npz gives (catalogue M, distance R, recorded PGV = peak
3-component velocity over P-6 s .. P+26 s). Fit on the TRAIN events (chronological 70/15/15, as everywhere):

  log10(PGV m/s) = a + b*M + c*log10(R) + d*R + e*log10(Vs30_station / Vs30_ref)

Vs30 (time-averaged shear-wave velocity of the top 30 m) is the USGS slope-proxy map cropped to SoCal
(app/public/vs30_socal.json, built 2026-10-07); Vs30_ref = the median at our 19 stations, so e is a site term
fitted from our own recordings (soft ground, low Vs30, shakes harder). PGV -> MMI uses Worden et al. (2012).
Reports the residual scatter on held-out TEST events. Writes data/processed/shaking_calibration.json, read by
src/eq/shaking.py (server wording + event term) and served to the app at /api/shaking-model (on-device estimates).

  python scripts/calibrate_shaking.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq import network  # noqa: E402

DATA = ROOT / "data" / "processed" / "v2" / "magnitude.npz"
VS30 = ROOT / "app" / "public" / "vs30_socal.json"
OUT = ROOT / "data" / "processed" / "shaking_calibration.json"
WORDEN_LO, WORDEN_HI, WORDEN_BRK = (3.78, 1.47), (2.89, 3.16), 0.53
ALERT_MMI = 3.0
FELT_PCTILE = 70
ENV_PCTILE = 90
R_DATA_MAX = 200.0


def split_chrono(t, fr=(0.7, 0.15)):
    o = np.argsort(t, kind="stable")
    n = len(o)
    return o[:int(n * fr[0])], o[int(n * fr[0]):int(n * (fr[0] + fr[1]))], o[int(n * (fr[0] + fr[1])):]


def load_rows():
    """(M, R_km, PGV_m/s, station Vs30, is_train, is_test) for every station record."""
    d = np.load(DATA)
    pgv = (np.abs(d["x"].astype(np.float32)) * d["s"][..., None]).max(axis=(1, 2))
    tr, _, te = split_chrono(d["ev_time"])
    vs = json.loads(VS30.read_text())["station_vs30"]
    sta_vs = np.array([vs[c] for c in network.CODES], float)[d["row_sta"]]
    M, R = d["mag"][d["row_ev"]].astype(float), d["row_dist"].astype(float)
    keep = (R > 0) & (pgv > 0)
    return (M[keep], R[keep], pgv[keep], sta_vs[keep], np.isin(d["row_ev"], tr)[keep], np.isin(d["row_ev"], te)[keep])


def design(M, R, V, vref):
    return np.column_stack([np.ones_like(M), M, np.log10(R), R, np.log10(V / vref)])


def pgv_to_mmi(pgv_ms):
    y = np.log10(max(pgv_ms, 1e-9) * 100.0)
    a, b = WORDEN_HI if y > WORDEN_BRK else WORDEN_LO
    return float(max(1.0, min(10.0, a + b * y)))


def felt_envelope(M, R, P, pgv_floor):
    """D_felt(mag) ~ p + q*mag: the ENV_PCTILE distance among records >= pgv_floor, per magnitude bin."""
    felt = P >= pgv_floor
    xs, ys = [], []
    for lo in np.arange(2.0, 7.5, 0.5):
        sel = felt & (M >= lo) & (M < lo + 0.5)
        if sel.sum() >= 5:
            xs.append(lo + 0.25)
            ys.append(np.percentile(R[sel], ENV_PCTILE))
    if len(xs) < 2:
        return dict(p=float(R_DATA_MAX), q=0.0)
    q, p = np.polyfit(xs, ys, 1)
    return dict(p=float(p), q=float(q))


def main():
    M, R, P, V, is_tr, is_te = load_rows()
    vref = float(np.median(list(json.loads(VS30.read_text())["station_vs30"].values())))
    A = design(M, R, V, vref)
    y = np.log10(P)
    coef, *_ = np.linalg.lstsq(A[is_tr], y[is_tr], rcond=None)
    res_te = y[is_te] - A[is_te] @ coef
    a, b, c, d_, e = (float(v) for v in coef)
    pgv_floor = float(np.percentile(P[is_tr], FELT_PCTILE))
    cal = {"gmpe": {"a": a, "b": b, "c": c, "d": d_}, "site": {"vs30_ref": vref, "e": e},
           "pgv_floor": pgv_floor, "alert_mmi": ALERT_MMI, "gmice": "worden2012_pgv", "r_data_max": R_DATA_MAX,
           "envelope": felt_envelope(M[is_tr], R[is_tr], P[is_tr], pgv_floor),
           "station_vs30": json.loads(VS30.read_text())["station_vs30"],
           "residual_sd_log10_test": float(res_te.std()), "fit": "v2 magnitude.npz train events (chronological)",
           "n_train_records": int(is_tr.sum()), "n_test_records": int(is_te.sum())}
    print(f"log10(PGV) = {a:.3f} + {b:.3f}*M + {c:.3f}*log10(R) + {d_:.5f}*R + {e:.3f}*log10(Vs30/{vref:.0f})")
    print(f"  train records {is_tr.sum()}, held-out test records {is_te.sum()}: residual sd {res_te.std():.3f} log10 "
          f"(factor {10 ** res_te.std():.2f}), bias {res_te.mean():+.3f}")
    print(f"  pgv_floor {pgv_floor * 100:.3f} cm/s (MMI {pgv_to_mmi(pgv_floor):.1f}); felt envelope "
          f"{cal['envelope']['p']:.0f} + {cal['envelope']['q']:.0f}*M km")
    OUT.write_text(json.dumps(cal, indent=2))
    print(f"saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
