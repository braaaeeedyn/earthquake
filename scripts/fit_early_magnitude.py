"""Fit the EARLY (preliminary) magnitude used for the first, fast push.

Classic early-warning amplitude scaling: from only the first T seconds after each station's picked P,
    M_station = a * log10(peak 3-C velocity in [P, P+T]) + b * log10(distance) + c
and the event estimate is the median over its stations. It needs ~T s of P (plus the 6 s response-
correction margin), so it is available ~10 s after P instead of the ~31 s the full sizing waits.

Fit on the TRAIN events of data/processed/v2/magnitude.npz (chronological 70/15/15, same split as the
magnitude ensemble), choose T on VALIDATION, report once on TEST. Writes data/processed/v2/early_mag.json.

  python scripts/fit_early_magnitude.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
NPZ = ROOT / "data" / "processed" / "v2" / "magnitude.npz"
OUT = ROOT / "data" / "processed" / "v2" / "early_mag.json"
SR, P_IDX = 100, 600                       # rows start 6 s before the (picked) P


def split_chrono(t, fr=(0.7, 0.15)):
    o = np.argsort(t, kind="stable")
    n = len(o)
    return o[:int(n * fr[0])], o[int(n * fr[0]):int(n * (fr[0] + fr[1]))], o[int(n * (fr[0] + fr[1])):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="where to write the fit (retrain.py: the challenger dir)")
    ap.add_argument("--T", type=float, help="force the P-window length (s) instead of choosing it on validation")
    args = ap.parse_args()
    out_fp = Path(args.out)
    d = np.load(NPZ)
    mag, ev_t = d["mag"], d["ev_time"]
    row_ev, dist, ok = d["row_ev"], d["row_dist"], d["row_pick_ok"]
    x = d["x"]                                          # (R, 3, 3200) float16 normalized; scale in s
    s = d["s"]
    tr, va, te = split_chrono(ev_t)
    sets = {k: np.isin(row_ev, v) & ok for k, v in (("tr", tr), ("va", va), ("te", te))}
    results = {}
    for T in (2.0, 3.0, 4.0):
        w = slice(P_IDX, P_IDX + int(T * SR))
        peak = np.abs(x[:, :, w].astype(np.float32) * s[:, :, None]).max(axis=(1, 2)) + 1e-12
        A = np.c_[np.log10(peak), np.log10(np.maximum(dist, 1.0)), np.ones(len(peak))]
        coef, *_ = np.linalg.lstsq(A[sets["tr"]], mag[row_ev[sets["tr"]]], rcond=None)
        est_row = A @ coef

        def event_est(rows):
            evs = np.unique(row_ev[rows])
            est = np.array([np.median(est_row[rows & (row_ev == e)]) for e in evs])
            return evs, est

        out = {}
        for k in ("va", "te"):
            evs, est = event_est(sets[k])
            err = est - mag[evs]
            big = mag[evs] >= 3.0
            out[k] = {"n": int(len(evs)), "mae": float(np.abs(err).mean()), "bias": float(err.mean()),
                      "est": est, "true": mag[evs]}
            out[k]["big_n"] = int(big.sum())
        results[T] = (coef, out)
        print(f"T={T:.0f}s  coef a={coef[0]:.3f} b={coef[1]:.3f} c={coef[2]:.3f}   "
              f"val MAE {out['va']['mae']:.3f} bias {out['va']['bias']:+.2f}   "
              f"test MAE {out['te']['mae']:.3f} bias {out['te']['bias']:+.2f}")
    T = args.T if args.T else min(results, key=lambda k: results[k][1]["va"]["mae"])   # chosen on VALIDATION
    coef, out = results[T]
    # early push threshold: chosen on VALIDATION so that >= 95% of M>=3 events pass the quick check
    va = out["va"]
    thr = float(np.quantile(va["est"][va["true"] >= 3.0], 0.05))
    te = out["te"]
    big, small = te["true"] >= 3.0, te["true"] < 2.5
    rep = {"T_s": T, "a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]), "early_min_mag": round(thr, 2),
           "val_mae": va["mae"], "test_mae": te["mae"], "test_bias": te["bias"], "test_n": te["n"],
           "test_M3_pass_rate": float(np.mean(te["est"][big] >= thr)) if big.any() else None,
           "test_M_lt_2.5_pass_rate": float(np.mean(te["est"][small] >= thr)) if small.any() else None,
           "fit_on": "train events, picked-P rows; T and threshold chosen on validation"}
    out_fp.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
