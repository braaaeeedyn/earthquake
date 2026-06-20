"""Superposed epoch analysis (SEA): model-free test for a pre-earthquake ULF signal.

For every M>=thresh quake within `radius` km of a station, line events up at their origin day
and average the common-mode-denoised daily ULF power as a function of days-relative-to-event.
A precursor (if any) appears as a bump in the days before t=0. Each event is z-scored to its own
pre-window baseline so per-event/station scale cancels. Significance: compare the real pre-window
mean to a null built from many random (non-event) day sets.

  python scripts/superposed_epoch.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
from eq.config import Config  # noqa: E402
from eq.data.real import FULL_START, build_region_inputs, region_config  # noqa: E402
from eq.features import BAND_MINUTE, feature_grid  # noqa: E402
from methodology_experiments import common_mode_residual  # noqa: E402

PRE = (-7, -1)          # precursor window (days before event)
BASE = (-30, -11)       # baseline window for per-event normalization
SPAN = (-30, 5)         # relative days to extract


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def daily_ulf(region):
    """(stations, daily ULF power array [n_stations, n_days]) from common-mode residual features."""
    readings, catalog, n_days, stations = build_region_inputs(region, min_magnitude=4.5)
    res = common_mode_residual(readings)
    cfg = region_config(Config(), stations, 300.0, 5.0, FULL_START, n_days)
    grids = feature_grid(res, cfg, BAND_MINUTE, samples_per_hour=60)
    daily = np.stack([np.nanmean(grids[s.code][:, :, 0], axis=1) for s in stations])  # ulf chan 0
    return stations, daily, catalog, n_days


def event_days(catalog, stations, radius_km, mag_min, n_days):
    """List of (day_index, nearest_station_index) for qualifying events."""
    start = pd.Timestamp(FULL_START)
    out = []
    for _, e in catalog[catalog["mag"] >= mag_min].iterrows():
        d = (pd.Timestamp(e["time"]).normalize() - start).days
        if not (0 <= d < n_days):
            continue
        dists = [_haversine_km(e["lat"], e["lon"], s.lat, s.lon) for s in stations]
        j = int(np.argmin(dists))
        if dists[j] <= radius_km:
            out.append((d, j))
    return out


def stack(daily, days, span=SPAN, base=BASE):
    """Per-event z-scored series over `span`; returns (n_events, span_len) with NaNs dropped."""
    rel = np.arange(span[0], span[1] + 1)
    rows = []
    for d, j in days:
        idx = d + rel
        ok = (idx >= 0) & (idx < daily.shape[1])
        s = np.full(len(rel), np.nan)
        s[ok] = daily[j, idx[ok]]
        bmask = (rel >= base[0]) & (rel <= base[1])
        b = s[bmask]
        if np.isfinite(b).sum() < 5:
            continue
        mu, sd = np.nanmean(b), np.nanstd(b)
        rows.append((s - mu) / (sd if sd > 1e-9 else 1.0))
    return rel, np.array(rows)


def pre_window_mean(rel, stacked):
    m = (rel >= PRE[0]) & (rel <= PRE[1])
    return float(np.nanmean(stacked[:, m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["california", "japan"])
    ap.add_argument("--radius", type=float, default=300.0)
    ap.add_argument("--mag", type=float, default=5.0)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "processed" / "sea_results.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    results = {}
    for region in args.regions:
        stations, daily, catalog, n_days = daily_ulf(region)
        evs = event_days(catalog, stations, args.radius, args.mag, n_days)
        rel, stacked = stack(daily, evs)
        obs = pre_window_mean(rel, stacked)

        # Null: same number of random (station, day) pairs, identical stacking/normalization.
        null = []
        n_st = daily.shape[0]
        for _ in range(args.nperm):
            fake = [(int(rng.integers(31, n_days - 6)), int(rng.integers(0, n_st)))
                    for _ in range(len(evs))]
            _, st = stack(daily, fake)
            if len(st):
                null.append(pre_window_mean(rel, st))
        null = np.array(null)
        p = float((null >= obs).mean())

        curve = np.nanmean(stacked, axis=0)
        sem = np.nanstd(stacked, axis=0) / np.sqrt(np.isfinite(stacked).sum(axis=0))
        results[region] = {
            "n_events": len(evs), "n_used": int(stacked.shape[0]),
            "pre_window_mean_z": obs, "null_mean": float(null.mean()),
            "null_std": float(null.std()), "p_value": p,
            "rel_days": rel.tolist(), "curve_z": curve.tolist(), "sem_z": sem.tolist(),
        }
        print(f"\n=== {region}: M>={args.mag} within {args.radius:.0f} km ===")
        print(f"  events used: {stacked.shape[0]}/{len(evs)}")
        print(f"  pre-window ([-7,-1]) mean z = {obs:+.3f}  "
              f"(null {null.mean():+.3f}±{null.std():.3f})  p = {p:.3f}")
        # compact curve print around the event
        for dd in range(-10, 4):
            i = dd - rel[0]
            bar = "#" * int(max(0, curve[i]) * 20)
            print(f"   day {dd:+d}: z={curve[i]:+.3f} {bar}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
