"""Reproduce / re-evaluate the live station network (src/eq/network.py).

  1. candidates = CI stations streaming HHZ on the public SeedLink server (what live can actually use)
  2. history    = first HHZ epoch at SCEDC (need years of archive to train on)
  3. noise      = median 2-8 Hz RMS ground velocity at 4 times of day over --days days
  4. score      = coverage of SoCal M3+ seismic areas (0.2-deg cells with >= 3 stations <= 100 km)
                  + major cities, under a minimum spacing, greedy; then compare with eq/network.py

  python scripts/select_network.py --days 3          # measure + report (does not edit network.py)

The chosen list is copied into src/eq/network.py by hand (with reasons), so a re-run can never
silently change the live network.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq import locate, network  # noqa: E402
from eq.catalog import load_catalog  # noqa: E402

OUT = ROOT / "data" / "raw" / "v2" / "network_selection.json"
CITIES = {"Los Angeles": (34.05, -118.24), "San Diego": (32.72, -117.16), "Riverside": (33.95, -117.40),
          "San Bernardino": (34.11, -117.29), "Irvine": (33.68, -117.83), "Bakersfield": (35.37, -119.02),
          "Santa Barbara": (34.42, -119.70), "Ventura": (34.27, -119.23), "Palm Springs": (33.83, -116.55),
          "Lancaster": (34.70, -118.14), "El Centro": (32.79, -115.56), "Ridgecrest": (35.62, -117.67),
          "Temecula": (33.49, -117.15), "San Luis Obispo": (35.28, -120.66), "Barstow": (34.90, -117.02)}


def candidates():
    from obspy.clients.fdsn import Client
    from obspy.clients.seedlink.basic_client import Client as SL
    info = SL("rtserve.iris.washington.edu", 18000, timeout=60).get_info(network="CI", station="*", level="channel")
    codes = sorted({r[1] for r in info if r[3] == "HHZ"})
    inv = Client("SCEDC", timeout=120).get_stations(network="CI", station=",".join(codes), channel="HHZ", level="channel")
    return {s.code: {"lat": s.latitude, "lon": s.longitude,
                     "first": str(min(c.start_date for c in s.channels))[:10]} for s in inv[0]}


def noise(code, days):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    c = Client("SCEDC", timeout=60)
    t0 = UTCDateTime() - 3 * 86400
    try:
        inv = c.get_stations(network="CI", station=code, channel="HHZ", level="response", starttime=t0 - days * 86400)
    except Exception:                                            # noqa: BLE001
        return None
    vals = []
    for d in range(days):
        for hh in (10, 17, 21, 3):                               # 03, 10, 14, 20 PDT
            t = t0 - d * 86400
            t = UTCDateTime(t.year, t.month, t.day, hh)
            try:
                st = c.get_waveforms("CI", code, "*", "HHZ", t, t + 120).merge(fill_value=0)
                tr = st[0]
                tr.detrend("demean"); tr.remove_sensitivity(inv); tr.filter("bandpass", freqmin=2, freqmax=8)
                vals.append(float(np.sqrt(np.median(tr.data ** 2))))
            except Exception:                                    # noqa: BLE001
                pass
    return float(np.median(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--n", type=int, default=19)
    ap.add_argument("--min-spacing", type=float, default=40.0)
    ap.add_argument("--max-noise", type=float, default=1e-7)
    ap.add_argument("--first-before", default="2010-12-31")
    args = ap.parse_args()
    cand = candidates()
    with ThreadPoolExecutor(8) as ex:
        nz = dict(zip(cand, ex.map(lambda c: noise(c, args.days), cand)))
    cat = load_catalog("2000-01-01", "2026-10-05", min_mag=1.0)
    cat = cat[cat.lat.between(32.55, 36.6) & cat.lon.between(-121, -114.3) & (cat.mag >= 3)]
    cells = np.unique(np.c_[np.floor(cat.lat / 0.2), np.floor(cat.lon / 0.2)], axis=0)
    glat, glon = (cells[:, 0] + 0.5) * 0.2, (cells[:, 1] + 0.5) * 0.2
    cp = np.array(list(CITIES.values()))

    def n_in(stas, la, lo):
        return sum(locate.haversine_km(la, lo, cand[s]["lat"], cand[s]["lon"]) <= 100 for s in stas)

    def score(stas):
        return {"cells_ge3": float(np.mean(n_in(stas, glat, glon) >= 3)),
                "cells_ge2": float(np.mean(n_in(stas, glat, glon) >= 2)),
                "cities": {c: int(n_in(stas, *p)) for c, p in CITIES.items()}}

    ok = [s for s in cand if nz.get(s) is not None and nz[s] <= args.max_noise and cand[s]["first"] <= args.first_before
          and 32.4 <= cand[s]["lat"] <= 36.6 and -121 <= cand[s]["lon"] <= -114]
    sel = []
    while len(sel) < args.n:
        best = None
        for s in ok:
            if s in sel or any(locate.haversine_km(cand[s]["lat"], cand[s]["lon"], cand[t]["lat"], cand[t]["lon"])
                               < args.min_spacing for t in sel):
                continue
            T = sel + [s]
            nc = n_in(T, cp[:, 0], cp[:, 1])
            v = (np.mean(n_in(T, glat, glon) >= 3) + 0.3 * np.mean(n_in(T, glat, glon) >= 2)
                 + 0.5 * np.mean(nc >= 3) + 0.2 * np.mean(nc >= 2) - 0.02 * np.log10(nz[s] / 1e-9))
            if best is None or v > best[0]:
                best = (v, s)
        if best is None:
            break
        sel.append(best[1])
    current = [c for c in network.CODES if c in cand]
    rep = {"greedy": sorted(sel), "greedy_score": score(sel), "current": network.CODES,
           "current_streamable": current, "current_missing": [c for c in network.CODES if c not in cand],
           "current_score": score(current), "noise": {c: nz.get(c) for c in network.CODES}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "noise"}, indent=1))
    print(pd.Series(rep["noise"]).map(lambda v: f"{v:.1e}" if v else "n/a").to_string())


if __name__ == "__main__":
    main()
