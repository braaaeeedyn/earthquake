"""Validate the shaking estimate against what people reported: USGS "Did You Feel It?" (DYFI).

For held-out Southern California quakes (2022-04-15 .. 2026-08-31, M >= 3.5, catalogue location and magnitude),
download DYFI's 10 km cells (community intensity CDI, number of responses) and compare with our estimate at each
cell centre: src/eq/shaking.py with the Vs30 at the cell. Cells with >= 3 responses inside the SoCal map frame are
scored. The chain underestimates what people report, so an MMI offset is fitted on the OLDER half of the quakes
(median residual) and accuracy is reported on the NEWER half only. This is done twice: without an event term
("mmi_offset") and with each quake's event term from our archived station records (v2 magnitude.npz, as live sizing
computes it; "mmi_offset_term"), because live confirmations apply one. Both offsets and their held-out summaries go
into shaking_calibration.json, which every estimator uses and /api/shaking-model serves to the app.

  python scripts/validate_mmi.py [--max-quakes 60]
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq import locate, network, shaking  # noqa: E402

FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
GRID = json.loads((ROOT / "app" / "public" / "vs30_socal.json").read_text())
OUT = ROOT / "data" / "processed" / "mmi_validation.json"
MAG_NPZ = ROOT / "data" / "processed" / "v2" / "magnitude.npz"


def event_terms():
    """term(id, time, mag): the quake's event term from our archived station records, or None if we have none."""
    d = np.load(MAG_NPZ)
    pgv = (np.abs(d["x"]).max(axis=2).astype(np.float32) * d["s"]).max(axis=1)   # peak |velocity| per record, m/s
    codes = np.array(network.CODES)[d["row_sta"]]
    idx = {e: i for i, e in enumerate(d["ev_id"])}

    def term(qid, t, mag):
        e = idx.get(qid)
        if e is None:                                            # same quake under another id: origin within 3 s
            dt = np.abs(d["ev_time"] - t)
            e = int(dt.argmin()) if dt.min() < 3 else None
        if e is None:
            return None
        r = np.flatnonzero(d["row_ev"] == e)
        return shaking.event_term(mag, d["row_dist"][r], pgv[r], codes[r])
    return term


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "seismicsocal-validate/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def vs30_at(lat, lon):
    i, j = round((lat - GRID["lat0"]) / GRID["step"]), round((lon - GRID["lon0"]) / GRID["step"])
    if 0 <= i < GRID["nlat"] and 0 <= j < GRID["nlon"]:
        return GRID["vs30"][i * GRID["nlon"] + j] or None
    return None


def centroid(geom):
    pts = np.asarray(geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0], float)
    return float(pts[:, 1].mean()), float(pts[:, 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-quakes", type=int, default=60)
    args = ap.parse_args()
    q = {"format": "geojson", "starttime": "2022-04-15", "endtime": "2026-09-01", "minmagnitude": 3.5,
         "minlatitude": 32.3, "maxlatitude": 36.9, "minlongitude": -121.3, "maxlongitude": -114.0,
         "producttype": "dyfi", "orderby": "magnitude", "limit": args.max_quakes}
    quakes = get(FDSN + "?" + urllib.parse.urlencode(q))["features"]
    term_of = event_terms()
    rows = []
    for f in quakes:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        term = term_of(f["id"], p["time"] / 1000.0, p["mag"])
        try:
            dyfi = get(p["detail"])["properties"]["products"]["dyfi"][0]["contents"]
            cells = get(dyfi["dyfi_geo_10km.geojson"]["url"])["features"]
        except Exception as e:                                   # noqa: BLE001
            print(f"  skip {p['title']}: {e!r}"[:120])
            continue
        n = 0
        for c in cells:
            cp = c["properties"]
            if (cp.get("nresp") or 0) < 3 or cp.get("cdi") is None:
                continue
            clat, clon = centroid(c["geometry"])
            v = vs30_at(clat, clon)
            if v is None:
                continue
            r = float(locate.haversine_km(lat, lon, clat, clon))
            rows.append((p["mag"], r, float(cp["cdi"]), shaking.estimate_mmi(p["mag"], r, v, offset=0.0),
                         shaking.estimate_mmi(p["mag"], r, offset=0.0), p["time"] / 1000.0,
                         np.nan if term is None else shaking.estimate_mmi(p["mag"], r, v, term, offset=0.0)))
            n += 1
        print(f"  {p['title'][:60]:60s} {n} cells", flush=True)
    a = np.array(rows)
    t_split = np.median(np.unique(a[:, 5]))                       # older half of the QUAKES fits the offset
    fit, ev = a[:, 5] <= t_split, a[:, 5] > t_split
    offset = float(np.median(a[fit, 2] - a[fit, 3]))
    obs, est = a[ev, 2], np.clip(a[ev, 3] + offset, 1, 10)
    est_nosite = np.clip(a[ev, 4] + float(np.median(a[fit, 2] - a[fit, 4])), 1, 10)
    raw = a[ev, 3]
    res = {"mmi_offset": round(offset, 2), "n_quakes_fit": int(len(np.unique(a[fit, 5]))),
           "n_quakes": int(len(np.unique(a[ev, 5]))), "n_cells": int(ev.sum()),
           "mae": round(float(np.abs(est - obs).mean()), 2), "bias": round(float((est - obs).mean()), 2),
           "within1": round(float(np.mean(np.abs(est - obs) <= 1.0)), 3),
           "mae_without_offset": round(float(np.abs(raw - obs).mean()), 2),
           "mae_without_site_term": round(float(np.abs(est_nosite - obs).mean()), 2),
           "source": "USGS DYFI 10 km cells, >= 3 responses, M>=3.5 SoCal 2022-04-15..2026-08-31; offset fitted on the "
                     "older half of the quakes, scores on the newer half"}
    for lo, hi in ((0, 50), (50, 100), (100, 400)):
        sel = (a[ev, 1] >= lo) & (a[ev, 1] < hi)
        if sel.any():
            res[f"mae_{lo}-{hi}km"] = round(float(np.abs(est[sel] - obs[sel]).mean()), 2)
    # the same fit and score for estimates WITH an event term (quakes our stations recorded)
    has = ~np.isnan(a[:, 6])
    offset_term = float(np.median(a[fit & has, 2] - a[fit & has, 6]))
    obs_t, est_t = a[ev & has, 2], np.clip(a[ev & has, 6] + offset_term, 1, 10)
    res["mmi_offset_term"] = round(offset_term, 2)
    res["with_event_term"] = {"mae": round(float(np.abs(est_t - obs_t).mean()), 2),
                              "bias": round(float((est_t - obs_t).mean()), 2),
                              "within1": round(float(np.mean(np.abs(est_t - obs_t) <= 1.0)), 3),
                              "n_cells": int(len(obs_t)), "n_quakes": int(len(np.unique(a[ev & has, 5]))),
                              "n_quakes_fit": int(len(np.unique(a[fit & has, 5])))}
    OUT.write_text(json.dumps(res, indent=1))
    cal = json.loads(shaking.CAL_PATH.read_text())
    cal["mmi_offset"] = res["mmi_offset"]
    cal["mmi_offset_term"] = res["mmi_offset_term"]
    cal["validation"] = {k: res[k] for k in ("mae", "bias", "within1", "n_cells", "n_quakes")}
    cal["validation_with_event_term"] = res["with_event_term"]
    shaking.CAL_PATH.write_text(json.dumps(cal, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
