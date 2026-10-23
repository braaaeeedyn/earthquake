"""Reviewer answers for the MMI model: data summary, DYFI staged comparison (raw / +event term / +offset).
Prints the numbers; writes only to data/processed/mmi_notes/ (DYFI cells cached there, fetched once from USGS).
Run before scripts/build_mmi_notes.py, which reads dyfi_cells_scored.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/mmi_notes"
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from eq import locate, network, shaking  # noqa: E402
import validate_mmi as vm  # noqa: E402

d = np.load(ROOT / "data/processed/v2/magnitude.npz")
pgv = (np.abs(d["x"].astype(np.float32)) * d["s"][..., None]).max(axis=(1, 2))
o = np.argsort(d["ev_time"], kind="stable"); n = len(o)
split = np.empty(n, "<U5")
split[o[:int(n * .7)]], split[o[int(n * .7):int(n * .85)]], split[o[int(n * .85):]] = "train", "val", "test"
rs = split[d["row_ev"]]
M, R = d["mag"][d["row_ev"]].astype(float), d["row_dist"].astype(float)
vs = shaking.CAL["station_vs30"]
codes = np.array(network.CODES)[d["row_sta"]]
keep = (R > 0) & (pgv > 0)
ts = lambda t: str(pd.Timestamp(t, unit="s"))[:10]

print("=== 1. GMPE data (v2 magnitude.npz) ===")
for s in ("train", "val", "test"):
    ev = np.flatnonzero(split == s); rr = keep & (rs == s)
    print(f"{s:5s} events {len(ev):5d}  {ts(d['ev_time'][ev].min())}..{ts(d['ev_time'][ev].max())}  "
          f"M {d['mag'][ev].min():.1f}-{d['mag'][ev].max():.1f}  records {rr.sum():6d}  "
          f"R {R[rr].min():.1f}-{R[rr].max():.0f} km  rec/event {rr.sum() / len(ev):.1f}")
mb, rb = [2, 3, 4, 5, 6, 7.2], [0, 5, 10, 20, 50, 100, 200, 400]
for s in ("train", "test"):
    rr = keep & (rs == s)
    t = pd.crosstab(pd.cut(M[rr], mb, right=False), pd.cut(R[rr], rb, right=False))
    print(f"\n{s} RECORDS, magnitude (rows) x epicentral distance km (cols):\n{t}")
    evM = d["mag"][split == s]
    print(f"{s} EVENTS by magnitude: {np.histogram(evM, mb)[0].tolist()} for bins {mb}")

# GMPE residual (obs - pred, log10) on TEST records by distance, without and with the event term
pred = np.array([np.log10(shaking.estimate_pgv(m, r, vs.get(c))) for m, r, c in zip(M, R, codes)])
res = np.log10(np.where(pgv > 0, pgv, np.nan)) - pred
dfr = pd.DataFrame({"ev": d["row_ev"], "M": M, "R": R, "res": res, "s": rs})[keep]
dfr["term"] = dfr.groupby("ev")["res"].transform(lambda x: np.clip(x.mean(), -.5, .5))
dfr["within"] = dfr["res"] - dfr["term"]
te = dfr[dfr.s == "test"]
print("\nTEST records, GMPE residual log10(obs/pred) by distance: n, mean, sd, sd after event term")
print(te.groupby(pd.cut(te.R, rb, right=False), observed=True).agg(
    n=("res", "size"), mean=("res", "mean"), sd=("res", "std"), sd_within=("within", "std")).round(3))
print(te.groupby(pd.cut(te.M, mb, right=False), observed=True).agg(n=("res", "size"), mean=("res", "mean")).round(3))
print(f"test residual sd {te.res.std():.3f}, after event term {te.within.std():.3f}; event-term sd {te.groupby('ev').term.first().std():.3f}")
print(f"depth (km) percentiles 5/50/95 all events: {np.nanpercentile(d['ev_depth'], [5, 50, 95]).round(1)}")

print("\n=== 2. DYFI validation set ===")
cache = OUT / "dyfi_cells.json"
if not cache.exists():
    q = {"format": "geojson", "starttime": "2022-04-15", "endtime": "2026-09-01", "minmagnitude": 3.5,
         "minlatitude": 32.3, "maxlatitude": 36.9, "minlongitude": -121.3, "maxlongitude": -114.0,
         "producttype": "dyfi", "orderby": "magnitude", "limit": 60}
    rows = []
    for f in vm.get(vm.FDSN + "?" + vm.urllib.parse.urlencode(q))["features"]:
        p = f["properties"]; lon, lat, dep = f["geometry"]["coordinates"][:3]
        try:
            dy = vm.get(p["detail"])["properties"]["products"]["dyfi"][0]["contents"]
            cells = vm.get(dy["dyfi_geo_10km.geojson"]["url"])["features"]
        except Exception as e:  # noqa: BLE001
            print("skip", p["title"], repr(e)[:80]); continue
        for c in cells:
            cp = c["properties"]
            if (cp.get("nresp") or 0) < 3 or cp.get("cdi") is None:
                continue
            clat, clon = vm.centroid(c["geometry"]); v = vm.vs30_at(clat, clon)
            if v is None:
                continue
            rows.append(dict(id=f["id"], mag=p["mag"], t=p["time"] / 1000, lat=lat, lon=lon, depth=dep, title=p["title"],
                             r=float(locate.haversine_km(lat, lon, clat, clon)), cdi=float(cp["cdi"]),
                             nresp=int(cp["nresp"]), vs30=v))
    cache.write_text(json.dumps(rows))
df = pd.DataFrame(json.loads(cache.read_text()))
qt = df.groupby("id").t.first()
t_split = np.median(qt.values)                                # same rule as validate_mmi: median quake time
df["fit"] = df.t <= t_split
for name, g in (("FIT (older)", df[df.fit]), ("EVAL (newer)", df[~df.fit])):
    q = g.groupby("id").agg(mag=("mag", "first"), t=("t", "first"), n=("r", "size"))
    print(f"{name}: {len(q)} quakes {ts(q.t.min())}..{ts(q.t.max())}, M {q.mag.min()}-{q.mag.max()} "
          f"(median {q.mag.median()}), {len(g)} cells, R {g.r.min():.1f}-{g.r.max():.0f} km, "
          f"cells/quake median {q.n.median():.0f} max {q.n.max()} ({q.n.max() / len(g):.0%} of cells)")
    print(pd.crosstab(pd.cut(g.mag, [3.5, 4, 4.5, 5, 6, 7.2], right=False), pd.cut(g.r, rb, right=False)))
    print("CDI distribution:", np.histogram(g.cdi, [1, 2, 3, 4, 5, 6, 7, 9])[0].tolist(), "bins 1..9")

# event term from OUR station records for each DYFI quake (catalogue M, epicentral R, recorded PGV)
idx = {e: i for i, e in enumerate(d["ev_id"])}
terms, nsta = {}, {}
for qid, g in df.groupby("id"):
    e = idx.get(qid)
    if e is None:                                              # fall back: same origin within 3 s
        dt = np.abs(d["ev_time"] - g.t.iloc[0]); e = int(dt.argmin()) if dt.min() < 3 else None
    if e is None:
        continue
    r = np.flatnonzero(d["row_ev"] == e)
    terms[qid] = shaking.event_term(g.mag.iloc[0], R[r], pgv[r], codes[r]); nsta[qid] = len(r)
df["term"] = df.id.map(terms)
print(f"\nevent term available for {len(terms)} / {df.id.nunique()} DYFI quakes "
      f"(stations per quake median {np.median(list(nsta.values())):.0f}); split of matched events: "
      f"{pd.Series([split[idx[k]] for k in terms if k in idx]).value_counts().to_dict()}")
print("event terms (log10):", pd.Series(terms).describe().round(3).to_dict())

raw = lambda g, term: np.array([shaking.estimate_mmi(m, r, v, t if term else 0.0, offset=0.0)
                                for m, r, v, t in zip(g.mag, g.r, g.vs30, g.term.fillna(0))])
df["mmi0"], df["mmi_t"] = raw(df, False), raw(df, True)
fit, ev = df[df.fit], df[~df.fit & df.term.notna()]
off0 = float(np.median(fit.cdi - fit.mmi0))
off_t = float(np.median(fit.dropna(subset=["term"]).cdi - fit.dropna(subset=["term"]).mmi_t))
print(f"\noffset refit on FIT quakes: without event term {off0:.3f} (production 0.76), with event term {off_t:.3f}")


def score(name, est, g):
    e = np.clip(est, 1, 10) - g.cdi.values
    per_q = pd.Series(np.abs(e)).groupby(g.id.values).mean()
    return dict(stage=name, MAE=round(np.abs(e).mean(), 2), bias=round(e.mean(), 2),
                RMSE=round(np.sqrt((e ** 2).mean()), 2), within1=round(np.mean(np.abs(e) <= 1), 3),
                MAE_per_quake=round(per_q.mean(), 2), n_cells=len(g), n_quakes=g.id.nunique())


evall = df[~df.fit]
rows = [score("A  GMPE+site+Worden (no term, no offset)", evall.mmi0, evall),
        score("C  + DYFI offset 0.76 (published)", evall.mmi0 + 0.76, evall),
        score("A' same quakes w/ term: no term, no offset", ev.mmi0, ev),
        score("B  + event term, no offset", ev.mmi_t, ev),
        score("C' no term + offset 0.76", ev.mmi0 + 0.76, ev),
        score(f"D  + event term + offset {off_t:.2f} (refit w/ term)", ev.mmi_t + off_t, ev),
        score("D' + event term + offset 0.76", ev.mmi_t + 0.76, ev)]
print(pd.DataFrame(rows).to_string(index=False))

print("\nEVAL residual (est - CDI) by distance, stage C (published) vs D (term + refit offset):")
ev = ev.assign(eC=np.clip(ev.mmi0 + .76, 1, 10) - ev.cdi, eD=np.clip(ev.mmi_t + off_t, 1, 10) - ev.cdi)
for col, bins in (("r", [0, 5, 10, 20, 50, 100, 200, 400]), ("mag", [3.5, 4, 4.5, 5, 7.2]), ("cdi", [1, 2, 3, 4, 5, 9])):
    print(ev.groupby(pd.cut(ev[col], bins, right=False), observed=True).agg(
        n=("eC", "size"), biasC=("eC", "mean"), maeC=("eC", lambda x: x.abs().mean()),
        biasD=("eD", "mean"), maeD=("eD", lambda x: x.abs().mean())).round(2))
low = ev.mmi0 < 1.0
print(f"\nlow-intensity: EVAL cells whose raw Worden MMI < 1 (clamped to 1): {low.sum()} of {len(ev)}; "
      f"their CDI median {ev.cdi[low].median() if low.any() else float('nan')}")
print(f"cells with raw MMI < 2 (Worden lower segment far below its data): {(ev.mmi0 < 2).sum()}")
df.to_csv(OUT / "dyfi_cells_scored.csv", index=False)
