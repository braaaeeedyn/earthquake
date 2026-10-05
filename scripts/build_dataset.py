"""Build the v2 training datasets (detection + magnitude) on the LIVE network, 2000 -> 2026-08.

Replaces seismic_build.py / seismic_build_multi.py (fixed 5-s offsets, catalog-proxy alignment,
stations that never streamed live, noise screened only against M>=2.5).

Stages (each resumable; fetched windows are cached under data/raw/v2/):
  check     every LIVE_NETWORK station must stream HH? on the SeedLink server (else abort)
  select    event / noise / hard-negative lists from the M1+ catalog
  fetch     download + response-remove (threaded, one FDSN request per event / noise time)
  assemble  pick P on every trace (same picker as live), fit the travel-time correction,
            write data/processed/v2/{magnitude,detection}.npz

Labels (see URGENT_PLAN.md section 2.2):
  magnitude  : catalog M>=3.0 events with network stations <= 200 km (light declustering: <= 25 per
               0.2-deg cell per month, largest kept, so one aftershock sequence can't dominate)
               + the M2-3 events with stations <= 100 km, so small live detections size correctly
  detection +: P-picked (SNR >= 3) windows of M>=3 events <= 200 km and M2-3 events <= 100 km
  detection -: random times (all hours) with NO catalogued M>=1 quake within 150 s / 150 km and no
               M>=3 within 300 s / 500 km of the station; plus HARD NEGATIVES = the live daemon's
               own false declarations (Sep 2026) that match no catalogued quake

  python scripts/build_dataset.py                 # all stages
  python scripts/build_dataset.py --stage fetch --workers 6
"""
import argparse
import json
import sys
from concurrent.futures import as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq import locate, network, seismic  # noqa: E402
from eq.quakecast import load_catalog  # noqa: E402

RAW = seismic.RAW
OUT = ROOT / "data" / "processed" / "v2"
LIVE_LOG = RAW / "live_declarations_2026-09.jsonl"     # copy of the VM events.jsonl (hard negatives)
SR = seismic.SR
PRE = int(seismic.PRE_S * SR)                           # predicted-P index in a long window (3200)

MAG_MIN, MAG_R = 3.0, 200.0
DETX_LO, DETX_R, DETX_N = 2.0, 100.0, 4000
NOISE_TIMES, NOISE_LEN = 900, 45.0
HARD_LEN = 80.0
REGION = (32.0, 36.8, -121.5, -114.0)                   # where events are taken from
PICK_SEARCH = int(6 * SR)                               # +-6 s around predicted P
SNR_DET, SNR_ALIGN = 3.0, 2.5
DET_PRE, DET_LEN = int(26 * SR), int(56 * SR)           # detection positive: P-26 s .. P+30 s
MAG_PRE, MAG_LEN = int(6 * SR), int(32 * SR)            # magnitude: P-6 s .. P+26 s (1 s crop slack)


# ---------------------------------------------------------------- check

def check_seedlink(server="rtserve.iris.washington.edu", port=18000):
    from obspy.clients.seedlink.basic_client import Client as SL
    info = SL(server, port, timeout=60).get_info(network=network.NET, station="*", level="channel")
    have = {(r[1], r[2]) for r in info if r[3] == "HHZ"}
    missing = [c for c in network.CODES if (c, network.LOC[c]) not in have]
    if missing:
        raise SystemExit(f"NOT streamable on {server}: {missing} -- fix eq/network.py first")
    print(f"check: all {len(network.CODES)} stations stream HHZ (pinned location codes) on {server}")


# ---------------------------------------------------------------- select

def station_dists(lat, lon):
    return locate.haversine_km(np.asarray(lat)[:, None], np.asarray(lon)[:, None],
                               network.COORDS[:, 0][None], network.COORDS[:, 1][None])


def select(args, cat):
    rng = np.random.default_rng(0)
    t = cat["time"]
    inwin = (t >= args.start) & (t < args.end)
    inreg = cat.lat.between(REGION[0], REGION[1]) & cat.lon.between(REGION[2], REGION[3])

    m = cat[inwin & inreg & (cat.mag >= MAG_MIN)].copy()
    d = station_dists(m.lat.values, m.lon.values)
    m = m[(d <= MAG_R).sum(1) >= 3].copy()
    m["cell"] = (np.floor(m.lat / 0.2).astype(int).astype(str) + "_" + np.floor(m.lon / 0.2).astype(int).astype(str)
                 + "_" + m.time.dt.strftime("%Y%m"))
    m = m.sort_values("mag", ascending=False).groupby("cell").head(25).sort_values("time")
    m.drop(columns="cell").to_csv(RAW / "sel_mag.csv", index=False)

    x = cat[inwin & inreg & (cat.mag >= DETX_LO) & (cat.mag < MAG_MIN)]
    dx = station_dists(x.lat.values, x.lon.values)
    x = x[(dx <= DETX_R).any(1)]
    x = x.iloc[np.sort(rng.choice(len(x), min(DETX_N, len(x)), replace=False))]
    x.to_csv(RAW / "sel_detx.csv", index=False)

    t0, t1 = pd.Timestamp(args.start).value / 1e9, pd.Timestamp(args.end).value / 1e9
    nt = pd.DataFrame({"t": np.sort(rng.uniform(t0, t1, NOISE_TIMES)).round(2)})
    nt.to_csv(RAW / "sel_noise.csv", index=False)
    print(f"select: magnitude events {len(m)} (M>={MAG_MIN}, declustered), detection-extra {len(x)} "
          f"(M{DETX_LO}-{MAG_MIN}), noise times {len(nt)}")


# ---------------------------------------------------------------- fetch

def _fetch_event(row, radius, inv):
    fp = RAW / "events" / f"{row.id}.npz"
    if fp.exists():
        return "cached"
    fp.parent.mkdir(parents=True, exist_ok=True)
    d = station_dists([row.lat], [row.lon])[0]
    idx = np.flatnonzero(d <= radius)
    tp = locate.travel_time(d[idx], max(float(np.nan_to_num(row.depth)), 0.0), corrected=False)
    origin = seismic.utc(row.time)
    t1, t2 = origin + tp.min() - seismic.PRE_S - 1, origin + tp.max() + seismic.POST_S + 1
    st = seismic.get_waveforms(seismic.get_client(), [network.CODES[i] for i in idx], t1, t2, channel="HH?,BH?")
    keep, xs = [], []
    for i, p in zip(idx, tp):
        c = network.CODES[i]
        w0 = origin + p - seismic.PRE_S
        x = seismic.to_zne(seismic.station_traces(st, c), inv, w0, seismic.NLONG) if len(st) else None
        if seismic.complete(x):
            keep.append(i); xs.append(x)
    if not keep:
        np.savez(fp, sta=np.zeros(0, int))
        return "empty"
    xh, s = seismic.pack(np.stack(xs))
    np.savez(fp, sta=np.asarray(keep), dist=d[keep].astype(np.float32), x=xh, s=s)
    return f"{len(keep)} sta"


def _screen(tt, codes_idx, cat_t, cat_lat, cat_lon, cat_mag, length):
    """Indices (into codes_idx) of stations whose window [tt, tt+length] has no catalogued quake
    nearby: M>=1 within 150 s / 150 km, or M>=3 within 300 s / 500 km."""
    lo, hi = np.searchsorted(cat_t, [tt - 300, tt + length + 300])
    if hi <= lo:
        return list(range(len(codes_idx)))
    et, ela, elo, em = cat_t[lo:hi], cat_lat[lo:hi], cat_lon[lo:hi], cat_mag[lo:hi]
    near_t = (et >= tt - 150) & (et <= tt + length + 30)
    ok = []
    for k, i in enumerate(codes_idx):
        dd = locate.haversine_km(network.COORDS[i, 0], network.COORDS[i, 1], ela, elo)
        bad = (near_t & (dd <= 150)) | ((em >= 3) & (dd <= 500))
        if not bad.any():
            ok.append(k)
    return ok


def _fetch_noise(tt, inv, cat_arrays):
    fp = RAW / "noise" / f"{tt:.2f}.npz"
    if fp.exists():
        return "cached"
    fp.parent.mkdir(parents=True, exist_ok=True)
    allidx = list(range(len(network.CODES)))
    ok = _screen(tt, allidx, *cat_arrays, NOISE_LEN)
    t1 = seismic.UTCDateTime(tt)
    st = seismic.get_waveforms(seismic.get_client(), [network.CODES[i] for i in ok], t1, t1 + NOISE_LEN,
                               channel="HHZ,BHZ") if ok else []
    keep, zs = [], []
    for k in ok:
        c = network.CODES[k]
        x = seismic.to_zne(seismic.station_traces(st, c), inv, t1, int(NOISE_LEN * SR), output="SENS") \
            if len(st) else None
        if seismic.complete(x):
            keep.append(k); zs.append(x[0])
    if not keep:
        np.savez(fp, sta=np.zeros(0, int))
        return "empty"
    zh, s = seismic.pack(np.stack(zs))
    np.savez(fp, sta=np.asarray(keep), x=zh, s=s)
    return f"{len(keep)} sta"


def _fetch_hard(k, rec, inv, cat_arrays):
    """A live false declaration: the declaring stations' Z for the 80 s before it, if no catalogued
    quake explains it. Station codes are the OLD live network (PASC, MWC, SVD, DGR, BAK)."""
    fp = RAW / "hardneg" / f"{k:05d}.npz"
    if fp.exists():
        return "cached"
    fp.parent.mkdir(parents=True, exist_ok=True)
    tt = rec["epoch"] - HARD_LEN + 5
    codes = rec["stations"] or [rec["proxy_station"]]
    # stations outside the new network are screened by distance from the declaration's proxy location
    lo, hi = np.searchsorted(cat_arrays[0], [tt - 300, tt + HARD_LEN + 300])
    et, ela, elo, em = (a[lo:hi] for a in cat_arrays)
    dd = locate.haversine_km(rec["proxy_lat"], rec["proxy_lon"], ela, elo)
    near_t = (et >= tt - 150) & (et <= tt + HARD_LEN + 30)
    if ((near_t & (dd <= 200)) | ((em >= 3) & (dd <= 500))).any():
        np.savez(fp, sta=np.zeros(0, "U8"))
        return "explained"
    t1 = seismic.UTCDateTime(tt)
    st = seismic.get_waveforms(seismic.get_client(), codes, t1, t1 + HARD_LEN, channel="HHZ")
    keep, zs = [], []
    for c in codes:
        x = seismic.to_zne(seismic.station_traces(st, c, loc="10" if c == "PASC" else ""), inv, t1,
                           int(HARD_LEN * SR), output="SENS") if len(st) else None
        if seismic.complete(x):
            keep.append(c); zs.append(x[0])
    if not keep:
        np.savez(fp, sta=np.zeros(0, "U8"))
        return "empty"
    zh, s = seismic.pack(np.stack(zs))
    np.savez(fp, sta=np.asarray(keep), x=zh, s=s, epoch=rec["epoch"])
    return f"{len(keep)} sta"


_W = {}


def _init_worker(catalog_end):
    """Process-pool initializer: each worker loads the response inventory + catalog once."""
    import warnings
    warnings.filterwarnings("ignore")
    _W["inv"] = seismic.load_inventory(extra=("MWC", "BAK"))
    cat = load_catalog("2000-01-01", catalog_end, min_mag=1.0)
    _W["cat"] = (cat.time.values.astype("datetime64[ms]").astype(np.int64) / 1e3,
                 cat.lat.values, cat.lon.values, cat.mag.values)


def _job(kind, payload):
    if kind == "event":
        row, radius = payload
        return _fetch_event(row, radius, _W["inv"])
    if kind == "noise":
        return _fetch_noise(payload, _W["inv"], _W["cat"])
    k, rec = payload
    return _fetch_hard(k, rec, _W["inv"], _W["cat"])


def fetch(args, cat):
    from concurrent.futures import ProcessPoolExecutor
    seismic.load_inventory(extra=("MWC", "BAK"))                  # cache it once before forking workers
    jobs = []
    if "events" in args.what:
        for r in pd.read_csv(RAW / "sel_mag.csv", parse_dates=["time"]).itertuples(index=False):
            if not (RAW / "events" / f"{r.id}.npz").exists():
                jobs.append(("event", (SimpleNamespace(**r._asdict()), MAG_R)))
    if "detx" in args.what:
        for r in pd.read_csv(RAW / "sel_detx.csv", parse_dates=["time"]).itertuples(index=False):
            if not (RAW / "events" / f"{r.id}.npz").exists():
                jobs.append(("event", (SimpleNamespace(**r._asdict()), DETX_R)))
    if "noise" in args.what:
        for tt in pd.read_csv(RAW / "sel_noise.csv")["t"]:
            if not (RAW / "noise" / f"{float(tt):.2f}.npz").exists():
                jobs.append(("noise", float(tt)))
    if "hardneg" in args.what and LIVE_LOG.exists():
        recs = [json.loads(x) for x in LIVE_LOG.read_text().splitlines() if x.strip()]
        for k, rec in enumerate(recs):
            if not (RAW / "hardneg" / f"{k:05d}.npz").exists():
                jobs.append(("hard", (k, rec)))
    print(f"fetch: {len(jobs)} requests with {args.workers} worker processes", flush=True)
    done, stats = 0, {}
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.catalog_end,)) as ex:
        futs = [ex.submit(_job, k, p) for k, p in jobs]
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:                                # noqa: BLE001
                r = f"error {type(e).__name__}"
            key = "ok" if r[0].isdigit() else r.split()[0]
            stats[key] = stats.get(key, 0) + 1
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)}  {stats}", flush=True)
    print(f"fetch done: {stats}", flush=True)


# ---------------------------------------------------------------- assemble

def assemble(args, cat):
    OUT.mkdir(parents=True, exist_ok=True)
    rows_tt = []                                             # (dist, depth, observed tt) for correction
    # ---- events (magnitude rows + detection positives) ----
    mag_ev, mag_rows, det_pos = [], [], []
    # BOTH lists feed the magnitude set: M>=3 events (stations <= 200 km) and M2-3 events (<= 100 km).
    # The live detector confirms many M1.5-3 quakes; a magnitude net that never saw them sizes them all
    # at ~M3 (validated in the replay harness), which breaks the felt-shaking push floor.
    for src, src_big in (("sel_mag.csv", True), ("sel_detx.csv", False)):
        sel = pd.read_csv(RAW / src, parse_dates=["time"])
        for r in sel.itertuples():
            fp = RAW / "events" / f"{r.id}.npz"
            if not fp.exists():
                continue
            z = np.load(fp)
            if not len(z["sta"]):
                continue
            x = seismic.unpack(z["x"], z["s"])                # (n, 3, NLONG) m/s
            otime = pd.Timestamp(r.time).value / 1e9
            ev_rows = []
            for k, (i, dist) in enumerate(zip(z["sta"], z["dist"])):
                if not np.isfinite(x[k]).all():
                    continue
                tp = float(locate.travel_time(dist, max(float(np.nan_to_num(r.depth)), 0.0), corrected=False))
                p, snr = locate.pick_p(x[k, 0], PRE - PICK_SEARCH, PRE + PICK_SEARCH)
                if p is not None and snr >= 5:
                    rows_tt.append((dist, (p - PRE) / SR + tp))
                aligned = p if (p is not None and snr >= SNR_ALIGN
                                and MAG_PRE <= p <= x.shape[2] - (MAG_LEN - MAG_PRE)) else PRE
                det_ok = (not args.skip_detection and p is not None and snr >= SNR_DET and (src_big or dist <= DETX_R)
                          and DET_PRE <= p <= x.shape[2] - (DET_LEN - DET_PRE))
                if det_ok:
                    det_pos.append((x[k, 0, p - DET_PRE:p - DET_PRE + DET_LEN].copy(), otime, i, r.mag, dist, snr))
                a = aligned - MAG_PRE
                ev_rows.append((i, dist, seismic.pack(x[k, :, a:a + MAG_LEN]), p is not None and snr >= SNR_ALIGN, snr))
            if len(ev_rows) >= 3:
                e = len(mag_ev)
                mag_ev.append((r.id, otime, r.lat, r.lon, r.depth, r.mag))
                for i, dist, w, ok, snr in ev_rows:
                    mag_rows.append((e, i, dist, w, ok, snr))
        print(f"assemble: {src} done -> {len(mag_ev)} magnitude events, {len(det_pos)} detection positives",
              flush=True)

    # ---- travel-time correction (median residual by 10 km bin) ----
    tt = np.array(rows_tt)
    bins = np.arange(0, 260, 10)
    table = []
    for b0 in bins:
        sel = (tt[:, 0] >= b0) & (tt[:, 0] < b0 + 10)
        if sel.sum() >= 20:
            pred = locate.travel_time(tt[sel, 0], locate.DEPTH_KM, corrected=False)
            table.append([b0 + 5.0, float(np.median(tt[sel, 1] - pred))])
    locate.TT_CORR.parent.mkdir(parents=True, exist_ok=True)
    locate.TT_CORR.write_text(json.dumps({"table": table, "n_picks": int(len(tt))}, indent=1))
    print(f"assemble: travel-time correction from {len(tt)} picks: {table[:4]} ...")

    Xm, Sm = np.stack([r[3][0] for r in mag_rows]), np.stack([r[3][1] for r in mag_rows])
    ev = np.array(mag_ev, dtype=object)
    np.savez(OUT / "magnitude.npz",
             ev_id=ev[:, 0].astype(str), ev_time=ev[:, 1].astype(float), ev_lat=ev[:, 2].astype(float),
             ev_lon=ev[:, 3].astype(float), ev_depth=ev[:, 4].astype(float), mag=ev[:, 5].astype(np.float32),
             row_ev=np.array([r[0] for r in mag_rows]), row_sta=np.array([r[1] for r in mag_rows]),
             row_dist=np.array([r[2] for r in mag_rows], np.float32),
             row_pick_ok=np.array([r[4] for r in mag_rows]), row_snr=np.array([r[5] for r in mag_rows], np.float32),
             x=Xm, s=Sm, coords=network.COORDS, stations=np.array(network.CODES))
    del Xm
    print(f"assemble: magnitude.npz {len(mag_ev)} events (M {ev[:, 5].astype(float).min():.1f}.."
          f"{ev[:, 5].astype(float).max():.1f}), {len(mag_rows)} station rows", flush=True)
    if args.skip_detection:
        meta_fp = OUT / "dataset_meta.json"
        meta = json.loads(meta_fp.read_text()) if meta_fp.exists() else {}
        meta.update({"mag_events": len(mag_ev), "mag_rows": len(mag_rows), "mag_includes_m2": True})
        meta_fp.write_text(json.dumps(meta, indent=1))
        return

    # ---- detection ----
    def norm16(w):
        w = w - w.mean(-1, keepdims=True)
        return (w / (w.std(-1, keepdims=True) + 1e-12)).astype(np.float16)

    pos = norm16(np.stack([p[0] for p in det_pos]))
    noise, ntime, nsta = [], [], []
    for fp in sorted((RAW / "noise").glob("*.npz")):
        z = np.load(fp)
        if len(z["sta"]):
            w = seismic.unpack(z["x"], z["s"])
            noise.append(norm16(w)); ntime += [float(fp.stem)] * len(w); nsta += list(z["sta"])
    hard, htime, hsta = [], [], []
    for fp in sorted((RAW / "hardneg").glob("*.npz")):
        z = np.load(fp)
        if len(z["sta"]):
            w = seismic.unpack(z["x"], z["s"])
            hard.append(norm16(w)); htime += [float(z["epoch"])] * len(w); hsta += [str(s) for s in z["sta"]]
    np.savez(OUT / "detection.npz",
             pos=pos, pos_time=np.array([p[1] for p in det_pos]), pos_sta=np.array([p[2] for p in det_pos]),
             pos_mag=np.array([p[3] for p in det_pos], np.float32),
             pos_dist=np.array([p[4] for p in det_pos], np.float32), p_index=DET_PRE,
             noise=np.concatenate(noise), noise_time=np.array(ntime), noise_sta=np.array(nsta),
             hard=np.concatenate(hard) if hard else np.zeros((0, int(HARD_LEN * SR)), np.float16),
             hard_time=np.array(htime), hard_sta=np.array(hsta), stations=np.array(network.CODES))
    print(f"assemble: detection pos={len(pos)} noise={sum(len(n) for n in noise)} hard={sum(len(h) for h in hard)}")
    meta = {"start": args.start, "end": args.end, "mag_events": len(mag_ev), "mag_rows": len(mag_rows),
            "det_pos": len(det_pos), "stations": network.CODES, "mag_includes_m2": True}
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "check", "select", "fetch", "assemble"])
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2026-09-01", help="event data end (Sep 2026 = hard negatives / live test)")
    ap.add_argument("--catalog-end", default="2026-10-05")
    ap.add_argument("--what", default="events,detx,noise,hardneg")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-detection", action="store_true", dest="skip_detection",
                    help="assemble only magnitude.npz (low memory)")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "check"):
        check_seedlink()
    cat = load_catalog("2000-01-01", args.catalog_end, min_mag=1.0) if args.stage != "check" else None
    if args.stage in ("all", "select"):
        select(args, cat)
    if args.stage in ("all", "fetch"):
        fetch(args, cat)
    if args.stage in ("all", "assemble"):
        assemble(args, cat)


if __name__ == "__main__":
    main()
