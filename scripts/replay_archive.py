"""Replay harness: run the EXACT live pipeline (src/eq/pipeline.py) over archived SCEDC data.

This is the acceptance test for the live system -- not test-set AUC. Two layers:

  scan   (expensive, cached)  fetch archived continuous vertical data hour by hour for the 19
         stations, apply the live preprocessing (demean + 18 Hz low-pass), score every 30 s window
         on a 2 s grid with the detector (DQ gate first), and cache P(quake) + the picker's result
         for every window that could trigger.  -> data/raw/v2/replay/<hour>.npz
  run    (cheap, per config)  feed the cached windows through Pipeline.triggerable/accept_pick, then
         associate / locate / size (3-C windows fetched on demand) / decide, exactly as live; write
         the declared events and SCORE them against the local catalog (strict match + a +1 h
         time-shifted CHANCE baseline).
  events (event-centric)      size + locate held-out TEST-period catalog events from the cached
         dataset windows (no fetch) -> magnitude bias/MAE and location error in live geometry.

Periods (never used to fit anything they evaluate):
  CALIBRATION = 2026-09-29..10-02 (validation live noise) + 2020-09-07..14 (validation-period week)
  TEST        = 2026-10-02..10-05 (held-out live noise; the old daemon ran these days too)
              + 2026-08-18..25 (test-period week, never trained on)

  python scripts/replay_archive.py scan      --start 2020-09-07,2026-08-18,2026-09-29 --end 2020-09-14,2026-08-25,2026-10-05
  python scripts/replay_archive.py calibrate --start 2026-09-29,2020-09-07 --end 2026-10-02,2020-09-14
  python scripts/replay_archive.py run       --start 2026-10-02,2026-08-18 --end 2026-10-05,2026-08-25
  python scripts/replay_archive.py compare-live --start 2026-10-02 --end 2026-10-05   # old VM log, same scorer
  python scripts/replay_archive.py events                                             # test-period events
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from eq import locate, network, seismic  # noqa: E402
from eq.pipeline import (NPTS, SR, ZNE_POST_S, ZNE_PRE_S, Config, MagnitudeEnsemble, Pipeline,  # noqa: E402
                         clean_window, detect_probs)
from eq.quakecast import load_catalog  # noqa: E402

CACHE = seismic.RAW / "replay"
OUT = ROOT / "data" / "processed" / "v2" / "replay"
STEP = 2.0
PICK_MIN_P = 0.2                     # cache picks for every window at/above this (any det_thresh >= it)
CAT_END = "2026-10-05"


def catalog():
    c = load_catalog("2000-01-01", CAT_END, min_mag=1.0)
    c["epoch"] = c.time.values.astype("datetime64[ms]").astype(np.int64) / 1e3
    return c


def ranges(args):
    """--start/--end accept comma-separated lists of matching dates -> [(start, end), ...]."""
    a, b = args.start.split(","), args.end.split(",")
    assert len(a) == len(b), "--start and --end need the same number of dates"
    return list(zip(a, b))


def coverage_mask(lat, lon, k=3, r=100.0):
    d = locate.haversine_km(np.asarray(lat)[:, None], np.asarray(lon)[:, None],
                            network.COORDS[:, 0][None], network.COORDS[:, 1][None])
    return (d <= r).sum(1) >= k


# ---------------------------------------------------------------- scan

def hour_window(h):
    return seismic.UTCDateTime(h.isoformat()) - NPTS / SR - 5, seismic.UTCDateTime(h.isoformat()) + 3600


def fetch_hour(h):
    t1, t2 = hour_window(h)
    return seismic.get_waveforms(seismic.get_client(), network.CODES, t1, t2, channel="HHZ")


def scan_hour(h, det, device, st):
    fp = CACHE / f"{h:%Y%m%dT%H}.npz"
    t1, t2 = hour_window(h)
    ends = np.arange(float(t1.timestamp) + NPTS / SR + 5, float(t2.timestamp) + 1e-6, STEP)
    rows_sta, rows_end, rows_p, rows_lag, rows_snr, avail = [], [], [], [], [], np.zeros(len(network.CODES), bool)
    for i, c in enumerate(network.CODES):
        tr = seismic.station_traces(st, c).select(channel="HHZ") if len(st) else []
        if not len(tr):
            continue
        tr = tr.copy().merge(fill_value=0)[0]
        x = seismic.lowpass(tr.data.astype(np.float64) - np.mean(tr.data))
        s0 = float(tr.stats.starttime.timestamp)
        wins, wend = [], []
        for e in ends:
            a = int(round((e - NPTS / SR - s0) * SR))
            if a < 0 or a + NPTS > len(x):
                continue
            w = x[a:a + NPTS]
            if clean_window(w):
                wins.append(w); wend.append(e)
        if not wins:
            continue
        avail[i] = True
        p = detect_probs(det, np.stack(wins), device)
        for w, e, pi in zip(wins, wend, p):
            lag, snr = np.nan, 0.0
            if pi >= PICK_MIN_P:
                k, snr = locate.pick_p(w, 50, len(w) - 50)
                lag = np.nan if k is None else (len(w) - k) / SR
            rows_sta.append(i); rows_end.append(e); rows_p.append(pi); rows_lag.append(lag); rows_snr.append(snr)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(fp, sta=np.array(rows_sta, np.int16), end=np.array(rows_end), p=np.array(rows_p, np.float16),
             lag=np.array(rows_lag, np.float32), snr=np.array(rows_snr, np.float32), avail=avail)
    return f"{len(rows_p)} windows, {int(avail.sum())} stations"


def cmd_scan(args):
    import torch
    from live_watch import load_detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from concurrent.futures import ThreadPoolExecutor
    det = load_detector(device)
    hours = [h for s, e in ranges(args) for h in pd.date_range(s, e, freq="h", inclusive="left")
             if not (CACHE / f"{h:%Y%m%dT%H}.npz").exists()]
    with ThreadPoolExecutor(6) as ex:                            # downloads overlap GPU scoring
        futs = [(h, ex.submit(fetch_hour, h)) for h in hours[:8]]   # bounded look-ahead (memory)
        nxt = 8
        while futs:
            h, fu = futs.pop(0)
            if nxt < len(hours):
                futs.append((hours[nxt], ex.submit(fetch_hour, hours[nxt])))
                nxt += 1
            try:
                print(f"scan {h:%Y-%m-%d %H}:00  {scan_hour(h, det, device, fu.result())}", flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"scan {h:%Y-%m-%d %H}:00  ERROR {e!r}", flush=True)


# ---------------------------------------------------------------- run

class ArchiveSource:
    """Replay waveform source: `now` is set by the driver; 3-C windows fetched on demand."""

    def __init__(self, inv):
        self.now, self.avail, self.inv = 0.0, np.zeros(len(network.CODES), bool), inv
        self.client = seismic.get_client()
        self.cache = {}

    def end(self, c):
        return self.now if self.avail[network.INDEX[c]] else None

    def z(self, c, t1, t2):
        return None

    def zne(self, c, t1, t2):
        key = (c, round(t1, 1))
        if key not in self.cache:
            a, b = seismic.UTCDateTime(t1) - ZNE_PRE_S, seismic.UTCDateTime(t2) + ZNE_POST_S
            try:
                st = seismic.get_waveforms(self.client, [c], a, b, channel="HH?")
                x = seismic.to_zne(seismic.station_traces(st, c), self.inv, a, int(round((b - a) * SR))) \
                    if len(st) else None
                i1 = int(round(ZNE_PRE_S * SR))
                self.cache[key] = None if x is None else x[:, i1:i1 + int(round((t2 - t1) * SR))]
            except Exception:                                    # noqa: BLE001
                self.cache[key] = None
        return self.cache[key]


def run_period(start, end, cfg, mag, inv):
    src = ArchiveSource(inv)
    events = []
    pipe = Pipeline(None, mag, network.COORDS, network.CODES, src, cfg, on_event=events.append,
                    early=None if mag is None else "auto")
    for h in pd.date_range(start, end, freq="h", inclusive="left"):
        fp = CACHE / f"{h:%Y%m%dT%H}.npz"
        if not fp.exists():
            continue
        z = np.load(fp)
        src.avail = z["avail"]
        keep = z["p"].astype(float) >= cfg.det_thresh             # only these can ever trigger
        order = np.argsort(z["end"][keep], kind="stable")
        sta, wend, p, lag, snr = (z["sta"][keep][order], z["end"][keep][order], z["p"][keep][order].astype(float),
                                  z["lag"][keep][order], z["snr"][keep][order])
        t = h.value / 1e9
        j = 0
        while t < h.value / 1e9 + 3600:
            t += STEP
            while j < len(wend) and wend[j] <= t:
                i = int(sta[j])
                if pipe.triggerable(i, wend[j], p[j]):
                    pipe.accept_pick(i, wend[j], None if np.isnan(lag[j]) else float(lag[j]), float(snr[j]), p[j])
                j += 1
            src.now = t
            pipe.advance(t)
    return events, pipe


def ev_record(ev, pipe):
    return {"epoch": ev.declared_at, "origin": ev.t0, "confirmed": ev.confirmed, "lat": ev.lat, "lon": ev.lon,
            "rms": ev.rms, "n_stations": len(ev.stations), "stations": [network.CODES[i] for i in ev.stations],
            "mag": ev.mag, "mag_spread": ev.mag_spread, "silent_near": ev.silent_near,
            "push_eligible": pipe.push_eligible(ev) if pipe is not None else None,
            "early_mag": ev.early_mag,
            "early_push": pipe.early_push_eligible(ev) if pipe is not None else None,
            "early_after_origin_s": None if ev.early_at is None else ev.early_at - ev.t0,
            "sized_after_origin_s": None if ev.sized_at is None else ev.sized_at - ev.t0}


def match(cat, t, lat, lon, dt=20.0, dkm=60.0):
    """Largest catalogued quake within dt seconds and dkm km of (t, lat, lon), or None."""
    lo, hi = np.searchsorted(cat.epoch.values, [t - dt, t + dt])
    if hi <= lo:
        return None
    sub = cat.iloc[lo:hi]
    d = locate.haversine_km(lat, lon, sub.lat.values, sub.lon.values)
    ok = d <= dkm
    if not ok.any():
        return None
    k = np.flatnonzero(ok)[np.argmax(sub.mag.values[ok])]
    return {"mag": float(sub.mag.values[k]), "lat": float(sub.lat.values[k]), "lon": float(sub.lon.values[k]),
            "epoch": float(sub.epoch.values[k]), "dist_km": float(d[k])}


def score(recs, cat, rngs, floor, tkey="origin", latkey="lat", lonkey="lon", label=""):
    """Precision (vs ANY catalogued M>=1 quake), push precision (vs M >= floor-0.5), recall of
    in-coverage M>=3, location + magnitude error -- each with a +1 h shifted chance baseline."""
    days = sum((pd.Timestamp(e) - pd.Timestamp(s)).total_seconds() / 86400 for s, e in rngs)
    conf = [r for r in recs if r.get("confirmed")]
    push = [r for r in conf if r.get("push_eligible", r.get("pushed"))]
    out = {"label": label, "days": round(days, 2), "declared": len(recs), "confirmed": len(conf), "push": len(push)}
    for name, subset in (("confirmed", conf), ("push", push)):
        real = [match(cat, r[tkey], r[latkey], r[lonkey]) for r in subset]
        chance = [match(cat, r[tkey] + 3600, r[latkey], r[lonkey]) for r in subset]
        if name == "push":
            real = [m if (m and m["mag"] >= floor - 0.5) else None for m in real]
            chance = [m if (m and m["mag"] >= floor - 0.5) else None for m in chance]
        n = max(len(subset), 1)
        out[f"{name}_precision"] = round(sum(m is not None for m in real) / n, 3) if subset else None
        out[f"{name}_chance"] = round(sum(m is not None for m in chance) / n, 3) if subset else None
        out[f"{name}_false_per_week"] = round(sum(m is None for m in real) / days * 7, 2)
        if name == "confirmed":
            pairs = [(r, m) for r, m in zip(subset, real) if m]
            out["loc_err_km_median"] = round(float(np.median([m["dist_km"] for _, m in pairs])), 1) if pairs else None
            mm = [(r["mag"], m["mag"]) for r, m in pairs if r.get("mag") is not None]
            if mm:
                e = np.array([a - b for a, b in mm])
                out["mag_bias"], out["mag_mae"], out["mag_n"] = round(float(e.mean()), 2), round(float(np.abs(e).mean()), 2), len(e)
    inr = np.zeros(len(cat), bool)
    for s, e in rngs:
        inr |= (cat.epoch.values >= pd.Timestamp(s).value / 1e9) & (cat.epoch.values < pd.Timestamp(e).value / 1e9)
    q = cat[inr & (cat.mag >= 3.0)]
    q = q[coverage_mask(q.lat.values, q.lon.values)]
    caught = [any(abs(r[tkey] - e) <= 20 and locate.haversine_km(r[latkey], r[lonkey], la, lo) <= 60 for r in conf)
              for e, la, lo in zip(q.epoch, q.lat, q.lon)]
    early = [r for r in conf if r.get("early_push")]
    if early or any("early_push" in r for r in recs):
        em = [match(cat, r[tkey], r[latkey], r[lonkey]) for r in early]
        out["early_push"] = len(early)
        out["early_push_real_M2.5+"] = sum(1 for m in em if m and m["mag"] >= floor - 0.5)
        out["early_retracted"] = sum(1 for r in early if not r.get("push_eligible"))
        out["final_push_without_early"] = sum(1 for r in push if not r.get("early_push"))
        lat_e = [r["early_after_origin_s"] for r in early if r.get("early_after_origin_s") is not None]
        lat_f = [r["sized_after_origin_s"] for r in push if r.get("sized_after_origin_s") is not None]
        out["early_push_s_after_origin_median"] = round(float(np.median(lat_e)), 1) if lat_e else None
        out["final_push_s_after_origin_median"] = round(float(np.median(lat_f)), 1) if lat_f else None
    out["m3_in_coverage"] = len(q)
    out["m3_recall"] = round(float(np.mean(caught)), 3) if len(q) else None
    return out


def cmd_run(args):
    import torch
    cfg = Config.load()
    for kv in args.set or []:
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mag = MagnitudeEnsemble(ROOT / "data" / "processed" / "magnitude_ensemble.pt", network.CODES, network.COORDS, device)
    inv = seismic.load_inventory()
    recs = []
    for s, e in ranges(args):
        events, pipe = run_period(s, e, cfg, mag, inv)
        recs += [ev_record(ev, pipe) for ev in events]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.start}_{args.end}"
    (OUT / f"events_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    s = score(recs, catalog(), ranges(args), cfg.alert_min_mag, label="new pipeline")
    s["config"] = asdict(cfg)
    (OUT / f"score_{tag}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def cmd_calibrate(args):
    """Grid-search the operating point on the VALIDATION period only (detection-only, no sizing):
    maximize real confirmed detections subject to <= --max-false-week false confirmations per week.
    Writes data/processed/v2/pipeline_config.json (read by live_watch and `run`)."""
    import itertools
    cat = catalog()
    grid = {"det_thresh": [0.5, 0.6, 0.7, 0.8, 0.9], "pick_snr": [3.0, 4.0, 6.0], "max_rms": [1.0, 1.5, 2.0],
            "max_silent_near": [0, 1], "min_stations": [3, 4]}
    days = sum((pd.Timestamp(e) - pd.Timestamp(s)).total_seconds() / 86400 for s, e in ranges(args))
    rows = []
    for vals in itertools.product(*grid.values()):
        cfg = Config(**dict(zip(grid, vals)))
        events = [ev for s, e in ranges(args) for ev in run_period(s, e, cfg, None, None)[0]]
        conf = [e for e in events if e.confirmed]
        real = sum(match(cat, e.t0, e.lat, e.lon) is not None for e in conf)
        rows.append({**dict(zip(grid, vals)), "confirmed": len(conf), "real": real,
                     "false_per_week": (len(conf) - real) / days * 7})
    df = pd.DataFrame(rows).sort_values(["real", "false_per_week"], ascending=[False, True])
    ok = df[df.false_per_week <= args.max_false_week]
    best = (ok if len(ok) else df.sort_values("false_per_week")).iloc[0]
    print(df.head(15).to_string(index=False))
    cfg = Config(**{k: (int(best[k]) if isinstance(getattr(Config(), k), int) else float(best[k])) for k in grid})
    from eq.pipeline import CONFIG_FILE
    CONFIG_FILE.write_text(json.dumps(asdict(cfg), indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "calibration.csv", index=False)
    print(f"chosen (validation {args.start}..{args.end}): {dict(best)}")
    print(f"wrote {CONFIG_FILE}")


def cmd_compare_live(args):
    """Score the OLD VM daemon's log for the same period with the same scorer (station proxy as
    location, declaration time as origin -- that is all the old system recorded)."""
    recs = [json.loads(x) for x in (seismic.RAW / "live_declarations_2026-09.jsonl").read_text().splitlines() if x]
    rg = [(pd.Timestamp(s).value / 1e9, pd.Timestamp(e).value / 1e9) for s, e in ranges(args)]
    recs = [dict(r, push_eligible=bool(r.get("pushed"))) for r in recs if any(a <= r["epoch"] < b for a, b in rg)]
    # the old daemon's declaration time lags the origin by travel time + latency; give it a wide window
    for r in recs:
        r["origin_guess"] = r["epoch"] - 20
    s = score(recs, catalog(), ranges(args), 3.0, tkey="origin_guess", latkey="proxy_lat", lonkey="proxy_lon",
              label="old live daemon (VM log)")
    print(json.dumps(s, indent=1))


# ---------------------------------------------------------------- event-centric

def cmd_events(args):
    """Locate + size held-out TEST-period events from cached dataset windows, in live geometry:
    picks from the shared picker on each station's cached vertical, location from those picks,
    windows cut at [P-5, P+25], distances from the LOCATED epicentre."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mag = MagnitudeEnsemble(ROOT / "data" / "processed" / "magnitude_ensemble.pt", network.CODES, network.COORDS, device)
    loc = locate.Locator(network.COORDS)
    d = np.load(ROOT / "data" / "processed" / "v2" / "magnitude.npz")
    t = d["ev_time"]
    test_start = np.sort(t)[int(len(t) * 0.85)]
    det_meta = np.load(ROOT / "data" / "processed" / "v2" / "detection.npz")
    tall = np.sort(np.concatenate([det_meta["pos_time"], det_meta["noise_time"]]))
    test_start = max(test_start, tall[int(len(tall) * 0.85)])
    ids = d["ev_id"]
    rows = []
    for e in np.flatnonzero(t >= test_start):
        fp = seismic.RAW / "events" / f"{ids[e]}.npz"
        z = np.load(fp)
        x = seismic.unpack(z["x"], z["s"])
        picks, pidx = {}, {}
        for k, i in enumerate(z["sta"]):
            p, snr = locate.pick_p(x[k, 0], 3200 - 600, 3200 + 600)
            if p is not None and snr >= Config().pick_snr:
                pidx[int(i)] = (k, p)
        if len(pidx) < 3:
            rows.append({"id": ids[e], "mag": float(d["mag"][e]), "located": False})
            continue
        # common time base: each cached window starts at origin + tt_pred - 32 s
        dd = locate.haversine_km(d["ev_lat"][e], d["ev_lon"][e], network.COORDS[:, 0], network.COORDS[:, 1])
        for i, (k, p) in pidx.items():
            tp = float(locate.travel_time(dd[i], max(float(np.nan_to_num(d["ev_depth"][e])), 0.0), corrected=False))
            picks[i] = tp - 32.0 + p / SR                      # seconds after the catalogue origin
        sol = loc.locate(picks)
        if sol is None or sol["rms"] > Config().max_rms:
            rows.append({"id": ids[e], "mag": float(d["mag"][e]), "located": False})
            continue
        dl = locate.haversine_km(sol["lat"], sol["lon"], network.COORDS[:, 0], network.COORDS[:, 1]).astype(np.float32)
        X = np.zeros((len(network.CODES), 3, NPTS), np.float32)
        M = np.zeros(len(network.CODES), bool)
        for k, i in enumerate(z["sta"]):
            if dl[i] > Config().size_radius_km:
                continue
            if int(i) in pidx:
                a = pidx[int(i)][1] - 500
            else:
                tp_loc = sol["t0"] + float(locate.travel_time(dl[i]))
                tp_pred = float(locate.travel_time(dd[i], max(float(np.nan_to_num(d["ev_depth"][e])), 0.0), corrected=False))
                a = int(round((tp_loc - (tp_pred - 32.0)) * SR)) - 500
            if 0 <= a and a + NPTS <= x.shape[2]:
                X[i], M[i] = x[k, :, a:a + NPTS], True
        m, spread = mag.predict(X, M, dl) if M.sum() else (None, None)
        rows.append({"id": ids[e], "mag": float(d["mag"][e]), "located": True, "est": m, "spread": spread,
                     "n_picks": len(sol["used"]), "rms": sol["rms"],
                     "loc_err_km": float(locate.haversine_km(d["ev_lat"][e], d["ev_lon"][e], sol["lat"], sol["lon"]))})
    df = pd.DataFrame(rows)
    ok = df[df.located & df.est.notna()]
    err = ok.est - ok.mag
    s = {"test_events": len(df), "located_frac": round(float(df.located.mean()), 3),
         "loc_err_km_median": round(float(ok.loc_err_km.median()), 1),
         "loc_err_km_p90": round(float(ok.loc_err_km.quantile(0.9)), 1),
         "mag_bias": round(float(err.mean()), 3), "mag_mae": round(float(err.abs().mean()), 3),
         "mag_r2": round(float(1 - np.sum(err ** 2) / np.sum((ok.mag - ok.mag.mean()) ** 2)), 3),
         "n_sized": len(ok)}
    for lo, hi in ((3.0, 3.5), (3.5, 4.5), (4.5, 9)):
        sel = (ok.mag >= lo) & (ok.mag < hi)
        if sel.any():
            s[f"bias_M{lo}-{hi}"] = round(float(err[sel].mean()), 2)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "events_test.csv", index=False)
    (OUT / "score_events_test.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "run", "events", "calibrate", "compare-live"])
    ap.add_argument("--max-false-week", type=float, default=1.0, dest="max_false_week")
    ap.add_argument("--start", default="2026-09-25")
    ap.add_argument("--end", default="2026-10-05")
    ap.add_argument("--set", nargs="*", help="override Config fields, e.g. det_thresh=0.7 max_rms=1.2")
    ap.add_argument("--tag")
    args = ap.parse_args()
    {"scan": cmd_scan, "run": cmd_run, "events": cmd_events, "calibrate": cmd_calibrate,
     "compare-live": cmd_compare_live}[args.cmd](args)


if __name__ == "__main__":
    main()
