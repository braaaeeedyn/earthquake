"""Score the live watcher's declared events against the USGS catalog -> a standing scorecard.

Reads the durable event log written by live_watch.py (data/processed/events.jsonl) and, for the
covered time window, asks the USGS FDSN catalog what really happened:

  PRECISION side  — each DECLARED event is a true detection if a real quake occurred close in
                    time and space to its (proxy) location, else a FALSE ALARM.
  RECALL side     — each REAL quake in the network region (>= --mag-floor) is CAUGHT if some
                    declared event matches it, else MISSED.

Run it by hand any time, or nightly (e.g. a cron / systemd timer on the host):
  python scripts/crosscheck_events.py
  python scripts/crosscheck_events.py --since 2026-09-01 --mag-floor 3.0 --json report.json

Honesty: the recall denominator assumes the network can detect quakes of >= --mag-floor within
--radius km of a station; that floor is a judgement call, so it is a CLI knob printed with the result.
Matching uses the LOCATED origin + epicentre (v2 logs; the old schema falls back to declaration time
+ station proxy) with tight defaults, and every precision is printed next to a CHANCE baseline (the
same declarations shifted +1 h): SoCal has ~50 catalogued M1+ quakes a day, so loose windows "match"
noise by coincidence -- the old generous 180 s / 100 km scorecard did exactly that.
"""
import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS_LOG = ROOT / "data" / "processed" / "events.jsonl"
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def haversine_km(lat1, lon1, lat2, lon2):
    r = math.radians
    return 2 * 6371 * math.asin(math.sqrt(
        math.sin(r(lat2 - lat1) / 2) ** 2
        + math.cos(r(lat1)) * math.cos(r(lat2)) * math.sin(r(lon2 - lon1) / 2) ** 2))


def load_events(since_epoch, log_path=EVENTS_LOG):
    """Declared events from the JSONL log, on/after since_epoch, oldest first."""
    if not log_path.exists():
        return []
    evs = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("epoch", 0) >= since_epoch:
            evs.append(e)
    return sorted(evs, key=lambda e: e["epoch"])


def station_coords():
    """(lats, lons) of the live network (src/eq/network.py), for the region bound."""
    sys.path.insert(0, str(ROOT / "src"))
    from eq import network
    return network.COORDS[:, 0].tolist(), network.COORDS[:, 1].tolist()


def usgs_query(params):
    params = {"format": "geojson", **params}
    req = urllib.request.Request(FDSN + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "seismicsocal-crosscheck/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def real_quakes_in_region(since_epoch, now_epoch, lats, lons, radius_km, mag_floor):
    """Real quakes >= mag_floor within radius_km of ANY station, over the coverage window."""
    margin = radius_km / 111.0                       # deg padding on the station bounding box
    g = usgs_query({"starttime": iso(since_epoch), "endtime": iso(now_epoch),
                    "minmagnitude": mag_floor,
                    "minlatitude": min(lats) - margin, "maxlatitude": max(lats) + margin,
                    "minlongitude": min(lons) - margin, "maxlongitude": max(lons) + margin})
    out = []
    for f in g.get("features", []):
        p = f.get("properties", {})
        c = (f.get("geometry") or {}).get("coordinates") or [None, None, None]
        if p.get("mag") is None or c[0] is None:
            continue
        lat, lon = c[1], c[0]
        if min(haversine_km(lat, lon, sa, so) for sa, so in zip(lats, lons)) <= radius_km:
            out.append({"id": f["id"], "mag": float(p["mag"]), "lat": lat, "lon": lon,
                        "place": p.get("place") or "", "epoch": p["time"] / 1000.0})
    return out


def ev_time_loc(ev):
    """(time, lat, lon) of a declaration: the LOCATED origin + epicentre when the log has them (v2
    pipeline), else the old schema's declaration time and station proxy."""
    if ev.get("origin") is not None and ev.get("lat") is not None:
        return ev["origin"], ev["lat"], ev["lon"]
    return ev["epoch"], ev["proxy_lat"], ev["proxy_lon"]


def match_real_for_declared(ev, time_tol, dist_tol, shift=0.0, min_mag=1.0):
    """The real quake (>= min_mag) nearest in time to a declared event within (time_tol, dist_tol),
    or None. `shift` moves the declaration in time -- with shift=3600 this measures CHANCE matches."""
    t, lat, lon = ev_time_loc(ev)
    t += shift
    g = usgs_query({"starttime": iso(t - time_tol), "endtime": iso(t + time_tol), "minmagnitude": min_mag,
                    "latitude": lat, "longitude": lon, "maxradiuskm": dist_tol})
    best = None
    for f in g.get("features", []):
        p = f.get("properties", {})
        c = (f.get("geometry") or {}).get("coordinates") or [None, None]
        if p.get("mag") is None:
            continue
        cand = {"id": f["id"], "mag": float(p["mag"]), "place": p.get("place") or "",
                "epoch": p["time"] / 1000.0, "dt": p["time"] / 1000.0 - t}
        if best is None or abs(cand["dt"]) < abs(best["dt"]):
            best = cand
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=EVENTS_LOG,
                    help=f"declared-event JSONL to score (default {EVENTS_LOG})")
    ap.add_argument("--since", help="ISO date/time (UTC); default = first logged event")
    ap.add_argument("--mag-floor", type=float, default=3.0, dest="mag_floor",
                    help="min magnitude counted as detectable for the recall denominator (default 3.0)")
    ap.add_argument("--radius", type=float, default=150.0, help="km from any station to count a quake (default 150)")
    ap.add_argument("--time-tol", type=float, default=30.0, dest="time_tol",
                    help="seconds a declared event may differ from a real origin time (default 180)")
    ap.add_argument("--match-mag", type=float, default=1.0, dest="match_mag",
                    help="smallest catalogued quake that counts as a real match (default 1.0)")
    ap.add_argument("--dist-tol", type=float, default=60.0, dest="dist_tol",
                    help="km a declared proxy may differ from a real epicentre (default 100)")
    ap.add_argument("--json", help="also write the full report to this JSON file")
    args = ap.parse_args()

    now_epoch = datetime.now(timezone.utc).timestamp()
    since_epoch = (datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp()
                   if args.since else 0.0)
    events = load_events(since_epoch, args.log)
    if not events:
        print(f"No declared events in {args.log}"
              + (f" since {args.since}" if args.since else "")
              + ".\n(The watcher writes this log on every declared event; none yet or wrong window.)")
        return
    if not args.since:
        since_epoch = events[0]["epoch"]

    # ---- precision: are the declared events real? ----
    declared = []
    for ev in events:
        m = match_real_for_declared(ev, args.time_tol, args.dist_tol, 0.0, args.match_mag)
        c = match_real_for_declared(ev, args.time_tol, args.dist_tol, 3600.0, args.match_mag)
        declared.append({"declared": ev, "match": m, "true": m is not None, "chance": c is not None})
    n_true = sum(d["true"] for d in declared)
    n_false = len(declared) - n_true

    # ---- recall: were the real quakes caught? ----
    lats, lons = station_coords()
    reals = real_quakes_in_region(since_epoch, now_epoch, lats, lons, args.radius, args.mag_floor)
    matched_ids = {d["match"]["id"] for d in declared if d["match"]}
    caught = [q for q in reals if q["id"] in matched_ids]
    missed = [q for q in reals if q["id"] not in matched_ids]

    span_days = (now_epoch - since_epoch) / 86400.0
    print(f"Coverage: {iso(since_epoch)} -> {iso(now_epoch)}  ({span_days:.1f} days)")
    print(f"Match tolerances: +/-{args.time_tol:.0f}s, {args.dist_tol:.0f}km from proxy\n")

    print(f"DECLARED events: {len(declared)}   true detections: {n_true}   false alarms: {n_false}")
    if declared:
        prec = n_true / len(declared)
        chance = sum(d["chance"] for d in declared) / len(declared)
        print(f"  precision = {prec:.2f}   (chance baseline, same events shifted +1 h: {chance:.2f})"
              f"   false-alarm rate = {n_false / span_days:.2f}/day")
    for d in declared:
        ev, m = d["declared"], d["match"]
        tag = f"REAL M{m['mag']:.1f} {m['place']} (dt={m['dt']:+.0f}s)" if m else "FALSE ALARM (no USGS match)"
        size = f"M{ev['mag']:.1f}" if ev.get("mag") is not None else "size n/a"
        print(f"    {ev['t']}  {ev['n_stations']}st near {ev['proxy_station']} {size}  -> {tag}")

    # ---- breakdown by declaration tier (the live log carries 'confirmed' / 'pushed'; the old
    #      schema does not, so only print this where the fields exist) ----
    if any("confirmed" in d["declared"] for d in declared):
        def tier_line(label, subset):
            n = len(subset)
            t = sum(s["true"] for s in subset)
            ch = sum(s["chance"] for s in subset)
            prec = f"precision {t / n:.2f} (chance {ch / n:.2f})" if n else "precision n/a"
            print(f"  {label}: {n}   true {t}   false {n - t}   {prec}")
        print()
        tier_line("CONFIRMED (push-eligible)", [d for d in declared if d["declared"].get("confirmed")])
        tier_line("TENTATIVE (lone, logged only)", [d for d in declared if not d["declared"].get("confirmed")])
        pushed = [d for d in declared if d["declared"].get("pushed")]
        if pushed:
            tp = sum(d["true"] for d in pushed)
            print(f"  PUSHED to devices: {len(pushed)}   of which real quakes: {tp}"
                  f"   false pushes: {len(pushed) - tp}")

    print(f"\nREAL quakes >= M{args.mag_floor} within {args.radius:.0f}km: {len(reals)}"
          f"   caught: {len(caught)}   missed: {len(missed)}")
    if reals:
        print(f"  recall = {len(caught) / len(reals):.2f}   (denominator assumes >= M{args.mag_floor} is detectable)")
    for q in missed:
        print(f"    MISSED  M{q['mag']:.1f}  {q['place']}  {iso(q['epoch'])}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "since": iso(since_epoch), "until": iso(now_epoch), "span_days": span_days,
            "tolerances": {"time_s": args.time_tol, "dist_km": args.dist_tol},
            "declared": declared,
            "precision": (n_true / len(declared)) if declared else None,
            "false_alarms_per_day": n_false / span_days if span_days else None,
            "real": {"count": len(reals), "caught": len(caught), "missed": missed,
                     "mag_floor": args.mag_floor, "radius_km": args.radius,
                     "recall": (len(caught) / len(reals)) if reals else None},
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
