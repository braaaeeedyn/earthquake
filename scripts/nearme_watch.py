"""Server-side "near me" earthquake watcher.

Polls the PUBLIC USGS realtime feed (real events, ~1-min latency), and for every NEW quake
notifies each subscriber whose location falls inside a MAGNITUDE-SCALED felt radius. All work
is server-side; a subscriber only supplies {location, contact} and receives a message.

First-test notifier is EMAIL (SMTP); SMS/native-push can be added behind the same notify()
seam later. No live seismic ML is involved here -- USGS already detects/locates events; the
project's ML models are the research demonstration layer and can enrich the message later.

  python scripts/nearme_watch.py --once --dry-run           # single poll, print (no send)
  python scripts/nearme_watch.py --selftest                 # deterministic: fake a quake on a subscriber
  python scripts/nearme_watch.py --feed 2.5_day --once --dry-run   # test against past-day real events
  python scripts/nearme_watch.py --interval 60              # live loop, email real subscribers

Email creds via env: SMTP_USER, SMTP_PASS (Gmail app password), optional SMTP_HOST/SMTP_PORT.
Subscribers: data/subscribers.json  ->  [{"name","lat","lon","email"}]
"""
import argparse
import json
import math
import os
import smtplib
import sys
import time
import urllib.request
from email.message import EmailMessage
from pathlib import Path

import shaking_model

ROOT = Path(__file__).resolve().parents[1]
SUBS = ROOT / "data" / "subscribers.json"
SEEN = ROOT / "data" / "processed" / "nearme_seen.json"
FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson"


def load_env():
    """Load KEY=VALUE lines from a gitignored .env at repo root into os.environ (no override)."""
    envf = ROOT / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


def felt_radius_km(mag: float) -> float:
    """Magnitude-scaled 'near' radius: ~M3->20km, M4.5->70km, M6->200km. Floored/capped."""
    return float(min(400.0, max(10.0, 10 ** (0.30 + 0.333 * mag))))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_usgs(feed: str):
    """Return list of events {id, mag, lat, lon, place, url} from a USGS summary feed."""
    with urllib.request.urlopen(FEED_URL.format(feed=feed), timeout=30) as resp:
        g = json.load(resp)
    out = []
    for f in g["features"]:
        p, c = f["properties"], f["geometry"]["coordinates"]
        if p.get("mag") is None:
            continue
        out.append({"id": f["id"], "mag": float(p["mag"]), "lat": float(c[1]),
                    "lon": float(c[0]), "place": p.get("place") or "unknown", "url": p.get("url", ""),
                    "time": p.get("time")})
    return out


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def matches(event, subs, min_mmi=3.0):
    """Subscribers where the model expects felt-or-stronger shaking, with distance + estimated MMI.

    The alert decision is now made on predicted shaking at each subscriber (estimated intensity
    >= min_mmi), not merely on being inside a fixed felt radius."""
    hits = []
    for s in subs:
        dist = haversine_km(event["lat"], event["lon"], s["lat"], s["lon"])
        mmi = shaking_model.estimate_mmi(event["mag"], dist)
        if mmi >= min_mmi:
            hits.append((s, dist, mmi))
    return hits


def send_email(to_addr, subject, body, dry_run):
    if dry_run or not os.environ.get("SMTP_USER"):
        print(f"    [dry-run email] to={to_addr}\n      subj: {subject}\n      body: {body}")
        return
    msg = EmailMessage()
    msg["From"] = os.environ["SMTP_USER"]; msg["To"] = to_addr
    msg["Subject"] = subject; msg.set_content(body)
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as srv:
        srv.starttls(); srv.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        srv.send_message(msg)
    print(f"    sent email -> {to_addr}")


def notify_all(event, hits, dry_run):
    for s, dist, mmi in hits:
        label, desc = shaking_model.describe(mmi)
        subject = f"Earthquake alert: M{event['mag']:.1f} — {label.lower()} shaking expected"
        body = (f"Hi {s['name']},\n\n"
                f"A magnitude {event['mag']:.1f} earthquake occurred {dist:.0f} km from your "
                f"location ({event['place']}).\n\n"
                f"Size: magnitude {event['mag']:.1f}.\n"
                f"Estimated shaking where you are: {label} — intensity {round(mmi)} of 10 "
                f"({desc}).\n\n"
                f"You're getting this alert because the model expects at least felt-level shaking "
                f"at your location.\n\n"
                f"Details: {event['url']}\n\n"
                f"(Research prototype - not an official warning. Magnitude from USGS; shaking is a "
                f"model estimate.)")
        send_email(s["email"], subject, body, dry_run)


def poll_once(feed, subs, seen, dry_run, min_mag, min_mmi):
    events = [e for e in fetch_usgs(feed) if e["mag"] >= min_mag]
    new = [e for e in events if e["id"] not in seen]
    print(f"  feed={feed}: {len(events)} events (>= M{min_mag}), {len(new)} new")
    for e in new:
        hits = matches(e, subs, min_mmi)
        tag = (f"-> {len(hits)} subscriber(s) alerted" if hits
               else "-> no subscribers reach the shaking threshold")
        print(f"  M{e['mag']:.1f} {e['place'][:44]:44s}  {tag}")
        if hits:
            notify_all(e, hits, dry_run)
        seen[e["id"]] = e["mag"]
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", default="all_hour",
                    help="USGS feed: all_hour, all_day, 2.5_day, 4.5_day, significant_day, ...")
    ap.add_argument("--interval", type=int, default=60, help="poll seconds (loop mode)")
    ap.add_argument("--once", action="store_true", help="single poll then exit")
    ap.add_argument("--dry-run", action="store_true", help="print instead of emailing")
    ap.add_argument("--min-mag", type=float, default=2.5)
    ap.add_argument("--min-mmi", type=float, default=3.0,
                    help="alert a subscriber when estimated shaking intensity (MMI) >= this")
    ap.add_argument("--selftest", action="store_true",
                    help="fabricate a quake at subscriber[0] and run the pipeline (dry-run)")
    args = ap.parse_args()

    subs = load_json(SUBS, [])
    if not subs:
        print(f"No subscribers. Create {SUBS.relative_to(ROOT)} : "
              f'[{{"name":"You","lat":34.05,"lon":-118.24,"email":"you@example.com"}}]')
        return

    if args.selftest:
        s0 = subs[0]
        fake = {"id": "selftest", "mag": 5.0, "lat": s0["lat"] + 0.05, "lon": s0["lon"],
                "place": "SELFTEST event near subscriber[0]", "url": "https://example.test"}
        d0 = haversine_km(fake['lat'], fake['lon'], s0['lat'], s0['lon'])
        print(f"[selftest] fabricated M5.0 ~{d0:.0f}km from {s0['name']} "
              f"(est. shaking MMI {round(shaking_model.estimate_mmi(5.0, d0))})")
        # sends a real email if SMTP creds are set and --dry-run not passed; else prints
        notify_all(fake, matches(fake, subs, args.min_mmi), dry_run=args.dry_run)
        return

    seen = load_json(SEEN, {})
    print(f"Watching {len(subs)} subscriber(s). Ctrl-C to stop.")
    while True:
        try:
            seen = poll_once(args.feed, subs, seen, args.dry_run, args.min_mag, args.min_mmi)
            SEEN.parent.mkdir(parents=True, exist_ok=True)
            SEEN.write_text(json.dumps(seen))
        except Exception as e:                      # network hiccups shouldn't kill the watcher
            print(f"  poll error: {e!r}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
