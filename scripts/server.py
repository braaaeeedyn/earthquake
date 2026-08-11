"""Minimal stdlib backend for the near-me feature: subscribe endpoint + live USGS events.

No framework (Flask/FastAPI not installed). The Vite dev server proxies /api/* here, so the
React app calls /api/subscribe and /api/events with no CORS fuss. All work is server-side;
the browser only sends {name, email, lat, lon}. Run alongside the app:

  python scripts/server.py            # http://localhost:8000
  cd app && npm run dev               # http://localhost:5173  (proxies /api -> :8000)
"""
import json
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import quake_archive  # noqa: E402
import push_fcm  # noqa: E402
from nearme_watch import SUBS, fetch_usgs, load_json, send_email  # noqa: E402

PORT = 8000
# Southern California only (the trained network's region): lat_min, lat_max, lon_min, lon_max
CA_BOUNDS = (32.0, 36.4, -121.5, -114.0)
CA_VIEWBOX = "-121.5,36.4,-114.0,32.0"      # Nominatim viewbox: left,top,right,bottom
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
WATCHER = None                               # the live_watch child process (set at startup)


def _in_ca(lat, lon):
    return CA_BOUNDS[0] <= lat <= CA_BOUNDS[1] and CA_BOUNDS[2] <= lon <= CA_BOUNDS[3]


def _is_ca_place(place):
    """The CA bounding box also clips western Nevada / Baja; the USGS place string disambiguates."""
    p = place or ""
    if "Baja" in p:
        return False
    return "California" in p or p.strip().endswith(", CA")


def _ca_window(window):
    """(start_datetime, min_magnitude, label) for a named time window. Month/year are CALENDAR
    (July, 2026), not rolling; day is today, week is the trailing 7 days, all is the full catalog."""
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if window == "day":
        return midnight, 1.0, "Today"
    if window == "week":
        return now - timedelta(days=7), 1.0, "Past 7 days"
    if window == "month":
        return midnight.replace(day=1), 2.5, now.strftime("%B %Y")
    if window == "year":
        return midnight.replace(month=1, day=1), 3.0, str(now.year)
    if window == "all":
        return datetime(1900, 1, 1, tzinfo=timezone.utc), 5.0, "All time"
    return None


def ca_top(window):
    """Top-5 largest California quakes in the given window, live from the USGS FDSN catalog."""
    w = _ca_window(window)
    if not w:
        return None
    start, minmag, label = w
    lat0, lat1, lon0, lon1 = CA_BOUNDS
    params = {"format": "geojson", "orderby": "magnitude", "limit": 20,
              "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"), "minmagnitude": minmag,
              "minlatitude": lat0, "maxlatitude": lat1, "minlongitude": lon0, "maxlongitude": lon1}
    req = urllib.request.Request(FDSN + "?" + urlencode(params),
                                 headers={"User-Agent": "earthquake-nearme-demo/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        g = json.load(r)
    events = []
    for f in g.get("features", []):
        p = f.get("properties", {})
        c = (f.get("geometry") or {}).get("coordinates") or [None, None]
        if p.get("mag") is None or not _is_ca_place(p.get("place")):
            continue
        events.append({"id": f["id"], "mag": float(p["mag"]), "lat": c[1], "lon": c[0],
                       "place": p.get("place") or "California", "url": p.get("url", ""),
                       "time": p.get("time")})
        if len(events) >= 5:
            break
    return {"window": window, "label": label, "events": events}


def send_confirmation(entry):
    """Email a new subscriber to confirm they're signed up for alerts at their location.

    Runs best-effort in a background thread so a slow/failed SMTP call never blocks or
    breaks the /api/subscribe response. Falls back to a console print when SMTP isn't
    configured (same behaviour as the watcher's send_email)."""
    subject = "You're subscribed to earthquake alerts"
    body = (
        f"Hi {entry['name']},\n\n"
        f"You're now signed up for earthquake alerts at your location "
        f"(lat {entry['lat']:.4f}, lon {entry['lon']:.4f}).\n\n"
        f"Our models watch the live Southern California seismic stream. When they detect a "
        f"quake and expect at least felt-level shaking where you are, we'll email you the "
        f"detection, its estimated size, and how hard it's likely to shake.\n\n"
        f"No action is needed. You'll only hear from us when a nearby quake happens.\n\n"
        f"(Research prototype, not an official warning. Rapid detection from live seismic "
        f"data, not sub-second pre-arrival warning.)"
    )
    try:
        send_email(entry["email"], subject, body, dry_run=False)
    except Exception as e:                               # SMTP hiccup shouldn't affect the user
        print(f"    confirmation email failed -> {entry['email']}: {e!r}")


def geocode(q):
    """Look up {lat, lon, name} for a California place via Nominatim, restricted to the CA box."""
    params = {"q": q, "format": "json", "limit": 1, "countrycodes": "us",
              "viewbox": CA_VIEWBOX, "bounded": 1}
    url = "https://nominatim.openstreetmap.org/search?" + urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "earthquake-nearme-demo/1.0 (research prototype)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        arr = json.load(r)
    if not arr:
        return None
    top = arr[0]
    lat, lon = float(top["lat"]), float(top["lon"])
    if not _in_ca(lat, lon):                     # double-check it really is inside California
        return None
    return {"lat": lat, "lon": lon, "name": top.get("display_name", q)}


def valid(sub):
    try:
        return (isinstance(sub.get("name"), str) and sub["name"].strip()
                and "@" in sub.get("email", "")
                and -90 <= float(sub["lat"]) <= 90 and -180 <= float(sub["lon"]) <= 180)
    except (KeyError, TypeError, ValueError):
        return False


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/events":
            # refresh the live feed, fold it into the daily archive, return today's top-5
            try:
                arc = quake_archive.update(fetch_usgs("2.5_day"))
                self._send(200, self._day(arc, quake_archive.today()))
            except Exception as e:                       # network hiccup
                self._send(502, {"error": str(e)})
        elif path == "/api/archive":
            # ?date=YYYY-MM-DD -> that day's top-5; no date -> list of available dates
            arc = quake_archive.load()
            date = parse_qs(urlparse(self.path).query).get("date", [None])[0]
            if date:
                self._send(200, self._day(arc, date))
            else:
                self._send(200, {"dates": sorted(arc.keys(), reverse=True)})
        elif path == "/api/ca":
            # ?window=day|week|month|year|all -> top-5 largest California quakes in that window
            window = parse_qs(urlparse(self.path).query).get("window", ["day"])[0]
            try:
                r = ca_top(window)
                self._send(200, r) if r else self._send(400, {"error": "bad window"})
            except Exception as e:
                self._send(502, {"error": str(e)})
        elif path == "/api/geocode":
            # ?q=<place> -> {lat, lon, name}, restricted to California
            q = parse_qs(urlparse(self.path).query).get("q", [""])[0].strip()
            if not q:
                return self._send(400, {"error": "need q"})
            try:
                r = geocode(q)
                self._send(200, r) if r else self._send(404, {"error": "no California match"})
            except Exception as e:
                self._send(502, {"error": str(e)})
        elif path == "/api/status":
            # the live SeedLink watcher is running -> the stream is live (it exits if the stream drops)
            self._send(200, {"live": WATCHER is not None and WATCHER.poll() is None})
        else:
            self._send(404, {"error": "not found"})

    @staticmethod
    def _day(arc, date):
        bucket = arc.get(date, {"world": [], "area": []})
        return {"date": date, "world": bucket["world"], "area": bucket["area"]}

    def do_POST(self):
        if self.path not in ("/api/subscribe", "/api/register-push"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            sub = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad JSON"})
        if self.path == "/api/register-push":
            return self._register_push(sub)
        if not valid(sub):
            return self._send(400, {"error": "need name, valid email, lat, lon"})
        subs = load_json(SUBS, [])
        entry = {"name": sub["name"].strip(), "email": sub["email"].strip(),
                 "lat": float(sub["lat"]), "lon": float(sub["lon"])}
        # de-dupe by (email, rounded location)
        key = (entry["email"], round(entry["lat"], 2), round(entry["lon"], 2))
        if any((s["email"], round(s["lat"], 2), round(s["lon"], 2)) == key for s in subs):
            return self._send(200, {"ok": True, "note": "already subscribed", "count": len(subs)})
        subs.append(entry)
        SUBS.write_text(json.dumps(subs, indent=2))
        # confirm the signup by email, off the request thread so SMTP latency never blocks
        threading.Thread(target=send_confirmation, args=(entry,), daemon=True).start()
        self._send(200, {"ok": True, "count": len(subs)})

    def _register_push(self, sub):
        """Store a mobile device's FCM token + location so the live watcher can push alerts."""
        try:
            token = sub["token"]
            lat, lon = float(sub["lat"]), float(sub["lon"])
        except (KeyError, TypeError, ValueError):
            return self._send(400, {"error": "need token, lat, lon"})
        if not isinstance(token, str) or not token.strip():
            return self._send(400, {"error": "need token, lat, lon"})
        toks = push_fcm.save_token(token.strip(), lat, lon, str(sub.get("name", "")).strip())
        self._send(200, {"ok": True, "count": len(toks)})

    def log_message(self, *a):                           # quiet default logging
        pass


def start_live_watcher():
    """Spawn the live SeedLink + model alert daemon so 'near me' alerts come from the trained
    detection/magnitude nets on the live stream -- never from USGS. Runs as a child process, so a
    stream/network failure in the watcher never affects this API. USGS is used only for the
    largest-quakes display (/api/ca)."""
    global WATCHER
    script = ROOT / "scripts" / "live_watch.py"
    try:
        proc = subprocess.Popen([sys.executable, str(script)])
        WATCHER = proc
        print(f"live alert watcher (SeedLink + models) started, pid {proc.pid}")
        return proc
    except Exception as e:                               # a watcher that won't start shouldn't kill the API
        print(f"could not start live_watch.py ({e!r}); API runs, but no live alerts")
        return None


if __name__ == "__main__":
    watcher = start_live_watcher()
    print(f"near-me backend on http://localhost:{PORT}  (POST /api/subscribe, GET /api/events)")
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    try:
        server.serve_forever()
    finally:
        if watcher and watcher.poll() is None:          # take the watcher down with the API
            watcher.terminate()
