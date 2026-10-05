"""Minimal stdlib backend for the near-me feature: push register/unregister + live USGS events.

No framework (Flask/FastAPI not installed). The Vite dev server proxies /api/* here, so the
app calls /api/register-push, /api/unregister-push and /api/events with no CORS fuss. Alerts
are push-only (app-only product); a device sends {token, stations, name} and receives FCM
pushes from the live watcher when one of its subscribed stations fires. Run alongside the app:

  python scripts/server.py            # http://localhost:8000
  cd app && npm run dev               # http://localhost:5173  (proxies /api -> :8000)
"""
import json
import os
import smtplib
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import quake_archive  # noqa: E402
import push_fcm  # noqa: E402
from nearme_watch import fetch_usgs, haversine_km  # noqa: E402  (import also loads .env into os.environ)
from eq import network  # noqa: E402

PORT = int(os.environ.get("PORT", "8000"))
# Bind address. Default 127.0.0.1 (safe: reach it through a reverse proxy that terminates TLS).
# Set HOST=0.0.0.0 to expose it directly (only behind a firewall/proxy — it speaks plain HTTP).
HOST = os.environ.get("HOST", "127.0.0.1")
# Support inbox for the contact form. SERVER-SIDE ONLY: never sent to the client, so the site
# never reveals the address. Contact messages are relayed here and nothing is persisted.
SUPPORT_TO = "braedynthompson@berkeley.edu"
# Southern California only (the trained network's region): lat_min, lat_max, lon_min, lon_max
CA_BOUNDS = (32.0, 36.4, -121.5, -114.0)
# The live station network (src/eq/network.py -- the single source of truth shared with the daemon,
# the dataset builder and the scorer). A device subscribes to the stations nearest it; the live
# watcher pushes it when a located event is within reach of one of them. This is also the region the
# "biggest SoCal quakes" browser is limited to.
STATIONS = network.as_api()
STATION_CODES = {s["code"] for s in STATIONS}
NET_COORDS = [(s["lat"], s["lon"]) for s in STATIONS]
# App version gate. `LATEST` = newest released app; `MIN` = lowest version allowed to run. The
# installed app compares itself: behind MIN on MAJOR or MINOR -> blocked (must re-download); a PATCH
# gap is only a soft notice. Bump LATEST every release; bump MIN (major/minor) to FORCE an update.
APP_LATEST_VERSION = "1.01.00"
APP_MIN_VERSION = "1.01.00"
# "Catchable" = where and how big the live pipeline can actually confirm a quake: >= 3 stations within
# 100 km (the confirmation rule's coverage) and M >= 2.0 (replay harness: ~80 % of in-coverage M2+
# quakes caught, ~48 % at M1.5-2, ~9 % below). The largest-quakes browser lists only these.
COVER_KM, COVER_MIN_STATIONS, CATCHABLE_MIN_MAG = 100.0, 3, 2.0
# Live v2 pipeline start (deploy of the 19-station network). Quakes after this can be marked caught /
# not caught against data/processed/events.jsonl; earlier ones get no mark.
LIVE_SINCE_MS = 1791191580000          # 2026-10-05 09:13 UTC
EVENTS_LOG = ROOT / "data" / "processed" / "events.jsonl"
_events_cache = {"mtime": None, "recs": []}
CA_VIEWBOX = "-121.5,36.4,-114.0,32.0"      # Nominatim viewbox: left,top,right,bottom
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
WATCHER = None                               # the live_watch child process (set at startup)


def _in_ca(lat, lon):
    return CA_BOUNDS[0] <= lat <= CA_BOUNDS[1] and CA_BOUNDS[2] <= lon <= CA_BOUNDS[3]


def _catchable(lat, lon):
    """True if >= COVER_MIN_STATIONS live stations are within COVER_KM -- where the pipeline can confirm."""
    return sum(haversine_km(lat, lon, sla, slo) <= COVER_KM for sla, slo in NET_COORDS) >= COVER_MIN_STATIONS


def _live_events():
    """Declared events from the live daemon's log (v2 records only), cached by file mtime."""
    try:
        m = EVENTS_LOG.stat().st_mtime
    except OSError:
        return []
    if _events_cache["mtime"] != m:
        recs = []
        for line in EVENTS_LOG.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("origin") is not None and r.get("lat") is not None:
                recs.append(r)
        _events_cache.update(mtime=m, recs=recs)
    return _events_cache["recs"]


def _caught(ev_ms, lat, lon):
    """Did the live pipeline catch this USGS quake? Match a declared event within 30 s of the origin and
    60 km of the epicentre. caught = confirmed (located + sized); seen = 1-2 stations only (tentative);
    missed = after go-live but no match; None = before the live pipeline existed."""
    if ev_ms is None or ev_ms < LIVE_SINCE_MS:
        return None
    t = ev_ms / 1000.0
    best = None
    for r in _live_events():
        if abs(r["origin"] - t) <= 30 and haversine_km(lat, lon, r["lat"], r["lon"]) <= 60:
            if r.get("confirmed") and (best is None or not best.get("confirmed")):
                best = r
            elif best is None:
                best = r
    if best is None:
        return {"status": "missed"}
    if best.get("confirmed"):
        return {"status": "caught", "mag": best.get("mag"), "n_stations": best.get("n_stations")}
    return {"status": "seen"}


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
    minmag = max(minmag, CATCHABLE_MIN_MAG)
    lat0, lat1, lon0, lon1 = CA_BOUNDS
    params = {"format": "geojson", "orderby": "magnitude", "limit": 60,
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
        if p.get("mag") is None or c[0] is None or c[1] is None:
            continue
        # keep only quakes the live pipeline could catch: SoCal by place (drops NV/Baja), inside the
        # 3-station coverage, M >= CATCHABLE_MIN_MAG (enforced in the query)
        if not _is_ca_place(p.get("place")) or not _catchable(c[1], c[0]):
            continue
        events.append({"id": f["id"], "mag": float(p["mag"]), "lat": c[1], "lon": c[0],
                       "place": p.get("place") or "California", "url": p.get("url", ""),
                       "time": p.get("time"), "caught": _caught(p.get("time"), c[1], c[0])})
        if len(events) >= 5:
            break
    return {"window": window, "label": label, "events": events}


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


def send_support_email(from_email, message):
    """Relay a support message to the hidden SUPPORT_TO inbox, with the sender's address as
    Reply-To so support can respond. Nothing is persisted: the sender's email lives only in
    this in-memory call and the outgoing message's Reply-To, never on disk. When SMTP isn't
    configured we no-op (and don't log the address) so the feature degrades quietly."""
    if not os.environ.get("SMTP_USER"):
        print("    [contact] SMTP not configured — message not sent (nothing stored)")
        return False                                     # caller reports an honest failure
    msg = EmailMessage()
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = SUPPORT_TO
    msg["Reply-To"] = from_email
    msg["Subject"] = f"SeismicSoCal support — {from_email}"
    msg.set_content(f"From: {from_email}\n\n{message}")
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as srv:
        srv.starttls()
        srv.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        srv.send_message(msg)
    return True


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # security headers: this is a JSON-only API, so lock down sniffing/embedding/referrers
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
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
        elif path == "/api/stations":
            return self._send(200, {"stations": STATIONS})
        elif path == "/api/version":
            return self._send(200, {"latest": APP_LATEST_VERSION, "min": APP_MIN_VERSION})
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
            # the live SeedLink watcher is running -> the stream is live (it exits if the stream drops);
            # per-station health (up / latency) comes from the daemon's status file (refreshed every 30 s)
            out = {"live": WATCHER is not None and WATCHER.poll() is None}
            try:
                st = json.loads((ROOT / "data" / "processed" / "live_status.json").read_text())
                if time.time() - st.get("t", 0) < 120:
                    out["stations"], out["push_enabled"] = st["stations"], st.get("push_enabled")
            except (OSError, ValueError):
                pass
            self._send(200, out)
        else:
            self._send(404, {"error": "not found"})

    @staticmethod
    def _day(arc, date):
        bucket = arc.get(date, {"world": [], "area": []})
        return {"date": date, "world": bucket["world"], "area": bucket["area"]}

    def do_POST(self):
        # Alerts are push-only (the app-only product): a device subscribes by registering its
        # FCM token + chosen stations, and unsubscribes by removing it. /api/contact relays a support
        # message to the hidden inbox. There is no email-based alert channel.
        if self.path not in ("/api/register-push", "/api/unregister-push", "/api/contact"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            sub = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad JSON"})
        if self.path == "/api/register-push":
            return self._register_push(sub)
        if self.path == "/api/contact":
            return self._contact(sub)
        return self._unregister_push(sub)

    def _register_push(self, sub):
        """Store a device's FCM token + the sensor stations it subscribes to (no coordinates) so the
        live watcher can push it whenever one of those stations fires."""
        token = sub.get("token")
        stations = sub.get("stations")
        if not isinstance(token, str) or not token.strip():
            return self._send(400, {"error": "need token and stations"})
        if not isinstance(stations, list) or not stations:
            return self._send(400, {"error": "pick at least one station"})
        stations = [str(s).strip() for s in stations]
        unknown = [s for s in stations if s not in STATION_CODES]
        if unknown:
            return self._send(400, {"error": f"unknown station(s): {', '.join(unknown)}"})
        toks = push_fcm.save_token(token.strip(), stations, str(sub.get("name", "")).strip())
        self._send(200, {"ok": True, "count": len(toks)})

    def _unregister_push(self, sub):
        """Remove a device's FCM token (unsubscribe). Idempotent: unknown token -> still ok."""
        token = sub.get("token")
        if not isinstance(token, str) or not token.strip():
            return self._send(400, {"error": "need token"})
        toks = push_fcm.remove_token(token.strip())
        self._send(200, {"ok": True, "count": len(toks)})

    def _contact(self, sub):
        """Relay a support message to the hidden inbox. The sender supplies their own email (used
        only as Reply-To) and a message; neither is stored. The support address is never returned."""
        email = str(sub.get("email", "")).strip()
        message = str(sub.get("message", "")).strip()
        if "@" not in email or "." not in email or len(email) > 254:
            return self._send(400, {"error": "enter a valid email"})
        if not (1 <= len(message) <= 5000):
            return self._send(400, {"error": "enter a message (max 5000 characters)"})
        try:
            sent = send_support_email(email, message)
        except Exception as e:                               # SMTP hiccup -> report, don't crash
            return self._send(502, {"error": f"could not send: {e}"})
        if not sent:                                         # creds missing -> don't claim success
            return self._send(503, {"error": "support is unavailable right now — please try again later"})
        self._send(200, {"ok": True})

    def log_message(self, *a):                           # quiet default logging
        pass


WATCHER_STOP = threading.Event()                         # set on shutdown to stop respawning


def start_live_watcher():
    """Spawn the live SeedLink + model alert daemon so 'near me' alerts come from the trained
    detection/magnitude nets on the live stream -- never from USGS. Runs as a child process, so a
    stream/network failure in the watcher never affects this API. USGS is used only for the
    largest-quakes display (/api/ca)."""
    global WATCHER
    script = ROOT / "scripts" / "live_watch.py"
    # -u: unbuffered child stdout, so its [EVENT] lines reach journald in real time. Without it
    # Python block-buffers the pipe and live events (and pushes) can sit hidden for a long time.
    proc = subprocess.Popen([sys.executable, "-u", str(script)])
    WATCHER = proc
    print(f"live alert watcher (SeedLink + models) started, pid {proc.pid}")
    return proc


def supervise_watcher(poll=10):
    """Keep the daemon up 24/7. The live watcher exits if its SeedLink stream drops; this thread
    notices and respawns it (short backoff so a persistent failure doesn't become a tight loop),
    so alerting resumes on its own without a human. The API keeps serving throughout."""
    delay = 5
    while not WATCHER_STOP.is_set():
        if WATCHER is None or WATCHER.poll() is not None:
            code = getattr(WATCHER, "returncode", None)
            print(f"live watcher not running (exit {code}); restarting in {delay}s")
            if WATCHER_STOP.wait(delay):
                break
            try:
                start_live_watcher()
                delay = 5                                # reset backoff after a clean (re)start
            except Exception as e:                       # a watcher that won't start shouldn't kill the API
                print(f"restart failed ({e!r}); backing off")
                delay = min(delay * 2, 120)
                continue
        if WATCHER_STOP.wait(poll):
            break


if __name__ == "__main__":
    try:
        start_live_watcher()
    except Exception as e:
        print(f"could not start live_watch.py ({e!r}); API runs, supervisor will retry")
    threading.Thread(target=supervise_watcher, daemon=True).start()
    print(f"near-me backend on http://{HOST}:{PORT}  (POST /api/register-push, GET /api/events)")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    finally:
        WATCHER_STOP.set()                              # stop the supervisor from respawning
        if WATCHER and WATCHER.poll() is None:          # take the watcher down with the API
            WATCHER.terminate()
