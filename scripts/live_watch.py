"""LIVE Southern-California earthquake detector — the models run continuously on a real-time
waveform stream (no USGS in the loop).

Architecture:
  SeedLink stream (10 CI/SCEDC SoCal stations)  -> rolling per-station buffers
    -> DETECTION model runs continuously on sliding 30 s vertical windows
    -> GRADED declaration: >= K stations agreeing within a short window (+ a move-out timing check)
       is a CONFIRMED event; a lone high-confidence station is a TENTATIVE, possibly-false alarm.
       Small quakes only 1-2 stations can feel still alert (labelled unconfirmed) — a missed quake
       is worse than a flagged false alarm — while the coincidence tier kills most noise.
    -> location proxy (strongest-triggering station) + MAGNITUDE model sizes CONFIRMED events
    -> devices subscribed to any triggering station get ONE combined push; alerts are push-only.
       Devices subscribe to sensor STATIONS (chosen near them at signup), not coordinates.

Honesty:
  - USGS is bypassed: the detector genuinely fires on the raw stream. But SeedLink latency is
    seconds-to-tens-of-seconds, so this is RAPID DETECTION, not sub-second pre-arrival warning.
  - Sizing uses the in-distribution 10-station magnitude ensemble. Event LOCATION is a proxy
    (the strongest-triggering station), not a true locator, so distance/shaking are estimates.
  - Coverage is the 10 SoCal stations the models were trained on. Statewide needs a retrain.

Run:
  python scripts/live_watch.py --selftest            # deterministic pipeline check (dry-run)
  python scripts/live_watch.py --replay              # verify detection + sizing on cached events
  python scripts/live_watch.py                       # LIVE (needs an always-on host + network)
  python scripts/live_watch.py --server rtserve.iris.washington.edu:18000 --min-stations 4
"""
import argparse
import datetime
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import push_fcm  # noqa: E402
import shaking_model  # noqa: E402
from nearme_watch import haversine_km  # noqa: E402

DETECTOR = ROOT / "data" / "processed" / "detector.pt"
MAG_CKPT = ROOT / "data" / "processed" / "magnitude_ensemble.pt"
PHASE2A = ROOT / "data" / "processed" / "seismic_phase2a_xl.npz"
PHASE1 = ROOT / "data" / "processed" / "seismic_phase1.npz"
EVENTS_LOG = ROOT / "data" / "processed" / "events.jsonl"   # durable audit log of declared events

SR = 100.0
NPTS = 3000                       # 30 s @ 100 Hz
SCALE = 7.775235e-4               # training amplitude scale = X[mask].std() over phase2a_xl (m/s)
NET = "CI"

# Tunables (also CLI flags)
DET_THRESH = 0.60                 # per-station detection probability to count as a trigger
MIN_STATIONS = 2                  # CONFIRM tier: stations that must agree for a corroborated event
LONE_THRESH = 0.85                # a SINGLE station alerts (tentative) only above this higher floor,
                                  # so lone triggers catch small quakes without pushing on plain noise
COINC_WIN = 12.0                  # seconds within which triggers count toward the same event
COOLDOWN = 120.0                  # seconds to suppress re-alerting the same event (keyed per station)
V_MIN = 2.0                       # km/s: slowest wave used to bound plausible inter-station move-out
PICK_JITTER = 4.0                 # s: slack for detection-window / timing jitter in the move-out check
# A device is alerted when any station it subscribes to triggers; no per-user distance is computed.


# ---------------------------------------------------------------- network + models

def load_network():
    """Canonical station list + coordinates (small arrays; .npz loads them without X)."""
    d = np.load(PHASE2A, allow_pickle=True)
    stations = [str(s) for s in d["stations"]]     # e.g. 'CI.CCC'
    coords = np.asarray(d["coords"], float)        # (10, 2) lat, lon — same order as the models
    return stations, coords


def load_detector():
    import torch
    from seismic_train import SeisModel
    ck = torch.load(DETECTOR, weights_only=False)
    m = SeisModel()
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def load_magnitude(coords):
    import torch
    from seismic_train_multi import MultiStationModel, adjacency
    ck = torch.load(MAG_CKPT, weights_only=False)
    Ahat = adjacency(coords)
    models = [MultiStationModel(Ahat, hybrid=True) for _ in ck["states"]]
    for mdl, st in zip(models, ck["states"]):
        mdl.load_state_dict(st)
        mdl.eval()
    return models, ck["am"], ck["asd"]


# ---------------------------------------------------------------- inference wrappers

def detect_prob(det_model, wave):
    """P(earthquake) for one 30 s vertical window (per-window normalized, as in training)."""
    import torch
    x = wave.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        e = det_model.backbone(torch.tensor(x)[None])
        return float(torch.sigmoid(det_model.det(e)).item())


def estimate_magnitude(models, am, asd, Xraw, mask, dist):
    """Deep network magnitude from response-removed velocity (m/s). Same path as demo_magnitude.

    Xraw: (10, 3, NPTS) velocity; mask: (10,) recorded; dist: (10,) km to the (proxy) epicentre.
    """
    import torch
    Xn = (Xraw / SCALE).astype(np.float32)[None]                       # (1,10,3,NPTS)
    m = mask.astype(bool)[None]
    logdist = np.where(dist > 0, np.log10(np.maximum(dist, 1.0)), 0.0).astype(np.float32)[None]
    rec = mask.astype(bool)
    peak = np.log10(np.abs(Xraw[rec]).reshape(rec.sum(), -1).max(1) + 1e-12)
    ld = np.log10(np.maximum(dist[rec], 1.0))
    feats = np.array([peak.mean(), peak.max(), ld.mean(), ld.min()], np.float32)
    afn = ((feats - am) / asd).astype(np.float32)[None]
    xs, ms, ls, a = (torch.tensor(Xn), torch.tensor(m), torch.tensor(logdist), torch.tensor(afn))
    with torch.no_grad():
        preds = [float(mdl(xs, ms, ls, a).item()) for mdl in models]
    return float(np.mean(preds))


# ---------------------------------------------------------------- event handling

def moveout_ok(first_time_by_sta, coords):
    """A single seismic source can only produce inter-station arrival-time differences up to the
    stations' separation divided by the wave speed (triangle inequality). So if any pair of
    triggering stations fired FARTHER apart in time than a wavefront at >= V_MIN could explain,
    the triggers cannot be one event -- reject them (this kills scattered multi-station noise).
    A necessary condition, not a locator: it never rejects a real event, only impossible ones."""
    stas = list(first_time_by_sta)
    for i in range(len(stas)):
        for j in range(i + 1, len(stas)):
            a, b = stas[i], stas[j]
            dt = abs(first_time_by_sta[a] - first_time_by_sta[b])
            dkm = haversine_km(coords[a][0], coords[a][1], coords[b][0], coords[b][1])
            if dt > dkm / V_MIN + PICK_JITTER:
                return False
    return True


def declare_graded(triggers, now, confirm_stations, coords):
    """Graded declaration over triggers within COINC_WIN. Returns (station_indices, strongest_idx,
    confirmed) or None:
      - >= confirm_stations distinct stations AND a physically consistent move-out -> confirmed=True
      - exactly one station, at prob >= LONE_THRESH                                -> confirmed=False
      - anything else (scattered multi-station noise, or a weak lone trigger)      -> None
    `triggers` is a deque of (time, sta_idx, prob)."""
    recent = [t for t in triggers if now - t[0] <= COINC_WIN]
    by_sta = {}                       # sta_idx -> best detection prob
    first = {}                        # sta_idx -> earliest trigger time (for the move-out check)
    for t, si, p in recent:
        by_sta[si] = max(by_sta.get(si, 0.0), p)
        first[si] = min(first.get(si, t), t)
    if not by_sta:
        return None
    strongest = max(by_sta, key=by_sta.get)
    if len(by_sta) >= confirm_stations and moveout_ok(first, coords):
        return sorted(by_sta), strongest, True
    if len(by_sta) == 1 and by_sta[strongest] >= LONE_THRESH:
        return sorted(by_sta), strongest, False
    return None


def log_event(names, coords, stas, strongest, mag, conf, pushed, confirmed):
    """Append one declared event to the JSONL audit log (durable, flushed -- independent of stdout
    buffering) so scripts/crosscheck_events.py can later score it against the USGS catalog."""
    rec = {
        "t": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "epoch": time.time(),
        "confirmed": bool(confirmed),
        "n_stations": len(stas),
        "stations": [names[i] for i in stas],
        "proxy_station": names[strongest],
        "proxy_lat": float(coords[strongest][0]),
        "proxy_lon": float(coords[strongest][1]),
        "mag": None if mag is None else round(float(mag), 2),
        "det_conf": round(float(conf), 3),
        "pushed": int(pushed),
    }
    EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def alert_push_devices(tokens, event_stations, strongest_name, station_coords, mag, nstations,
                       confirmed, dry_run):
    """Push each device subscribed to ANY of the event's stations exactly once. The message is
    personalized by DISTANCE from the event to the user's NEAREST subscribed station — a proxy for
    how far the quake is from them, derived from station coordinates (we store no user location).
    `station_coords` maps station code -> (lat, lon); `event_stations`/`strongest_name` are codes.

      confirmed  -> "nearest to <strongest> (~D km from you), M<mag>, <intensity> shaking expected"
      tentative  -> "one sensor <D km from you> — unconfirmed, may be a false alarm" """
    event_set = set(event_stations)
    slat, slon = station_coords[strongest_name]
    sent = 0
    for t in tokens:
        subs = [s for s in t.get("stations", []) if s in station_coords]
        if event_set.isdisjoint(subs):
            continue                                  # device isn't subscribed to any triggering station
        # distance from the (proxy) epicentre to the user's nearest subscribed sensor
        d = min(haversine_km(slat, slon, station_coords[s][0], station_coords[s][1]) for s in subs)
        where = (f"at the {strongest_name} station near you" if d < 15
                 else f"nearest to the {strongest_name} station, about {d:.0f} km from you")
        if confirmed:
            size = f" Estimated M{mag:.1f}." if mag is not None else ""
            shake = ""
            if mag is not None:
                label, _ = shaking_model.describe(shaking_model.estimate_mmi(mag, d))
                shake = (" Likely too far to be felt at your area." if label == "Not felt"
                         else f" {label} shaking expected at your area.")
            title = f"Earthquake detected near {strongest_name}"
            body = (f"{nstations} sensors agree — {where}.{size}{shake} "
                    f"Rapid detection, not an official warning.")
        else:
            title = f"Possible quake near {strongest_name}"
            body = (f"One sensor detected possible shaking {where} — unconfirmed and may be a false "
                    f"alarm (no other station agrees yet). Not an official warning.")
        if push_fcm.send_push(t["token"], title, body, dry_run):
            sent += 1
    return sent


# ---------------------------------------------------------------- live SeedLink

def run_live(args):
    import obspy
    from obspy.clients.fdsn import Client
    from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

    stations, coords = load_network()
    names = [s.split(".")[1] for s in stations]
    idx_of = {n: i for i, n in enumerate(names)}
    coord_of = {n: (float(coords[i][0]), float(coords[i][1])) for i, n in enumerate(names)}
    det = load_detector()
    try:
        mag_models, am, asd = load_magnitude(coords)
    except Exception as e:
        print(f"  magnitude model unavailable ({e!r}); detections will alert without size")
        mag_models = None
    print(f"Fetching station responses for {len(names)} stations...")
    inv = Client("IRIS").get_stations(network=NET, station=",".join(names),
                                      channel="HH?", level="response")

    buffers = {n: obspy.Stream() for n in names}     # per-station rolling 3C stream
    lock = threading.Lock()
    triggers = deque(maxlen=200)
    last_alert = {}                                  # strongest sta_idx -> time of its last alert

    def scan():
        while True:
            time.sleep(args.scan)
            now = time.time()
            with lock:
                snap = {n: buffers[n].copy() for n in names}
            for n, st in snap.items():
                z = st.select(channel="*Z")
                if not len(z):
                    continue
                tr = z.merge(fill_value=0)[0]
                if tr.stats.npts < NPTS:
                    continue
                w = tr.data[-NPTS:].astype(np.float32)
                p = detect_prob(det, w)
                if p >= args.det_thresh:
                    triggers.append((now, idx_of[n], p))
            decl = declare_graded(triggers, now, args.min_stations, coords)
            if decl and now - last_alert.get(decl[1], 0.0) > COOLDOWN:
                stas, strongest, confirmed = decl
                last_alert[strongest] = now
                epi_lat, epi_lon = coords[strongest]
                conf = max((p for t, si, p in triggers if now - t <= COINC_WIN and si in stas),
                           default=0.0)
                # size only CONFIRMED events (the magnitude ensemble needs the multi-station data)
                mag = size_event(snap, names, coords, epi_lat, epi_lon, inv, mag_models, am, asd) \
                    if (confirmed and mag_models) else None
                event_names = [names[i] for i in stas]
                # reload device tokens each event so mobile users who just signed up are covered
                psent = alert_push_devices(push_fcm.load_tokens(), event_names, names[strongest],
                                           coord_of, mag, len(stas), confirmed, args.dry_run)
                log_event(names, coords, stas, strongest, mag, conf, psent, confirmed)
                tag = "EVENT" if confirmed else "TENTATIVE"
                msize = f"M{mag:.1f}" if mag is not None else ("size n/a" if confirmed else "unconfirmed")
                print(f"[{tag}] {len(stas)} station(s), near {names[strongest]} "
                      f"({epi_lat:.2f},{epi_lon:.2f}) {msize} -> {psent} push alerted", flush=True)

    class Client_(EasySeedLinkClient):
        def on_data(self, trace):
            with lock:
                buffers[trace.stats.station] += trace
                buffers[trace.stats.station].merge(fill_value=0)
                buffers[trace.stats.station].trim(starttime=obspy.UTCDateTime() - 120)

    threading.Thread(target=scan, daemon=True).start()
    print(f"Connecting to SeedLink {args.server} ...  (Ctrl-C to stop)")
    # obspy 1.5.x bug: EasySeedLinkClient auto-connects in __init__ with SeedLinkConnection.timeout
    # left at None, and connect() passes that None into is_connected(), crashing the `< timeout`
    # comparison. Create without auto-connecting, set an explicit timeout, then connect.
    cli = Client_(args.server, autoconnect=False)
    cli.conn.timeout = 30
    cli.connect()
    for n in names:
        cli.select_stream(NET, n, "HH?")
    cli.run()


def size_event(snap, names, coords, epi_lat, epi_lon, inv, mag_models, am, asd):
    """Build response-removed 3C windows for the 10 stations from live buffers, then size."""
    import obspy
    Xraw = np.zeros((len(names), 3, NPTS), np.float32)
    mask = np.zeros(len(names), bool)
    dist = np.array([haversine_km(epi_lat, epi_lon, c[0], c[1]) for c in coords], np.float32)
    for i, n in enumerate(names):
        st = snap[n].copy()
        if len(st) < 3:
            continue
        try:
            st.merge(fill_value=0)
            st.detrend("demean")
            st.remove_response(inventory=inv, output="VEL", water_level=60)
            if abs(st[0].stats.sampling_rate - SR) > 1e-6:
                st.resample(SR)
            comps = {}
            for tr in st:
                comps[tr.stats.channel[-1]] = tr.data[-NPTS:]
            order = [c for c in ("Z", "N", "E") if c in comps] or list(comps)[:3]
            for j, c in enumerate(order[:3]):
                d = comps[c].astype(np.float32)
                Xraw[i, j, -len(d):] = d[-NPTS:]
            mask[i] = True
        except Exception:
            continue
    if mask.sum() == 0:
        return None
    return estimate_magnitude(mag_models, am, asd, Xraw, mask, dist)


# ---------------------------------------------------------------- verification modes

def replay(args):
    """Verify the model wrappers on cached data (no network): detection on phase1 windows and
    the deep magnitude path on a held-out phase2a event."""
    det = load_detector()
    d1 = dict(np.load(PHASE1, allow_pickle=True))
    ev = np.where(d1["ydet"] == 1)[0][0]
    no = np.where(d1["ydet"] == 0)[0][0]
    pe = detect_prob(det, d1["waves"][ev])
    pn = detect_prob(det, d1["waves"][no])
    print(f"[replay] detection  event-window P={pe:.2f}   noise-window P={pn:.2f}  "
          f"(event should be high, noise low)")

    stations, coords = load_network()
    mag_models, am, asd = load_magnitude(coords)
    d2 = dict(np.load(PHASE2A, allow_pickle=True))
    gi = int(np.argmax(d2["mag"]))                      # the largest held-out event
    est = estimate_magnitude(mag_models, am, asd, d2["X"][gi], d2["mask"][gi], d2["dist"][gi])
    print(f"[replay] magnitude  true M{d2['mag'][gi]:.1f}  ->  deep-ensemble estimate M{est:.1f}")


def selftest(args):
    """Deterministic pipeline check (dry-run): both alert tiers, and station-based targeting."""
    stations, coords = load_network()
    names = [s.split(".")[1] for s in stations]
    coord_of = {n: (float(coords[i][0]), float(coords[i][1])) for i, n in enumerate(names)}
    now = time.time()

    # CONFIRMED: K stations agree -> combined push to a device subscribed to one of them.
    trig = deque((now, i, 0.9) for i in range(args.min_stations))
    decl = declare_graded(trig, now, args.min_stations, coords)
    assert decl and decl[2] is True, "K agreeing stations should confirm"
    stas, strongest, confirmed = decl
    event_names = [names[i] for i in stas]
    # a subscriber whose only station is a FAR one (so the message shows a real distance from them)
    far = max(names, key=lambda n: haversine_km(*coord_of[names[strongest]], *coord_of[n]))
    subbed = {"token": "selftest-token", "stations": [far], "name": "subbed"}
    other = {"token": "other-token", "stations": [names[(strongest + 5) % len(names)]], "name": "other"}
    # targeting: only devices subscribed to a triggering station are recipients (dry-run send returns
    # False, so we assert on the membership filter directly rather than on the sent count).
    recipients = [t["name"] for t in (subbed, other) if not set(event_names).isdisjoint(t["stations"])]
    assert recipients == [], "a subscriber to a non-triggering station must NOT be alerted"
    subbed["stations"] = [names[strongest], far]      # now they follow a triggering station too
    alert_push_devices([subbed, other], event_names, names[strongest], coord_of, 5.2, len(stas), True, dry_run=True)
    print(f"[selftest] CONFIRMED near {names[strongest]}: message shows distance to the user's "
          f"nearest subscribed sensor + estimated shaking")

    # TENTATIVE: a single high-confidence station -> unconfirmed push to its subscribers only.
    lone = deque([(now, 0, 0.92)])
    decl2 = declare_graded(lone, now, args.min_stations, coords)
    assert decl2 and decl2[2] is False, "a lone strong station should be tentative"
    alert_push_devices([{"token": "t", "stations": [names[0]], "name": "s"}],
                       [names[0]], names[0], coord_of, None, 1, False, dry_run=True)
    print(f"[selftest] TENTATIVE near {names[0]}: unconfirmed push to its subscribers (may be false)")

    # A weak lone station must NOT alert.
    assert declare_graded(deque([(now, 0, 0.7)]), now, args.min_stations, coords) is None, \
        "a weak lone station should not declare"
    print("[selftest] weak lone trigger correctly suppressed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="rtserve.iris.washington.edu:18000", help="SeedLink host:port")
    ap.add_argument("--min-stations", type=int, default=MIN_STATIONS, dest="min_stations",
                    help="stations that must agree to CONFIRM an event (fewer => a tentative alert)")
    ap.add_argument("--det-thresh", type=float, default=DET_THRESH, dest="det_thresh")
    ap.add_argument("--scan", type=float, default=2.0, help="seconds between detection scans")
    ap.add_argument("--dry-run", action="store_true", help="print instead of pushing")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
    elif args.replay:
        replay(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
