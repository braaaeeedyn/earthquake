"""LIVE Southern-California earthquake detector: the trained models on a real-time SeedLink stream.

The engine is src/eq/pipeline.py (shared with the offline replay harness, so what is measured
offline is exactly what runs here):

  SeedLink (19 CI stations of eq/network.py, pinned location codes)  -> per-station buffers
    -> data-quality gate -> DETECT (30 s windows, data time) -> PICK the P onset
    -> LOCATE: >= 3 picks that fit one source = CONFIRMED (1-2 stations / poor fit = TENTATIVE, log only)
    -> SIZE once P+25 s has arrived: windows cut [P-5, P+25] per station, distance from the LOCATED
       epicentre (training geometry) -> magnitude ensemble (+ spread)
    -> QUICK CHECK ~10 s after P (first 4 s of P amplitude + located distance): a provisional push
       ("detected, confirming size") iff confirmed and the quick estimate clears its validated threshold
    -> FINAL push iff CONFIRMED and full-window M >= alert floor -- replaces the provisional one (same
       notification tag); a provisional push whose full size falls below the floor is RETRACTED.
       All pushes need PUSH_ENABLED=1 (env) and go to devices subscribed to a station within reach.

QuakeOps: one window per station every 30 s is logged as scale-free drift features (drift_check.py);
the daemon exits when data/processed/models.json names a newly deployed model version (tracking.py pull),
and server.py's supervisor respawns it on the new checkpoint.

Honesty: SeedLink latency is seconds to tens of seconds and sizing waits for P+25 s, so pushes go out
roughly 30-60 s after origin: RAPID DETECTION, not pre-arrival early warning. Coverage is strongest
where >= 3 stations are within ~100 km (LA basin, Inland Empire, Mojave, Ridgecrest, Kern).

  python scripts/live_watch.py --selftest      # deterministic checks (no network)
  python scripts/live_watch.py                 # LIVE (pushes only if PUSH_ENABLED=1)
"""
import argparse
import datetime
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import push_fcm  # noqa: E402
from eq import locate, network, seismic, shaking  # noqa: E402
from eq.pipeline import (ZNE_POST_S, ZNE_PRE_S, Config, Event, MagnitudeEnsemble, Pipeline, det_prep,  # noqa: E402
                         window_features)

DETECTOR = ROOT / "data" / "processed" / "detector.pt"
MAG_CKPT = ROOT / "data" / "processed" / "magnitude_ensemble.pt"
EVENTS_LOG = ROOT / "data" / "processed" / "events.jsonl"
STATUS = ROOT / "data" / "processed" / "live_status.json"
MODELS_JSON = ROOT / "data" / "processed" / "models.json"     # written by tracking.py pull (QuakeOps)
FEAT_DIR = ROOT / "data" / "processed" / "features"
FEATURE_EVERY = 15                # log 1 in 15 scanned windows per station (= one per 30 s) for drift_check.py
BUFFER_S = 300.0                  # seconds of 3-component history kept per station
STALE_S = 90.0                    # a station whose newest sample is older than this is "down"
ALERT_REACH_KM = 150.0            # devices subscribed to a station within this of the epicentre are alerted


def load_detector(device="cpu", path=DETECTOR):
    import torch
    from eq.models import DetectorNet
    ck = torch.load(path, weights_only=False, map_location=device)
    if list(ck["stations"]) != network.CODES:
        raise RuntimeError(f"detector trained on {list(ck['stations'])}, network is {network.CODES}")
    m = DetectorNet().to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


# ---------------------------------------------------------------- live waveform source

class LiveSource:
    """Rolling per-station buffers fed by SeedLink. Times are epoch seconds of the DATA."""

    def __init__(self, inv):
        import obspy
        self.inv = inv
        self.lock = threading.Lock()
        self.buf = {c: obspy.Stream() for c in network.CODES}
        self.last_packet = {}

    def add(self, tr):
        c = tr.stats.station
        if c not in self.buf or tr.stats.location != network.LOC[c]:
            return
        with self.lock:
            self.buf[c] += tr
            self.buf[c].merge(fill_value=0)
            self.buf[c].trim(starttime=self.buf[c][0].stats.endtime - BUFFER_S)
            self.last_packet[c] = time.time()

    def _snap(self, c):
        with self.lock:
            return self.buf[c].copy()

    def end(self, c):
        with self.lock:
            z = self.buf[c].select(channel="??Z")
            if not len(z):
                return None
            e = float(z[0].stats.endtime.timestamp)
        return e if time.time() - e < STALE_S else None

    def z(self, c, t1, t2):
        z = self._snap(c).select(channel="??Z")
        if not len(z):
            return None
        tr = z[0]
        x = seismic.lowpass(tr.data.astype(np.float64) - np.mean(tr.data))
        i1 = int(round((t1 - tr.stats.starttime.timestamp) * seismic.SR))
        n = int(round((t2 - t1) * seismic.SR))
        return x[i1:i1 + n] if i1 >= 0 and i1 + n <= len(x) else None

    def zne(self, c, t1, t2, sens=False):
        """3-C velocity for [t1, t2]. sens=True (quick check): sensitivity-scaled on [t1 - ZNE_PRE_S, t2], no
        post-window wait. Otherwise the response is removed on [t1 - ZNE_PRE_S, t2 + ZNE_POST_S] only --
        the same segment the replay harness corrects -- NOT on the whole 300 s buffer, whose end taper
        (5 % = 15 s) used to attenuate the newest seconds of the sizing window and bias magnitudes low."""
        import obspy
        a, b = obspy.UTCDateTime(t1 - ZNE_PRE_S), obspy.UTCDateTime(t2 + (0.0 if sens else ZNE_POST_S))
        st = self._snap(c).slice(a, b)
        if len(st) < 3 or min(tr.stats.endtime for tr in st) < b - 0.05:
            return None
        n_seg = int(round((b - a) * seismic.SR))
        x = seismic.to_zne(st, self.inv, a, n_seg, output="SENS" if sens else "VEL")
        if x is None:
            return None
        i1, n = int(round(ZNE_PRE_S * seismic.SR)), int(round((t2 - t1) * seismic.SR))
        return x[:, i1:i1 + n]

    def health(self):
        now = time.time()
        out = {}
        for c in network.CODES:
            e = None
            with self.lock:
                z = self.buf[c].select(channel="??Z")
                if len(z):
                    e = float(z[0].stats.endtime.timestamp)
            out[c] = {"up": e is not None and now - e < STALE_S,
                      "latency_s": None if e is None else round(now - e, 1)}
        return out


# ---------------------------------------------------------------- QuakeOps: drift features + model version

class FeatureLog:
    """Appends (ts, station, prob, crest, hf_ratio) for every FEATURE_EVERY-th clean window per station.
    Never raises: drift logging must not be able to break detection."""

    def __init__(self):
        self.n = {}

    def __call__(self, i, end, w, p):
        k = self.n.get(i, 0)
        self.n[i] = k + 1
        if k % FEATURE_EVERY:
            return
        try:
            crest, hf = window_features(det_prep(np.asarray(w)[None]))
            fp = FEAT_DIR / f"{datetime.datetime.fromtimestamp(end, datetime.timezone.utc):%Y-%m-%d}.csv"
            new = not fp.exists()
            FEAT_DIR.mkdir(parents=True, exist_ok=True)
            with open(fp, "a", encoding="utf-8") as f:
                if new:
                    f.write("ts,station,prob,crest,hf_ratio\n")
                f.write(f"{end:.1f},{network.CODES[i]},{p:.5f},{crest[0]:.4f},{hf[0]:.4f}\n")
        except Exception:                                # noqa: BLE001
            pass


def deployed_versions():
    """{'detector': v, 'magnitude': v} from models.json (None when the registry isn't in use)."""
    try:
        m = json.loads(MODELS_JSON.read_text())
        return {k: (m.get(k) or {}).get("deployed") for k in ("detector", "magnitude")}
    except (OSError, ValueError):
        return {"detector": None, "magnitude": None}


# ---------------------------------------------------------------- events: log + push

def nearest_station(lat, lon):
    d = locate.haversine_km(lat, lon, network.COORDS[:, 0], network.COORDS[:, 1])
    i = int(np.argmin(d))
    return network.CODES[i], float(d[i])


def log_event(ev: Event, pushed, eligible):
    code, dkm = nearest_station(ev.lat, ev.lon)
    rec = {
        "t": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "epoch": time.time(),
        "id": ev.id,
        "origin": round(ev.t0, 2),
        "confirmed": bool(ev.confirmed),
        "n_stations": len(ev.stations),
        "stations": [network.CODES[i] for i in ev.stations],
        "picks": ev.picks,
        "lat": round(ev.lat, 3), "lon": round(ev.lon, 3),
        "rms_s": None if ev.rms != ev.rms else round(ev.rms, 2),
        "silent_near": ev.silent_near,
        "nearest_station": code, "nearest_km": round(dkm, 1),
        "early_mag": None if ev.early_mag is None else round(ev.early_mag, 2),
        "early_pushed": int(getattr(ev, "early_sent", {}).get("standard", 0)),
        "early_after_origin_s": None if ev.early_at is None else round(ev.early_at - ev.t0, 1),
        # the fast alert-speed profile's quick check (2 s of P, no margin) -- sent only to "fast" subscribers
        "early_fast_mag": None if "fast" not in ev.early else round(ev.early["fast"]["mag"], 2),
        "early_fast_pushed": int(getattr(ev, "early_sent", {}).get("fast", 0)),
        "early_fast_after_origin_s": None if "fast" not in ev.early else round(ev.early["fast"]["at"] - ev.t0, 1),
        "mag": None if ev.mag is None else round(ev.mag, 2),
        "mag_spread": None if ev.mag_spread is None else round(ev.mag_spread, 2),
        "sized_after_origin_s": None if ev.sized_at is None else round(ev.sized_at - ev.t0, 1),
        "sized_stations": ev.sized_stations,
        "pgv_term": round(ev.pgv_term, 3),
        "push_eligible": bool(eligible),
        "pushed": int(pushed),
        # legacy fields read by crosscheck_events.py
        "proxy_station": code, "proxy_lat": round(ev.lat, 3), "proxy_lon": round(ev.lon, 3),
    }
    EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def push_message(ev: Event, user_stations, stage="final", mode="standard"):
    """Message for one device; distance = located epicentre -> the device's nearest subscribed sensor.
      stage 'early'   : provisional notice from the quick check of the device's alert-speed `mode` (size pending)
      stage 'final'   : confirmed magnitude (replaces the provisional notice)
      stage 'retract' : the full sizing came in below the felt floor (replaces the provisional notice)"""
    subs = [s for s in user_stations if s in network.INDEX]
    d = min(float(locate.haversine_km(ev.lat, ev.lon, *network.COORDS[network.INDEX[s]])) for s in subs)
    code, _ = nearest_station(ev.lat, ev.lon)
    region = dict((s[0], s[4]) for s in network.LIVE_NETWORK)[code]
    if stage == "early":
        q = ev.early[mode]["mag"]
        if mode == "fast":
            title = f"Fast alert: earthquake detected ({region})"
            body = (f"{len(ev.stations)} sensors located a quake about {d:.0f} km from your nearest sensor. "
                    f"Rough size M~{q:.1f} (fast setting: earlier, less certain); the confirmed magnitude follows "
                    f"in under a minute. Rapid detection, not an official warning.")
        else:
            title = f"Earthquake detected ({region}) — confirming size"
            body = (f"{len(ev.stations)} sensors located a quake about {d:.0f} km from your nearest sensor. "
                    f"Preliminary size M~{q:.1f}; the confirmed magnitude follows in under a minute. "
                    f"Rapid detection, not an official warning.")
        return title, body
    if stage == "retract":
        size = f"M{ev.mag:.1f}" if ev.mag is not None else "a size that could not be confirmed"
        title = f"Update: smaller quake ({region})"
        body = (f"The full measurement came in at {size}, below the level people usually feel. "
                f"You can disregard the earlier alert.")
        return title, body
    near = min(subs, key=lambda s: locate.haversine_km(ev.lat, ev.lon, *network.COORDS[network.INDEX[s]]))
    label, _ = shaking.describe(shaking.estimate_mmi(ev.mag, d, shaking.CAL.get("station_vs30", {}).get(near), ev.pgv_term))
    shake = " Likely too far to be felt where you are." if label == "Not felt" else f" {label} shaking possible near you."
    title = f"M{ev.mag:.1f} earthquake confirmed ({region})"
    body = (f"{len(ev.stations)} sensors located it about {d:.0f} km from your nearest sensor.{shake} "
            f"Rapid detection, not an official warning.")
    return title, body


def alert_devices(ev: Event, tokens, dry_run, stage="final", mode=None):
    """Push each device subscribed to any station within ALERT_REACH_KM of the epicentre, once per
    stage. Every stage of one event carries the same notification tag, so later stages REPLACE earlier
    ones on the device instead of stacking. `mode` limits the push to devices on that alert-speed
    profile (subscriptions without one are 'standard')."""
    d = locate.haversine_km(ev.lat, ev.lon, network.COORDS[:, 0], network.COORDS[:, 1])
    reach = {network.CODES[i] for i in np.flatnonzero(d <= ALERT_REACH_KM)}
    sent = 0
    for t in tokens:
        subs = [s for s in t.get("stations", []) if s in network.INDEX]
        if not subs or reach.isdisjoint(subs) or (mode and t.get("mode", "standard") != mode):
            continue
        title, body = push_message(ev, subs, stage, mode or "standard")
        if push_fcm.send_push(t["token"], title, body, dry_run, tag=f"quake-{ev.id}", data=quake_data(ev, stage, mode),
                              data_only="local_text" in t.get("caps", [])):
            sent += 1
    return sent


def quake_data(ev: Event, stage, mode):
    """Hidden fields sent with every push: the app uses them to open the quake and estimate shaking at the user's
    home on the device (and, for 'local_text' apps, to write the notification). The quick size stands in for the
    magnitude until the full sizing exists."""
    mag = ev.mag if stage != "early" or ev.mag is not None else ev.early.get(mode or "standard", {}).get("mag")
    code, _ = nearest_station(ev.lat, ev.lon)
    region = dict((s[0], s[4]) for s in network.LIVE_NETWORK)[code]
    return {"type": "quake", "stage": stage, "mode": mode or "standard", "id": ev.id, "lat": round(ev.lat, 3),
            "lon": round(ev.lon, 3), "t0": round(ev.t0, 1), "mag": None if mag is None else round(mag, 1),
            "pgv_term": round(ev.pgv_term, 3), "region": region}


def handle_early(pipe, ev, mode, dry_run, push_enabled):
    """Stage 1 for one alert-speed profile: its quick check ran. Provisional push to that profile's devices
    iff confirmed location and its quick M >= its threshold."""
    if not hasattr(ev, "early_sent"):
        ev.early_sent = {}
    ok = pipe.early_push_eligible(ev, mode)
    ev.early_sent[mode] = alert_devices(ev, push_fcm.load_tokens(), dry_run, "early", mode) if ok and push_enabled else 0
    print(f"[EARLY:{mode}] {ev.id} quick-check M~{ev.early[mode]['mag']:.1f} eligible={ok} "
          f"pushed={ev.early_sent[mode]}", flush=True)


def handle_event(pipe, ev, dry_run, push_enabled):
    """Stage 2: full sizing done (or a tentative event). Confirm with the magnitude, or retract a
    provisional notice that turned out below the felt floor."""
    eligible = pipe.push_eligible(ev)
    sent = 0
    if push_enabled and eligible:
        sent = alert_devices(ev, push_fcm.load_tokens(), dry_run, "final")       # every profile gets the confirmation
    elif push_enabled:                                                             # retract only where a provisional went
        for mode, n in getattr(ev, "early_sent", {}).items():
            if n:
                sent += alert_devices(ev, push_fcm.load_tokens(), dry_run, "retract", mode)
    log_event(ev, sent, eligible)
    tag = "EVENT" if ev.confirmed else "TENTATIVE"
    size = f"M{ev.mag:.1f}±{ev.mag_spread:.1f}" if ev.mag is not None else "unsized"
    print(f"[{tag}] {len(ev.stations)} sta {','.join(network.CODES[i] for i in ev.stations)} "
          f"at ({ev.lat:.2f},{ev.lon:.2f}) rms={ev.rms:.2f}s {size} eligible={eligible} pushed={sent}", flush=True)


# ---------------------------------------------------------------- live

def run_live(args):
    from obspy.clients.fdsn import Client
    from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
    cfg = Config.load()
    det = load_detector()
    mag = MagnitudeEnsemble(MAG_CKPT, network.CODES, network.COORDS)
    print(f"Fetching responses for {len(network.CODES)} stations...", flush=True)
    inv = Client("IRIS").get_stations(network=network.NET, station=",".join(network.CODES),
                                      channel="HH?", level="response")
    src = LiveSource(inv)
    push_enabled = os.environ.get("PUSH_ENABLED", "0") == "1"
    pipe = Pipeline(det, mag, network.COORDS, network.CODES, src, cfg,
                    on_event=lambda ev: handle_event(pipe, ev, args.dry_run, push_enabled),
                    on_early=lambda ev, mode: handle_early(pipe, ev, mode, args.dry_run, push_enabled))
    flog, on_window = FeatureLog(), pipe.on_window
    pipe.on_window = lambda i, end, w, p: (flog(i, end, w, p), on_window(i, end, w, p))
    models = deployed_versions()
    print(f"models {models}", flush=True)
    print(f"alert-speed profiles: {sorted(pipe.early) or 'none (no early_mag*.json) -> final magnitude only'}",
          flush=True)
    print(f"config {cfg}\npush {'ENABLED' if push_enabled else 'DISABLED (shadow mode)'}", flush=True)

    def loop():
        last_status = 0.0
        while True:
            time.sleep(1.0)
            ends = [e for e in (src.end(c) for c in network.CODES) if e is not None]
            if not ends:
                continue
            try:
                pipe.step(max(ends))
            except Exception as e:                       # noqa: BLE001  never let one bad step kill alerting
                print(f"step error: {e!r}", flush=True)
            if time.time() - last_status > 30:
                last_status = time.time()
                STATUS.write_text(json.dumps({"t": time.time(), "stations": src.health(),
                                              "push_enabled": push_enabled, "models": models}))
                if deployed_versions() != models:        # a new champion was installed: restart on it
                    print(f"models changed {models} -> {deployed_versions()}; exiting for respawn", flush=True)
                    os._exit(3)

    class Client_(EasySeedLinkClient):
        def on_data(self, trace):
            src.add(trace)

    threading.Thread(target=loop, daemon=True).start()
    print(f"Connecting to SeedLink {args.server} ...", flush=True)
    cli = Client_(args.server, autoconnect=False)
    cli.conn.timeout = 30                                # obspy 1.5 EasySeedLinkClient timeout=None bug
    cli.connect()
    for c in network.CODES:
        cli.select_stream(network.NET, c, f"{network.LOC[c]}HH?" if network.LOC[c] else "HH?")
    cli.run()


# ---------------------------------------------------------------- selftest

class _FakeSource:
    def __init__(self, now):
        self.now = now

    def end(self, c):
        return self.now

    def z(self, c, t1, t2):
        return None

    def zne(self, c, t1, t2, sens=False):
        return None


def selftest(args):
    """Deterministic checks: location from synthetic picks, the 3-station confirmation rule, the
    tentative path, the push floor and message targeting. No network, no models."""
    cfg = Config()
    src = _FakeSource(1000.0)
    got = []
    pipe = Pipeline(None, None, network.COORDS, network.CODES, src, cfg, on_event=got.append)
    lat, lon, t0 = 34.05, -117.55, 900.0                     # synthetic quake in the Inland Empire
    d = locate.haversine_km(lat, lon, network.COORDS[:, 0], network.COORDS[:, 1])
    near = np.argsort(d)[:4]
    for i in near:
        pipe.accept_pick(int(i), t0 + float(locate.travel_time(d[i])) + 3, 3.0, 10.0, 0.95)
    pipe.associate(t0 + 40)
    ev = pipe.events[-1]
    err = float(locate.haversine_km(lat, lon, ev.lat, ev.lon))
    assert ev.confirmed and len(ev.stations) >= 3 and err < 10, (ev, err)
    print(f"[selftest] 4 synthetic picks -> CONFIRMED, located {err:.1f} km from truth, rms {ev.rms:.2f}s")

    pipe2 = Pipeline(None, None, network.COORDS, network.CODES, src, cfg, on_event=got.append)
    for i in near[:2]:
        pipe2.accept_pick(int(i), t0 + float(locate.travel_time(d[i])) + 3, 3.0, 10.0, 0.95)
    pipe2.associate(t0 + 60)
    assert pipe2.events and not pipe2.events[-1].confirmed, "2 stations must stay TENTATIVE"
    print("[selftest] 2 agreeing stations -> TENTATIVE (logged, never pushed)")

    ev.mag = cfg.alert_min_mag + 0.4
    assert pipe.push_eligible(ev)
    ev.mag = cfg.alert_min_mag - 0.1
    assert not pipe.push_eligible(ev)
    assert not pipe2.push_eligible(pipe2.events[-1])
    print(f"[selftest] push floor M{cfg.alert_min_mag:g}: only confirmed events at/above it are eligible")

    ev.mag = 4.2
    title, body = push_message(ev, [network.CODES[int(near[0])]])
    assert "M4.2" in title and "sensors" in body
    far_only = [c for c in network.CODES if locate.haversine_km(ev.lat, ev.lon, *network.COORDS[network.INDEX[c]])
                > ALERT_REACH_KM][:1]
    assert alert_devices(ev, [{"token": "x", "stations": far_only}], dry_run=True) == 0
    print(f"[selftest] message: {title!r} / {body!r}")
    print("[selftest] devices subscribed only to far stations are not alerted")

    early = {"T_s": 4.0, "a": 0.7, "b": 1.5, "c": 4.0, "early_min_mag": 3.04}
    pipe3 = Pipeline(None, None, network.COORDS, network.CODES, src, cfg, early=early)
    ev.early_mag = 3.4
    assert pipe3.early_push_eligible(ev)
    ev.early_mag = 2.8
    assert not pipe3.early_push_eligible(ev)
    ev.early_mag, ev.mag = 3.4, 2.6
    t1, _ = push_message(ev, [network.CODES[int(near[0])]], "early")
    t2, b2 = push_message(ev, [network.CODES[int(near[0])]], "retract")
    assert "confirming" in t1 and "disregard" in b2
    print(f"[selftest] two-stage: provisional {t1!r} -> retraction {t2!r} (same notification tag)")
    ev.early["fast"] = {"mag": 3.3, "at": ev.t0 + 25}
    tf, bf = push_message(ev, [network.CODES[int(near[0])]], "early", "fast")
    assert "Fast alert" in tf and "less certain" in bf
    near_sub = [network.CODES[int(near[0])]]
    devs = [{"token": "s", "stations": near_sub}, {"token": "f", "stations": near_sub, "mode": "fast"}]
    assert alert_devices(ev, devs, dry_run=True, stage="early", mode="fast") == 0     # dry run: counts real sends
    print(f"[selftest] fast profile: {tf!r} goes only to devices whose mode is 'fast'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="rtserve.iris.washington.edu:18000")
    ap.add_argument("--dry-run", action="store_true", help="print pushes instead of sending")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
