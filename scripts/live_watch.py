"""LIVE Southern-California earthquake detector — the models run continuously on a real-time
waveform stream (no USGS in the loop).

Architecture:
  SeedLink stream (10 CI/SCEDC SoCal stations)  -> rolling per-station buffers
    -> DETECTION model runs continuously on sliding 30 s vertical windows
    -> COINCIDENCE: an event is declared only when >= K stations trigger within a short window
       (this is what kills single-station false alarms)
    -> location proxy (strongest-triggering station) + MAGNITUDE model sizes the event
    -> subscribers whose estimated shaking clears a threshold get emailed detection + size + shaking

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
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import shaking_model  # noqa: E402
from nearme_watch import SUBS, haversine_km, load_json, send_email  # noqa: E402

DETECTOR = ROOT / "data" / "processed" / "detector.pt"
MAG_CKPT = ROOT / "data" / "processed" / "magnitude_ensemble.pt"
PHASE2A = ROOT / "data" / "processed" / "seismic_phase2a_xl.npz"
PHASE1 = ROOT / "data" / "processed" / "seismic_phase1.npz"

SR = 100.0
NPTS = 3000                       # 30 s @ 100 Hz
SCALE = 7.775235e-4               # training amplitude scale = X[mask].std() over phase2a_xl (m/s)
NET = "CI"

# Tunables (also CLI flags)
DET_THRESH = 0.60                 # per-station detection probability to count as a trigger
MIN_STATIONS = 4                  # COINCIDENCE: stations that must agree to declare an event
COINC_WIN = 12.0                  # seconds within which triggers count toward the same event
COOLDOWN = 120.0                  # seconds to suppress re-alerting the same event
ALERT_MMI = 3.0                   # alert a subscriber when estimated shaking (MMI) >= this


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

def declare_from_triggers(triggers, now, min_stations):
    """Return (station_indices, strongest_idx) if >= min_stations distinct stations triggered
    within COINC_WIN, else None. `triggers` is a deque of (time, sta_idx, prob)."""
    recent = [t for t in triggers if now - t[0] <= COINC_WIN]
    by_sta = {}
    for _, si, p in recent:
        by_sta[si] = max(by_sta.get(si, 0.0), p)
    if len(by_sta) < min_stations:
        return None
    strongest = max(by_sta, key=by_sta.get)
    return sorted(by_sta), strongest


def alert_subscribers(subs, epi_lat, epi_lon, mag, det_conf, nstations, dry_run):
    """Email every subscriber whose estimated shaking reaches the threshold. Returns count sent."""
    sent = 0
    for s in subs:
        dist = haversine_km(epi_lat, epi_lon, s["lat"], s["lon"])
        mmi = shaking_model.estimate_mmi(mag, dist) if mag is not None else 0.0
        if mag is not None and mmi < ALERT_MMI:
            continue
        if mag is None:
            continue                                  # no size -> no shaking-based decision
        label, desc = shaking_model.describe(mmi)
        subject = f"Live quake detected: M{mag:.1f} — {label.lower()} shaking expected"
        body = (f"Hi {s['name']},\n\n"
                f"Our model just detected an earthquake on the live Southern-California seismic "
                f"stream (detection confidence {det_conf:.0%}), about {dist:.0f} km from you.\n\n"
                f"Detection: confirmed by the model on {nstations} stations.\n"
                f"Size: estimated magnitude {mag:.1f}.\n"
                f"Estimated shaking where you are: {label} — intensity {round(mmi)} of 10 ({desc}).\n\n"
                f"(Research prototype - rapid detection, not an official warning. Location is a "
                f"station-based estimate; shaking is a model estimate.)")
        send_email(s["email"], subject, body, dry_run)
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
    last_alert = [0.0]
    subs = load_json(SUBS, [])

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
            decl = declare_from_triggers(triggers, now, args.min_stations)
            if decl and now - last_alert[0] > COOLDOWN:
                stas, strongest = decl
                last_alert[0] = now
                epi_lat, epi_lon = coords[strongest]
                conf = max(p for t, si, p in triggers if now - t <= COINC_WIN)
                mag = size_event(snap, names, coords, epi_lat, epi_lon, inv, mag_models, am, asd) \
                    if mag_models else None
                nsent = alert_subscribers(subs, epi_lat, epi_lon, mag, conf, len(stas), args.dry_run)
                msize = f"M{mag:.1f}" if mag is not None else "size n/a"
                print(f"[EVENT] {len(stas)} stations, near {names[strongest]} "
                      f"({epi_lat:.2f},{epi_lon:.2f}) {msize} -> {nsent} subscriber(s) alerted")

    class Client_(EasySeedLinkClient):
        def on_data(self, trace):
            with lock:
                buffers[trace.stats.station] += trace
                buffers[trace.stats.station].merge(fill_value=0)
                buffers[trace.stats.station].trim(starttime=obspy.UTCDateTime() - 120)

    threading.Thread(target=scan, daemon=True).start()
    print(f"Connecting to SeedLink {args.server} ...  (Ctrl-C to stop)")
    cli = Client_(args.server)
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
    """Deterministic pipeline check: fake >=K coincident triggers -> declare -> alert (dry-run)."""
    subs = load_json(SUBS, [])
    if not subs:
        print(f"No subscribers in {SUBS.relative_to(ROOT)}.")
        return
    stations, coords = load_network()
    now = time.time()
    trig = deque((now, i, 0.9) for i in range(args.min_stations))   # K stations agree
    decl = declare_from_triggers(trig, now, args.min_stations)
    assert decl, "coincidence should declare with K triggers"
    _, strongest = decl
    s0 = subs[0]
    epi_lat, epi_lon = s0["lat"] + 0.2, s0["lon"]              # a quake ~22 km from subscriber[0]
    mag = 5.2
    print(f"[selftest] declared event near {stations[strongest]}, placed M{mag} "
          f"~{haversine_km(epi_lat, epi_lon, s0['lat'], s0['lon']):.0f} km from {s0['name']}")
    n = alert_subscribers(subs, epi_lat, epi_lon, mag, 0.95, args.min_stations, dry_run=True)
    print(f"[selftest] {n} subscriber(s) would be alerted (dry-run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="rtserve.iris.washington.edu:18000", help="SeedLink host:port")
    ap.add_argument("--min-stations", type=int, default=MIN_STATIONS, dest="min_stations")
    ap.add_argument("--det-thresh", type=float, default=DET_THRESH, dest="det_thresh")
    ap.add_argument("--scan", type=float, default=2.0, help="seconds between detection scans")
    ap.add_argument("--dry-run", action="store_true", help="print instead of emailing")
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
