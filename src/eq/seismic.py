"""Phase 1 seismic-waveform data: detection + magnitude from single-station traces.

Builds a small, cached, labeled dataset from SCEDC via ObsPy FDSN, driven by the USGS
California catalog. Event windows (labeled with magnitude) + noise windows (negatives),
as fixed-length vertical-component traces. Reusable; Phase 2 extends to multi-station + GNN.

Design notes:
  - The CNN sees each window std-normalized (learns transient SHAPE, not loudness).
  - Magnitude also needs scale, so we keep log10(peak amplitude) and log10(distance) as
    auxiliary scalars (this is exactly what the amplitude baseline uses).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance (km); accepts array or scalar args."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(np.asarray(lat2) - np.asarray(lat1)), np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


SR = 100.0                       # target sampling rate (Hz)
WIN_S = 30.0                     # window length (s)
NPTS = int(SR * WIN_S)           # 3000 samples
LEAD_S = 5.0                     # seconds of pre-P lead in an event window
VP_KMS = 6.5                     # crustal P velocity for a rough arrival estimate
CHANS = ("HHZ", "BHZ", "EHZ")    # vertical-channel preference

STATIONS = [("CI", "CCC"), ("CI", "CLC"), ("CI", "TOW2"),
            ("CI", "WBM"), ("CI", "PASC"), ("CI", "PFO")]
# Phase 2: a denser SoCal broadband network (for multi-station + GNN).
STATIONS2 = [("CI", "CCC"), ("CI", "CLC"), ("CI", "TOW2"), ("CI", "WBM"),
             ("CI", "PASC"), ("CI", "SVD"), ("CI", "RIO"), ("CI", "MWC"),
             ("CI", "DGR"), ("CI", "BAK")]
_CACHE = Path(__file__).resolve().parents[2] / "data" / "raw" / "seismic"
_CACHE3 = Path(__file__).resolve().parents[2] / "data" / "raw" / "seismic3c"


def get_client() -> Client:
    return Client("SCEDC", timeout=60)


def station_coords(client: Client, net: str, sta: str) -> tuple[float, float] | None:
    try:
        inv = client.get_stations(network=net, station=sta, level="station")
        s = inv[0][0]
        return float(s.latitude), float(s.longitude)
    except Exception:
        return None


def fetch_window(client: Client, net: str, sta: str, t_start: UTCDateTime) -> np.ndarray | None:
    """Vertical-component window -> (NPTS,) float32 at SR Hz; cached as .npy. None on failure."""
    cdir = _CACHE / f"{net}.{sta}"
    cdir.mkdir(parents=True, exist_ok=True)
    fp = cdir / f"{int(t_start.timestamp)}.npy"
    if fp.exists():
        return np.load(fp)
    for cha in CHANS:
        try:
            st = client.get_waveforms(net, sta, "*", cha, t_start, t_start + WIN_S + 1)
        except Exception:
            continue
        if not len(st):
            continue
        tr = st.merge(fill_value=0)[0]
        tr.detrend("demean")
        if abs(tr.stats.sampling_rate - SR) > 1e-6:
            tr.resample(SR)
        data = tr.data.astype(np.float32)
        w = np.zeros(NPTS, np.float32)
        n = min(NPTS, len(data))
        w[:n] = data[:n]
        np.save(fp, w)
        return w
    np.save(fp, np.zeros(0, np.float32))   # mark as tried-empty so we don't refetch
    return None


def fetch_window_3c(client, net, sta, t_start, inv):
    """3-component window (Z,N,E), response-removed to velocity (m/s) -> (3, NPTS); cached.
    Returns None on failure (and caches an empty sentinel so we don't refetch)."""
    cdir = _CACHE3 / f"{net}.{sta}"
    cdir.mkdir(parents=True, exist_ok=True)
    fp = cdir / f"{int(t_start.timestamp)}.npy"
    if fp.exists():
        a = np.load(fp)
        return a if a.ndim == 2 else None
    for band in ("HH", "BH", "EH"):
        try:
            st = client.get_waveforms(net, sta, "*", band + "?", t_start, t_start + WIN_S + 1)
        except Exception:
            continue
        if len(st) < 3:
            continue
        try:
            st.merge(fill_value=0); st.detrend("demean")
            st.remove_response(inventory=inv, output="VEL", water_level=60)
            if abs(st[0].stats.sampling_rate - SR) > 1e-6:
                st.resample(SR)
        except Exception:
            continue
        comp = {}
        for tr in st:
            d = tr.data.astype(np.float32)
            w = np.zeros(NPTS, np.float32); w[:min(NPTS, len(d))] = d[:min(NPTS, len(d))]
            comp[tr.stats.channel[-1]] = w
        if not all(k in comp for k in "ZNE"):
            continue
        arr = np.stack([comp["Z"], comp["N"], comp["E"]])
        np.save(fp, arr); return arr
    np.save(fp, np.zeros(0, np.float32)); return None


def build_multistation(catalog, min_mag=3.5, max_dist_km=200.0, min_stations=3,
                       max_events=400, stations=STATIONS2, seed=0):
    """Per-event multi-station 3-component samples for magnitude + the GNN.

    Returns X (N,S,3,NPTS), mask (N,S), mag (N,), dist (N,S), wtime (N,), coords (S,2)."""
    rng = np.random.default_rng(seed)
    client = get_client()
    invs, coords = {}, {}
    for net, sta in stations:
        try:
            inv = client.get_stations(network=net, station=sta, channel="*", level="response")
            invs[(net, sta)] = inv
            s = inv[0][0]; coords[(net, sta)] = (float(s.latitude), float(s.longitude))
        except Exception:
            print(f"  {net}.{sta}: no response/coords, skip", flush=True)
    stas = [ns for ns in stations if ns in coords]
    slat = np.array([coords[ns][0] for ns in stas]); slon = np.array([coords[ns][1] for ns in stas])
    S = len(stas)

    big = catalog[catalog["mag"] >= min_mag].reset_index(drop=True)
    ev_t = big["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    order = rng.permutation(len(big))
    X, mask, mag, dist, wtime = [], [], [], [], []
    n_ok = 0
    for k in order:
        if n_ok >= max_events:
            break
        elat, elon, emag = big["lat"][k], big["lon"][k], big["mag"][k]
        d = haversine_km(elat, elon, slat, slon)
        arr = np.zeros((S, 3, NPTS), np.float32); msk = np.zeros(S, bool)
        for j, ns in enumerate(stas):
            if d[j] > max_dist_km:
                continue
            origin = UTCDateTime(int(ev_t[k]))
            w = fetch_window_3c(client, ns[0], ns[1], origin + d[j] / VP_KMS - LEAD_S, invs[ns])
            if w is not None and w.shape == (3, NPTS) and np.abs(w).max() > 0:
                arr[j] = w; msk[j] = True
        if msk.sum() >= min_stations:
            X.append(arr); mask.append(msk); mag.append(float(emag))
            dist.append(d.astype(np.float32)); wtime.append(float(ev_t[k])); n_ok += 1
            if n_ok % 25 == 0:
                print(f"  events with >= {min_stations} stations: {n_ok}", flush=True)
    return {
        "X": np.asarray(X, np.float32), "mask": np.asarray(mask),
        "mag": np.asarray(mag, np.float32), "dist": np.asarray(dist, np.float32),
        "wtime": np.asarray(wtime, np.float64),
        "coords": np.asarray([coords[ns] for ns in stas], np.float32),
        "stations": np.asarray([f"{n}.{s}" for n, s in stas]),
    }


def build(catalog, min_mag=3.0, max_dist_km=120.0, max_event_per_sta=200,
          max_noise_per_sta=200, stations=STATIONS, seed=0):
    """Assemble event + noise windows. Returns dict of arrays + station metadata."""
    rng = np.random.default_rng(seed)
    client = get_client()
    ev_t = catalog["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    ev_lat = catalog["lat"].to_numpy(); ev_lon = catalog["lon"].to_numpy()
    ev_mag = catalog["mag"].to_numpy()

    waves, ydet, mags, dists, logamp, staid = [], [], [], [], [], []
    wtime, evidx = [], []
    coords = {}
    for sidx, (net, sta) in enumerate(stations):
        c = station_coords(client, net, sta)
        if c is None:
            print(f"  {net}.{sta}: no coords, skip", flush=True)
            continue
        coords[f"{net}.{sta}"] = c
        slat, slon = c
        d = haversine_km(ev_lat, ev_lon, slat, slon)
        near = np.where((ev_mag >= min_mag) & (d <= max_dist_km))[0]
        rng.shuffle(near)
        n_ev = 0
        for i in near:
            if n_ev >= max_event_per_sta:
                break
            origin = UTCDateTime(int(ev_t[i]))
            tp = origin + d[i] / VP_KMS
            w = fetch_window(client, net, sta, tp - LEAD_S)
            if w is None or not len(w) or np.allclose(w, 0):
                continue
            waves.append(w); ydet.append(1); mags.append(float(ev_mag[i]))
            dists.append(float(d[i])); logamp.append(float(np.log10(np.abs(w).max() + 1.0)))
            staid.append(sidx); wtime.append(float((tp - LEAD_S).timestamp)); evidx.append(int(i))
            n_ev += 1
        # noise windows: random times with no catalog event within +-60s / 200km
        n_no = 0; tries = 0
        tmin, tmax = ev_t.min(), ev_t.max()
        while n_no < max_noise_per_sta and tries < max_noise_per_sta * 6:
            tries += 1
            t = int(rng.integers(tmin, tmax))
            dd = haversine_km(ev_lat, ev_lon, slat, slon)
            if np.any((np.abs(ev_t - t) < 60) & (dd <= 200)):
                continue
            w = fetch_window(client, net, sta, UTCDateTime(t))
            if w is None or not len(w) or np.allclose(w, 0):
                continue
            waves.append(w); ydet.append(0); mags.append(-1.0)
            dists.append(-1.0); logamp.append(float(np.log10(np.abs(w).max() + 1.0)))
            staid.append(sidx); wtime.append(float(t)); evidx.append(-1); n_no += 1
        print(f"  {net}.{sta}: events={n_ev} noise={n_no}", flush=True)

    return {
        "waves": np.asarray(waves, np.float32),
        "ydet": np.asarray(ydet, np.int64),
        "mag": np.asarray(mags, np.float32),
        "dist": np.asarray(dists, np.float32),
        "logamp": np.asarray(logamp, np.float32),
        "staid": np.asarray(staid, np.int64),
        "wtime": np.asarray(wtime, np.float64),
        "evidx": np.asarray(evidx, np.int64),
        "stations": np.asarray([f"{n}.{s}" for n, s in stations]),
    }
