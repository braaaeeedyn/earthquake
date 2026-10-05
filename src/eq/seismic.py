"""Seismic waveform access for the v2 dataset (SCEDC FDSN), built on the LIVE network only.

One FDSN request per event covers every network station within range; each station's 3-component
record is response-removed to ground velocity (m/s) and cut to a LONG window around its predicted
P arrival (P-32 s .. P+36 s). Long windows let training (a) re-align on the SAME picker the live
pipeline uses and (b) place the P onset anywhere in a detection crop (onset-position augmentation).
Windows are cached compactly (float16 + a per-trace scale) under data/raw/v2/.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from obspy import UTCDateTime, read_inventory
from obspy.clients.fdsn import Client
from scipy.signal import butter, sosfilt

from .locate import haversine_km  # noqa: F401  (re-exported: older scripts import it from here)
from .network import CODES, LOC, NET

SR = 100.0
PRE_S, POST_S = 32.0, 36.0                     # long event window around predicted P
NLONG = int(SR * (PRE_S + POST_S))              # 6800 samples
RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "v2"
INV_FILE = RAW / "inventory.xml"
LOWPASS_HZ = 18.0                               # common band: pre-2010 archive is 40 Hz BH, live is 100 Hz HH
_LP = butter(4, LOWPASS_HZ, btype="lowpass", fs=SR, output="sos")


def lowpass(x):
    """Causal 18 Hz low-pass applied to EVERY trace in training and live, so 40 Hz (BH) and 100 Hz
    (HH) sources look alike to the models. Causal, so it never moves energy before an onset."""
    return sosfilt(_LP, np.asarray(x, np.float64), axis=-1).astype(np.float32)


def get_client() -> Client:
    return Client("SCEDC", timeout=120)


def load_inventory(extra=()):
    """Response inventory (all epochs) for the network + any extra station codes; cached."""
    if INV_FILE.exists():
        return read_inventory(str(INV_FILE))
    RAW.mkdir(parents=True, exist_ok=True)
    inv = get_client().get_stations(network=NET, station=",".join(list(CODES) + list(extra)),
                                    channel="HH?,BH?", level="response")
    inv.write(str(INV_FILE), format="STATIONXML")
    return inv


def get_waveforms(client, codes, t1, t2, channel="HH?", tries=3):
    """One multi-station request; [] on no data. Retries transient failures."""
    for k in range(tries):
        try:
            return client.get_waveforms(NET, ",".join(codes), "*", channel, t1, t2)
        except Exception as e:                               # noqa: BLE001
            if "No data" in str(e) or "204" in str(e) or type(e).__name__ == "FDSNNoDataException":
                return []
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def station_traces(st, code, loc=None):
    """Traces of one station: the 100 Hz HH band if present (else the 40 Hz BH archive band),
    pinned to the station's location code when it has several sensors."""
    loc = LOC.get(code, "") if loc is None else loc
    sel = st.select(station=code)
    sel = sel.select(channel="HH?") if len(sel.select(channel="HH?")) else sel.select(channel="BH?")
    if loc and len(sel.select(location=loc)):
        return sel.select(location=loc)
    locs = sorted({tr.stats.location for tr in sel})
    return sel.select(location=locs[0]) if locs else sel


def to_zne(st, inv, t1, n, output="VEL"):
    """Station stream -> (3, n) float32 array [Z, N|1, E|2] from t1, response-removed (or None)."""
    if not len(st):
        return None
    st = st.copy()
    try:
        st.merge(fill_value=0)
        st.detrend("demean")
        if output == "VEL":
            st.remove_response(inventory=inv, output="VEL", water_level=60)
        else:
            st.remove_sensitivity(inv)
        for tr in st:
            if abs(tr.stats.sampling_rate - SR) > 1e-6:
                tr.resample(SR)
            tr.data = lowpass(tr.data)
    except Exception:                                         # noqa: BLE001
        return None
    out = np.zeros((3, n), np.float32)
    have = 0
    for tr in st:
        c = tr.stats.channel[-1]
        j = {"Z": 0, "N": 1, "1": 1, "E": 2, "2": 2}.get(c)
        if j is None:
            continue
        off = int(round((tr.stats.starttime - t1) * SR))
        d = tr.data.astype(np.float32)
        a, b = max(0, off), min(n, off + len(d))
        if b > a:
            out[j, a:b] = d[a - off:b - off]
            have |= 1 << j
    return out if have & 1 else None                          # Z is mandatory


def complete(x, max_zero_frac=0.02):
    """Reject gap-filled/dead traces (same spirit as the live clean_window gate)."""
    return (x is not None and bool(np.isfinite(x).all()) and np.mean(x[0] == 0.0) <= max_zero_frac
            and np.abs(x[0]).max() > 0)


def pack(x):
    """float32 -> (float16 normalized, float32 scale) per trace (last axis = time)."""
    s = np.abs(x).max(axis=-1, keepdims=True).astype(np.float32)
    s[s == 0] = 1.0
    return (x / s).astype(np.float16), s.squeeze(-1)


def unpack(xh, s):
    return xh.astype(np.float32) * s[..., None]


def utc(ts) -> UTCDateTime:
    return UTCDateTime(str(ts))
