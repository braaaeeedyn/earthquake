"""P-wave picking, travel times and grid-search location -- shared by training AND live.

Why this exists: training windows were cut knowing the catalog answer (P at 5 s, catalog
epicentre), while the old live loop sized events on wall-clock windows with a station-as-epicentre
proxy, which collapsed every live magnitude to ~M2.5. Both the dataset builder and the live
pipeline now align windows on the SAME picker and measure distance from a LOCATED epicentre, so
the models see the same kind of input in training and in production.

  pick_p(z, ...)            -> onset index + SNR (STA/LTA trigger, AIC refinement)
  travel_time(dist_km, z)   -> P travel time (s), simple Pg/Pn crustal model (+ optional correction)
  Locator.locate(picks)     -> (lat, lon, t0, rms, used) by grid search, with outlier rejection
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

SR = 100.0
VP_CRUST = 6.1          # km/s, average crustal P (Pg)
VP_MANTLE = 7.9         # km/s, Pn
PN_INTERCEPT = 6.0      # s, Pn intercept for a ~30 km SoCal crust
DEPTH_KM = 8.0          # fixed hypocentre depth for live location (typical SoCal seismogenic depth)

_SOS = butter(4, [2.0, 10.0], btype="bandpass", fs=SR, output="sos")
TT_CORR = Path(__file__).resolve().parents[2] / "data" / "processed" / "v2" / "tt_correction.json"


def haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    a = (np.sin(r(np.asarray(lat2) - lat1) / 2) ** 2
         + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(r(np.asarray(lon2) - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------- travel time

_corr = None


def _correction():
    """Empirical residual (s) vs epicentral distance, fit from training picks by the builder."""
    global _corr
    if _corr is None:
        _corr = (np.array([[0.0, 0.0], [1000.0, 0.0]])
                 if not TT_CORR.exists() else np.array(json.loads(TT_CORR.read_text())["table"]))
    return _corr


def travel_time(dist_km, depth_km=DEPTH_KM, corrected=True):
    """First-arriving P time (s): min(direct Pg, head-wave Pn), plus the empirical correction."""
    d = np.asarray(dist_km, float)
    tg = np.sqrt(d ** 2 + depth_km ** 2) / VP_CRUST
    tn = d / VP_MANTLE + PN_INTERCEPT
    t = np.minimum(tg, tn)
    if corrected:
        c = _correction()
        t = t + np.interp(d, c[:, 0], c[:, 1])
    return t


# ---------------------------------------------------------------- picker

def bandpass(x):
    """Causal 2-10 Hz band-pass (causal so a pick is never pulled before the true onset)."""
    return sosfilt(_SOS, np.asarray(x, float) - np.mean(x))


def _sta_lta(x2, nsta, nlta):
    cs = np.concatenate(([0.0], np.cumsum(x2)))
    sta = np.zeros_like(x2)
    lta = np.zeros_like(x2)
    sta[nsta - 1:] = (cs[nsta:] - cs[:-nsta]) / nsta
    lta[nlta - 1:] = (cs[nlta:] - cs[:-nlta]) / nlta
    out = np.zeros_like(x2)
    ok = lta > 0
    out[ok] = sta[ok] / lta[ok]
    out[:nlta] = 0.0
    return out


def _aic(x):
    """Akaike picker: index minimizing k*log(var(x[:k])) + (n-k-1)*log(var(x[k:]))."""
    n = len(x)
    if n < 20:
        return None
    k = np.arange(5, n - 5)
    c1, c2 = np.cumsum(x), np.cumsum(x ** 2)
    m1 = c1[k - 1] / k
    v1 = c2[k - 1] / k - m1 ** 2
    m2 = (c1[-1] - c1[k - 1]) / (n - k)
    v2 = (c2[-1] - c2[k - 1]) / (n - k) - m2 ** 2
    aic = k * np.log(np.maximum(v1, 1e-30)) + (n - k - 1) * np.log(np.maximum(v2, 1e-30))
    return int(k[np.argmin(aic)])


def pick_p(z, lo=None, hi=None, on=3.5, sta_s=0.5, lta_s=5.0):
    """Pick the first P onset in z[lo:hi] (sample indices). Returns (index, snr) or (None, 0.0).

    STA/LTA on the filtered energy finds the first exceedance of `on`; AIC within [-2 s, +0.5 s]
    of it refines the onset. SNR = RMS(2 s after) / RMS(5 s before) on the filtered trace."""
    f = bandpass(z)
    cf = _sta_lta(f ** 2, int(sta_s * SR), int(lta_s * SR))
    lo = int(lta_s * SR) if lo is None else max(int(lo), int(lta_s * SR))
    hi = len(z) if hi is None else min(int(hi), len(z))
    if hi - lo < 10:
        return None, 0.0
    above = np.flatnonzero(cf[lo:hi] >= on)
    if not len(above):
        return None, 0.0
    i = lo + int(above[0])
    a, b = max(0, i - int(2.0 * SR)), min(len(f), i + int(0.5 * SR))
    j = _aic(f[a:b])
    p = a + j if j is not None else i
    pre = f[max(0, p - int(5 * SR)):p]
    post = f[p:p + int(2 * SR)]
    if len(pre) < SR or len(post) < SR / 2:
        return None, 0.0
    snr = float(np.sqrt(np.mean(post ** 2)) / (np.sqrt(np.mean(pre ** 2)) + 1e-30))
    return p, snr


# ---------------------------------------------------------------- locator

class Locator:
    """Grid-search epicentre (fixed depth) from P picks at known stations.

    Coarse 0.05 deg grid over SoCal, then a 0.01 deg refinement. Origin time is the median of
    (pick - travel time) at each node (robust to one bad pick); misfit is the RMS residual. With
    >= 4 picks, the worst pick is dropped while that cuts the RMS a lot (outlier rejection)."""

    def __init__(self, coords, bounds=(31.8, 37.0, -121.6, -113.8), step=0.05):
        self.coords = np.asarray(coords, float)
        lats = np.arange(bounds[0], bounds[1] + 1e-9, step)
        lons = np.arange(bounds[2], bounds[3] + 1e-9, step)
        g = np.array(np.meshgrid(lats, lons, indexing="ij")).reshape(2, -1).T
        self.grid = g
        d = haversine_km(g[:, :1], g[:, 1:], self.coords[:, 0][None], self.coords[:, 1][None])
        self.tt = travel_time(d)                                    # (G, S)

    @staticmethod
    def _fit(tt, sta, t):
        r = t[None, :] - tt[:, sta]                                 # (G, n) implied origin times
        t0 = np.median(r, axis=1)
        res = r - t0[:, None]
        return t0, np.sqrt(np.mean(res ** 2, axis=1)), res

    def _solve(self, sta, t):
        t0, rms, _ = self._fit(self.tt, sta, t)
        k = int(np.argmin(rms))
        la, lo = self.grid[k]
        fl = np.arange(la - 0.06, la + 0.0601, 0.01)
        fo = np.arange(lo - 0.06, lo + 0.0601, 0.01)
        g = np.array(np.meshgrid(fl, fo, indexing="ij")).reshape(2, -1).T
        d = haversine_km(g[:, :1], g[:, 1:], self.coords[sta, 0][None], self.coords[sta, 1][None])
        tt = travel_time(d)
        r = t[None, :] - tt
        t0f = np.median(r, axis=1)
        res = r - t0f[:, None]
        rmsf = np.sqrt(np.mean(res ** 2, axis=1))
        j = int(np.argmin(rmsf))
        return float(g[j, 0]), float(g[j, 1]), float(t0f[j]), float(rmsf[j]), res[j]

    def locate(self, picks, drop_ratio=0.5):
        """picks: {station_index: arrival_time (s, any common epoch)}. Needs >= 3 picks.
        Returns dict(lat, lon, t0, rms, used=[station indices], residuals) or None."""
        sta = np.array(sorted(picks), int)
        if len(sta) < 3:
            return None
        t = np.array([picks[s] for s in sta], float)
        lat, lon, t0, rms, res = self._solve(sta, t)
        while len(sta) > 3:
            w = int(np.argmax(np.abs(res)))
            keep = np.arange(len(sta)) != w
            cand = self._solve(sta[keep], t[keep])
            if cand[3] < drop_ratio * rms:                          # dropping it halves the misfit
                sta, t = sta[keep], t[keep]
                lat, lon, t0, rms, res = cand
            else:
                break
        return {"lat": lat, "lon": lon, "t0": t0, "rms": rms,
                "used": [int(s) for s in sta], "residuals": [float(x) for x in res]}
