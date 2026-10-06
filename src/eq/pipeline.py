"""The detection -> pick -> locate -> size -> decide engine, shared by the live daemon and replay.

Everything runs on DATA time (waveform timestamps), never wall-clock time, and every model input
is cut the way its training data was cut:

  DETECT    each station's newest 30 s vertical window (after the data-quality gate) -> P(quake)
  PICK      on a station's first trigger, the shared STA/LTA+AIC picker finds the P onset
  LOCATE    >= MIN_STATIONS picks that fit ONE source (grid search, RMS <= MAX_RMS) AND pass the
            negative-evidence check -> CONFIRMED. With exactly 3 picks the fit is exactly determined
            (lat, lon, t0), so RMS alone proves nothing; what does is that no healthy station CLOSER
            to the epicentre than the picking ones stayed silent, and that a picking station is within
            MAX_NEAREST_KM. 1-2 stations (a 2-pick "fit" always fits) are TENTATIVE: logged, never pushed
  SIZE      once P+25 s has arrived at the stations needed, each station's window is cut at
            [P-5 s, P+25 s] (its pick, or the located P), distances come from the LOCATED epicentre,
            and the magnitude ensemble runs -- exactly the training geometry
  DECIDE    push iff CONFIRMED and M >= alert_min_mag

A `source` object supplies waveforms (live: SeedLink buffers; replay: archived FDSN data):
  source.z(code, t1, t2)        -> raw vertical samples at 100 Hz (np.ndarray) or None
  source.zne(code, t1, t2, sens=False) -> (3, n) velocity (m/s) or None: response-removed (sizing), or with
                                  sens=True scaled by the station's sensitivity only (quick check, no taper wait)
  source.end(code)              -> latest data time available for that station (epoch s) or None
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from . import locate

SR = 100.0
NPTS = 3000
# 3-C sizing windows are response-corrected on [t1 - ZNE_PRE_S, t2 + ZNE_POST_S] -- the SAME segment shape
# in live and replay. obspy's correction tapers ~5 % of the segment at each end (~3.3 s here), so the
# margins keep the taper off the 30 s window the model sees (training windows were untapered too).
ZNE_PRE_S, ZNE_POST_S = 30.0, 6.0
CONFIG_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "v2" / "pipeline_config.json"
EARLY_FILE = CONFIG_FILE.parent / "early_mag.json"     # fitted by scripts/fit_early_magnitude.py
# Alert-speed profiles for the FIRST (provisional) message; each subscriber picks one (push_tokens.json "mode").
# Both run for every confirmed event. (fit file, post-window margin s, sensitivity-only instead of response removal)
#   standard: 4 s of P, full response removal + the 6 s taper margin  -- most safeguards, first msg ~33-35 s
#   fast:     2 s of P, sensitivity-scaled, no margin (1-18 Hz is flat for broadband sensors, so no taper wait)
#             -- first msg ~26 s, quick size less accurate, more retractions
# Measured on 20 replayed days: replay_archive.py early-variants -> HOW_IT_WORKS.md section 5.3.
EARLY_PROFILES = {"standard": (EARLY_FILE, ZNE_POST_S, False),
                  "fast": (CONFIG_FILE.parent / "early_mag_T2.json", 0.0, True)}


@dataclass
class Config:
    det_thresh: float = 0.5        # per-window P(quake) that counts as a trigger
    pick_snr: float = 3.0          # minimum pick SNR to use a pick
    min_stations: int = 3          # picks that must fit one source for CONFIRMED
    max_rms: float = 1.5           # s, location misfit allowed for CONFIRMED
    max_silent_near: int = 0       # healthy stations nearer than the farthest picking one that may stay silent
    max_nearest_km: float = 120.0  # the closest picking station must be within this of the epicentre
    assoc_window: float = 90.0     # s, picks considered together
    refractory: float = 60.0       # s, one P pick per station per arrival (no S / coda re-picks)
    tentative_after: float = 40.0  # s, an unassociated pick older than this is logged as tentative
    event_sep_s: float = 120.0     # a new event needs this much origin-time separation (big-quake coda re-triggers) ...
    event_sep_km: float = 100.0    # ... or this much distance from the previous one
    size_radius_km: float = 200.0  # stations used for magnitude (training used <= 200 km)
    size_max_wait: float = 30.0    # s past the ideal sizing time before sizing with what has arrived
    alert_min_mag: float = 3.0     # felt-shaking push floor (final, full-window magnitude)
    early_max_wait: float = 20.0   # s past the ideal quick-check time before giving up on it
    scan_step: float = 2.0         # s between detection windows per station

    @classmethod
    def load(cls, path=CONFIG_FILE):
        if Path(path).exists():
            return cls(**{**asdict(cls()), **json.loads(Path(path).read_text())})
        return cls()


def clean_window(w):
    """Data-quality gate: reject gap-fill zeros, stuck/flat runs, clipping and lone glitch spikes --
    telemetry artefacts the detector never saw in training."""
    if len(w) < NPTS:
        return False
    if np.mean(w == 0.0) > 0.05:
        return False
    change = np.flatnonzero(np.diff(w) != 0)
    run = len(w) if change.size == 0 else int(np.diff(np.concatenate(([-1], change, [len(w) - 1]))).max())
    if run > 50:
        return False
    amax = float(np.max(np.abs(w)))
    if amax == 0.0 or np.mean(np.abs(w) >= 0.999 * amax) > 0.01:
        return False
    med = np.median(w)
    z = np.abs(w - med) / (1.4826 * (np.median(np.abs(w - med)) + 1e-9))
    if z.max() > 30.0 and int(np.sum(z > 10.0)) < 3:
        return False
    return True


_HP = butter(2, 1.0, btype="highpass", fs=SR, output="sos")


def det_prep(windows):
    """Detector input, IDENTICAL in training, replay and live: demean, causal 1 Hz high-pass, unit std.
    The high-pass removes the sub-1 Hz band where response-removed velocity (training positives) and
    sensitivity-scaled counts (noise, live) differ -- otherwise the model can learn the preprocessing
    instead of the earthquake. Broadband responses are flat across 1-18 Hz, where local quakes live."""
    x = np.asarray(windows, np.float64)
    x = sosfilt(_HP, x - x.mean(-1, keepdims=True), axis=-1)
    return ((x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)).astype(np.float32)


def window_features(x):
    """Scale-free drift features of det_prep'd windows (N, 3000): crest = log10(max|x| / rms) and
    hf_ratio = share of 1-18 Hz power above 5 Hz. Identical for training and live windows (QuakeOps drift)."""
    x = np.asarray(x, np.float64)
    crest = np.log10(np.abs(x).max(-1) / (np.sqrt((x ** 2).mean(-1)) + 1e-12) + 1e-12)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1 / SR)
    hf = spec[:, (f >= 5) & (f < 18)].sum(-1) / (spec[:, (f >= 1) & (f < 18)].sum(-1) + 1e-30)
    return crest, hf


def detect_probs(model, windows, device="cpu"):
    """P(quake) for a batch of raw 30 s vertical windows (det_prep, as in training)."""
    import torch
    x = det_prep(windows)
    with torch.no_grad():
        out = []
        for i in range(0, len(x), 512):
            out.append(torch.sigmoid(model(torch.tensor(x[i:i + 512], device=device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


@dataclass
class Pick:
    sta: int
    t: float
    prob: float
    snr: float
    used: bool = False


@dataclass
class Event:
    lat: float
    lon: float
    t0: float
    rms: float
    stations: list
    picks: dict
    confirmed: bool
    declared_at: float
    mag: float | None = None
    mag_spread: float | None = None
    sized_stations: list = field(default_factory=list)
    silent_near: int = 0
    early: dict = field(default_factory=dict)   # quick check per profile: mode -> {"mag", "at" (data time)}
    sized_at: float | None = None         # data time the full sizing ran

    @property
    def id(self):
        return f"{int(self.t0)}_{self.lat:.2f}_{self.lon:.2f}"

    # the standard profile's quick check (the original single-stage fields, kept for logs and callers)
    @property
    def early_mag(self):
        return self.early.get("standard", {}).get("mag")

    @early_mag.setter
    def early_mag(self, v):
        self.early.setdefault("standard", {})["mag"] = v

    @property
    def early_at(self):
        return self.early.get("standard", {}).get("at")


class Pipeline:
    def __init__(self, det_model, mag, coords, codes, source, cfg=None, device="cpu", on_event=None,
                 on_early=None, early="auto"):
        self.det, self.mag, self.device = det_model, mag, device
        self.coords, self.codes = np.asarray(coords, float), list(codes)
        self.source, self.cfg = source, cfg or Config.load()
        self.loc = locate.Locator(self.coords)
        self.on_event = on_event or (lambda ev: None)
        self.on_early = on_early or (lambda ev, mode: None)
        # early: "auto" = every EARLY_PROFILES fit that exists; None = no quick check; a dict = one standard-profile
        # fit (tests); or {mode: fit} with optional "post"/"sens" keys
        if isinstance(early, str):
            early = {m: {**json.loads(f.read_text()), "post": post, "sens": sens}
                     for m, (f, post, sens) in EARLY_PROFILES.items() if f.exists()}
        elif isinstance(early, dict) and "T_s" in early:
            early = {"standard": early}
        self.early = {m: {"post": ZNE_POST_S, "sens": False, **e} for m, e in (early or {}).items()}
        self.early_pending: list[tuple[Event, str]] = []
        self.picks: list[Pick] = []
        self.last_pick = {}                           # sta -> time of last pick (refractory)
        self.last_scan = {}                           # sta -> data time of last scanned window end
        self.events: list[Event] = []
        self.pending: list[Event] = []                # confirmed, awaiting sizing

    # ------------------------------------------------------------ detection + picking
    def scan(self, probs_fn=None):
        """Score every station's newest unscanned window. `probs_fn(windows)` defaults to the model."""
        wins, meta = [], []
        for i, c in enumerate(self.codes):
            end = self.source.end(c)
            if end is None or end - self.last_scan.get(i, -1e18) < self.cfg.scan_step:
                continue
            self.last_scan[i] = end
            w = self.source.z(c, end - NPTS / SR, end)
            if w is None or not clean_window(w):
                continue
            wins.append(w); meta.append((i, end))
        if not wins:
            return
        probs = (probs_fn or (lambda W: detect_probs(self.det, W, self.device)))(wins)
        for (i, end), w, p in zip(meta, wins, probs):
            self.on_window(i, end, w, float(p))

    def triggerable(self, i, end, p):
        return p >= self.cfg.det_thresh and end - self.last_pick.get(i, -1e18) >= self.cfg.refractory

    def on_window(self, i, end, w, p):
        if self.triggerable(i, end, p):
            k, snr = locate.pick_p(w, 50, len(w) - 50)
            self.accept_pick(i, end, None if k is None else (len(w) - k) / SR, snr, p)

    def accept_pick(self, i, end, lag_s, snr, p):
        """A triggered window's pick: `lag_s` = seconds from the onset to the window end (None if no
        onset was found). Replay feeds cached (lag, snr) through here, so the logic is identical."""
        if lag_s is None or snr < self.cfg.pick_snr:
            return
        t = end - lag_s
        self.last_pick[i] = t
        self.picks.append(Pick(i, t, p, snr))

    # ------------------------------------------------------------ association
    def associate(self, now):
        cfg = self.cfg
        self.picks = [p for p in self.picks if now - p.t <= cfg.assoc_window + 60]
        free = [p for p in self.picks if not p.used and now - p.t <= cfg.assoc_window]
        best = {}
        for p in free:                                 # one (earliest) pick per station
            if p.sta not in best or p.t < best[p.sta].t:
                best[p.sta] = p
        if len(best) >= cfg.min_stations:
            sol = self.loc.locate({s: p.t for s, p in best.items()})
            if sol and len(sol["used"]) >= cfg.min_stations and sol["rms"] <= cfg.max_rms and self._duplicate(sol):
                for s in sol["used"]:                  # later phases / coda of an event already declared
                    best[s].used = True
                return
            if sol and len(sol["used"]) >= cfg.min_stations and sol["rms"] <= cfg.max_rms:
                ev = Event(sol["lat"], sol["lon"], sol["t0"], sol["rms"], sol["used"],
                           {self.codes[s]: round(best[s].t, 2) for s in sol["used"]}, True, now)
                ev.silent_near = self._silent_near(ev)
                d = locate.haversine_km(ev.lat, ev.lon, self.coords[sol["used"], 0], self.coords[sol["used"], 1])
                if ev.silent_near <= cfg.max_silent_near and d.min() <= cfg.max_nearest_km:
                    for s in sol["used"]:
                        best[s].used = True
                    self.events.append(ev)
                    self.pending.append(ev)
                    self.early_pending += [(ev, m) for m in self.early]
                    return
        # tentative: picks that waited long enough without forming a confirmed event
        stale = [p for p in free if now - p.t > cfg.tentative_after]
        if stale:
            for p in stale:
                p.used = True
            sta = sorted({p.sta for p in stale})
            s0 = max(stale, key=lambda p: p.prob).sta
            ev = Event(float(self.coords[s0, 0]), float(self.coords[s0, 1]), min(p.t for p in stale), float("nan"),
                       sta, {self.codes[p.sta]: round(p.t, 2) for p in stale}, False, now)
            self.events.append(ev)
            self.on_event(ev)

    def _duplicate(self, sol):
        for e in self.events[-20:]:
            if e.confirmed and abs(e.t0 - sol["t0"]) < self.cfg.event_sep_s and \
                    locate.haversine_km(e.lat, e.lon, sol["lat"], sol["lon"]) < self.cfg.event_sep_km:
                return True
        return False

    def _silent_near(self, ev):
        """Negative evidence: healthy stations closer to the epicentre than the FARTHEST picking
        station that did not pick. A real quake reaches nearer stations first, so silence there means
        the 'event' is coincident noise at scattered stations (the 3-pick fit is exactly determined)."""
        d = locate.haversine_km(ev.lat, ev.lon, self.coords[:, 0], self.coords[:, 1])
        far = np.max(d[ev.stations])
        n = 0
        for i in np.flatnonzero(d < far):
            if i not in ev.stations and self.source.end(self.codes[i]) is not None:
                n += 1
        return n

    # ------------------------------------------------------------ sizing
    def size_ready(self, now):
        if self.mag is None:                          # detection-only mode (replay calibration)
            for ev in self.pending:
                self.on_event(ev)
            self.pending.clear()
            return
        for ev in list(self.pending):
            d = locate.haversine_km(ev.lat, ev.lon, self.coords[:, 0], self.coords[:, 1])
            near = np.flatnonzero(d <= self.cfg.size_radius_km)
            p_at = {i: ev.picks.get(self.codes[i], ev.t0 + float(locate.travel_time(d[i]))) for i in near}
            need = sorted(ev.stations, key=lambda i: d[i])[:self.cfg.min_stations]
            ideal = max(p_at[i] for i in need) + 25.0 + ZNE_POST_S
            if now < ideal:
                continue
            X = np.zeros((len(self.codes), 3, NPTS), np.float32)
            mask = np.zeros(len(self.codes), bool)
            for i in near:
                t1 = p_at[i] - 5.0
                end = self.source.end(self.codes[i])
                if end is None or end < t1 + NPTS / SR + ZNE_POST_S:
                    continue
                x = self.source.zne(self.codes[i], t1, t1 + NPTS / SR)
                if x is not None and x.shape == (3, NPTS) and np.mean(x[0] == 0) < 0.05:
                    X[i], mask[i] = x, True
            if mask.sum() < self.cfg.min_stations and now < ideal + self.cfg.size_max_wait:
                continue
            self.pending.remove(ev)
            ev.sized_at = now
            if mask.sum():
                ev.mag, ev.mag_spread = self.mag.predict(X, mask, d.astype(np.float32))
                ev.sized_stations = [self.codes[i] for i in np.flatnonzero(mask)]
            self.on_event(ev)

    # ------------------------------------------------------------ quick check (preliminary size)
    def early_ready(self, now):
        """Preliminary magnitude from only the first T s after P at the PICKED stations (classic early-
        warning amplitude scaling, fitted by scripts/fit_early_magnitude.py), once per alert-speed profile:
        T s (+ that profile's margin) after the third pick. Decides whether that profile's first, provisional
        push goes out."""
        cfg = self.cfg
        for ev, mode in list(self.early_pending):
            e = self.early[mode]
            picked = sorted(ev.stations, key=lambda i: ev.picks[self.codes[i]])
            ideal = ev.picks[self.codes[picked[cfg.min_stations - 1]]] + e["T_s"] + e["post"]
            if now < ideal:
                continue
            d = locate.haversine_km(ev.lat, ev.lon, self.coords[:, 0], self.coords[:, 1])
            est = []
            for i in picked:
                tp = ev.picks[self.codes[i]]
                end = self.source.end(self.codes[i])
                if end is None or end < tp + e["T_s"] + e["post"]:
                    continue
                x = self.source.zne(self.codes[i], tp, tp + e["T_s"], sens=e["sens"])
                if x is None or not np.isfinite(x).all():
                    continue
                peak = float(np.abs(x).max()) + 1e-12
                est.append(e["a"] * np.log10(peak) + e["b"] * np.log10(max(float(d[i]), 1.0)) + e["c"])
            if len(est) < cfg.min_stations and now < ideal + cfg.early_max_wait:
                continue
            self.early_pending.remove((ev, mode))
            if est:
                ev.early[mode] = {"mag": float(np.median(est)), "at": now}
                self.on_early(ev, mode)

    def early_push_eligible(self, ev, mode="standard"):
        """First, provisional push for one alert-speed profile: confirmed location AND that profile's quick check
        clears its validated threshold."""
        e, r = self.early.get(mode), ev.early.get(mode, {})
        return bool(e and ev.confirmed and r.get("mag") is not None and r["mag"] >= e["early_min_mag"])

    def advance(self, now):
        """Everything after detection, in order: associate/locate -> quick check -> full sizing.
        Live (`step`) and the replay harness both call this, so they can never drift apart."""
        self.associate(now)
        if self.early:
            self.early_ready(now)
        self.size_ready(now)

    def step(self, now, probs_fn=None):
        self.scan(probs_fn)
        self.advance(now)

    def push_eligible(self, ev):
        return bool(ev.confirmed and ev.mag is not None and ev.mag >= self.cfg.alert_min_mag)


class MagnitudeEnsemble:
    """The multi-station magnitude ensemble with its normalizers, loaded from one checkpoint that
    also carries the amplitude scale and the station list it was trained on."""

    def __init__(self, ckpt_path, codes, coords, device="cpu"):
        import torch
        from .models import MultiStationModel, adjacency
        ck = torch.load(ckpt_path, weights_only=False, map_location=device)
        if list(ck["stations"]) != list(codes):
            raise RuntimeError(f"magnitude checkpoint stations {list(ck['stations'])} != network {list(codes)}")
        A = adjacency(np.asarray(coords, np.float32))
        self.models = []
        for st in ck["states"]:
            m = MultiStationModel(A, hybrid=True, amp_feature=True).to(device)
            m.load_state_dict(st)
            m.eval()
            self.models.append(m)
        self.la_mu, self.la_sd, self.am, self.asd = ck["la_mu"], ck["la_sd"], ck["am"], ck["asd"]
        self.device = device

    def predict(self, X, mask, dist):
        """X: (S, 3, 3000) velocity (m/s), windows [P-5, P+25]; mask: recorded; dist: km from epicentre.
        Same representation as training: unit-peak waveforms + standardized log10 peak per station."""
        import torch
        rec = mask.astype(bool)
        peak = np.abs(X).max(axis=(1, 2)) + 1e-12
        la = np.log10(peak).astype(np.float32)
        xn = (X / peak[:, None, None]).astype(np.float32) * rec[:, None, None]
        ld = np.log10(np.maximum(dist[rec], 1.0))
        feats = np.array([la[rec].mean(), la[rec].max(), ld.mean(), ld.min()], np.float32)
        afn = ((feats - self.am) / self.asd).astype(np.float32)[None]
        logdist = np.where(dist > 0, np.log10(np.maximum(dist, 1.0)), 0.0).astype(np.float32)[None]
        las = (((la - self.la_mu) / self.la_sd) * rec).astype(np.float32)[None]
        t = lambda a: torch.tensor(a, device=self.device)  # noqa: E731
        xs, ms, ls, a, lt = t(xn[None]), t(rec[None]), t(logdist), t(afn), t(las)
        with torch.no_grad():
            preds = [float(m(xs, ms, ls, a, lt).item()) for m in self.models]
        return float(np.mean(preds)), float(np.std(preds))
