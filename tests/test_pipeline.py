import numpy as np

from eq import locate, network
from eq.pipeline import Config, Pipeline, clean_window


class Src:
    def __init__(self, now=0.0):
        self.now = now

    def end(self, c):
        return self.now

    def z(self, *a):
        return None

    def zne(self, *a):
        return None


def _pipe():
    return Pipeline(None, None, network.COORDS, network.CODES, Src(), Config())


def _picks(pipe, lat, lon, t0, n):
    d = locate.haversine_km(lat, lon, network.COORDS[:, 0], network.COORDS[:, 1])
    for i in np.argsort(d)[:n]:
        pipe.accept_pick(int(i), t0 + float(locate.travel_time(d[i])) + 2.0, 2.0, 8.0, 0.9)


def test_clean_window_gate():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, 3000)
    assert clean_window(w)
    z = w.copy(); z[:400] = 0
    assert not clean_window(z)                            # gap fill
    s = w.copy(); s[1000:1100] = 3.3
    assert not clean_window(s)                            # stuck run
    g = w.copy(); g[1500] = 500
    assert not clean_window(g)                            # lone glitch


def test_three_stations_confirm():
    p = _pipe()
    _picks(p, 34.1, -117.4, 0.0, 3)
    p.associate(40.0)
    assert p.events and p.events[-1].confirmed and len(p.pending) == 1


def test_two_stations_only_tentative():
    p = _pipe()
    _picks(p, 34.1, -117.4, 0.0, 2)
    p.associate(20.0)
    assert not p.events
    p.associate(80.0)
    assert p.events and not p.events[-1].confirmed and not p.push_eligible(p.events[-1])


def test_inconsistent_picks_do_not_confirm():
    """Three stations firing at times no single source can explain (shared noise) must not confirm."""
    p = _pipe()
    far = [network.INDEX["BAR"], network.INDEX["MPM"], network.INDEX["SMM"]]
    for k, i in enumerate(far):
        p.accept_pick(i, 10.0 + 1.0 * k, 0.0, 8.0, 0.9)    # near-simultaneous across ~400 km
    p.associate(15.0)
    assert not any(e.confirmed for e in p.events)


def test_refractory_blocks_repicks():
    p = _pipe()
    assert p.triggerable(0, 100.0, 0.9)
    p.accept_pick(0, 100.0, 1.0, 8.0, 0.9)
    assert not p.triggerable(0, 120.0, 0.95)


def test_push_floor():
    p = _pipe()
    _picks(p, 34.1, -117.4, 0.0, 4)
    p.associate(40.0)
    ev = p.events[-1]
    ev.mag = p.cfg.alert_min_mag
    assert p.push_eligible(ev)
    ev.mag = p.cfg.alert_min_mag - 0.2
    assert not p.push_eligible(ev)
    ev.mag = None
    assert not p.push_eligible(ev)


EARLY = {"T_s": 4.0, "a": 0.7, "b": 1.5, "c": 4.0, "early_min_mag": 3.04}


class AmpSrc(Src):
    """Every station shows the same peak velocity in its quick-check window."""
    def __init__(self, now, peak):
        super().__init__(now)
        self.peak = peak

    def zne(self, c, t1, t2):
        x = np.zeros((3, int(round((t2 - t1) * 100))))
        x[0, 10] = self.peak
        return x


def test_quick_check_runs_after_P_plus_T_and_gates_the_first_push():
    src = AmpSrc(0.0, 1e-4)
    p = Pipeline(None, None, network.COORDS, network.CODES, src, Config(), early=EARLY)
    _picks(p, 34.1, -117.4, 0.0, 4)
    p.associate(40.0)
    ev = p.events[-1]
    third = sorted(ev.picks.values())[2]
    src.now = third + EARLY["T_s"]                      # too early: needs the +6 s response margin
    p.early_ready(src.now)
    assert ev.early_mag is None
    src.now = third + EARLY["T_s"] + 6.5
    p.early_ready(src.now)
    assert ev.early_mag is not None and p.early_push_eligible(ev) == (ev.early_mag >= EARLY["early_min_mag"])


def test_no_quick_check_without_a_fit():
    p = Pipeline(None, None, network.COORDS, network.CODES, Src(), Config(), early=None)
    _picks(p, 34.1, -117.4, 0.0, 4)
    p.advance(40.0)
    assert p.events and not p.early_pending and not p.early_push_eligible(p.events[-1])
