import numpy as np

from eq import locate, network


def test_picker_finds_onset():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 3000)
    z[1800:] += 12 * np.sin(np.arange(1200) * 2 * np.pi * 5 / 100) * np.exp(-np.arange(1200) / 400)
    k, snr = locate.pick_p(z)
    assert k is not None and abs(k - 1800) < 30 and snr > 3


def test_picker_rejects_noise():
    z = np.random.default_rng(1).normal(0, 1, 3000)
    k, _ = locate.pick_p(z)
    assert k is None


def test_locator_recovers_epicentre():
    loc = locate.Locator(network.COORDS)
    lat, lon, t0 = 35.6, -117.6, 100.0
    d = locate.haversine_km(lat, lon, network.COORDS[:, 0], network.COORDS[:, 1])
    picks = {int(i): t0 + float(locate.travel_time(d[i])) for i in np.argsort(d)[:5]}
    sol = loc.locate(picks)
    assert locate.haversine_km(lat, lon, sol["lat"], sol["lon"]) < 5 and abs(sol["t0"] - t0) < 1


def test_locator_rejects_outlier_pick():
    loc = locate.Locator(network.COORDS)
    lat, lon = 34.0, -117.2
    d = locate.haversine_km(lat, lon, network.COORDS[:, 0], network.COORDS[:, 1])
    near = np.argsort(d)[:5]
    picks = {int(i): float(locate.travel_time(d[i])) for i in near}
    picks[int(near[-1])] += 15.0                          # one wild pick
    sol = loc.locate(picks)
    assert int(near[-1]) not in sol["used"] and sol["rms"] < 1.0


def test_two_picks_cannot_locate():
    assert locate.Locator(network.COORDS).locate({0: 1.0, 1: 2.0}) is None
