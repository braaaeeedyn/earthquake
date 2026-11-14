"""crosscheck_events: local matching against one catalog, and the retry on a USGS 429."""
import io
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import crosscheck_events as cc  # noqa: E402


def feat(id_, epoch, lat, lon, mag=2.0):
    return {"id": id_, "properties": {"mag": mag, "time": epoch * 1000, "place": id_},
            "geometry": {"coordinates": [lon, lat, 5.0]}}


EV = {"origin": 1_000_000.0, "lat": 34.0, "lon": -117.0}
CAT = [feat("far", 1_000_005, 36.0, -117.0), feat("late", 1_000_100, 34.0, -117.0),
       feat("near", 1_000_010, 34.1, -117.0), feat("nearer", 1_000_004, 34.2, -117.1),
       feat("small", 1_000_001, 34.0, -117.0, mag=0.5), feat("hour", 1_003_602, 34.0, -117.0)]


def test_matches_locally_nearest_in_time_within_both_tolerances():
    assert cc.match_real_for_declared(EV, 30, 60, catalog=CAT)["id"] == "nearer"
    assert cc.match_real_for_declared(EV, 30, 60, 3600.0, catalog=CAT)["id"] == "hour"
    assert cc.match_real_for_declared(EV, 30, 60, 7200.0, catalog=CAT) is None


def test_one_request_covers_every_event_and_its_chance_copy(monkeypatch):
    calls = []
    monkeypatch.setattr(cc, "usgs_query", lambda p: calls.append(p) or {"features": CAT})
    later = {"epoch": 1_050_000.0, "proxy_lat": 33.0, "proxy_lon": -116.0}
    assert cc.catalog_for([EV, later], 30, 60, 1.0) == CAT
    (p,) = calls
    assert p["starttime"] <= cc.iso(1_000_000 - 30) and p["endtime"] >= cc.iso(1_050_000 + 3600 + 30)
    assert p["minlatitude"] < 33.0 - 0.5 and p["maxlatitude"] > 34.0 + 0.5
    assert p["minlongitude"] < -117.0 - 0.6 and p["maxlongitude"] > -116.0 + 0.6


def test_a_429_is_retried(monkeypatch):
    answers = [urllib.error.HTTPError("u", 429, "Too Many Requests", None, None), io.BytesIO(b'{"features": []}')]

    def urlopen(req, timeout):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(cc.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)
    assert cc.usgs_query({}) == {"features": []}
