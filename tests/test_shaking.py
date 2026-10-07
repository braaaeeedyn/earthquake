import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import push_fcm  # noqa: E402

from eq import shaking  # noqa: E402


def test_event_term_is_zero_for_a_quake_that_shook_as_predicted():
    dists, codes = [20.0, 60.0, 120.0], ["PASC", "BFS", "SVD"]
    vs = shaking.CAL.get("station_vs30", {})
    obs = [shaking.estimate_pgv(4.0, r, vs.get(c)) for r, c in zip(dists, codes)]
    assert abs(shaking.event_term(4.0, dists, obs, codes)) < 1e-9
    assert shaking.event_term(4.0, dists, [p * 100 for p in obs], codes) == shaking.EVENT_TERM_CLIP   # clipped


def test_soft_ground_and_closer_means_stronger_shaking():
    rock, soft = shaking.estimate_mmi(4.5, 30, vs30=760), shaking.estimate_mmi(4.5, 30, vs30=250)
    assert soft > rock
    assert shaking.estimate_mmi(4.5, 10) > shaking.estimate_mmi(4.5, 100)
    assert shaking.describe(4.4)[0] == "Light"


def test_push_message_shapes():
    data = {"id": "e1", "lat": 34.1, "mag": 3.6, "pgv_term": None}
    normal = push_fcm.build_message("t", "T", "B", "quake-e1", data)["message"]
    assert normal["notification"] == {"title": "T", "body": "B"} and normal["data"] == {"id": "e1", "lat": "34.1", "mag": "3.6"}
    assert normal["android"]["notification"]["tag"] == "quake-e1"
    only = push_fcm.build_message("t", "T", "B", "quake-e1", data, data_only=True)["message"]
    assert "notification" not in only and only["data"]["title"] == "T" and only["data"]["tag"] == "quake-e1"


def test_rounding_matches_the_app():
    assert shaking.describe(2.5)[0] == "Weak" and shaking.describe(2.49)[0] == "Not felt"


def test_pre_v2_calibration_file_falls_back_to_the_v2_fit(tmp_path, monkeypatch):
    old = tmp_path / "shaking_calibration.json"
    old.write_text('{"gmpe": {"a": 0, "b": 0, "c": 0, "d": 0}, "pgv_floor": 1e-5}')   # no site term
    monkeypatch.setattr(shaking, "CAL_PATH", old)
    assert shaking.load() is shaking._DEFAULT
