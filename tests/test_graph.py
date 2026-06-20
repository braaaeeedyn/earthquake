"""Station graph tests."""
import numpy as np

from eq.config import Config
from eq.graph import build_station_graph, haversine_km, normalized_adjacency


def test_haversine_one_degree_latitude():
    d = haversine_km(35.0, -118.0, 36.0, -118.0)
    assert 100.0 < d < 120.0  # ~111 km per degree of latitude


def test_graph_is_symmetric_and_weighted():
    cfg = Config()
    a = build_station_graph(cfg.stations, cfg)
    n = len(cfg.stations)
    assert a.shape == (n, n)
    assert np.allclose(a, a.T)              # symmetric
    assert np.all(np.diag(a) == 0)          # no self-loops in raw adjacency
    assert np.all((a >= 0) & (a <= 1))      # Gaussian weights in (0, 1]


def test_default_cluster_is_connected():
    cfg = Config()
    a = build_station_graph(cfg.stations, cfg)
    assert np.all(a.sum(axis=1) > 0)        # every station has at least one neighbor


def test_normalized_adjacency_has_self_loops_and_is_finite():
    cfg = Config()
    a = build_station_graph(cfg.stations, cfg)
    an = normalized_adjacency(a)
    assert an.shape == a.shape
    assert np.all(np.isfinite(an))
    assert np.all(np.diag(an) > 0)          # self-loops added
