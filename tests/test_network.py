import numpy as np

from eq import locate, network


def test_codes_unique_and_pinned():
    assert len(set(network.CODES)) == len(network.CODES)
    assert network.LOC["PASC"] == "10"                      # PASC streams two sensors; pin the STS-2


def test_no_retired_stations():
    for c in ("CCC", "CLC", "TOW2", "WBM", "RIO", "MWC", "BAK", "SCZ2"):
        assert c not in network.CODES


def test_min_spacing():
    """Neighbours closer than ~40 km share local noise and could 'confirm' each other."""
    c = network.COORDS
    d = locate.haversine_km(c[:, 0][:, None], c[:, 1][:, None], c[:, 0][None], c[:, 1][None])
    d[np.diag_indices_from(d)] = 1e9
    assert d.min() > 40.0
