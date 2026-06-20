"""Station graph for the GNN (MVP §3.1 station graph structure).

Edges are defined by geographic distance: stations within ``connect_km`` are
connected, with a Gaussian edge weight ``exp(-d^2 / 2 sigma^2)``.
"""
from __future__ import annotations

import numpy as np

from .config import Config, Station

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def build_station_graph(stations: tuple[Station, ...], cfg: Config) -> np.ndarray:
    """Symmetric weighted adjacency matrix (zero diagonal) for the station cluster."""
    n = len(stations)
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(stations[i].lat, stations[i].lon, stations[j].lat, stations[j].lon)
            if d <= cfg.graph.connect_km:
                w = float(np.exp(-(d ** 2) / (2.0 * cfg.graph.sigma_km ** 2)))
                a[i, j] = w
                a[j, i] = w
    return a


def normalized_adjacency(a: np.ndarray) -> np.ndarray:
    """Symmetric-normalized adjacency with self-loops: D^-1/2 (A + I) D^-1/2 (GCN form)."""
    a_hat = a + np.eye(a.shape[0])
    deg = a_hat.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return d_inv_sqrt @ a_hat @ d_inv_sqrt
