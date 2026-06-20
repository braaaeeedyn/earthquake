"""Single source of truth for the LOCKED rules (MVP §3.1).

Everything downstream (pipeline, labeling, graph, splits) reads from here so the
rules are explicit, version-controlled, and changed in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabelingRules:
    magnitude_threshold: float = 5.0   # M >= this counts as a target event
    proximity_km: float = 300.0        # event must be within this distance of ANY cluster station
    horizon_days: int = 7              # forward window we are predicting
    window_days: int = 27              # trailing matrix width  (24 x 27)
    hours_per_day: int = 24            # matrix height


@dataclass(frozen=True)
class PreprocessRules:
    min_minute_fraction: float = 0.5        # >= this fraction of an hour's minutes must be present, else hour = NaN
    max_sample_missing_fraction: float = 0.3  # drop a 24x27 sample if more than this fraction of cells are missing
    standardize: str = "per_station"        # normalizer fit on TRAIN ONLY (leakage-safe)


@dataclass(frozen=True)
class GraphRules:
    connect_km: float = 500.0   # connect two stations if within this distance
    sigma_km: float = 300.0     # Gaussian edge-weight length scale: w = exp(-d^2 / 2 sigma^2)


@dataclass(frozen=True)
class SplitRules:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    # Embargo (days) enforced between splits so no input(27d)+label(7d) window straddles a
    # boundary. 27 + 7 = 34 guarantees temporally disjoint coverage between splits.
    embargo_days: int = 34


@dataclass(frozen=True)
class Station:
    code: str
    lat: float
    lon: float


# MVP starter cluster: a geographically coherent set in Southern California
# (all pairwise distances < ~300 km, so the graph is connected at connect_km=500).
DEFAULT_STATIONS: tuple[Station, ...] = (
    Station("HOL", 35.00, -118.00),
    Station("SBA", 34.40, -119.70),
    Station("TEH", 35.10, -118.40),
    Station("RIV", 33.95, -117.40),
    Station("GOL", 35.40, -116.90),
)


@dataclass(frozen=True)
class Config:
    labeling: LabelingRules = field(default_factory=LabelingRules)
    preprocess: PreprocessRules = field(default_factory=PreprocessRules)
    graph: GraphRules = field(default_factory=GraphRules)
    split: SplitRules = field(default_factory=SplitRules)
    stations: tuple[Station, ...] = field(default_factory=lambda: DEFAULT_STATIONS)
    start_date: str = "2015-01-01"
    n_days: int = 540  # length of the synthetic record; real data sets this from the catalog span
