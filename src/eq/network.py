"""The live station network -- the ONE place the station list is defined.

Every consumer imports this table: the dataset builder, the live daemon, the API (/api/stations),
the cross-check scorer and the replay harness. The old per-file lists drifted apart and included
five stations that never streamed live (CCC, CLC, TOW2, WBM, RIO are archived at SCEDC but NOT
relayed on IRIS's public SeedLink), which is how the live network silently shrank to 5 stations.

Selection rules (reproduce with scripts/select_network.py):
  - streams HH? on rtserve.iris.washington.edu (checked by the builder before every build);
  - recording since <= 2010 (>= 15 years of training data);
  - quiet: median 2-8 Hz noise <= 1e-7 m/s over 4 times of day (BAK, ADO, VTV, USC ... excluded);
  - >= ~44 km apart (most pairs > 50 km), so neighbours cannot "confirm" each other on shared local (cultural) noise;
  - greedy coverage of SoCal's M3+ seismic areas + major cities, hand-adjusted for San Diego/Imperial.
  SCZ2 (Santa Cruz Island) was dropped: its real-time feed disappeared from rtserve during selection;
  SNCC (San Nicolas Island) covers the offshore / Channel Islands instead.

`loc` pins the location code where a station streams more than one sensor (PASC: '10' = the
STS-2 with continuous 2006->now history; '00' changed instrument in 2023).
"""
from __future__ import annotations

import numpy as np

NET = "CI"

# (station, location code, lat, lon, region)
LIVE_NETWORK = [
    ("PASC", "10", 34.17141, -118.18523, "LA basin"),
    ("BFS",  "",   34.23883, -117.65853, "LA basin"),
    ("SVD",  "",   34.10647, -117.09822, "Inland Empire"),
    ("DGR",  "",   33.65001, -117.00947, "Inland Empire"),
    ("BAR",  "",   32.68005, -116.67215, "San Diego / Imperial"),
    ("IKP",  "",   32.65012, -116.10948, "San Diego / Imperial"),
    ("SWS",  "",   32.94508, -115.79988, "San Diego / Imperial"),
    ("BEL",  "",   34.00060, -115.99820, "San Diego / Imperial"),
    ("GSC",  "",   35.30177, -116.80574, "Mojave"),
    ("GMR",  "",   34.78457, -115.65994, "Mojave"),
    ("EDW2", "",   34.88110, -117.99388, "Mojave"),
    ("LRL",  "",   35.47954, -117.68212, "Ridgecrest"),
    ("MPM",  "",   36.05799, -117.48901, "Ridgecrest"),
    ("ISA",  "",   35.66278, -118.47403, "Kern"),
    ("ARV",  "",   35.12690, -118.83009, "Kern"),
    ("SMM",  "",   35.31420, -119.99581, "Central coast / offshore"),
    ("MPP",  "",   34.88848, -119.81362, "Central coast / offshore"),
    ("SNCC", "",   33.24787, -119.52437, "Central coast / offshore"),
    ("CIA",  "",   33.40190, -118.41502, "Central coast / offshore"),
]

CODES = [s[0] for s in LIVE_NETWORK]
LOC = {s[0]: s[1] for s in LIVE_NETWORK}
COORDS = np.array([(s[2], s[3]) for s in LIVE_NETWORK], dtype=np.float64)   # (S, 2) lat, lon
INDEX = {c: i for i, c in enumerate(CODES)}


def as_api():
    """Station list for /api/stations (code, lat, lon, region)."""
    return [{"code": s[0], "lat": s[2], "lon": s[3], "region": s[4]} for s in LIVE_NETWORK]
