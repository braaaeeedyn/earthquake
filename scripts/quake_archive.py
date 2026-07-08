"""Daily top-5 earthquake archive.

Tracks the five largest quakes of each UTC calendar day, for the WORLD and for the California
"area", using a size-5 min-heap (push every event, and once the heap holds five, evict the
smallest — so what remains is always the largest five). Buckets are keyed by the event's own
UTC date, so past days stay frozen and users can browse the biggest quakes of any day, even
after those events roll out of USGS's live 24-hour feed.
"""
import heapq
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "processed" / "quake_archive.json"
TOP_K = 5
KEEP_DAYS = 90
# California bounding box: (lat_min, lat_max, lon_min, lon_max)
CA = (32.0, 42.2, -124.6, -114.1)


def _utc_date(ms):
    return datetime.fromtimestamp((ms or 0) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _in_california(e):
    return CA[0] <= e["lat"] <= CA[1] and CA[2] <= e["lon"] <= CA[3]


def _top5(events):
    """Largest TOP_K events by magnitude via a size-limited min-heap. De-duplicates by id."""
    by_id = {e["id"]: e for e in events}
    heap = []  # (mag, id) — a min-heap, so heap[0] is the smallest of the ones we're keeping
    for e in by_id.values():
        heapq.heappush(heap, (e["mag"], e["id"]))
        if len(heap) > TOP_K:
            heapq.heappop(heap)  # eject the smallest of the current largest five
    keep = {i for _, i in heap}
    return sorted((by_id[i] for i in keep), key=lambda e: -e["mag"])


def _slim(e):
    return {k: e[k] for k in ("id", "mag", "lat", "lon", "place", "url", "time") if k in e}


def update(feed, path=ARCHIVE):
    """Merge a USGS feed into the per-day archive and persist it. Returns the archive dict."""
    arc = json.loads(path.read_text()) if path.exists() else {}
    by_date = defaultdict(list)
    for e in feed:
        by_date[_utc_date(e.get("time"))].append(_slim(e))
    for d, evs in by_date.items():
        bucket = arc.setdefault(d, {"world": [], "area": []})
        bucket["world"] = _top5(bucket["world"] + evs)
        bucket["area"] = _top5(bucket["area"] + [e for e in evs if _in_california(e)])
    for d in sorted(arc.keys())[:-KEEP_DAYS]:  # bound the file to the most recent KEEP_DAYS
        del arc[d]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(arc))
    return arc


def load(path=ARCHIVE):
    return json.loads(path.read_text()) if path.exists() else {}


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
