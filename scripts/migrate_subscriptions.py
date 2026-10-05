"""One-time migration of push subscriptions from the old 10-station list to eq/network.py.

Every subscribed code that is no longer in the network is replaced by the NEAREST current station
(so a Ridgecrest subscriber who followed CCC now follows LRL). Codes still in the network are kept.
Dry-run by default; --apply writes the file after saving a timestamped backup next to it.

  python scripts/migrate_subscriptions.py                       # show what would change
  python scripts/migrate_subscriptions.py --apply
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq import locate, network  # noqa: E402

TOKENS = ROOT / "data" / "processed" / "push_tokens.json"
OLD = {"CCC": (35.5249, -117.3645), "CLC": (35.8157, -117.5975), "TOW2": (35.8086, -117.7649),
       "WBM": (35.6084, -117.8905), "PASC": (34.1714, -118.1852), "SVD": (34.1065, -117.0982),
       "RIO": (34.1047, -117.9796), "MWC": (34.2236, -118.0583), "DGR": (33.6500, -117.0095),
       "BAK": (35.3444, -119.1044)}


def nearest_new(code):
    lat, lon = OLD[code]
    d = locate.haversine_km(lat, lon, network.COORDS[:, 0], network.COORDS[:, 1])
    return network.CODES[int(d.argmin())], float(d.min())


def migrate(tokens):
    changed = 0
    for t in tokens:
        new = []
        for c in t.get("stations", []):
            if c in network.INDEX:
                m = c
            elif c in OLD:
                m, _ = nearest_new(c)
            else:
                continue
            if m not in new:
                new.append(m)
        if new != t.get("stations"):
            changed += 1
            t["stations"] = new
    return tokens, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(TOKENS))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    for c in OLD:
        if c not in network.INDEX:
            m, d = nearest_new(c)
            print(f"  {c:5s} -> {m:5s} ({d:.0f} km)")
    fp = Path(args.file)
    if not fp.exists():
        print(f"no {fp}; nothing to migrate")
        return
    tokens, changed = migrate(json.loads(fp.read_text()))
    print(f"{changed} of {len(tokens)} subscriptions change")
    if args.apply and changed:
        shutil.copy(fp, fp.with_suffix(f".bak{int(time.time())}.json"))
        fp.write_text(json.dumps(tokens, indent=1))
        print(f"wrote {fp} (backup saved)")


if __name__ == "__main__":
    main()
