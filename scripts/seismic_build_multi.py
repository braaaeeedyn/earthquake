"""Phase 2a, Step 1: build the multi-station 3-component event dataset (for magnitude + GNN).

  python scripts/seismic_build_multi.py --max-events 6 --min-stations 2   # smoke
  python scripts/seismic_build_multi.py                                   # full
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eq.quakecast import load_catalog  # noqa: E402
from eq.seismic import build_multistation  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "seismic_phase2a.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mag", type=float, default=3.5)
    ap.add_argument("--max-dist", type=float, default=200.0)
    ap.add_argument("--min-stations", type=int, default=3)
    ap.add_argument("--max-events", type=int, default=400)
    ap.add_argument("--start", default="2010-01-01", help="catalog start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2023-12-31", help="catalog end date (YYYY-MM-DD)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    cat = load_catalog(args.start, args.end, min_mag=2.5)
    print(f"catalog {len(cat)} events; building multi-station 3C samples "
          f"(M>={args.min_mag}, <= {args.max_dist}km, >= {args.min_stations} stations)...", flush=True)
    ds = build_multistation(cat, min_mag=args.min_mag, max_dist_km=args.max_dist,
                            min_stations=args.min_stations, max_events=args.max_events)
    X, mask, mag = ds["X"], ds["mask"], ds["mag"]
    print(f"\nDataset: {len(mag)} events  shape={X.shape}  (events,stations,3comp,samples)")
    if len(mag):
        print(f"  stations: {list(ds['stations'])}")
        print(f"  stations/event: mean {mask.sum(1).mean():.1f} (min {mask.sum(1).min()}, "
              f"max {mask.sum(1).max()})")
        print(f"  magnitude: {mag.min():.1f}..{mag.max():.1f} (mean {mag.mean():.2f})")
        print(f"  peak velocity (m/s) over recorded traces: "
              f"{np.abs(X[mask]).max():.2e} max, {np.median(np.abs(X[mask]).max(axis=(1,2))):.2e} median")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **ds)
        print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
