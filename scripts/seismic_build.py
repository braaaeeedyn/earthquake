"""Phase 1, Step 1: build + cache the single-station detection/magnitude dataset.

  python scripts/seismic_build.py --max-event 20 --max-noise 20 --nstations 2   # smoke
  python scripts/seismic_build.py                                               # full
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eq.quakecast import load_catalog  # noqa: E402
from eq.seismic import STATIONS, build  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "seismic_phase1.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mag", type=float, default=3.0)
    ap.add_argument("--max-dist", type=float, default=120.0)
    ap.add_argument("--max-event", type=int, default=200)
    ap.add_argument("--max-noise", type=int, default=200)
    ap.add_argument("--nstations", type=int, default=len(STATIONS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    cat = load_catalog("2010-01-01", "2023-12-31", min_mag=2.5)
    print(f"catalog: {len(cat)} events; building from {args.nstations} stations "
          f"(M>={args.min_mag}, <= {args.max_dist} km, "
          f"<= {args.max_event} ev + {args.max_noise} noise/station)...", flush=True)
    ds = build(cat, min_mag=args.min_mag, max_dist_km=args.max_dist,
               max_event_per_sta=args.max_event, max_noise_per_sta=args.max_noise,
               stations=STATIONS[:args.nstations])

    w, y = ds["waves"], ds["ydet"]
    print(f"\nDataset: {len(y)} windows  ({int((y==1).sum())} event / {int((y==0).sum())} noise)")
    if len(y):
        ev = ds["mag"][y == 1]
        print(f"  shape: {w.shape}  (samples/window={w.shape[1] if w.ndim>1 else 0})")
        print(f"  magnitude range: {ev.min():.1f}..{ev.max():.1f}  mean {ev.mean():.2f}")
        print(f"  event peak-amp (log10 counts): mean {ds['logamp'][y==1].mean():.2f}, "
              f"noise: {ds['logamp'][y==0].mean():.2f}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **ds)
        print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
