"""Phase 1: fetch the California catalog and sanity-check the forecasting task.

Verifies the input catalog is complete enough (low Mc) and that the target base rate is
learnable (not <2%) before any modeling.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eq.quakecast import CA_BBOX, daily_features, daily_targets, load_catalog  # noqa: E402

START, END = "2010-01-01", "2023-12-31"
HORIZON = 7
MC = 2.5  # input completeness magnitude


def main():
    n_days = (np.datetime64(END) - np.datetime64(START)).astype(int)
    cat = load_catalog(START, END, min_mag=MC)
    print(f"California box {CA_BBOX}, {START}..{END} ({n_days} days)")
    print(f"Input catalog M>={MC}: {len(cat)} events  "
          f"(M>=4: {int((cat['mag']>=4).sum())}, M>=4.5: {int((cat['mag']>=4.5).sum())}, "
          f"M>=5: {int((cat['mag']>=5).sum())})")
    print(f"  magnitude range {cat['mag'].min():.1f}..{cat['mag'].max():.1f}\n")

    print(f"Target base rates (>=1 event in next {HORIZON}d):")
    for tmag in (4.0, 4.5, 5.0):
        y, pos_day = daily_targets(cat, START, n_days, tmag, HORIZON)
        print(f"  M>={tmag}: {int(pos_day.sum())} target-event-days, "
              f"base_rate={y.mean():.3f} ({int(y.sum())}/{len(y)} positive issue-days)")

    # feature sanity for the default target
    feats = daily_features(cat, START, n_days, target_mag=4.0, mc=MC)
    print(f"\nFeature frame: {feats.shape[0]} days x {feats.shape[1]-2} features")
    print("  columns:", [c for c in feats.columns if c not in ("day", "date")])
    print(feats.drop(columns=["day", "date"]).describe().loc[["mean", "std", "min", "max"]].T.round(2))


if __name__ == "__main__":
    main()
