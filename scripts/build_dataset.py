"""Materialize a labeled dataset to data/processed/dataset.npz.

Currently uses the synthetic fixture; Phase 3 swaps in real INTERMAGNET + USGS data
behind the same ``build_dataset`` interface.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eq.config import Config          # noqa: E402
from eq.pipeline import build_dataset  # noqa: E402


def main() -> None:
    cfg = Config()
    data = build_dataset(cfg, seed=0)
    x_tr, y_tr, a_tr = data["train"]
    x_va, y_va, a_va = data["val"]
    x_te, y_te, a_te = data["test"]

    out = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        Xtr=x_tr, ytr=y_tr, atr=a_tr,
        Xva=x_va, yva=y_va, ava=a_va,
        Xte=x_te, yte=y_te, ate=a_te,
    )
    print(f"Wrote {out}")
    print(f"  train: X={x_tr.shape} pos_rate={y_tr.mean():.3f}")
    print(f"  val  : X={x_va.shape} pos_rate={y_va.mean():.3f}")
    print(f"  test : X={x_te.shape} pos_rate={y_te.mean():.3f}")


if __name__ == "__main__":
    main()
