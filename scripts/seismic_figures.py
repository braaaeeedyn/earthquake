"""Consolidated results figure for the seismic pivot: deep CNN+GNN+Transformer vs classic baselines.

Numbers are the final, verified runs (California SCEDC network):
  detection  : scripts/seismic_train.py        (1000 ev / 1000 noise, single-station)
  magnitude  : scripts/seismic_train_multi.py --hybrid   (794 events, multi-station)
  EEW        : scripts/seismic_eew_ensemble.py --k 5     (794 events, multi-station)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
APP = Path(__file__).resolve().parents[1] / "app" / "public"

DEEP, BASE = "#534AB7", "#888780"


def main():
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))

    # 1. Detection AUC
    ax[0].bar(["Deep\n(CNN+Tr)", "STA/LTA"], [0.988, 0.625], color=[DEEP, BASE])
    ax[0].axhline(0.5, ls="--", c="k", lw=0.8); ax[0].set_ylim(0, 1)
    ax[0].set_title("Detection (AUC)"); ax[0].set_ylabel("ROC-AUC")
    for i, v in enumerate([0.988, 0.625]):
        ax[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # 2. Magnitude R2 (hybrid) + GNN ablation
    ax[1].bar(["Deep\nhybrid", "amp+dist\nbaseline", "Deep\n1-station"],
              [0.771, 0.690, 0.253], color=[DEEP, BASE, "#B7AED0"])
    ax[1].set_ylim(0, 1); ax[1].set_title("Magnitude (R²)"); ax[1].set_ylabel("R²")
    for i, v in enumerate([0.771, 0.690, 0.253]):
        ax[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # 3. EEW regression R2
    ax[2].bar(["Deep\nensemble", "early-amp\n+dist"], [0.728, 0.720], color=[DEEP, BASE])
    ax[2].set_ylim(0, 1); ax[2].set_title("EEW: PGV prediction (R²)"); ax[2].set_ylabel("R²")
    for i, v in enumerate([0.728, 0.720]):
        ax[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # 4. EEW alert: recall (the operational metric)
    ax[3].bar(["Deep\nensemble", "baseline"], [0.65, 0.55], color=[DEEP, BASE])
    ax[3].set_ylim(0, 1); ax[3].set_title("EEW: strong-shaking alert (recall)")
    ax[3].set_ylabel("recall")
    for i, v in enumerate([0.65, 0.55]):
        ax[3].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    fig.suptitle("Seismic CNN+GNN+Transformer vs. classic baselines  (California SCEDC network)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for d in (OUT, APP):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "seismic_results.png", dpi=140)
    print(f"wrote {OUT/'seismic_results.png'} and {APP/'seismic_results.png'}")


if __name__ == "__main__":
    main()
