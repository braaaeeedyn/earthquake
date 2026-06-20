"""Repeated-seed evaluation with confidence intervals (de-risking small-sample results).

A single train/eval run on a small test set is noisy. ``repeated_evaluate`` retrains from
scratch across several seeds and reports mean ± 95% CI per metric, so we can see how much
of a result is signal vs. seed luck. Thresholds are still chosen on validation only.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .metrics import auc_metrics, best_threshold, binary_metrics
from .train import predict_proba, train_model

_KEYS = ("precision", "recall", "f1", "f2", "accuracy")
_AUC_KEYS = ("roc_auc", "pr_auc", "pr_auc_over_base")


def evaluate_model(model, loaders, *, beta: float = 2.0, pos_weight=None, epochs: int = 80) -> dict:
    """Train one model and evaluate on test using a validation-selected threshold."""
    train_model(model, loaders["train"], epochs=epochs, lr=1e-3, pos_weight=pos_weight,
                weight_decay=1e-4, val_loader=loaders["val"], select_beta=beta)
    yv, pv = predict_proba(model, loaders["val"])
    t = best_threshold(yv, pv, beta=beta)
    yt, pt = predict_proba(model, loaders["test"])
    return binary_metrics(yt, pt, t)


def repeated_evaluate(make_model, loaders, *, seeds=(0, 1, 2, 3, 4), beta: float = 2.0,
                      pos_weight=None, epochs: int = 80) -> dict:
    """Retrain across seeds; return per-metric mean/std/95% CI plus the raw runs."""
    runs = []
    for s in seeds:
        torch.manual_seed(s)
        runs.append(evaluate_model(make_model(), loaders, beta=beta,
                                   pos_weight=pos_weight, epochs=epochs))
    n = len(runs)
    agg: dict = {"n_test": runs[0]["n"], "positives": runs[0]["positives"], "seeds": n, "runs": runs}
    for k in _KEYS:
        vals = np.array([r[k] for r in runs], dtype=float)
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        agg[k] = {
            "mean": float(vals.mean()),
            "std": std,
            "ci": 1.96 * std / math.sqrt(n) if n > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return agg


def repeated_auc(make_model, loaders, *, seeds=(0, 1, 2, 3, 4), pos_weight=None,
                 epochs: int = 40) -> dict:
    """Retrain across seeds; report threshold-free ROC-AUC / PR-AUC mean ± 95% CI.

    AUCs need no threshold. Also returns the seed-averaged test probabilities so a single
    pooled PR curve can be drawn per grid cell.
    """
    per_seed, probs = [], []
    y_test = None
    for s in seeds:
        torch.manual_seed(s)
        model = make_model()
        train_model(model, loaders["train"], epochs=epochs, lr=1e-3, pos_weight=pos_weight,
                    weight_decay=1e-4, val_loader=loaders["val"], select_beta=2.0)
        yt, pt = predict_proba(model, loaders["test"])
        y_test = yt
        probs.append(pt)
        per_seed.append(auc_metrics(yt, pt))

    n = len(per_seed)
    mean_prob = np.mean(np.stack(probs), axis=0)
    pooled = auc_metrics(y_test, mean_prob)
    agg: dict = {
        "n_test": int(len(y_test)), "positives": int(y_test.sum()),
        "base_rate": pooled["base_rate"], "seeds": n,
        "y_test": y_test, "mean_prob": mean_prob,
    }
    for k in _AUC_KEYS:
        vals = np.array([r[k] for r in per_seed], dtype=float)
        vals = vals[np.isfinite(vals)]
        m = len(vals)
        std = float(vals.std(ddof=1)) if m > 1 else 0.0
        agg[k] = {
            "mean": float(vals.mean()) if m else float("nan"),
            "ci": 1.96 * std / math.sqrt(m) if m > 1 else 0.0,
        }
    return agg


def format_row(name: str, agg: dict) -> str:
    def cell(k):
        a = agg[k]
        return f"{a['mean']:.3f}±{a['ci']:.3f}"
    return (f"  {name:<12} R={cell('recall')}  P={cell('precision')}  "
            f"F1={cell('f1')}  acc={cell('accuracy')}")
