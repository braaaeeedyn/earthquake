"""Classification metrics (MVP §5).

Recall on the positive (earthquake) class matters most under heavy imbalance, so we
always report precision/recall/F1 — not accuracy alone. ``best_threshold`` is selected
on validation, never on test, to keep the held-out evaluation honest.
"""
from __future__ import annotations

import numpy as np


def fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    """F-beta: beta>1 weights recall over precision (recall matters most here)."""
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": fbeta(precision, recall, 1.0),
        "f2": fbeta(precision, recall, 2.0),
        "accuracy": accuracy,
        "threshold": threshold,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def auc_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """Threshold-free metrics. PR-AUC's no-skill reference is the base rate, not 0.5, so
    ``pr_auc_over_base`` > 1 indicates skill above chance for the positive class."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    base = float(y.mean()) if len(y) else 0.0
    both = 0 < y.sum() < len(y)
    roc = float(roc_auc_score(y, p)) if both else float("nan")
    pr = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "base_rate": base,
        "pr_auc_over_base": pr / base if base > 0 else float("nan"),
        "n": int(len(y)),
        "positives": int(y.sum()),
    }


def best_threshold(y_true: np.ndarray, prob: np.ndarray, beta: float = 1.0) -> float:
    """Threshold maximizing F-beta over the observed probabilities (tie -> lower thresh).

    beta>1 favors recall, the operating point we care about most.
    """
    cands = np.unique(np.concatenate([[0.0], np.asarray(prob), [1.0]]))
    best_t, best_score = 0.5, -1.0
    for t in cands:
        m = binary_metrics(y_true, prob, float(t))
        score = fbeta(m["precision"], m["recall"], beta)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t
