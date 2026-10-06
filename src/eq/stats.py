"""Confidence intervals for the headline metrics and for model-vs-model comparisons.

Percentile bootstrap over the test set. Detection CLUSTERS by event: every station window of one quake
(and every station's noise window at one random time) shares a cluster id, because resampling single
windows treats correlated windows as independent and makes the interval too narrow.
`paired_bootstrap` scores two models on the SAME resample each time -- the comparison the promotion gate
uses (deep vs baseline, challenger vs champion).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


def _resamples(n, clusters, n_boot, seed):
    """Index arrays for each bootstrap resample (rows, or whole clusters of rows)."""
    rng = np.random.default_rng(seed)
    if clusters is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, n)
        return
    _, inv = np.unique(clusters, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    k = len(members)
    for _ in range(n_boot):
        yield np.concatenate([members[j] for j in rng.integers(0, k, k)])


def _score(metric, y, p, idx):
    try:
        v = metric(y[idx], p[idx])
    except ValueError:                                   # e.g. AUC on a one-class resample
        return np.nan
    return v if np.isfinite(v) else np.nan


def bootstrap_ci(metric, y, p, clusters=None, n_boot=2000, alpha=0.05, seed=0):
    """(point, lo, hi) of metric(y, p) with a (cluster) percentile bootstrap CI."""
    y, p = np.asarray(y), np.asarray(p)
    vals = np.array([_score(metric, y, p, i) for i in _resamples(len(y), clusters, n_boot, seed)])
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, p)), float(lo), float(hi)


def paired_bootstrap(metric, y, p_a, p_b, clusters=None, n_boot=2000, alpha=0.05, seed=0):
    """Delta = metric(a) - metric(b) on the same rows: (delta, lo, hi, P(delta < 0))."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    d = []
    for i in _resamples(len(y), clusters, n_boot, seed):
        d.append(_score(metric, y, p_a, i) - _score(metric, y, p_b, i))
    d = np.asarray(d)
    lo, hi = np.nanquantile(d, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, p_a) - metric(y, p_b)), float(lo), float(hi), float(np.nanmean(d < 0))


def seed_ci(values, alpha=0.05):
    """(mean, lo, hi): t-interval of a metric across training seeds (seed-to-seed variance)."""
    v = np.asarray(values, float)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = float(_st.t.ppf(1 - alpha / 2, len(v) - 1) * se)
    return float(m), float(m - h), float(m + h)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))
