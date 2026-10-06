import numpy as np
from sklearn.metrics import roc_auc_score

from eq import stats


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(0)
    y = rng.normal(2.0, 1.0, 400)
    point, lo, hi = stats.bootstrap_ci(lambda a, b: float(np.mean(a)), y, y, n_boot=500)
    assert lo < 2.0 < hi and lo < point < hi


def test_paired_identical_models_give_zero_delta():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0, 0.3, 300), 0, 1)
    d, lo, hi, _ = stats.paired_bootstrap(roc_auc_score, y, p, p, n_boot=300)
    assert d == 0 and lo <= 0 <= hi


def test_paired_detects_a_better_model():
    rng = np.random.default_rng(2)
    y = rng.normal(3, 0.5, 500)
    good, bad = y + rng.normal(0, 0.1, 500), y + rng.normal(0, 0.4, 500)
    d, lo, hi, p_worse = stats.paired_bootstrap(stats.r2, y, good, bad, n_boot=300)
    assert d > 0 and lo > 0 and p_worse == 0


def test_cluster_bootstrap_is_wider_for_correlated_rows():
    """10 copies of each value: row resampling thinks n is 10x larger than it is."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 40)
    y = np.repeat(base, 10)
    cl = np.repeat(np.arange(40), 10)
    m = lambda a, b: float(np.mean(a))  # noqa: E731
    _, lo_r, hi_r = stats.bootstrap_ci(m, y, y, n_boot=400)
    _, lo_c, hi_c = stats.bootstrap_ci(m, y, y, clusters=cl, n_boot=400)
    assert (hi_c - lo_c) > 2 * (hi_r - lo_r)


def test_seed_ci():
    m, lo, hi = stats.seed_ci([0.94, 0.95, 0.96, 0.95, 0.95])
    assert abs(m - 0.95) < 1e-9 and lo < 0.95 < hi
