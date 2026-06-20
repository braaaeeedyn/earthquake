"""Phases 2-4: honest out-of-sample evaluation of catalog-based forecasting.

Time-ordered split. Compares climatology (the reference), an ETAS-lite recent-rate baseline,
and ML (logistic regression, gradient boosting) on catalog features. Headline metric is the
Brier skill score vs climatology (>0 = real skill); also reports CSEP-style information gain,
ROC-AUC, and threshold metrics (precision/recall/MCC) at a train-selected operating point.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import matthews_corrcoef, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from eq.quakecast import daily_features, daily_targets, load_catalog  # noqa: E402

START, END = "2010-01-01", "2023-12-31"
HORIZON, MC, BURN_IN = 7, 2.5, 365
FEATS = ["cnt_1", "cnt_7", "cnt_30", "cnt_90", "cnt_365", "logmoment_30",
         "days_since_any", "days_since_target", "maxmag_30", "bvalue_365"]


def _metrics(y, p, clim):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    brier = np.mean((p - y) ** 2)
    brier_clim = np.mean((clim - y) ** 2)
    bss = 1 - brier / brier_clim
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll_c = np.sum(y * np.log(clim) + (1 - y) * np.log(1 - clim))
    ig = (ll - ll_c) / (len(y) * np.log(2))           # bits/day vs climatology
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    return brier, bss, ig, auc


def _thresh_metrics(ytr, ptr, yte, pte):
    grid = np.quantile(ptr, np.linspace(0.5, 0.99, 50))
    thr = max(grid, key=lambda t: matthews_corrcoef(ytr, ptr >= t))
    yh = (pte >= thr).astype(int)
    tp = int(((yh == 1) & (yte == 1)).sum()); fp = int(((yh == 1) & (yte == 0)).sum())
    fn = int(((yh == 0) & (yte == 1)).sum()); tn = int(((yh == 0) & (yte == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return rec, prec, far, matthews_corrcoef(yte, yh)


def run(target_mag):
    n_days = (np.datetime64(END) - np.datetime64(START)).astype(int)
    cat = load_catalog(START, END, min_mag=MC)
    y, pos_day = daily_targets(cat, START, n_days, target_mag, HORIZON)
    feats = daily_features(cat, START, n_days, target_mag, MC)
    X = feats[FEATS].to_numpy(dtype=float)

    # ETAS-lite: recent target-event rate -> Poisson prob of >=1 in next HORIZON days.
    tcsum = np.concatenate([[0], np.cumsum(pos_day.astype(int))])
    lam30 = np.array([(tcsum[t] - tcsum[max(t - 30, 0)]) / 30.0 for t in range(n_days)])
    p_etas = 1 - np.exp(-lam30 * HORIZON)

    idx = np.arange(BURN_IN, n_days)                  # drop warm-up
    cut = BURN_IN + int(0.7 * len(idx))
    tr, te = np.arange(BURN_IN, cut), np.arange(cut, n_days)
    clim = y[tr].mean()                               # climatology = train base rate
    clim_te = np.full(len(te), clim)

    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "gbdt": GradientBoostingClassifier(random_state=0),
    }
    preds = {"climatology": clim_te, "etas_lite": p_etas[te]}
    trpreds = {"climatology": np.full(len(tr), clim), "etas_lite": p_etas[tr]}
    for name, clf in models.items():
        clf.fit(Xtr, y[tr])
        preds[name] = clf.predict_proba(Xte)[:, 1]
        trpreds[name] = clf.predict_proba(Xtr)[:, 1]

    print(f"\n=== California  target M>={target_mag}, {HORIZON}d horizon ===")
    print(f"  test base rate={y[te].mean():.3f}  (train={clim:.3f})  "
          f"n_test={len(te)} days, {int(y[te].sum())} positive")
    print(f"  {'model':<12}{'Brier':>8}{'BSS':>8}{'infogain':>10}{'AUC':>7}"
          f"{'recall':>8}{'prec':>7}{'FAR':>7}{'MCC':>8}")
    for name, p in preds.items():
        brier, bss, ig, auc = _metrics(y[te], p, clim_te)
        rec, prec, far, mcc = _thresh_metrics(y[tr], trpreds[name], y[te], p)
        print(f"  {name:<12}{brier:>8.3f}{bss:>+8.3f}{ig:>+10.4f}{auc:>7.3f}"
              f"{rec:>8.2f}{prec:>7.2f}{far:>7.2f}{mcc:>+8.3f}")


def rolling_run(target_mag, init_days=3 * 365, step=90):
    """Rolling-origin (expanding-window) forecast: refit every `step` days on all prior data,
    forecast the next block. Climatology reference is the train-so-far base rate (adaptive),
    so skill is measured above the recent average -> robust to non-stationary seismicity."""
    n_days = (np.datetime64(END) - np.datetime64(START)).astype(int)
    cat = load_catalog(START, END, min_mag=MC)
    y, pos_day = daily_targets(cat, START, n_days, target_mag, HORIZON)
    X = daily_features(cat, START, n_days, target_mag, MC)[FEATS].to_numpy(dtype=float)
    tcsum = np.concatenate([[0], np.cumsum(pos_day.astype(int))])
    lam30 = np.array([(tcsum[t] - tcsum[max(t - 30, 0)]) / 30.0 for t in range(n_days)])
    p_etas_all = 1 - np.exp(-lam30 * HORIZON)

    origin0 = BURN_IN + init_days
    coll = {k: [] for k in ("y", "clim", "etas", "logreg", "gbdt")}
    for o in range(origin0, n_days, step):
        tr = np.arange(BURN_IN, o)
        te = np.arange(o, min(o + step, n_days))
        if not len(te) or y[tr].sum() < 5:
            continue
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        gb = GradientBoostingClassifier(random_state=0).fit(sc.transform(X[tr]), y[tr])
        coll["y"].append(y[te])
        coll["clim"].append(np.full(len(te), y[tr].mean()))
        coll["etas"].append(p_etas_all[te])
        coll["logreg"].append(lr.predict_proba(sc.transform(X[te]))[:, 1])
        coll["gbdt"].append(gb.predict_proba(sc.transform(X[te]))[:, 1])
    Y = np.concatenate(coll["y"]); clim = np.clip(np.concatenate(coll["clim"]), 1e-6, 1 - 1e-6)

    print(f"\n=== ROLLING-ORIGIN  California  target M>={target_mag}, {HORIZON}d "
          f"(refit/{step}d, adaptive climatology) ===")
    print(f"  forecast days={len(Y)}, positive={int(Y.sum())} ({Y.mean():.3f})")
    print(f"  {'model':<12}{'Brier':>8}{'BSS':>8}{'infogain':>10}{'AUC':>7}"
          f"{'recall':>8}{'prec':>7}{'FAR':>7}{'MCC':>8}")
    for name in ("climatology", "etas", "logreg", "gbdt"):
        p = clim if name == "climatology" else np.clip(np.concatenate(coll[name]), 1e-6, 1 - 1e-6)
        brier, bss, ig, auc = _metrics(Y, p, clim)
        yh = (p > clim).astype(int)                   # alarm when prob exceeds recent climatology
        tp = int(((yh == 1) & (Y == 1)).sum()); fp = int(((yh == 1) & (Y == 0)).sum())
        fn = int(((yh == 0) & (Y == 1)).sum()); tn = int(((yh == 0) & (Y == 0)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        far = fp / (fp + tn) if (fp + tn) else 0.0
        mcc = matthews_corrcoef(Y, yh) if len(set(yh)) > 1 else 0.0
        print(f"  {name:<12}{brier:>8.3f}{bss:>+8.3f}{ig:>+10.4f}{auc:>7.3f}"
              f"{rec:>8.2f}{prec:>7.2f}{far:>7.2f}{mcc:>+8.3f}")


def main():
    for tmag in (4.5, 5.0):
        run(tmag)
    for tmag in (4.5, 5.0):
        rolling_run(tmag)


if __name__ == "__main__":
    main()
