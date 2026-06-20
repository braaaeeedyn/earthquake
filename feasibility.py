"""
Geomagnetic earthquake-precursor FEASIBILITY HARNESS
=====================================================

Purpose: BEFORE building any CNN / GNN / Transformer, answer one question
empirically: is there a detectable, out-of-sample, controlled-false-alarm
signal in geomagnetic data that precedes earthquakes?

This script runs end-to-end on a synthetic dataset so you can see the
methodology and the output figures immediately. To run it on real data,
replace `load_data()` (the ONLY function you must change) with a loader that
reads your INTERMAGNET station files + USGS/ISC earthquake catalog and returns
the same structures. Everything downstream is data-agnostic.

The harness deliberately uses ONLY simple, interpretable features and a
logistic-regression / gradient-boosting baseline. The logic: if a dumb model
finds no signal under honest validation, three neural nets will not either.
If a dumb model DOES find signal, you have justification to build them.

Outputs (saved to ./out/):
  1. feature_separation.png  - do pre-quake windows differ from controls?
  2. storm_confounder.png     - is the "signal" just global magnetic storms?
  3. roc_oos.png              - out-of-sample ROC + skill vs. always-no baseline
  4. molchanov_diagram.png    - prediction-gain / Molchanov error diagram
  5. feasibility_report.txt   - machine-readable verdict + numbers
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, matthews_corrcoef,
                             precision_recall_curve, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
OUT = "out"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# CONFIG  --  these mirror your stated design (24h x 27d matrices, 7-day label)
# ----------------------------------------------------------------------------
HOURS = 24
DAYS = 27               # 27-day window feeding each prediction
LEAD_DAYS = 7           # predict EQ in the NEXT 7 days
MAG_MIN = 5.0           # qualifying earthquake magnitude
RADIUS_KM = 200.0       # qualifying epicentral distance from station
# Realistic class balance: qualifying weeks are RARE. This is the whole game.
BASE_RATE = 0.06        # ~6% of windows are "pre-quake". Tune to your catalog.

# Synthetic-only knobs: how strong is the planted precursor, 0 = none.
# Set TRUE_EFFECT = 0.0 to confirm the harness correctly reports "no signal".
TRUE_EFFECT = 0.35      # effect size of planted ULF-power anomaly (in sigma)
STORM_CONFOUND = 0.8    # how much global storms leak into features (confounder)


# ----------------------------------------------------------------------------
# DATA LOADING  --  *** REPLACE THIS FUNCTION FOR REAL DATA ***
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    X: np.ndarray          # (N, n_features) engineered daily/window features
    y: np.ndarray          # (N,) 1 = EQ in next 7d, 0 = not
    feat_names: list        # length n_features
    times: np.ndarray      # (N,) window end-time as np.datetime64 (for time split)
    kp: np.ndarray         # (N,) global geomagnetic index (Kp/Dst proxy) per window
    station: np.ndarray    # (N,) station id per window


def load_data():
    """REAL-DATA loader (this is the only function changed from the shipped harness).

    Reuses the project's existing pipeline pieces:
      - eq.data.real.build_region_components -> cached INTERMAGNET H/Z/F minute series
      - eq.features._hour_matrix             -> the same 24-hour reshaping the pipeline uses
      - eq.graph.haversine_km                -> station<->epicentre distance for labeling
      - scripts/storm_experiment.fetch_kp_daily -> local GFZ planetary Kp index (global)

    For each station it builds the 24x27 hourly matrices, computes the SIMPLE window
    features (no deep nets), labels by M>=MAG_MIN within RADIUS_KM and the next LEAD_DAYS,
    attaches per-window mean Kp, and drops windows with insufficient geomagnetic coverage
    (no interpolation, no label imputation). Rows are NOT shuffled.
    """
    import sys
    import warnings
    from pathlib import Path

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from eq.data.real import FULL_START, build_region_components
    from eq.features import _hour_matrix
    from eq.graph import haversine_km
    from storm_experiment import fetch_kp_daily

    # Region/feature selection via env vars (diagnostic controls for the geographic /
    # station-identity confound). Defaults reproduce the shipped pooled run.
    REGIONS = [r.strip() for r in os.environ.get("EQ_REGIONS", "california,japan").split(",")
               if r.strip()]
    DROP = {f.strip() for f in os.environ.get("EQ_DROP_FEATS", "").split(",") if f.strip()}
    EPS = 1e-9
    MIN_COVERAGE = 0.70                 # keep windows with >= 70% finite hourly cells
    base_ts = pd.Timestamp(FULL_START)
    base_dt = np.datetime64(FULL_START)

    feat_names = ["H_mean", "H_var", "Z_mean", "Z_var",
                  "ZH_polarization", "ULF_power", "var_ratio_7v20"]

    def hourly(series, n_days):
        """Minute series -> (n_days, 24) hourly means (NaN where missing)."""
        rows = _hour_matrix(series, base_ts, n_days, 60)        # (n_days*24, 60)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hr = np.nanmean(rows, axis=1)
        return hr.reshape(n_days, 24)

    Xr, yr, tr, kpr, str_ = [], [], [], [], []
    code_to_id, n_total, n_drop_cov, n_drop_nan = {}, 0, 0, 0

    for region in REGIONS:
        comps, catalog, n_days, stations = build_region_components(region, min_magnitude=4.5)
        kp_daily = fetch_kp_daily(FULL_START, n_days)           # (n_days,) global Kp
        cat = catalog[catalog["mag"] >= MAG_MIN]

        for st in stations:
            code_to_id.setdefault(st.code, len(code_to_id))
            sid = code_to_id[st.code]
            Hh, Zh = hourly(comps[st.code]["H"], n_days), hourly(comps[st.code]["Z"], n_days)
            Fh = np.sqrt(Hh ** 2 + Zh ** 2)

            # qualifying-event day indices within RADIUS_KM of THIS station
            qdays = np.zeros(n_days, dtype=bool)
            for _, ev in cat.iterrows():
                if haversine_km(ev["lat"], ev["lon"], st.lat, st.lon) <= RADIUS_KM:
                    d = (pd.Timestamp(ev["time"]).normalize() - base_ts).days
                    if 0 <= d < n_days:
                        qdays[d] = True

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for t in range(DAYS - 1, n_days - LEAD_DAYS):     # window end-day, label observable
                    n_total += 1
                    Hw, Zw, Fw = Hh[t - 26:t + 1], Zh[t - 26:t + 1], Fh[t - 26:t + 1]
                    if np.isfinite(Hw).mean() < MIN_COVERAGE:
                        n_drop_cov += 1
                        continue
                    Hf, Zf, Ff = Hw.ravel(), Zw.ravel(), Fw.ravel()
                    dH, dZ, dF = np.diff(Hf), np.diff(Zf), np.diff(Ff)
                    last7 = np.diff(Fh[t - 6:t + 1].ravel())
                    prior20 = np.diff(Fh[t - 26:t - 6].ravel())
                    feats = [
                        np.nanmean(Hf), np.nanvar(Hf), np.nanmean(Zf), np.nanvar(Zf),
                        np.nanvar(dZ) / (np.nanvar(dH) + EPS),        # ZH_polarization (ULF power ratio)
                        np.nanvar(dF),                                # ULF_power proxy
                        np.nanvar(last7) / (np.nanvar(prior20) + EPS),  # var_ratio_7v20
                    ]
                    if not np.all(np.isfinite(feats)):
                        n_drop_nan += 1
                        continue
                    label = int(qdays[t + 1:t + 1 + LEAD_DAYS].any())
                    Xr.append(feats); yr.append(label)
                    tr.append(base_dt + np.timedelta64(int(t), "D"))
                    kpr.append(float(np.nanmean(kp_daily[t - 26:t + 1])))
                    str_.append(sid)

    keep = [i for i, nm in enumerate(feat_names) if nm not in DROP]
    feat_names = [feat_names[i] for i in keep]
    X = np.asarray(Xr, dtype=float)[:, keep] if Xr else np.empty((0, len(keep)))
    y = np.asarray(yr, dtype=int)
    times = np.asarray(tr, dtype="datetime64[D]")
    kp = np.nan_to_num(np.asarray(kpr, dtype=float), nan=float(np.nanmean(kpr)))
    station = np.asarray(str_, dtype=int)

    print(f"[load_data] windows considered={n_total}  "
          f"dropped(coverage<{MIN_COVERAGE:.0%})={n_drop_cov}  dropped(nan feats)={n_drop_nan}  "
          f"kept={len(y)}  stations={code_to_id}")
    return Dataset(X=X, y=y, feat_names=feat_names, times=times, kp=kp, station=station)


# ----------------------------------------------------------------------------
# TEST 1 — in-sample feature separation (necessary, not sufficient)
# ----------------------------------------------------------------------------
def test_feature_separation(ds: Dataset):
    rows = []
    for j, name in enumerate(ds.feat_names):
        a = ds.X[ds.y == 1, j]
        b = ds.X[ds.y == 0, j]
        # Mann-Whitney U: nonparametric, robust to non-normality
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Cliff's delta effect size from U
        delta = 2 * u / (len(a) * len(b)) - 1
        rows.append((name, a.mean() - b.mean(), delta, p))
    df = pd.DataFrame(rows, columns=["feature", "mean_diff", "cliffs_delta", "p"])
    df = df.sort_values("p")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1D9E75" if p < 0.05 else "#888780" for p in df["p"]]
    ax.barh(df["feature"], df["cliffs_delta"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Cliff's delta (pre-quake vs control)   |   green = p<0.05")
    ax.set_title("Test 1: Do features separate pre-quake windows from controls?")
    fig.tight_layout()
    fig.savefig(f"{OUT}/feature_separation.png", dpi=130)
    plt.close(fig)
    return df


# ----------------------------------------------------------------------------
# TEST 2 — storm confounder check (the killer of naive studies)
# ----------------------------------------------------------------------------
def test_storm_confounder(ds: Dataset):
    """
    Is the 'precursor' just global magnetic storms? Two checks:
      (a) correlation of each feature with the global Kp/Dst index
      (b) does the EQ signal SURVIVE after regressing out Kp from features?
    If features are mostly storm-driven and the signal vanishes after
    de-storming, the precursor is an artifact.
    """
    storm_z = (ds.kp - ds.kp.mean()) / ds.kp.std()

    # (a) feature-storm correlations
    corrs = [np.corrcoef(ds.X[:, j], storm_z)[0, 1] for j in range(len(ds.feat_names))]

    # (b) de-storm: residualize each feature on Kp, re-test EQ separation
    Xres = ds.X.copy()
    for j in range(Xres.shape[1]):
        beta = np.polyfit(storm_z, ds.X[:, j], 1)
        Xres[:, j] = ds.X[:, j] - (beta[0] * storm_z + beta[1])

    raw_auc, res_auc = [], []
    for j in range(Xres.shape[1]):
        raw_auc.append(roc_auc_score(ds.y, ds.X[:, j]))
        res_auc.append(roc_auc_score(ds.y, Xres[:, j]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.barh(ds.feat_names, corrs, color="#D85A30")
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("corr(feature, global Kp index)")
    ax1.set_title("(a) How storm-driven is each feature?")

    width = 0.4
    yidx = np.arange(len(ds.feat_names))
    ax2.barh(yidx - width/2, [abs(a-0.5) for a in raw_auc], width,
             label="raw", color="#888780")
    ax2.barh(yidx + width/2, [abs(a-0.5) for a in res_auc], width,
             label="after de-storming", color="#1D9E75")
    ax2.set_yticks(yidx); ax2.set_yticklabels(ds.feat_names)
    ax2.set_xlabel("EQ-separating power |AUC-0.5|")
    ax2.set_title("(b) Does signal survive removing storms?")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/storm_confounder.png", dpi=130)
    plt.close(fig)

    survives = np.mean([abs(r-0.5) for r in res_auc]) > 0.5 * np.mean([abs(r-0.5) for r in raw_auc])
    return dict(feature_storm_corr=dict(zip(ds.feat_names, corrs)),
                raw_auc=raw_auc, residual_auc=res_auc, signal_survives_destorm=bool(survives))


# ----------------------------------------------------------------------------
# TEST 3 — honest out-of-sample evaluation (time-ordered split)
# ----------------------------------------------------------------------------
def test_out_of_sample(ds: Dataset):
    order = np.argsort(ds.times)
    cut = int(0.7 * len(order))
    tr, te = order[:cut], order[cut:]

    scaler = StandardScaler().fit(ds.X[tr])
    Xtr, Xte = scaler.transform(ds.X[tr]), scaler.transform(ds.X[te])

    # Two baselines: linear and gradient-boosted (your real comparison point).
    results = {}
    for name, clf in [("logreg", LogisticRegression(max_iter=2000,
                                                     class_weight="balanced")),
                      ("gbdt", GradientBoostingClassifier(random_state=0))]:
        clf.fit(Xtr, ds.y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(ds.y[te], p)

        # Pick threshold maximizing MCC on TRAIN, apply to TEST (no peeking).
        ptr = clf.predict_proba(Xtr)[:, 1]
        thr_grid = np.quantile(ptr, np.linspace(0.5, 0.99, 50))
        best_thr = max(thr_grid, key=lambda t: matthews_corrcoef(ds.y[tr], ptr >= t))
        yhat = (p >= best_thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(ds.y[te], yhat).ravel()
        recall = tp / (tp + fn) if (tp+fn) else 0.0
        far = fp / (fp + tn) if (fp+tn) else 0.0       # false-alarm rate
        precision = tp / (tp + fp) if (tp+fp) else 0.0
        mcc = matthews_corrcoef(ds.y[te], yhat)
        acc = (tp + tn) / len(yhat)
        results[name] = dict(auc=auc, recall=recall, far=far, precision=precision,
                             mcc=mcc, accuracy=acc, thr=float(best_thr),
                             probs=p, ytrue=ds.y[te])

    # Always-no baseline for context
    always_no_acc = 1 - ds.y[te].mean()

    # ROC plot
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in results:
        fpr, tpr, _ = roc_curve(results[name]["ytrue"], results[name]["probs"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random (AUC=0.5)")
    ax.set_xlabel("False-alarm rate"); ax.set_ylabel("Recall (hit rate)")
    ax.set_title("Test 3: Out-of-sample ROC (time-ordered split)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/roc_oos.png", dpi=130)
    plt.close(fig)

    return results, always_no_acc


# ----------------------------------------------------------------------------
# TEST 4 — Molchanov-style prediction-gain diagram
# ----------------------------------------------------------------------------
def test_molchanov(results):
    """
    Plot alarm-time fraction vs missed-event fraction. The diagonal is random
    guessing; a useful predictor sits well below it. This is the standard way
    seismology evaluates precursor claims, and what reviewers will expect.
    """
    r = results["gbdt"]
    fpr, tpr, _ = roc_curve(r["ytrue"], r["probs"])
    tau = fpr            # fraction of time under alarm (proxy)
    nu = 1 - tpr          # fraction of missed events

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(tau, nu, color="#534AB7", lw=2, label="model")
    ax.plot([0, 1], [1, 0], "k--", lw=0.8, label="random")
    ax.set_xlabel("Fraction of time under alarm (tau)")
    ax.set_ylabel("Fraction of missed earthquakes (nu)")
    ax.set_title("Test 4: Molchanov prediction error diagram")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/molchanov_diagram.png", dpi=130)
    plt.close(fig)

    # Prediction gain at the operating point: how much better than random.
    op = np.argmin(tau + nu)
    gain = (1 - nu[op]) / max(tau[op], 1e-9)
    return dict(operating_tau=float(tau[op]), operating_nu=float(nu[op]),
                prediction_gain=float(gain))


# ----------------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------------
def verdict(sep_df, storm, oos, always_no, molch):
    lines = []
    add = lines.append
    add("=" * 70)
    add("GEOMAGNETIC EARTHQUAKE-PRECURSOR FEASIBILITY REPORT")
    add("=" * 70)
    add(f"Config: M>={MAG_MIN}, R<={RADIUS_KM}km, lead={LEAD_DAYS}d, "
        f"base_rate={BASE_RATE:.1%}")
    add(f"(Synthetic ground-truth effect size = {TRUE_EFFECT} sigma; "
        f"set 0 to test the null.)")
    add("")
    add("TEST 1  Feature separation (in-sample, necessary not sufficient)")
    sig = sep_df[sep_df.p < 0.05]
    add(f"  {len(sig)}/{len(sep_df)} features separate at p<0.05.")
    for _, row in sep_df.iterrows():
        add(f"    {row.feature:18s} delta={row.cliffs_delta:+.3f}  p={row.p:.2e}")
    add("")
    add("TEST 2  Storm confounder (the make-or-break check)")
    add(f"  Signal survives de-storming: {storm['signal_survives_destorm']}")
    add("  -> If False, the 'precursor' is largely global magnetic storms,")
    add("     not earthquake physics. This is the #1 false-positive trap.")
    add("")
    add("TEST 3  Out-of-sample skill (time-ordered 70/30 split)")
    add(f"  Always-'no' accuracy baseline: {always_no:.3f}  <-- beating THIS")
    add("     with accuracy is trivial and meaningless.")
    for name, r in oos.items():
        add(f"  {name:7s} AUC={r['auc']:.3f}  recall={r['recall']:.2f}  "
            f"FAR={r['far']:.2f}  precision={r['precision']:.2f}  "
            f"MCC={r['mcc']:+.3f}")
    add("")
    add("TEST 4  Molchanov prediction gain")
    add(f"  operating tau={molch['operating_tau']:.2f}  "
        f"nu={molch['operating_nu']:.2f}  gain={molch['prediction_gain']:.2f}x")
    add("  (gain=1 is random; >1 means better than chance at that operating pt)")
    add("")
    add("-" * 70)
    best_auc = max(r["auc"] for r in oos.values())
    best_mcc = max(r["mcc"] for r in oos.values())
    go = (best_auc > 0.60 and best_mcc > 0.1 and storm["signal_survives_destorm"])
    add("VERDICT")
    if go:
        add("  GO (conditional): out-of-sample AUC and MCC exceed noise AND the")
        add("  signal survives the storm confounder. There is enough real signal")
        add("  to justify building the CNN/GNN/Transformer, which may extract more.")
        add(f"  Best OOS AUC={best_auc:.3f}, MCC={best_mcc:+.3f}.")
    else:
        add("  NO-GO (so far): with simple features under honest validation, the")
        add("  signal is absent, vanishes after de-storming, or has no skill above")
        add("  the trivial baseline. A deep net is unlikely to manufacture signal")
        add("  that simple separable features can't even hint at. Revisit the")
        add("  labeling rules, station selection, magnitude/radius, or accept that")
        add("  short-term prediction may not be achievable from this data.")
        add(f"  Best OOS AUC={best_auc:.3f}, MCC={best_mcc:+.3f}.")
    add("-" * 70)

    report = "\n".join(lines)
    with open(f"{OUT}/feasibility_report.txt", "w") as f:
        f.write(report)
    print(report)
    return go


def main():
    ds = load_data()
    print(f"Loaded {len(ds.y)} windows, {ds.y.sum()} positive "
          f"({ds.y.mean():.1%} base rate), {len(ds.feat_names)} features.\n")
    sep = test_feature_separation(ds)
    storm = test_storm_confounder(ds)
    oos, always_no = test_out_of_sample(ds)
    molch = test_molchanov(oos)
    verdict(sep, storm, oos, always_no, molch)
    print(f"\nFigures + report written to ./{OUT}/")


if __name__ == "__main__":
    main()
