"""QuakeOps drift monitor: is the live stream still the kind of data the detector was trained on?

Features (all on det_prep'd 30 s windows -- identical code in training and live -- so they are scale-free
and directly comparable; raw amplitude is not: training noise is normalized, live is raw counts):
  prob      detector P(quake)
  crest     log10(max|x| / rms)            spikiness / glitches
  hf_ratio  share of 1-18 Hz power > 5 Hz  spectral content (cultural noise, a failing sensor)
live_watch.py logs them for one window per station every 30 s -> data/processed/features/<date>.csv.

REFERENCE = the same features on the TRAINING-split noise windows of detection.npz (time-random, per
station), scored by the champion. Noise only: the live stream is almost all noise.
Daily, per station: Evidently DataDriftPreset (normed Wasserstein, threshold 0.1) of yesterday vs the
reference -> insufficient (< 500 rows) / ok (0 drifted features) / watch (1) / drifting (>= 2).
Email when a station is drifting on 2 consecutive days (once per streak). 90-day retention.

  python scripts/drift_check.py --build-reference [--det CKPT] [--out CSV]   # PC (and on promotion)
  python scripts/drift_check.py [--day YYYY-MM-DD]                            # VM, daily (systemd timer)
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from eq import network  # noqa: E402

PROC = ROOT / "data" / "processed"
FEAT_DIR = PROC / "features"
REF = PROC / "drift_reference.csv"
STATUS = PROC / "drift_status.json"
HISTORY = PROC / "drift_history.jsonl"
HTML_DIR = PROC / "drift"
FEATURES = ["prob", "crest", "hf_ratio"]
MIN_ROWS, KEEP_DAYS = 500, 90


def build_reference(det_ckpt=PROC / "detector.pt", out=REF):
    """Training-split noise windows (first 30 s, as evaluated) -> station, prob, crest, hf_ratio."""
    import torch
    from demo_detect import NPTS, load
    from live_watch import load_detector
    from eq.pipeline import det_prep, detect_probs, window_features
    d, _, noise, _, sp, _, _ = load()
    ni = sp["tr"][1]
    w = noise[ni, :NPTS].astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prob = detect_probs(load_detector(device, det_ckpt), w, device)
    crest, hf = window_features(det_prep(w))
    df = pd.DataFrame({"station": np.array(network.CODES)[d["noise_sta"][ni]], "prob": prob, "crest": crest,
                       "hf_ratio": hf})
    df.to_csv(out, index=False)
    print(f"reference: {len(df)} training noise windows, {df.station.nunique()} stations -> {out}")
    return out


def drifted_columns(cur, ref, html=None):
    from evidently import Report
    from evidently.presets import DataDriftPreset
    snap = Report([DataDriftPreset(method="wasserstein", threshold=0.1)]).run(current_data=cur, reference_data=ref)
    if html:
        snap.save_html(str(html))
    out = []
    for m in snap.dict()["metrics"]:
        col = m.get("config", {}).get("column")
        if col and m["value"] >= m["config"]["threshold"]:
            out.append(col)
    return out


def check(day):
    fp = FEAT_DIR / f"{day}.csv"
    cur = pd.read_csv(fp) if fp.exists() else pd.DataFrame(columns=["ts", "station"] + FEATURES)
    ref = pd.read_csv(REF)
    (HTML_DIR / day).mkdir(parents=True, exist_ok=True)
    stations = {}
    for code in network.CODES:
        c, r = cur[cur.station == code][FEATURES], ref[ref.station == code][FEATURES]
        if len(c) < MIN_ROWS or len(r) < 50:
            stations[code] = {"status": "insufficient", "n": int(len(c)), "drifted": []}
            continue
        cols = drifted_columns(c.reset_index(drop=True), r.reset_index(drop=True), HTML_DIR / day / f"{code}.html")
        stations[code] = {"status": ["ok", "watch"][len(cols)] if len(cols) < 2 else "drifting",
                          "n": int(len(c)), "drifted": cols}
    prev = json.loads(STATUS.read_text()) if STATUS.exists() else {"stations": {}}
    streak = {}
    for code, s in stations.items():
        p = prev.get("stations", {}).get(code, {})
        s["streak"] = p.get("streak", 0) + 1 if s["status"] == "drifting" else 0
        if s["status"] == "drifting" and s["streak"] == 2:                  # email once per streak
            streak[code] = s["drifted"]
    out = {"date": day, "reference": "training noise (champion detector)", "method": "Evidently normed Wasserstein > 0.1",
           "stations": stations}
    STATUS.write_text(json.dumps(out, indent=1))
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": day, **{c: s["status"] for c, s in stations.items()}}) + "\n")
    counts = pd.Series([s["status"] for s in stations.values()]).value_counts().to_dict()
    print(f"drift {day}: {counts}")
    if streak:
        import tracking
        tracking.ops_email("QuakeOps: station drift", f"Drifting 2 days running ({day}): "
                           + "; ".join(f"{c} ({', '.join(v)})" for c, v in streak.items())
                           + "\nReports: data/processed/drift/" + day)
    return out


def prune(today):
    cutoff = (today - timedelta(days=KEEP_DAYS)).isoformat()
    for fp in FEAT_DIR.glob("*.csv"):
        if fp.stem < cutoff:
            fp.unlink()
    for dp in HTML_DIR.glob("*"):
        if dp.is_dir() and dp.name < cutoff:
            shutil.rmtree(dp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-reference", action="store_true", dest="build")
    ap.add_argument("--det", default=str(PROC / "detector.pt"))
    ap.add_argument("--out", default=str(REF))
    ap.add_argument("--day", help="UTC date to check (default: yesterday)")
    args = ap.parse_args()
    if args.build:
        build_reference(Path(args.det), Path(args.out))
        return
    today = datetime.now(timezone.utc).date()
    check(args.day or (today - timedelta(days=1)).isoformat())
    prune(today)


if __name__ == "__main__":
    main()
