"""QuakeOps monthly retrain + promotion gate (gaps #4, #7, #11, #12, #13, #18).

Stages (each resumable: data/processed/retrain/<month>/state.json records what finished):
  data     build_dataset.py --append through the end of <month> (check -> select -> fetch -> assemble)
  train    demo_detect.py / demo_magnitude.py --retrain into the month dir, each --compare'd PAIRED with the
           champion on the challenger's chronological test split (neither model has seen it); quick-check fit,
           travel-time table and drift reference alongside; logged to MLflow + registered @challenger
  replay   the acceptance test: the 10 held-out replay days with (champion det, champion mag),
           (challenger det, champion mag) and (champion det, challenger mag) + event-centric sizing
  gate     rules G1-G6 per model (HOW_IT_WORKS.md section 12), pytest + selftest -> gate.json + MLflow run
  promote  passing models -> @champion, installed at the PC's live paths, promotions.jsonl, seismic.json, email

  python scripts/retrain.py --month 2026-09                 # all stages
  python scripts/retrain.py --month 2026-09 --stage gate    # one stage (re-runs it)
  python scripts/retrain.py --month 2026-09 --dry-run       # everything except moving aliases / installing
  python scripts/retrain.py rollback --model detector --to 1

Detect and size are decided independently. The calibrated pipeline_config.json is never changed here: a
detector that needs a new trigger threshold fails G3 and is calibrated by hand (replay_archive.py calibrate).
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tracking  # noqa: E402

PROC = ROOT / "data" / "processed"
V2 = PROC / "v2"
REPLAY = V2 / "replay"
PROMOTIONS = PROC / "promotions.jsonl"
CHAMP = {"detector": PROC / "detector.pt", "magnitude": PROC / "magnitude_ensemble.pt"}
# held-out replay days (never used for calibration) -- the acceptance test
TEST_START, TEST_END = "2026-10-02,2026-08-18", "2026-10-05,2026-08-25"
STAGES = ["data", "train", "replay", "gate", "promote"]
PY = sys.executable


def mdir(month):
    d = PROC / "retrain" / month
    d.mkdir(parents=True, exist_ok=True)
    return d


def sh(*args):
    print("$", " ".join(str(a) for a in args), flush=True)
    subprocess.run([PY, *map(str, args)], cwd=ROOT, check=True)


def month_end(month):
    return (pd.Period(month, "M") + 1).start_time.strftime("%Y-%m-%d")


def jload(fp):
    return json.loads(Path(fp).read_text())


# ---------------------------------------------------------------- stages

def stage_data(month, d):
    end = month_end(month)
    if pd.Timestamp(jload(V2 / "dataset_meta.json")["end"]) >= pd.Timestamp(end):
        print(f"data: dataset already runs to {end}")
        return
    shutil.copyfile(V2 / "tt_correction.json", d / "tt_correction_prev.json")   # restored if size is rejected
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sh("scripts/build_dataset.py", "--stage", "check")
    sh("scripts/build_dataset.py", "--stage", "select", "--append", "--end", end, "--catalog-end", today)
    sh("scripts/build_dataset.py", "--stage", "fetch", "--catalog-end", today)
    sh("scripts/build_dataset.py", "--stage", "assemble", "--end", end, "--catalog-end", today)


def stage_train(month, d, args):
    sh("scripts/demo_detect.py", "--retrain", "--seeds", args.seeds, "--out", d / "detector.pt",
       "--compare", CHAMP["detector"])
    sh("scripts/demo_magnitude.py", "--retrain", "--seeds", args.seeds, "--out", d / "magnitude_ensemble.pt",
       "--compare", CHAMP["magnitude"])
    sh("scripts/fit_early_magnitude.py", "--out", d / "early_mag.json")
    sh("scripts/fit_early_magnitude.py", "--T", "2", "--out", d / "early_mag_T2.json")    # fast alert-speed profile
    shutil.copyfile(V2 / "tt_correction.json", d / "tt_correction.json")
    sh("scripts/drift_check.py", "--build-reference", "--det", d / "detector.pt", "--out", d / "drift_reference.csv")
    if tracking.enabled():
        for name, sidecars in (("detector", ["drift_reference.csv"]),
                               ("magnitude", ["early_mag.json", "early_mag_T2.json", "tt_correction.json"])):
            ckpt = d / tracking.BUNDLES[name][0]
            rid = jload(ckpt.with_suffix(".json"))["mlflow_run_id"]
            tracking.log_files([d / s for s in sidecars], run_id=rid)
            v = tracking.register(name, rid, ckpt, {"month": month})
            tracking.set_alias(name, "challenger", v)
            print(f"{name}: registered v{v} @challenger")


def stage_replay(month, d):
    det_c, mag_c, early_c = d / "detector.pt", d / "magnitude_ensemble.pt", d / "early_mag.json"
    rng = ["--start", TEST_START, "--end", TEST_END]
    sh("scripts/replay_archive.py", "scan", *rng)                               # champion cache (usually complete)
    sh("scripts/replay_archive.py", "scan", "--det", det_c, *rng)
    sh("scripts/replay_archive.py", "run", *rng, "--tag", f"{month}_champion")
    sh("scripts/replay_archive.py", "run", "--det", det_c, *rng, "--tag", f"{month}_detector")
    sh("scripts/replay_archive.py", "run", "--mag", mag_c, "--early", early_c, *rng, "--tag", f"{month}_magnitude")
    sh("scripts/replay_archive.py", "events", "--tag", f"{month}_champion")
    sh("scripts/replay_archive.py", "events", "--mag", mag_c, "--tag", f"{month}_magnitude")


# ---------------------------------------------------------------- gate

def rule(name, ok, detail):
    return {"rule": name, "result": "SKIPPED" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail}


def replay_rules(sc, base, pushed_mag_err):
    """G3 shared part: precision vs chance, no false pushes (provisional or final), M3 recall >= champion."""
    early_false = (sc.get("early_push") or 0) - (sc.get("early_push_real_M2.5+") or 0)
    return [
        rule("G3 confirmed precision >= 0.85 and >= chance + 0.5",
             sc["confirmed_precision"] is not None and sc["confirmed_precision"] >= 0.85
             and sc["confirmed_precision"] >= (sc["confirmed_chance"] or 0) + 0.5,
             f"{sc['confirmed_precision']} (chance {sc['confirmed_chance']}, n={sc['confirmed']})"),
        rule("G3 no false pushes", (sc["push_precision"] in (None, 1.0)) and early_false == 0,
             f"final {sc['push']} (precision {sc['push_precision']}), provisional false {early_false}"),
        rule("G3 in-coverage M3 recall >= champion", (sc["m3_recall"] or 0) >= (base["m3_recall"] or 0),
             f"{sc['m3_recall']} vs {base['m3_recall']}"),
        rule("G3 pushed |mag - catalogue| <= 0.3", None if pushed_mag_err is None else pushed_mag_err <= 0.3,
             "no pushes" if pushed_mag_err is None else f"max {pushed_mag_err:.2f}"),
    ]


def pushed_mag_error(month, tag):
    """Max |final magnitude - catalogue| over the replay's push-eligible events (matched to the catalogue)."""
    import replay_archive as R
    fp = REPLAY / f"events_{month}_{tag}.jsonl"
    recs = [json.loads(x) for x in fp.read_text().splitlines() if x] if fp.exists() else []
    cat = R.catalog()
    errs = [abs(r["mag"] - m["mag"]) for r in recs if r.get("push_eligible")
            for m in [R.match(cat, r["origin"], r["lat"], r["lon"])] if m]
    return max(errs) if errs else None


def lineage_rules(new, old_ck, new_ck):
    first_test_day = min(pd.Timestamp(x).timestamp() for x in TEST_START.split(","))
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    end_new, end_old = new_ck["dataset"].get("end"), old_ck["dataset"].get("end")
    return [
        rule("G5 clean git tree", not dirty, "clean" if not dirty else dirty.replace("\n", "; ")[:200]),
        rule("G5 lineage recorded", bool(new_ck.get("git") and new_ck["dataset"].get("version") and new_ck.get("seeds")),
             f"git {new_ck.get('git')}, data {new_ck['dataset'].get('version')}, seeds {new_ck.get('seeds')}"),
        rule("G5 replay test days inside the challenger's test period", new["test_start"] <= first_test_day,
             f"test starts {datetime.fromtimestamp(new['test_start'], timezone.utc):%Y-%m-%d}"),
    ], end_new and end_old and pd.Timestamp(end_new) > pd.Timestamp(end_old), f"data end {end_old} -> {end_new}"


def stage_gate(month, d):
    import torch
    load = lambda p: torch.load(p, weights_only=False, map_location="cpu")  # noqa: E731
    tests = subprocess.run([PY, "-m", "pytest", "-q"], cwd=ROOT).returncode == 0 and \
        subprocess.run([PY, "scripts/live_watch.py", "--selftest"], cwd=ROOT, capture_output=True).returncode == 0
    base = jload(REPLAY / f"score_{month}_champion.json")
    ev_base = jload(REPLAY / f"score_events_test_{month}_champion.json")
    out = {"month": month, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    s = jload(d / "detector.json")
    vs = s["vs"]
    sc = jload(REPLAY / f"score_{month}_detector.json")
    lin, newer, why = lineage_rules(s, load(CHAMP["detector"]), load(d / "detector.pt"))
    f_new, f_old = s["fpr_test_noise_at_trigger"], vs["fpr_test_noise_at_trigger_other"]
    l_new, l_old = s["fpr_live_noise_at_trigger"], vs["fpr_live_noise_at_trigger_other"]
    rules = [
        rule("G1 beats STA/LTA (paired dAUC CI low > 0)", s["d_auc_vs_sta_lta"][1] > 0, f"{s['d_auc_vs_sta_lta']}"),
        rule("G2 dAUC >= -0.001 and CI high >= 0", vs["d_auc"][0] >= -0.001 and vs["d_auc"][2] >= 0, f"{vs['d_auc']}"),
        rule("G2 dMCC >= -0.02", vs["d_mcc"][0] >= -0.02, f"{vs['d_mcc']}"),
        # false triggers per window at the LIVE trigger threshold (pipeline_config.json), both models
        rule("G2 test-noise FPR at the live trigger <= champion + 0.005", f_new <= f_old + 0.005,
             f"{f_new:.4f} vs {f_old:.4f} (trigger {s['trigger']})"),
        rule("G2 live-noise FPR at the live trigger <= champion + 0.0025",
             None if l_old is None else l_new <= l_old + 0.0025, f"{l_new} vs {l_old}"),
        *replay_rules(sc, base, pushed_mag_error(month, "detector")),
        rule("G4 pytest + selftest", tests, "green" if tests else "failing"),
        *lin,
        rule("G6 reason to switch (newer data or a significant gain)",
             bool(newer or vs["d_auc"][1] > 0 or vs["d_mcc"][1] > 0), why),
    ]
    out["detector"] = {"rules": rules, "pass": all(r["result"] != "FAIL" for r in rules)}

    s = jload(d / "magnitude_ensemble.json")
    vs = s["vs"]
    sc = jload(REPLAY / f"score_{month}_magnitude.json")
    ev = jload(REPLAY / f"score_events_test_{month}_magnitude.json")
    lin, newer, why = lineage_rules(s, load(CHAMP["magnitude"]), load(d / "magnitude_ensemble.pt"))
    rules = [
        rule("G1 beats amp+dist (paired dR2 CI low > 0)", s["d_r2_vs_baseline"][1] > 0, f"{s['d_r2_vs_baseline']}"),
        rule("G2 dR2 >= -0.01 and CI high >= 0", vs["d_r2"][0] >= -0.01 and vs["d_r2"][2] >= 0, f"{vs['d_r2']}"),
        rule("G2 dMAE <= +0.01", vs["d_mae"][0] <= 0.01, f"{vs['d_mae']}"),
        *[r for r in replay_rules(sc, base, pushed_mag_error(month, "magnitude")) if "recall" not in r["rule"]],
        rule("G3 event-centric MAE <= champion + 0.02", ev["mag_mae"] <= ev_base["mag_mae"] + 0.02,
             f"{ev['mag_mae']} vs {ev_base['mag_mae']} (bias {ev['mag_bias']:+.3f})"),
        rule("G4 pytest + selftest", tests, "green" if tests else "failing"),
        *lin,
        rule("G6 reason to switch (newer data or a significant gain)",
             bool(newer or vs["d_r2"][1] > 0 or vs["d_mae"][2] < 0), why),
    ]
    out["magnitude"] = {"rules": rules, "pass": all(r["result"] != "FAIL" for r in rules)}
    (d / "gate.json").write_text(json.dumps(out, indent=1))
    for name in ("detector", "magnitude"):
        print(f"\n{name}: {'PASS' if out[name]['pass'] else 'REJECT'}")
        for r in out[name]["rules"]:
            print(f"  {r['result']:7s} {r['rule']}  [{r['detail']}]")
    with tracking.run("quakeops-gate", f"gate-{month}", {"month": month}) as rid:
        if rid:
            import mlflow
            mlflow.log_metrics({f"{n}_pass": float(out[n]["pass"]) for n in ("detector", "magnitude")})
            mlflow.log_artifact(str(d / "gate.json"))
            out["gate_run_id"] = rid
            (d / "gate.json").write_text(json.dumps(out, indent=1))


def stage_promote(month, d, dry_run):
    gate = jload(d / "gate.json")
    lines = []
    for name, files in (("detector", ["detector.pt", "drift_reference.csv"]),
                        ("magnitude", ["magnitude_ensemble.pt", "early_mag.json", "early_mag_T2.json", "tt_correction.json"])):
        g = gate[name]
        failed = [r["rule"] for r in g["rules"] if r["result"] == "FAIL"]
        ver = tracking.alias_version(name, "challenger") if tracking.enabled() else None
        if not g["pass"]:
            lines.append(f"{name}: REJECTED ({'; '.join(failed)})")
            if ver and not dry_run:
                tracking.tag(name, ver, rejected_reasons="; ".join(failed))
            if name == "magnitude" and (d / "tt_correction_prev.json").exists() and not dry_run:
                shutil.copyfile(d / "tt_correction_prev.json", V2 / "tt_correction.json")
            continue
        reason = "passed G1-G6: " + next(r["detail"] for r in g["rules"] if r["rule"].startswith("G6"))
        lines.append(f"{name}: PROMOTED v{ver} ({reason})")
        if dry_run:
            continue
        prev = tracking.alias_version(name, "champion") if tracking.enabled() else None
        if ver:
            tracking.set_alias(name, "champion", ver)
            tracking.tag(name, ver, promoted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         promotion_reason=reason, gate_run_id=gate.get("gate_run_id", ""), previous_champion=prev)
        for f in files:                                  # the PC's live paths hold the champion
            dst = V2 / f if f in ("early_mag.json", "early_mag_T2.json", "tt_correction.json") else PROC / f
            shutil.copyfile(d / f, dst)
        shutil.copyfile(d / Path(files[0]).with_suffix(".json").name,
                        PROC / ("detection_demo.json" if name == "detector" else "magnitude_demo.json"))
        with open(PROMOTIONS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"model": name, "version": ver, "previous": prev, "month": month,
                                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "reason": reason, "gate": gate[name]["rules"]}) + "\n")
    if not dry_run and any("PROMOTED" in x for x in lines):
        sh("scripts/make_figures.py", "publish")
    msg = f"QuakeOps retrain {month}" + (" (dry run)" if dry_run else "") + "\n" + "\n".join(lines)
    print(msg)
    if not dry_run:
        tracking.ops_email(f"QuakeOps retrain {month}", msg)


def rollback(args):
    old = tracking.alias_version(args.model, "champion")
    tracking.set_alias(args.model, "champion", args.to)
    tracking.tag(args.model, args.to, rolled_back_from=old,
                 promoted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 promotion_reason=f"rollback from v{old}")
    print(f"{args.model}: @champion v{old} -> v{args.to} (the VM installs it on its next pull --apply)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "rollback"])
    ap.add_argument("--month", help="YYYY-MM to add (default: last month)")
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--seeds", default="5")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--model", choices=["detector", "magnitude"])
    ap.add_argument("--to", type=int)
    args = ap.parse_args()
    if args.cmd == "rollback":
        return rollback(args)
    month = args.month or (pd.Period(datetime.now(timezone.utc).strftime("%Y-%m"), "M") - 1).strftime("%Y-%m")
    d = mdir(month)
    sf = d / "state.json"
    state = jload(sf) if sf.exists() else {}
    for st in ([args.stage] if args.stage else STAGES):
        if not args.stage and state.get(st):
            print(f"[{month}] {st}: done {state[st]} -- skipped")
            continue
        print(f"\n[{month}] stage {st}", flush=True)
        {"data": lambda: stage_data(month, d), "train": lambda: stage_train(month, d, args),
         "replay": lambda: stage_replay(month, d), "gate": lambda: stage_gate(month, d),
         "promote": lambda: stage_promote(month, d, args.dry_run)}[st]()
        if not (st == "promote" and args.dry_run):
            state[st] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            sf.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
