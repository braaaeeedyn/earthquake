"""QuakeOps experiment tracking + model registry (MLflow). The ONLY module that imports mlflow.

Everything is a no-op when MLFLOW_TRACKING_URI is unset, so training/evaluation scripts run offline and
in CI exactly as before. With it set (PC: https://mlflow.<host> + MLFLOW_TRACKING_USERNAME/PASSWORD for
Caddy basic auth; VM: http://127.0.0.1:5000), every training run records params, lineage (git commit,
dirty flag, dataset version, torch/CUDA), metrics with CI bounds, and its checkpoint + sidecars under the
run's `model/` artifact directory. Registered models use ALIASES (champion / challenger), not stages.

  python scripts/tracking.py register-legacy     # today's detector.pt + magnitude_ensemble.pt -> v1 @champion
  python scripts/tracking.py pull [--apply]      # (VM, daily) fetch @champion, verify sha256, write models.json
  python scripts/tracking.py status              # aliases + versions
"""
import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nearme_watch  # noqa: E402,F401  (import loads .env into os.environ)

PROC = ROOT / "data" / "processed"
MODELS_JSON = PROC / "models.json"
# registered model -> (main checkpoint, sidecars), paths relative to data/processed; deployed to the same paths
BUNDLES = {
    "detector": ("detector.pt", ["drift_reference.csv"]),
    "magnitude": ("magnitude_ensemble.pt", ["v2/early_mag.json", "v2/early_mag_T2.json", "v2/tt_correction.json"]),
}
EXPERIMENT = {"detector": "detect", "magnitude": "size"}


def enabled():
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def lineage():
    """Code + data + environment a model was trained from (gap #18)."""
    import torch
    meta = PROC / "v2" / "dataset_meta.json"
    m = json.loads(meta.read_text()) if meta.exists() else {}
    return {"git_commit": _git("rev-parse", "--short", "HEAD"), "git_dirty": str(bool(_git("status", "--porcelain"))),
            "dataset_version": m.get("version", "unversioned"), "dataset_end": m.get("end", ""),
            "torch": torch.__version__, "cuda": str(torch.cuda.is_available())}


@contextlib.contextmanager
def run(experiment, run_name, params=None):
    """MLflow run (or a no-op without MLFLOW_TRACKING_URI). Yields the run id or None."""
    if not enabled():
        yield None
        return
    import mlflow
    for s in (sys.stdout, sys.stderr):                   # MLflow prints emoji; a cp1252 console would crash
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="backslashreplace")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as r:
        mlflow.log_params(params or {})
        mlflow.set_tags(lineage())
        yield r.info.run_id


def active_run_id():
    if not enabled():
        return None
    import mlflow
    r = mlflow.active_run()
    return r.info.run_id if r else None


def log_metrics(d, prefix=""):
    """Log every numeric (or [lo, hi] CI pair) entry of a summary dict to the active run."""
    if not enabled():
        return
    import mlflow
    out = {}
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[prefix + k] = float(v)
        elif isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            out[prefix + k + "_lo"], out[prefix + k + "_hi"] = float(v[0]), float(v[1])
    mlflow.log_metrics({k: v for k, v in out.items() if v == v})


def log_files(paths, run_id=None):
    """Attach files to a run's model/ directory (the active run, or `run_id` after the fact)."""
    if not enabled():
        return
    import mlflow
    for p in paths:
        if run_id:
            mlflow.MlflowClient().log_artifact(run_id, str(p), "model")
        else:
            mlflow.log_artifact(str(p), "model")


def client():
    import mlflow
    return mlflow.MlflowClient()


def register(name, run_id, ckpt, tags=None):
    """New registry version from a run's model/ directory, tagged with the checkpoint sha256."""
    import mlflow
    c = client()
    try:
        c.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        c.create_registered_model(name)
    src = c.get_run(run_id).info.artifact_uri + "/model"
    v = c.create_model_version(name, src, run_id=run_id, tags={"sha256": sha256(ckpt), **(tags or {})})
    return int(v.version)


def alias_version(name, alias):
    """Version number behind an alias, or None."""
    import mlflow
    try:
        return int(client().get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def set_alias(name, alias, version):
    client().set_registered_model_alias(name, alias, str(version))


def tag(name, version, **tags):
    for k, v in tags.items():
        client().set_model_version_tag(name, str(version), k, str(v))


def run_metrics(name, version):
    mv = client().get_model_version(name, str(version))
    return mv, client().get_run(mv.run_id)


def download(name, version, dst):
    """Download a version's model/ directory to dst; returns dst."""
    import mlflow
    mv = client().get_model_version(name, str(version))
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="model", dst_path=str(dst))
    return dst / "model"


# ---------------------------------------------------------------- commands

def register_legacy(args):
    """Register the checkpoints the live site runs today as v1 @champion (idempotent)."""
    import drift_check
    summaries = {"detector": PROC / "detection_demo.json", "magnitude": PROC / "magnitude_demo.json"}
    for name, (ckpt, sidecars) in BUNDLES.items():
        if alias_version(name, "champion") is not None:
            print(f"{name}: already has @champion v{alias_version(name, 'champion')} -- skipped")
            continue
        if name == "detector":
            drift_check.build_reference(PROC / ckpt, PROC / "drift_reference.csv")
        import torch
        ck = torch.load(PROC / ckpt, weights_only=False, map_location="cpu")
        with run(EXPERIMENT[name], f"{name}-v1-legacy", {"legacy": True}) as rid:
            import mlflow
            mlflow.set_tags({"git_commit": ck.get("git", "unknown"), "trained": ck.get("trained", ""),
                             "dataset_version": ck.get("dataset", {}).get("version", "v2-2026-09-01")})
            log_metrics(json.loads(summaries[name].read_text()))
            log_files([PROC / ckpt] + [PROC / s for s in sidecars if (PROC / s).exists()])
        v = register(name, rid, PROC / ckpt, {"legacy": "true"})
        set_alias(name, "champion", v)
        tag(name, v, promoted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            promotion_reason="v2 models live since 2026-10-05 (replay-accepted, registered as v1)")
        print(f"{name}: registered v{v} @champion")


def _summary(name, version):
    mv, r = run_metrics(name, version)
    keep = {"detector": ("test_auc", "test_mcc", "sta_lta_auc", "fpr_test_noise_at_trigger", "fpr_live_noise_at_trigger", "n_test"),
            "magnitude": ("ens_r2", "ens_mae", "baseline_r2", "live_like_r2", "n_test")}[name]
    m = r.data.metrics
    out = {"version": int(version), "run_id": mv.run_id, "sha256": mv.tags.get("sha256"),
           "git_commit": r.data.tags.get("git_commit"), "dataset_version": r.data.tags.get("dataset_version"),
           "trained": r.data.tags.get("trained") or datetime.fromtimestamp(
               r.info.start_time / 1000, timezone.utc).isoformat(timespec="seconds"),
           "promoted_at": mv.tags.get("promoted_at"), "reason": mv.tags.get("promotion_reason"), "metrics": {}}
    for k in keep:
        if k in m:
            out["metrics"][k] = m[k]
            if f"{k}_ci_lo" in m:
                out["metrics"][k + "_ci"] = [m[f"{k}_ci_lo"], m[f"{k}_ci_hi"]]
    return out


def pull(args):
    """VM, daily: resolve @champion, write models.json; with --apply (or QUAKEOPS_AUTO_DEPLOY=1) download
    a new champion, verify its sha256, install it at the live paths. The daemon notices the new
    `deployed` version in models.json and exits; server.py's supervisor respawns it on the new model."""
    apply = args.apply or os.environ.get("QUAKEOPS_AUTO_DEPLOY") == "1"
    cur = json.loads(MODELS_JSON.read_text()) if MODELS_JSON.exists() else {}
    out = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "history": []}
    note = []
    for name, (ckpt, sidecars) in BUNDLES.items():
        v = alias_version(name, "champion")
        if v is None:
            out[name] = cur.get(name)
            continue
        info = _summary(name, v)
        deployed = (cur.get(name) or {}).get("deployed")
        if deployed is None and (PROC / ckpt).exists() and sha256(PROC / ckpt) == info["sha256"]:
            deployed = v                                     # first pull: the files already on disk are v
        if deployed != v and apply:
            src = download(name, v, PROC / "models" / name / f"v{v}")
            if sha256(src / Path(ckpt).name) != info["sha256"]:
                raise SystemExit(f"{name} v{v}: sha256 mismatch -- not installed")
            for rel in [ckpt] + sidecars:
                f = src / Path(rel).name
                if f.exists():
                    tmp = PROC / (rel + ".new")
                    shutil.copyfile(f, tmp)
                    os.replace(tmp, PROC / rel)              # atomic swap at the live path
            deployed = v
            note.append(f"{name} v{v} installed")
        elif deployed != v:
            note.append(f"{name} v{v} is @champion but v{deployed} is deployed (pull --apply to install)")
        info["deployed"] = deployed
        out[name] = info
        for mv in client().search_model_versions(f"name='{name}'"):
            if mv.tags.get("promoted_at"):
                out["history"].append({"model": name, "version": int(mv.version), "promoted_at": mv.tags["promoted_at"],
                                       "reason": mv.tags.get("promotion_reason", ""),
                                       "rolled_back_from": mv.tags.get("rolled_back_from")})
    out["history"].sort(key=lambda h: h["promoted_at"], reverse=True)
    out["notes"] = note
    MODELS_JSON.write_text(json.dumps(out, indent=1))
    print("\n".join(note) or "deployed models are the current champions")
    if note and note != cur.get("notes"):
        ops_email("QuakeOps: model registry", "\n".join(note))


def ops_email(subject, body):
    to = os.environ.get("OPS_EMAIL_TO") or os.environ.get("SMTP_USER")
    if to:
        nearme_watch.send_email(to, subject, body, dry_run=False)
    else:
        print(f"[ops email, no recipient configured] {subject}: {body}")


def status(args):
    for name in BUNDLES:
        print(f"{name}: champion v{alias_version(name, 'champion')}  challenger v{alias_version(name, 'challenger')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register-legacy", "pull", "status"])
    ap.add_argument("--apply", action="store_true", help="pull: install a new champion")
    args = ap.parse_args()
    if not enabled():
        raise SystemExit("set MLFLOW_TRACKING_URI (in .env) first")
    {"register-legacy": register_legacy, "pull": pull, "status": status}[args.cmd](args)


if __name__ == "__main__":
    main()
