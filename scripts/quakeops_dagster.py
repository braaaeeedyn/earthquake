"""Dagster orchestration for the QuakeOps monthly retrain (gap #11). Runs on the PC, where the GPU is.

One monthly-partitioned asset per retrain.py stage; each runs `retrain.py --month <partition> --stage <s>`
in a subprocess, so Dagster only orders, schedules, retries and records runs -- the stage logic lives in
retrain.py and runs identically by hand. Partition 2026-09 = add September 2026 to the dataset.

  dagster dev -f scripts/quakeops_dagster.py          # UI on http://localhost:3000; runs the schedule while up
  dagster job execute -f scripts/quakeops_dagster.py -j retrain_monthly --tags '{"dagster/partition": "2026-09"}'

Schedule: day 3 of each month, 03:00 local (late catalogue revisions have settled), for the previous month.
When `dagster dev` isn't running, Windows Task Scheduler runs the same job (see DEPLOY.md, QuakeOps).
"""
import subprocess
import sys
from pathlib import Path

import dagster as dg

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if (ROOT / ".venv" / "Scripts").exists() else sys.executable
months = dg.MonthlyPartitionsDefinition(start_date="2026-09-01", end_offset=0)


def _stage(context, stage):
    month = context.partition_key[:7]
    context.log.info(f"retrain.py --month {month} --stage {stage}")
    subprocess.run([PY, str(ROOT / "scripts" / "retrain.py"), "--month", month, "--stage", stage], cwd=ROOT, check=True)
    return dg.MaterializeResult(metadata={"month": month})


@dg.asset(partitions_def=months, description="Dataset grown through <month> (build_dataset.py --append)")
def dataset(context):
    return _stage(context, "data")


@dg.asset(partitions_def=months, deps=[dataset], description="Challenger detector + magnitude, paired vs champion")
def challengers(context):
    return _stage(context, "train")


@dg.asset(partitions_def=months, deps=[challengers], description="Replay acceptance on the held-out days")
def replay(context):
    return _stage(context, "replay")


@dg.asset(partitions_def=months, deps=[replay], description="Promotion gate G1-G6 -> gate.json")
def gate_decision(context):
    return _stage(context, "gate")


@dg.asset(partitions_def=months, deps=[gate_decision], description="Alias moves, promotions.jsonl, seismic.json")
def promotion(context):
    return _stage(context, "promote")


retrain_monthly = dg.define_asset_job("retrain_monthly", selection="*", partitions_def=months)

defs = dg.Definitions(
    assets=[dataset, challengers, replay, gate_decision, promotion],
    jobs=[retrain_monthly],
    schedules=[dg.build_schedule_from_partitioned_job(retrain_monthly, day_of_month=3, hour_of_day=3)],
)
