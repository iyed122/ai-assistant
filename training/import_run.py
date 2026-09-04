#!/usr/bin/env python3
"""
Import a training run into the project's MLflow store.

The adapter fine-tune is the one step that cannot run on a 6 GB card (4-bit
weights alone are ~4.5 GB before gradients and optimiser state), so it executes
on a hosted GPU. This brings the resulting run back into training/mlflow.db under
the same experiment, using the parameter and metric names training/pipeline.py
logs, so the lineage is continuous with every other run.

Reads mlflow_run.json as produced by the training notebook.

usage:
    python -m training.import_run path/to/mlflow_run.json [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{TRAINING_DIR / 'mlflow.db'}")
MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT", "weaver-finetune")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = json.loads(Path(a.run_json).read_text(encoding="utf-8"))
    params  = data.get("params", {})
    metrics = data.get("metrics", {})
    series  = data.get("series", [])
    verdict = data.get("verdict", "unknown")

    name = a.run_name or f"{params.get('method','dpo')}_{datetime.now():%Y%m%d_%H%M%S}"

    print(f"tracking : {MLFLOW_URI}")
    print(f"experiment: {MLFLOW_EXP}")
    print(f"run name : {name}")
    print(f"params {len(params)} | metrics {len(metrics)} | series {len(series)} | verdict {verdict}")

    if a.dry_run:
        print("\n-- dry run, nothing written --")
        for k, v in sorted(params.items()):
            print(f"   param  {k:24s} {v}")
        for k, v in sorted(metrics.items()):
            print(f"   metric {k:24s} {v}")
        return

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXP)

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        for pt in series:
            try:
                mlflow.log_metric(pt["key"], float(pt["value"]), step=int(pt.get("step", 0)))
            except Exception:
                continue
        mlflow.set_tag("verdict", verdict)
        mlflow.set_tag("imported_from", Path(a.run_json).name)
        # Provenance: the adapter fine-tune ran on a hosted GPU because the local
        # card cannot hold an 8B model plus gradients. Recorded so the run's
        # origin is auditable rather than implied.
        mlflow.set_tag("compute", str(params.get("compute", "unknown")))
        mlflow.set_tag("execution_host", str(params.get("platform", "unknown")))
        print(f"\nlogged run_id={run.info.run_id}")

    # confirm it landed alongside the existing runs
    import sqlite3
    db = MLFLOW_URI.replace("sqlite:///", "")
    if Path(db).exists():
        c = sqlite3.connect(db)
        n = c.execute("select count(*) from runs").fetchone()[0]
        latest = c.execute("select name, status from runs order by start_time desc limit 3").fetchall()
        print(f"mlflow.db now holds {n} runs; most recent: {latest}")


if __name__ == "__main__":
    main()
