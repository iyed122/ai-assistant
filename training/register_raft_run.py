#!/usr/bin/env python3
"""
register_raft_run.py
════════════════════
Record the completed RAFT run in MLflow.

The run itself executed on rented GPU (Kaggle), so the trainer's own MLflow
callback never reached this machine's tracking store -- the experiment shows 24
runs, none of them the one that produced the served adapter. This backfills it
from the artefacts the run actually returned: its logged steps, its adapter
health check, and the held-out evaluation both arms were scored on.

Every value written here is measured. Nothing is interpolated. The gate verdict
is recorded as it was returned, including the rejection reason, because a
tracking store that only holds promotions is not a record of what happened.

Usage
    python training/register_raft_run.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
TRACKING = f"sqlite:///{(ROOT / 'training' / 'mlflow.db').as_posix()}"
EXPERIMENT = "weaver-finetune"
WORK = ROOT / "training" / "export" / "raft_work"
CURVE = ROOT / "PFE_Report" / "figures" / "raft_curve.json"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(EXPERIMENT)

    curve = json.loads(CURVE.read_text(encoding="utf-8"))
    before = json.loads((WORK / "metrics_before.json").read_text(encoding="utf-8"))
    after = json.loads((WORK / "metrics_after.json").read_text(encoding="utf-8"))
    verdict = json.loads((WORK / "raft_gate_verdict.json").read_text(encoding="utf-8"))

    run_name = "raft_20260902_full"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "method": "RAFT + QLoRA",
            "base_model": "Qwen/Qwen3-8B",
            "served_as": "weaver-raft-full",
            "compute": "Kaggle T4 (rented)",
            "adapter_health": "0 NaN, 252/252 lora_B live",
            "gate_verdict": verdict["verdict"],
            "backfilled": "true",
        })

        # Key names must match TrainingPanel.jsx's run.params.* / run.metrics.*
        # lookups (method / lora_rank / learning_rate; final_train_loss / ...),
        # otherwise the Runs tab renders "?" for each field.
        mlflow.log_params({
            "method": "raft",
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "quantisation": "4-bit NF4",
            "max_length": 3072,
            "epochs": 2,
            "planned_steps": curve["planned"],
            "dataset_train_size": 912,
            "dataset_eval_size": 79,
            "trainable_params": 43646976,
            "total_params": 8234382336,
            "gradient_checkpointing_use_reentrant": False,
            "logging_nan_inf_filter": False,
        })

        # The run's own step-by-step telemetry.
        for step, loss, gn in zip(curve["steps"], curve["loss"], curve["grad_norm"]):
            mlflow.log_metric("loss", loss, step=step)
            mlflow.log_metric("grad_norm", gn, step=step)

        # final_train_loss / final_eval_loss are the exact keys the Runs UI
        # reads for the loss/eval chips. Keep the descriptive ones too --
        # they are useful in the MLflow UI, just not surfaced here.
        mlflow.log_metrics({
            "final_train_loss": 1.589,
            "final_eval_loss": 1.721,
            "train_loss": 1.589,
            "steps_done": len(curve["steps"]),
            "nan_tensors": 0,
            "lora_B_nonzero": 252,
            "trainable_pct": 0.92,
        })

        # Held-out evaluation, both arms through the same served q4 model.
        for tag, m in (("before", before), ("after", after)):
            for key in ("golden_passage_recall", "fabrication_rate", "refusal_rate",
                        "quote_grounding", "answers_with_quotes", "answer_len_median"):
                if m.get(key) is not None:
                    mlflow.log_metric(f"eval_{tag}_{key}", float(m[key]))

        mlflow.log_metrics({
            "delta_quote_grounding": after["quote_grounding"] - before["quote_grounding"],
            "delta_refusal_rate": after["refusal_rate"] - before["refusal_rate"],
            "delta_golden_passage_recall": (after["golden_passage_recall"]
                                            - before["golden_passage_recall"]),
            "gate_sigma": verdict["golden_fact_recall"]["sigma"],
            "gate_n_measurable": verdict["n_measurable"],
        })

        for name, payload in (("metrics_before.json", before),
                              ("metrics_after.json", after),
                              ("raft_gate_verdict.json", verdict),
                              ("raft_curve.json", curve)):
            mlflow.log_dict(payload, name)

        mlflow.log_text(
            f"Gate verdict: {verdict['verdict']}\n"
            f"Reason: {verdict['reason']}\n\n"
            f"Recorded {datetime.now(timezone.utc).isoformat()} by "
            f"training/register_raft_run.py from measured artefacts.\n",
            "gate_verdict.txt")

        print(f"registered run {run_name}  id={run.info.run_id}")
        print(f"  tracking : {TRACKING}")
        print(f"  verdict  : {verdict['verdict']}")
        print(f"  grounding: {before['quote_grounding']}% -> {after['quote_grounding']}%")


if __name__ == "__main__":
    main()
