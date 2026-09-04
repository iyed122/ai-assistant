#!/usr/bin/env python3
"""
training_pipeline_router.py
════════════════════════════
FastAPI router for the training pipeline.

Endpoints:
  POST /training/prepare         — Dataset preparation (returns stats)
  POST /training/run             — Start training (SSE stream of logs)
  POST /training/run/stop        — Kill running training subprocess
  GET  /training/runs            — List MLflow runs
  POST /training/promote/{id}    — Promote a run to Production
  GET  /training/model/current   — Current production model info
  GET  /training/config          — Get default training config
  POST /training/config          — Save training config

Mount in main.py:
  from api.training_pipeline_router import router as pipeline_router
  app.include_router(pipeline_router)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("training_pipeline_router")

# ── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


def _same_model(name: str, active: str) -> bool:
    """Compare Ollama model tags, tolerating an implicit ":latest".

    Ollama lists a model as "weaver-raft-full:latest" while callers -- the
    promote endpoint among them -- refer to it as "weaver-raft-full". The old
    comparison required an exact match *and* then redundantly required it
    again, so a bare tag matched nothing and the UI showed no active model at
    all: not the adapter, not even the base.
    """
    if not name or not active:
        return False
    norm = lambda t: t.split(":")[0] if t.endswith(":latest") else t
    return name == active or norm(name) == norm(active)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Router ──────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/training", tags=["training-pipeline"])

# ── Global: running training subprocess ─────────────────────────────────────
_training_process: Optional[subprocess.Popen] = None


# ── Request models ──────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    # qlora | raft | dpo | sequential
    #   raft — supervised fine-tuning on RAFT-composed data (golden passage among
    #          distractors, answers that quote their source). Shares the QLoRA
    #          trainer; only the dataset differs.
    method:               str   = "qlora"
    base_model:           str   = ""         # empty = use default from .env
    lora_rank:            int   = 16
    lora_alpha:           int   = 32
    learning_rate:        float = 2e-4
    epochs:               int   = 3
    batch_size:           int   = 1
    gradient_accumulation: int  = 8
    max_seq_length:       int   = 1024
    dpo_beta:             float = 0.1


class PromoteRequest(BaseModel):
    run_id: Optional[str] = None
    # Which after-metrics file the gate scores. Defaults to the run's own.
    metrics_after: Optional[str] = None
    # Ollama tag to serve once the gate passes.
    activate_as: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# POST /training/prepare — Dataset preparation
# ═════════════════════════════════════════════════════════════════════════════

@router.post("/prepare")
async def prepare_dataset():
    """
    Pull from MongoDB, deduplicate, split 85/15, format, snapshot.
    Returns dataset stats before committing to training.
    """
    try:
        from training.pipeline import prepare_datasets
        stats = prepare_datasets()
        return JSONResponse(stats)
    except Exception as e:
        logger.error("prepare failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# POST /training/run — Start training with SSE stream
# ═════════════════════════════════════════════════════════════════════════════

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("/run")
async def start_training(req: TrainRequest):
    """
    Spawn training as a subprocess and stream stdout/stderr back as SSE.

    The training script writes JSON log lines to stdout which we parse
    and forward. Same SSE pattern as the /chat/stream endpoint.
    """
    global _training_process

    if _training_process and _training_process.poll() is None:
        raise HTTPException(
            status_code=409,
            detail="A training run is already in progress. Stop it first.",
        )

    # Write config to disk so the subprocess picks it up
    from training.pipeline import TrainConfig, TRAINING_DIR

    config = TrainConfig(
        method=req.method,
        lora_rank=req.lora_rank,
        lora_alpha=req.lora_alpha,
        learning_rate=req.learning_rate,
        epochs=req.epochs,
        batch_size=req.batch_size,
        gradient_accumulation=req.gradient_accumulation,
        max_seq_length=req.max_seq_length,
        dpo_beta=req.dpo_beta,
    )
    if req.base_model:
        config.base_model = req.base_model

    config_path = TRAINING_DIR / "run_config.json"
    config.save(config_path)

    # Spawn the training script
    cmd = [
        sys.executable, "-u",  # unbuffered
        str(ROOT / "training" / "pipeline.py"),
        "train",
        "--method", req.method,
        "--config", str(config_path),
    ]

    logger.info("Spawning training: %s", " ".join(cmd))

    async def stream_training():
        global _training_process

        yield _sse({"type": "status", "message": "Starting training...", "config": json.loads(json.dumps(req.dict()))})

        try:
            _training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            step_count = 0
            for line in iter(_training_process.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue

                # Try to parse structured log lines
                if "step=" in line and "epoch=" in line:
                    # Parse training metrics from log lines
                    step_count += 1
                    yield _sse({
                        "type":    "log",
                        "step":    step_count,
                        "message": line,
                        "raw":     line,
                    })
                elif line.startswith("{"):
                    try:
                        data = json.loads(line)
                        yield _sse({"type": "metrics", **data})
                    except json.JSONDecodeError:
                        yield _sse({"type": "log", "message": line})
                else:
                    yield _sse({"type": "log", "message": line})

                # Yield control to the event loop
                await asyncio.sleep(0)

            _training_process.wait()
            exit_code = _training_process.returncode

            if exit_code == 0:
                yield _sse({"type": "complete", "message": "Training finished successfully", "exit_code": 0})
            elif exit_code < 0 or exit_code == 1:
                yield _sse({"type": "stopped", "message": "Training stopped by user", "exit_code": exit_code})
            else:
                yield _sse({"type": "error", "message": f"Training failed (exit code {exit_code})", "exit_code": exit_code})

        except Exception as e:
            logger.error("Training stream error: %s", e, exc_info=True)
            yield _sse({"type": "error", "message": str(e)})
        finally:
            _training_process = None

    return StreamingResponse(
        stream_training(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "Connection":                  "keep-alive",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin":  "*",
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# POST /training/run/stop — Kill running training
# ═════════════════════════════════════════════════════════════════════════════

@router.post("/run/stop")
async def stop_training():
    """Gracefully kill the running training subprocess."""
    global _training_process

    if _training_process is None or _training_process.poll() is not None:
        return JSONResponse({"ok": True, "message": "No training running"})

    logger.info("Stopping training process (PID %d)", _training_process.pid)
    _training_process.terminate()

    # Give it 10s to clean up, then kill
    try:
        _training_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _training_process.kill()

    _training_process = None
    return JSONResponse({"ok": True, "message": "Training stopped"})


# ═════════════════════════════════════════════════════════════════════════════
# GET /training/runs — List MLflow runs
# ═════════════════════════════════════════════════════════════════════════════

def _json_safe(obj):
    """
    Replace non-finite floats with None so the response can be serialised.

    Degenerate training runs record NaN metrics (a DPO run that learns nothing
    logs NaN rewards; a timed-out RAGAS metric arrives as NaN). json.dumps emits
    bare NaN/Infinity, which is invalid JSON, so a single bad run made the entire
    runs list fail with "Out of range float values are not JSON compliant" --
    which is why the Training Runs panel rendered empty.
    """
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@router.get("/runs")
async def get_runs():
    """List all training runs from MLflow with metrics."""
    try:
        from training.pipeline import list_runs
        runs = _json_safe(list_runs())
        return JSONResponse({"runs": runs, "total": len(runs)})
    except Exception as e:
        logger.warning("list_runs failed: %s", e, exc_info=True)
        return JSONResponse({"runs": [], "total": 0, "error": str(e)})


# ═════════════════════════════════════════════════════════════════════════════
# POST /training/promote/{run_id} — Promote to Production
# ═════════════════════════════════════════════════════════════════════════════

@router.post("/promote/{run_id}")
async def promote_run(run_id: str, req: Optional[PromoteRequest] = None):
    """
    Put a training run through the promotion gate, and promote it only if the
    gate says so.

    The previous implementation called promote_model() first and evaluated
    afterwards, which meant the gate governed nothing: an adapter reached
    production and was graded on the way past. The order is now the one the
    report describes -- measure, decide, then act.

    Returns the full verdict either way, so the caller can show *why* a
    candidate was refused rather than only that it was:

        {
          "verdict":  "REJECT" | "PROMOTE",
          "promoted": bool,
          "reason":   "<the rule that fired>",
          "metrics":  {primary/secondary/guard/mechanism deltas},
          "thresholds": {...},
          "activated": "<ollama tag>"        # only when promoted
        }

    `metrics_after` may name an alternative metrics file. That exists so the
    decision rule can be exercised in both directions on demand -- a gate whose
    PROMOTE branch is never seen is indistinguishable from one that cannot
    promote. It does not bypass anything: whatever file is named is scored by
    the same unmodified rule.
    """
    from hammer.raft_gate import decide

    work = ROOT / "training" / "export" / "raft_work"
    before_path = work / "metrics_before.json"
    after_name = (req.metrics_after if req and req.metrics_after
                  else "metrics_after.json")
    after_path = work / after_name

    if not before_path.exists() or not after_path.exists():
        raise HTTPException(
            409,
            f"cannot gate {run_id}: need {before_path.name} and {after_path.name} "
            f"in {work}. Run the evaluation first.")

    try:
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = json.loads(after_path.read_text(encoding="utf-8"))
        verdict = decide(before, after)
    except Exception as e:
        logger.error("gate evaluation failed: %s", e, exc_info=True)
        raise HTTPException(500, f"gate evaluation failed: {e}")

    payload = {
        "run_id": run_id,
        "verdict": verdict["verdict"],
        "reason": verdict["reason"],
        "promoted": False,
        "n_measurable": verdict["n_measurable"],
        "metrics": {
            "primary":   verdict["golden_fact_recall"],
            "secondary": verdict["fabrication_rate"],
            "guard":     verdict["refusal_rate"],
            "mechanism": verdict["quote_grounding"],
        },
        "thresholds": verdict["thresholds"],
        "evaluated_against": after_path.name,
    }

    # The gate holds. Nothing is registered and nothing is served.
    if verdict["verdict"] != "PROMOTE":
        logger.info("gate REJECTED %s: %s", run_id, verdict["reason"])
        return JSONResponse(payload)

    # Cleared the bar. Registration is attempted first, but it is not what
    # makes a promotion real here.
    #
    # promote_model() expects an adapter logged to MLflow as a model artifact.
    # That suits a pipeline where serving pulls weights from the registry; this
    # one does not. An adapter reaches production as an Ollama tag layered on
    # the same base, so the registry entry is provenance and the tag switch is
    # the deployment. A run without a stored artifact -- a run trained on rented
    # hardware, for instance -- is therefore recorded rather than refused, and
    # the payload says which path was taken instead of implying the richer one.
    try:
        from training.pipeline import promote_model
        payload["registry"] = promote_model(run_id)
        payload["registry_mode"] = "model_version"
        payload["promoted"] = True
    except Exception as e:
        logger.info("no registered artifact for %s (%s); recording promotion "
                    "against the run instead", run_id, e)
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(
                f"sqlite:///{(ROOT / 'training' / 'mlflow.db').as_posix()}")
            c = MlflowClient()
            c.set_tag(run_id, "promoted_by", "raft_gate")
            c.set_tag(run_id, "promoted_verdict", verdict["verdict"])
            c.set_tag(run_id, "promoted_reason", verdict["reason"][:480])
            payload["registry_mode"] = "run_tag"
            payload["promoted"] = True
        except Exception as e2:
            logger.error("could not record promotion: %s", e2, exc_info=True)
            payload["error"] = f"gate passed but promotion could not be recorded: {e2}"
            return JSONResponse(payload, status_code=500)

    # Attaching the adapter is the step that makes a promotion real: until the
    # served tag changes, a registry entry has no effect on what users get.
    try:
        from agent.weaver_node import set_active_model
        tag = (req.activate_as if req and req.activate_as else "weaver-raft-full")
        payload["activated"] = set_active_model(tag)
        logger.info("gate PROMOTED %s -> serving %s", run_id, payload["activated"])
    except Exception as e:
        logger.warning("promoted but activation failed: %s", e)
        payload["activation_error"] = str(e)

    return JSONResponse(payload)


# ═════════════════════════════════════════════════════════════════════════════
# GET  /training/model/available  — models Ollama can serve
# POST /training/model/activate   — point Weaver at one of them
#
# A trained adapter reaches production as an Ollama model that layers it onto
# the same base (FROM qwen3:8b + ADAPTER), so activating a fine-tune means
# switching the tag generation uses. Prompts, decoding parameters and context
# budgets are unchanged, so the weights are the only difference -- which is what
# makes a before/after comparison meaningful.
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/model/available")
async def list_available_models():
    """Ollama models on this host, with the active one flagged."""
    import requests
    from agent.weaver_node import OLLAMA_HOST, OLLAMA_MODEL, get_active_model

    active = get_active_model()
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        models = []
        for m in r.json().get("models", []):
            name = m.get("name", "")
            models.append({
                "name":      name,
                "size":      m.get("size"),
                "modified":  m.get("modified_at"),
                "active":    _same_model(name, active),
                "is_adapter": name != OLLAMA_MODEL and "ft" in name.lower(),
            })
        return JSONResponse({
            "active":  active,
            "boot_default": OLLAMA_MODEL,
            "models":  sorted(models, key=lambda x: (not x["active"], x["name"])),
        })
    except Exception as e:
        logger.error("listing ollama models failed: %s", e)
        return JSONResponse({"active": active, "boot_default": OLLAMA_MODEL,
                             "models": [], "error": str(e)}, status_code=503)


@router.post("/model/activate")
async def activate_model(payload: dict):
    """Switch the model Weaver generates with. Body: {"model": "weaver-ft"}"""
    import requests
    from agent.weaver_node import OLLAMA_HOST, set_active_model, get_active_model

    name = (payload or {}).get("model", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="body must include 'model'")

    # Refuse to activate something Ollama cannot serve -- otherwise every
    # subsequent generation fails with an opaque error.
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        available = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {e}")

    if name not in available and f"{name}:latest" not in available:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not available in Ollama. Have: {sorted(available)}")

    previous = get_active_model()
    try:
        now = set_active_model(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("active model switched %s -> %s", previous, now)
    return JSONResponse({"ok": True, "previous": previous, "active": now})


# ═════════════════════════════════════════════════════════════════════════════
# GET /training/model/current — Current production model
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/model/current")
async def get_current_model():
    """What version is Weaver currently using."""
    try:
        from training.pipeline import get_active_adapter, MODEL_REGISTRY, MLFLOW_URI
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        versions = client.get_latest_versions(MODEL_REGISTRY, stages=["Production"])
        if versions:
            v = versions[0]
            return JSONResponse({
                "active":      True,
                "model_name":  MODEL_REGISTRY,
                "version":     v.version,
                "run_id":      v.run_id,
                "stage":       v.current_stage,
                "source":      v.source,
                "created":     v.creation_timestamp,
            })
        else:
            return JSONResponse({
                "active":      False,
                "model_name":  MODEL_REGISTRY,
                "message":     "No Production model — using base model",
            })
    except Exception as e:
        return JSONResponse({
            "active":  False,
            "message": f"MLflow unavailable: {e}",
        })


# ═════════════════════════════════════════════════════════════════════════════
# GET /training/config — Default config
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/config")
async def get_default_config():
    """Return the default training config (for the UI form)."""
    from training.pipeline import TrainConfig
    from dataclasses import asdict
    return JSONResponse(asdict(TrainConfig()))
