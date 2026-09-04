#!/usr/bin/env python3
"""
raft_eval_router.py
═══════════════════
FastAPI router for the post-training half of the RAFT loop, so the whole thing is
drivable from the UI rather than only from a terminal.

  POST /raft/adapter/export    — PEFT adapter -> GGUF -> Ollama model
  POST /raft/evaluate/run      — generate both arms, score them, run the gate (SSE)
  GET  /raft/evaluate/status   — what exists on disk right now
  GET  /raft/evaluate/verdict  — the last gate verdict

WHY THIS DISPATCHES ACROSS VENVS
Three environments exist and each step must run in the right one; the router picks,
so no caller has to know:

  venv          project + inference. Talks to Ollama over HTTP, so it is all that
                generation needs.
  venv_hammer   the evaluator: transformers 5.8.1, which is the only environment
                here that recognises the Qwen3 architecture -- required both for the
                GGUF conversion and for the metrics.

No third "training venv" is needed: local 8B training is impractical on a 6 GB card
regardless of environment, so training runs on rented GPU and only its adapter comes
back. Everything after that already works across the two venvs above.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("raft_eval_router")
router = APIRouter(prefix="/raft", tags=["raft"])

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "venv" / "Scripts" / "python.exe"
VENV_HAMMER = ROOT / "venv_hammer" / "Scripts" / "python.exe"
SCRATCH = Path(os.getenv("RAFT_SCRATCH", str(ROOT / "training" / "export" / "raft_work")))
LLAMA_CPP = os.getenv("LLAMA_CPP", r"C:/Users/iyedm/llama.cpp")

_running: Optional[subprocess.Popen] = None


class ExportRequest(BaseModel):
    adapter_path: str
    name: str = "weaver-raft"
    base: str = "qwen3:8b"


class EvaluateRequest(BaseModel):
    after_model: str = "weaver-raft"
    before_model: str = "qwen3:8b"
    payload: str = "measurable_payload.jsonl"
    limit: int = 0


def _sse(event: str, **data) -> str:
    return f"data: {json.dumps({'event': event, **data})}\n\n"


@router.post("/adapter/export")
async def export_adapter(req: ExportRequest):
    """Convert a PEFT adapter to GGUF and register it as an Ollama model.

    Runs in venv_hammer: the converter loads the base config through transformers,
    and the project venv predates the Qwen3 architecture.
    """
    adapter = Path(req.adapter_path)
    if not (adapter / "adapter_config.json").exists():
        raise HTTPException(404, f"{adapter} is not a PEFT adapter directory")

    # Never register an adapter that did not learn -- a completed run once wrote a
    # file in which 497 of 504 tensors were NaN.
    try:
        import torch
        import safetensors.torch as st

        w = st.load_file(str(adapter / "adapter_model.safetensors"))
        nan = sum(1 for t in w.values() if torch.isnan(t).any())
        inf = sum(1 for t in w.values() if torch.isinf(t).any())
        B = [t for k, t in w.items() if "lora_B" in k]
        live = sum(1 for t in B if not torch.isnan(t).any() and float(t.abs().sum()) > 0)
        if nan or inf or live != len(B):
            raise HTTPException(
                422,
                f"adapter is unusable: NaN {nan}, Inf {inf}, live lora_B {live}/{len(B)}")
    except HTTPException:
        raise
    except Exception as e:  # torch missing in this venv is not fatal for the export
        logger.warning("health check skipped: %s", e)

    env = {**os.environ, "LLAMA_CPP": LLAMA_CPP}
    cmd = [str(VENV_HAMMER), "-m", "training.export_to_ollama",
           "--adapter", str(adapter), "--name", req.name, "--from", req.base]
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=3600)
    if p.returncode != 0:
        raise HTTPException(500, (p.stderr or p.stdout)[-1500:])
    return JSONResponse({"ok": True, "model": req.name,
                         "log": (p.stdout or "")[-2000:]})


@router.post("/evaluate/run")
async def run_evaluation(req: EvaluateRequest):
    """Generate both arms, score each, and run the promotion gate.

    Both arms are generated through the same served model at the same
    quantisation, so the only difference between them is the adapter.
    """
    global _running
    if _running and _running.poll() is None:
        raise HTTPException(409, "an evaluation is already running")

    SCRATCH.mkdir(parents=True, exist_ok=True)

    async def stream():
        steps = [
            ("generate:before", [str(VENV), "local_generate.py",
                                 "--out", "local_before.jsonl",
                                 "--model", req.before_model,
                                 "--payload", req.payload, "--pause-every", "0"]),
            ("generate:after", [str(VENV), "local_generate.py",
                                "--out", "local_after.jsonl",
                                "--model", req.after_model,
                                "--payload", req.payload, "--pause-every", "0"]),
            ("metrics:before", [str(VENV_HAMMER), "raft_metrics.py",
                                "local_before.jsonl", req.payload, "before"]),
            ("metrics:after", [str(VENV_HAMMER), "raft_metrics.py",
                               "local_after.jsonl", req.payload, "after"]),
            ("gate", [str(VENV_HAMMER), "-m", "hammer.raft_gate",
                      "--before", "metrics_before.json",
                      "--after", "metrics_after.json"]),
        ]

        for name, cmd in steps:
            yield _sse("step_start", step=name)
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(SCRATCH),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    yield _sse("log", step=name, line=line[:400])
            rc = await proc.wait()
            if rc != 0:
                yield _sse("error", step=name, code=rc)
                return
            yield _sse("step_done", step=name)

        verdict_file = SCRATCH / "raft_gate_verdict.json"
        verdict = json.loads(verdict_file.read_text(encoding="utf-8")) \
            if verdict_file.exists() else None
        yield _sse("done", verdict=verdict)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/evaluate/status")
async def evaluate_status():
    """What of the pipeline already exists on disk."""
    def count(name: str) -> Optional[int]:
        p = SCRATCH / name
        if not p.exists():
            return None
        return sum(1 for _ in p.open(encoding="utf-8"))

    return JSONResponse({
        "scratch": str(SCRATCH),
        "before_answers": count("local_before.jsonl"),
        "after_answers": count("local_after.jsonl"),
        "metrics_before": (SCRATCH / "metrics_before.json").exists(),
        "metrics_after": (SCRATCH / "metrics_after.json").exists(),
        "verdict": (SCRATCH / "raft_gate_verdict.json").exists(),
    })


@router.get("/evaluate/verdict")
async def evaluate_verdict():
    p = SCRATCH / "raft_gate_verdict.json"
    if not p.exists():
        raise HTTPException(404, "no verdict yet -- run /raft/evaluate/run")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))
