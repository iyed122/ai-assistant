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
from datetime import datetime
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
    # 0 = the whole holdout (79 questions per arm, ~45 min). A small limit is
    # what makes this demonstrable in a UI: the pipeline is identical, only
    # shorter.
    limit: int = 0
    # Where the run writes. NEVER the scratch root: that directory holds the
    # measured evidence for the delivered experiment (metrics_before.json,
    # metrics_after.json, both answer files, the verdict), and a UI-triggered
    # run that overwrote them would destroy the result it is demonstrating.
    # Each run gets its own subdirectory instead.
    out_dir: str = ""


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

    # The payload is named relative to the scratch root, where the rebuilt
    # holdout lives; an absolute path is honoured as given.
    payload = Path(req.payload)
    if not payload.is_absolute():
        payload = SCRATCH / payload
    if not payload.exists():
        raise HTTPException(400, f"payload not found: {payload}")

    # Each run writes into its own directory. The scratch root itself is
    # read-only from here -- see EvaluateRequest.out_dir.
    run_dir = Path(req.out_dir) if req.out_dir else \
        SCRATCH / "ui_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    before_a = run_dir / "local_before.jsonl"
    after_a = run_dir / "local_after.jsonl"
    m_before = run_dir / "metrics_before.json"
    m_after = run_dir / "metrics_after.json"
    verdict_file = run_dir / "raft_gate_verdict.json"

    lim = ["--limit", str(req.limit)] if req.limit else []

    async def stream():
        # Modules, not bare filenames: these live in hammer/ and are run with
        # the project root as cwd so their package imports resolve. Generation
        # only needs an HTTP client, so it runs in the project venv; scoring and
        # the gate run in venv_hammer.
        steps = [
            ("generate:before", [str(VENV), "-m", "hammer.raft_generate",
                                 "--model", req.before_model,
                                 "--payload", str(payload),
                                 "--out", str(before_a)] + lim),
            ("generate:after", [str(VENV), "-m", "hammer.raft_generate",
                                "--model", req.after_model,
                                "--payload", str(payload),
                                "--out", str(after_a)] + lim),
            ("metrics:before", [str(VENV_HAMMER), "-m", "hammer.raft_metrics",
                                "--answers", str(before_a),
                                "--payload", str(payload),
                                "--tag", "before", "--out", str(m_before)]),
            ("metrics:after", [str(VENV_HAMMER), "-m", "hammer.raft_metrics",
                               "--answers", str(after_a),
                               "--payload", str(payload),
                               "--tag", "after", "--out", str(m_after)]),
            ("gate", [str(VENV_HAMMER), "-m", "hammer.raft_gate",
                      "--before", str(m_before), "--after", str(m_after),
                      "--out", str(verdict_file)]),
        ]

        yield _sse("run_dir", path=str(run_dir))

        for name, cmd in steps:
            yield _sse("step_start", step=name)
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(ROOT),
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

        verdict = json.loads(verdict_file.read_text(encoding="utf-8")) \
            if verdict_file.exists() else None
        yield _sse("done", verdict=verdict, run_dir=str(run_dir))

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
