#!/usr/bin/env python3
"""
raft_generate.py
════════════════
Generate one evaluation arm by replaying the holdout prompts through a served
Ollama model.

Both arms -- base and adapter -- go through the same server at the same
quantisation, so the only difference between them is the adapter. That is the
whole point: an earlier comparison ran the base in fp16 and the adapter in q4,
which measured the quantiser as much as the training.

Deliberately dependency-light (requests only) so it runs in the project venv,
which is the environment that talks to Ollama.

Usage
    python -m hammer.raft_generate --model qwen3:8b        --out .../local_before.jsonl
    python -m hammer.raft_generate --model weaver-raft-full --out .../local_after.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

# Console output is UTF-8 regardless of the terminal's own encoding.
# Windows consoles default to cp1252, which cannot encode the box-drawing and
# check characters used in this project's banners. An unguarded print then
# raises UnicodeEncodeError from inside module import or setup, and the caller
# sees an unrelated failure -- in one case retrieval silently returned zero
# sources and every answer became a refusal.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


OLLAMA = "http://localhost:11434/api/generate"


def generate(model: str, prompt: str, timeout: int = 300) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(OLLAMA, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Greedy: the arms must differ by the adapter, not by sampling noise.
        #
        # num_ctx is sized, not generous. The longest holdout prompt is ~3.2k
        # tokens and generation is capped at 512, so 4096 cannot truncate. An
        # earlier 8192 setting cost ~83 s/question -- nearly all of it prefill,
        # because the oversized KV cache spilled off the GPU. Both arms must of
        # course run at the same setting for the comparison to mean anything.
        "options": {"temperature": 0.0, "top_p": 1.0, "num_predict": 512,
                    "num_ctx": 4096},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", ""), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--payload", default="training/export/raft_work/measurable_payload.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.payload).open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: a 45-minute run should not restart from zero.
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                pass
        print(f"resuming: {len(done)} already generated")

    fails = 0
    with out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            if r["qid"] in done:
                continue
            try:
                ans, dt = generate(args.model, r["prompt"])
            except Exception as e:
                fails += 1
                print(f"[{i}/{len(rows)}] qid={r['qid']} FAILED: {e}", flush=True)
                continue
            fh.write(json.dumps({"qid": r["qid"], "uid": r.get("uid"),
                                 "model": args.model, "answer": ans,
                                 "seconds": round(dt, 2)}, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] {dt:.1f}s  last_len={len(ans)}", flush=True)

    total = sum(1 for _ in out.open(encoding="utf-8"))
    print(f"wrote {out}  answers={total}  failures={fails}")


if __name__ == "__main__":
    main()
