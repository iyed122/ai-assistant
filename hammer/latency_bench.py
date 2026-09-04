#!/usr/bin/env python
"""
Generation-latency benchmark across local Ollama models.

Answers one question with real numbers: how much of the response time is the
choice of generation model?  That is the variable being changed, so generation
latency is what this measures - not end-to-end, which also carries retrieval and
live-API time and would confound the comparison.

Every model is warmed up first (the first call after load is dominated by
weight loading and is not representative), then each prompt is issued N times.

    python -m hammer.latency_bench
    python -m hammer.latency_bench --models qwen3:8b mistral phi3.5 --repeats 3

Writes hammer/latency_bench.json, which PFE_Report/figures/make_latency_bench.py
turns into a figure.  Nothing here touches the corpus or the client APIs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests

OLLAMA = "http://localhost:11434"

# Representative of the assistant's real workload: short factual lookups,
# a refusal, and two long structured syntheses.
PROMPTS = [
    "List the steps to migrate a video encoder from firmware 4.4 to 4.5.",
    "What is the purpose of a redundancy switch in a broadcast delivery chain?",
    "A ticket asks about a component that does not exist in the documentation. "
    "Reply honestly that you cannot find it, in one sentence.",
    "Explain, with numbered steps, how an input redundancy mechanism recovers "
    "from the loss of a primary transport stream.",
    "Summarise the difference between a hotfix release and a maintenance release "
    "for an embedded product line, and say when each is appropriate.",
]


def one_call(model: str, prompt: str, timeout: int = 240) -> dict | None:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json=body, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        print(f"    ! {model}: {type(e).__name__}: {e}")
        return None
    wall = time.perf_counter() - t0
    d = r.json()
    out_tokens = d.get("eval_count") or 0
    return {
        "wall_s": wall,
        "out_tokens": out_tokens,
        "tok_per_s": out_tokens / wall if wall > 0 else 0.0,
    }


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(int(round((p / 100) * (len(xs) - 1))), len(xs) - 1)
    return xs[k]


def bench(models: list[str], repeats: int) -> dict:
    results: dict[str, dict] = {}
    for model in models:
        print(f"\n=== {model} ===")
        print("  warm-up...", end="", flush=True)
        if one_call(model, "Reply with the single word: ready.") is None:
            print(" unavailable, skipping")
            continue
        print(" ok")

        calls = []
        for i, prompt in enumerate(PROMPTS, 1):
            for r in range(repeats):
                res = one_call(model, prompt)
                if res:
                    calls.append(res)
                    print(f"  prompt {i}/{len(PROMPTS)} run {r + 1}/{repeats}: "
                          f"{res['wall_s']:6.2f}s  {res['tok_per_s']:5.1f} tok/s")

        if not calls:
            continue
        walls = [c["wall_s"] for c in calls]
        results[model] = {
            "n": len(calls),
            "mean_s": statistics.fmean(walls),
            "p50_s": pct(walls, 50),
            "p95_s": pct(walls, 95),
            "max_s": max(walls),
            "mean_tok_per_s": statistics.fmean([c["tok_per_s"] for c in calls]),
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["qwen3:8b", "mistral", "phi3.5"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    print(f"Benchmarking {len(args.models)} models x {len(PROMPTS)} prompts "
          f"x {args.repeats} repeats")
    results = bench(args.models, args.repeats)

    out = Path(__file__).resolve().parent / "latency_bench.json"
    out.write_text(json.dumps({
        "prompts": len(PROMPTS),
        "repeats": args.repeats,
        "measures": "generation latency only (Ollama /api/generate, non-streaming)",
        "models": results,
    }, indent=2), encoding="utf-8")

    print(f"\n{'model':16} {'n':>4} {'mean':>8} {'P50':>8} {'P95':>8} {'tok/s':>8}")
    for m, s in sorted(results.items(), key=lambda kv: kv[1]["mean_s"]):
        print(f"{m:16} {s['n']:4} {s['mean_s']:7.2f}s {s['p50_s']:7.2f}s "
              f"{s['p95_s']:7.2f}s {s['mean_tok_per_s']:7.1f}")
    if len(results) > 1:
        best = min(results.values(), key=lambda s: s["mean_s"])["mean_s"]
        worst = max(results.values(), key=lambda s: s["mean_s"])["mean_s"]
        print(f"\nspread: {worst / best:.2f}x between fastest and slowest model")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
