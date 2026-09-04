#!/usr/bin/env python3
"""
Pick between trained adapters on evidence rather than on their training losses.

Two runs trained on different subsets at different sequence lengths report losses
that cannot be compared: v28 saw 667 examples in a 2048-token window, the Colab run
saw all 914 in 3072. A lower number there means "an easier set", not "a better
adapter".

So score every candidate the same way: mean token-level loss on the SAME held-out
file, in the same window, on the same base. That is a like-for-like ranking, and it
costs one model load plus a few seconds per adapter rather than a 500-question
generation run per candidate.

This is a cheap SELECTION step, not the evaluation. The real verdict still comes
from generating the holdout answers with the winner and running the metrics and the
promotion gate.

usage:
    python training/compare_adapters.py \
        --eval training/datasets/raft_eval.jsonl \
        --adapter training/export/raft_v28/raft_adapter \
        --adapter training/export/raft_colab/raft_adapter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE = "Qwen/Qwen3-8B"


def health(adapter: Path) -> tuple[bool, str]:
    """An adapter full of NaN still loads and still produces text. Check first."""
    import safetensors.torch as st

    f = adapter / "adapter_model.safetensors"
    if not f.exists():
        return False, "no adapter_model.safetensors"
    w = st.load_file(str(f))
    nan = sum(1 for t in w.values() if torch.isnan(t).any())
    inf = sum(1 for t in w.values() if torch.isinf(t).any())
    B = [t for k, t in w.items() if "lora_B" in k]
    live = sum(1 for t in B if not torch.isnan(t).any() and float(t.abs().sum()) > 0)
    if nan or inf:
        return False, f"NaN {nan} / Inf {inf}"
    if live != len(B):
        return False, f"only {live}/{len(B)} lora_B are non-zero"
    return True, f"clean, {live}/{len(B)} lora_B live"


@torch.no_grad()
def mean_loss(model, tok, rows: list, max_len: int) -> float:
    """Mean per-token loss over the eval file, supervising the assistant turn only."""
    total, n_tok = 0.0, 0
    for r in rows:
        text = tok.apply_chat_template(r["messages"], tokenize=False)
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len).to(model.device)
        out = model(**ids, labels=ids["input_ids"])
        k = int(ids["input_ids"].numel())
        total += float(out.loss) * k
        n_tok += k
    return total / max(n_tok, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("training/datasets/raft_eval.jsonl"))
    ap.add_argument("--adapter", type=Path, action="append", required=True,
                    help="repeat for each candidate")
    ap.add_argument("--max-length", type=int, default=3072,
                    help="same window for every candidate, or the comparison is void")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.eval.open(encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    print(f"eval set: {a.eval}  ({len(rows)} examples, max_length={a.max_length})\n")

    candidates = []
    for p in a.adapter:
        ok, why = health(p)
        print(f"{'PASS' if ok else 'SKIP'}  {p}  --  {why}")
        if ok:
            candidates.append(p)
    if not candidates:
        sys.exit("\nno healthy adapter to compare")

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    print(f"\nloading {BASE} in 4-bit ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.float16, trust_remote_code=True)
    base.eval()

    results = {}
    results["base (no adapter)"] = mean_loss(base, tok, rows, a.max_length)
    print(f"  base (no adapter)     loss {results['base (no adapter)']:.4f}", flush=True)

    for p in candidates:
        m = PeftModel.from_pretrained(base, str(p))
        m.eval()
        results[str(p)] = mean_loss(m, tok, rows, a.max_length)
        print(f"  {p.parent.name:20s}  loss {results[str(p)]:.4f}", flush=True)
        m.unload()          # restore the base weights before the next candidate

    print("\nranking (lower is better):")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        delta = v - results["base (no adapter)"]
        print(f"  {v:.4f}  ({delta:+.4f} vs base)  {k}")

    best = min((k for k in results if k != "base (no adapter)"), key=lambda k: results[k])
    if results[best] >= results["base (no adapter)"]:
        print("\nWARNING: no adapter beats the untrained base on held-out loss.")
    else:
        print(f"\nbest: {best}")
    Path("adapter_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
