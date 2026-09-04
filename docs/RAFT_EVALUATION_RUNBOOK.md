# RAFT evaluation — reproducible runbook

Every command needed to go from a trained adapter to a promotion verdict, run
entirely on the local machine. No Kaggle account, no Colab, no GPU rental.

Written 20 Aug 2026 while executing it. Times are measured, not estimated.

## Why local, and why only 79 questions

**Local.** Both arms are generated through Ollama, which is the stack that
actually serves the assistant. Earlier attempts generated the baseline on Kaggle
with Hugging Face fp16 while the adapter would be served as q4_K_M GGUF — a
quantisation difference that confounds the comparison. Running both arms through
the same local Ollama removes it, and measures the model as deployed.

**79 questions.** The primary metric, `golden_fact_recall`, is defined only where
retrieval returned the answering passage: 79 of the 500-question holdout. The
other 421 cannot move it at any sample size. Which 79 is decided by retrieval,
which is identical in both arms and independent of the adapter, so restricting to
them introduces no selection bias — it is the metric's own domain.

At 96 s/question on a rented T4 the full 500 needed 13.3 h against Kaggle's 12 h
ceiling. Locally through Ollama it is ~37 s/question, so 79 questions is ~50
minutes per arm.

## Prerequisites

| | |
|---|---|
| Ollama running | `curl http://localhost:11434/api/tags` |
| Base model pulled | `qwen3:8b` |
| MongoDB running | holds retrieved contexts (`kb_synth`) |
| `venv` | project + inference |
| `venv_hammer` | evaluator (transformers 5.8.1) |
| `$LLAMA_CPP` | a built llama.cpp checkout, for the GGUF conversion |

## 1. Attach the adapter to Ollama

Converts the PEFT adapter to GGUF and creates an Ollama model that is
`FROM qwen3:8b` + `ADAPTER`. Creates a **new tag**; the served model is untouched
until you activate it.

    set LLAMA_CPP=C:/Users/iyedm/llama.cpp
    venv_hammer\Scripts\python.exe -m training.export_to_ollama ^
        --adapter training/export/raft_colab/raft_adapter/checkpoint-60 ^
        --name weaver-colab --from qwen3:8b

Use `venv_hammer`, not `venv`: the conversion loads the base config through
transformers, and `venv` has 4.46.3 which does not recognise the `qwen3`
architecture.

If the adapter was trained on a pre-quantised base, `adapter_config.json` must
name a base llama.cpp can resolve — edit `base_model_name_or_path` to
`Qwen/Qwen3-8B` before converting.

Verify:

    curl -s http://localhost:11434/api/tags

## 2. Build the measurable subset

The 79 qids are those where the evidence sentence appears among the retrieved
passages and carries at least one extractable fact — the same condition
`raft_metrics.py` applies.

    python -c "see scratchpad/measurable_qids.json"   # 79 qids
    # -> measurable_payload.jsonl (79 rows of holdout_payload.jsonl)

## 3. Generate both arms

Identical prompts, identical decoding, same machine, same quantisation — the arms
differ only by the adapter. `local_generate.py` uses Weaver's own prompt builder,
so the prompts match production exactly. Both are resumable and skip qids already
answered.

    # before
    venv\Scripts\python.exe local_generate.py --out local_before.jsonl ^
        --model qwen3:8b --payload measurable_payload.jsonl --split v5 --pause-every 0

    # after
    venv\Scripts\python.exe local_generate.py --out local_after.jsonl ^
        --model weaver-colab --payload measurable_payload.jsonl --split v5 --pause-every 0

~50 minutes per arm at ~37 s/question.

## 4. Metrics

Nine metrics, fixed before any adapter existed. Deterministic — no LLM judge, so
no quota and no judge bias.

    venv_hammer\Scripts\python.exe raft_metrics.py local_before.jsonl measurable_payload.jsonl before
    venv_hammer\Scripts\python.exe raft_metrics.py local_after.jsonl  measurable_payload.jsonl after

## 5. Promotion gate

Applies the pre-registered decision rule: recall must rise by at least 2 standard
errors, refusals must not rise more than 15 points, fabrication must not rise more
than 2 points.

    venv_hammer\Scripts\python.exe -m hammer.raft_gate ^
        --before metrics_before.json --after metrics_after.json

`hammer/adapter_eval.py` implements the same rule against MongoDB's 134-question
benchmark; that set has **zero qid overlap** with the RAFT holdout, which is why
`raft_gate.py` exists.

## 6. Screenshots, then restore

Capture the UI at every successful step — dataset prepared, run configured, run
completed, adapter attached, gate verdict. These are report figures and the state
cannot be reconstructed once the serving model is switched back.

Activating changes what the live assistant serves. **Restore afterwards and verify
against the API, not the click:**

    POST /training/model/activate   {"model": "qwen3:8b"}

The key is `model`, not `name`; a wrong key 400s silently.

## What can go wrong

| Symptom | Cause |
|---|---|
| `model type qwen3 not recognized` | transformers < 4.51 — use `venv_hammer` |
| `no held-out questions answered by both models` | `adapter_eval.py` against the RAFT holdout; use `raft_gate.py` |
| adapter converts but answers are unchanged | check the Ollama model was activated, not just created |
| loss curve fine but adapter unusable | run the NaN / dead-`lora_B` health check; a completed run is not evidence |
