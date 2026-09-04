# RAFT fine-tuning — training configuration and what it took to get there

Kaggle notebook `iyedmediouni/raft-qlora-training`, private. Base `Qwen/Qwen3-8B`,
1 × Tesla T4 (sm_75, 14.56 GB).

## Status — both candidates rejected by the gate; dataset rebuilt; retraining

Two numerically healthy adapters were trained on the **first** version of the
training set, and **the promotion gate rejected both**, for opposite reasons:

| | colab ckpt-60 | v28 (Kaggle) |
|---|---|---|
| verdict | **REJECT** — degenerate abstention | **REJECT** — fabrication rose |
| refusals | rose sharply | *fell* sharply |
| what it learned | recite the abstention template | answer regardless, invent when unsure |

The two bracket the same underlying defect from opposite sides. The higher
learning rate and larger sample memorised the abstention string; the lower
learning rate and smaller sample never learned abstention at all. Neither is a
training-dynamics problem — see the root cause below.

**Root cause: one hardcoded abstention string used for every golden-absent
example**, 20% of the training set as a single identical target and by far its
largest duplicate. The model learned the string as a high-probability default
rather than the behaviour, and reproduced it verbatim on held-out questions whose
evidence was present. The same recitation appeared on two independent stacks
(Kaggle T4 / HF fp16, and local Ollama q4), which is what establishes it as the
data rather than the run.

This is the documented failure mode of refusal-aware instruction tuning: SFT on a
fixed refusal pattern memorises the pattern instead of the decision boundary and
over-refuses on answerable questions.

**Corrected set (v2), now training:** every abstention is derived from its own
passages — naming the releases actually retrieved and the one that is missing — so
no fixed string exists to memorise.

| | v1 (rejected) | **v2** |
|---|---|---|
| examples | 914 | **912** |
| unique targets | 735 / 914 | **912 / 912** |
| most-repeated target | **180× (20%)** | **1× (0.1%)** |
| `##begin_quote##` coverage | 72% | **78%** |
| quotes marked verbatim that are not | 42 spans | **0** — de-quoted |
| max tokens | — | 3,071 (fits 3072) |

| | v28 (Kaggle) | **colab ckpt-60** |
|---|---|---|
| optimizer steps | 82 / 82 (complete) | 60 / 114 (partial) |
| training examples | 667 — 73% of the set | **914 — 100%** |
| `max_length` | 2048 | **3072** |
| learning rate | 5e-5 | **2e-4** |
| loss at final logged step | 1.64 – 1.86 | **1.35 – 1.50** |
| median `lora_B` norm | 0.233 | **0.562** |
| NaN / Inf | 0 / 0 | 0 / 0 |
| `lora_B` live | 252 / 252 | 252 / 252 |

Head-to-head on the same 20 held-out questions, generated through Ollama on the
same served base:

| | baseline | v28 | **colab ckpt-60** |
|---|---|---|---|
| `##Reason` emitted | 0 / 20 | 20 / 20 | **20 / 20** |
| `##begin_quote##` (verbatim evidence) | 0 / 20 | 0 / 20 | **4 / 20** |
| `##Answer` emitted | 0 / 20 | 13 / 20 | **20 / 20** |
| refusals | 15 / 20 | 15 / 20 | 15 / 20 |
| median answer length | 501 | 405 | 265 |

Only five of those twenty questions had retrievable evidence — the rest are
correct abstentions — so four quotes represents most of the answerable set. The
identical refusal count across all three arms is the guard result that matters:
the adapter did not buy its scores with silence.

**Why a partial checkpoint.** The Colab run completed all 114 steps and reported a
healthy adapter, but Google Drive was at 99% capacity: checkpoint-60 wrote in full,
checkpoints 70 and 80 lost their `adapter_model.safetensors`, and the final save
had nowhere to go. Checkpoint-60 is a complete and valid adapter at 1.04 epochs —
under-trained relative to plan, not damaged. Retraining was considered and declined:
the measured behaviour already exceeds the completed v28 run, and the ceiling on
the headline metric is retrieval coverage, not training length.

## Training configuration (v27)

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3-8B |
| Objective | RAFT — supervised on chain-of-thought targets that quote their evidence |
| Parameter method | QLoRA — 4-bit NF4, double quantisation, fp16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 43,646,976 of 4,761,498,624 (0.92%) |
| Epochs / optimizer steps | 2 / 108 |
| Batch size / grad accumulation | 1 / 16 (effective batch 16) |
| Learning rate / schedule | 5e-5, cosine, 5-step warmup |
| Gradient clipping | 0.3 |
| Optimizer | paged_adamw_8bit |
| Max sequence length | 2560 tokens |
| Mixed precision | fp16 (AMP on) |
| Gradient checkpointing | on, **`use_reentrant=False`** |
| Loaded footprint | 5.96 GB across 252 `Linear4bit` layers |

## Dataset — `raft-trainset-914` (private)

| | train | eval |
|---|---|---|
| Examples | 914 | 79 |
| Golden passage present (RAFT's *P*) | 734 — 80% | 60 |
| Distractor-only → abstention | 180 | 19 |
| Targets containing a verbatim quote | 670 | 46 |
| Distractors per example | 3 | 3 |

Token lengths (Qwen3 tokenizer): min 308, median 1,730, p90 2,371, max 3,071.

The `max_length` window decides how much of the set survives, because an example
longer than the window is dropped rather than truncated — truncating it would cut
the answer off and leave nothing to learn from:

| `max_length` | examples retained |
|---|---|
| 2048 | 667 / 914 (73%) |
| **2560** | **869 / 914 (95%)** |
| 3072 | 914 / 914 (100%) — OOMs |

2560 is the best point available: 3072 admits everything but the logits tensor
(sequence × 151,936-token vocabulary) does not fit alongside the model.

## Root cause of the NaN failures

**`use_reentrant=True` (the PyTorch default) combined with PEFT gradient
checkpointing.** Under the default, checkpointed segments do not track gradients
correctly for inputs that did not originally require grad — precisely the state
`prepare_model_for_kbit_training` creates when it calls `enable_input_require_grads()`
on a frozen quantised base. The documented symptom is a silent NaN loss within the
first few optimizer steps.

That matched every observation: deterministic, always at step 3, identical loss
values, and completely unaffected by precision, sequence length, learning rate or
optimizer choice. Setting `gradient_checkpointing_kwargs={"use_reentrant": False}`
in both `prepare_model_for_kbit_training` and `SFTConfig` fixed it, and the very
next run logged finite gradient norms (1.19, 1.37) for the first time.

### Diagnoses that were wrong

Recorded because a reader will otherwise assume they were the cause:

- **fp16 overflow in Qwen3-8B.** Wrong. A run with mixed precision switched off
  entirely — no autocast, no GradScaler, fp32 throughout — diverged at the same
  step with the same losses.
- **Truncation destroying the training signal.** Wrong as the cause. It is real
  (19% of examples had zero supervised tokens at `max_length=2048`) and worth
  fixing, but a run with a pre-flight check confirming `0/914` zero-label examples
  still diverged identically.
- **Learning rate, and the 8-bit optimizer.** Wrong. Lowering the LR to 5e-5,
  extending warmup, and switching to `adamw_torch` changed nothing.

## Genuine environment constraints (independent of the above)

1. **`torch_dtype=torch.float16` must be passed at load.** Omitting it, or asking
   for bfloat16, loads the checkpoint in its stored bf16 and the 4-bit quantizer
   never applies — 8.19B × 2 bytes ≈ 16 GB onto a 14.56 GB card, `OutOfMemoryError`
   at `_materialize_copy` before training starts.

2. **bf16 is numerically fine but emulated on sm_75.** A bf16 run produced no
   output inside Kaggle's 12-hour ceiling and was killed at 43,200 s, exit 137.

3. **Library versions matter.** Kaggle's "Latest Container Image" ships
   transformers v5, whose behaviours (a `grad_dtype` attribute defaulting to
   bfloat16 on fp32 LoRA parameters, a `_materialize_copy` loader path, renamed
   arguments) broke a recipe written for the 4.x era. The notebook pins
   `transformers==4.51.3, peft==0.15.2, trl==0.17.0, accelerate==1.6.0,
   bitsandbytes==0.45.5` — 4.51 being the earliest that knows the Qwen3
   architecture.

## The silent-failure trap

One run reported `train_loss: 0.0415`, completed all 116 steps, and set
`stopped_early: false`. That reads as excellent convergence. It is the mean of two
real losses and 114 zeros from a model that died at step 3, and its adapter had
497 of 504 tensors containing NaN.

The zeros were not real either: `logging_nan_inf_filter` defaults to **True** and
rewrites NaN/Inf losses to 0.0 in the log, so the framework was hiding the failure
it was reporting. It is now disabled.

Four safeguards run on every attempt:

- **Pre-flight token check** — every example is measured against `max_length`
  before the GPU is spent; the run aborts if any has zero supervised tokens.
- **`logging_nan_inf_filter=False`** — the loss shown is the loss computed.
- **Divergence abort** — training stops if the loss is zero or non-finite for two
  consecutive steps. On its first outing it stopped a diverged run after 11.6
  minutes instead of 4.2 hours.
- **Weight health check** — the saved adapter is scanned for NaN and Inf, and
  `raft_run.json` records the counts beside the metrics.

## Verification required before any result is reported

1. `raft_run.json` → `nan_tensors == 0`, `inf_tensors == 0`
2. `lora_B_nonzero == lora_B_tensors` — an all-zero B means nothing was learned
3. A loss curve that descends, with finite gradient norms
4. Only then: post-adapter generation on the 500-question holdout, the nine
   pre-registered metrics, and the promotion gate

The evaluation notebook enforces 1 and 2 with assertions and refuses to run on a
broken adapter.

## Reporting caveat

`raft_run.json` from v26 reports `epochs: 2` when the run did **one**. That field
was a hardcoded literal in the summary dict, not a reading of the config. It now
reads from `sft_cfg`, and additionally records `max_length` and how many examples
the length filter dropped. Any figure quoted from a v26-or-earlier `raft_run.json`
should be checked against the log.

## GPU quota

Kaggle allows 30 GPU-hours per week. Roughly 21.5 h consumed — 12 h by the bf16
timeout, 4.2 h by the fp16 divergence, 1.3 h by v26, the rest by short failures.
v27 needs ~2.5–3 h and the post-adapter generation ~1.7 h.
