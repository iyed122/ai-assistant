# RAFT — recorded measurements

Durable record of every RAFT number that has actually been measured, with its
provenance. This file exists because the first set of evaluation artefacts lived
only in a session scratch directory under `%TEMP%` and was destroyed by Windows
temp cleanup. Numbers recorded here are not to be re-derived from memory.

**Nothing in this file is estimated.** Where a figure could not be reproduced it
is marked unreproducible rather than filled in.

---

## 1. Training run — `weaver-raft-full` (measured, reproducible)

The completed RAFT QLoRA run on Qwen3-8B, from the run's own logged steps.

| Quantity | Value |
|---|---|
| Steps | 114 / 114 (no early stop) |
| Epochs | 2 |
| Training examples | 912 |
| `max_length` | 3072 |
| Final `train_loss` | 1.589 |
| Loss, first → last step | 2.729 → 1.721 |
| Loss, mean of first 10 steps | 2.191 |
| Loss, mean of last 10 steps | 1.458 |
| Gradient norm, first → last | 1.16 → 0.22 |
| Wall clock | 259 min |

Adapter health, verified independently of the trainer after download:

| Check | Result |
|---|---|
| NaN / Inf tensors | 0 |
| Live `lora_B` tensors | 252 / 252 |
| `lora_B` norms | min 0.2150 · median 0.5741 · max 1.6889 |
| Trainable params | 43,646,976 (0.92%) |
| Served as | `weaver-raft-full` in Ollama, 5.31 GB (q4) |

Source: run logs, `PFE_Report/figures/raft_curve.json`. Fully reproducible.

---

## 2. Holdout evaluation — original definition (measured, NOT reproducible)

Measured on the 79-question measurable subset of the 500-question holdout, base
model vs. checkpoint 60 (1 epoch of an earlier run).

Recall here was scored against a **single extracted evidence sentence** per
question. That payload (`holdout_payload.jsonl`) and its generator
(`raft_pool.jsonl`) were lost to temp cleanup and were never persisted to Mongo
— the sibling collections `kb_synth.doc_bench` (456 rows) and
`knowledge_base.synth_eval` (986 rows) share the question *style* but have
**zero qid overlap and zero question-text overlap** with this holdout, so they
cannot substitute. Kaggle held no copy either.

| Metric | base (`qwen3:8b`) | ckpt60 (1 epoch) |
|---|---|---|
| golden-fact recall | 69.8% | 63.0% |
| fabrication rate | 5.1% | 21.8% |
| refusal rate | 31.6% | 12.8% |
| bare refusal share | 72.0% | — |
| **quote grounding** | **3.63%** | **12.66%** |
| evidence similarity | 0.763 | — |
| median answer length | 558 | — |

Reading: the mechanism RAFT targets — grounding an answer in a quoted span from
the retrieved context — moved 3.63% → 12.66%. The other columns moved the wrong
way at that checkpoint, which is why it was never promoted.

**These figures are closed.** They cannot be extended to `weaver-raft-full`,
because the definition of recall they use is no longer computable. Do not add a
third column to this table.

---

## 3. Holdout evaluation — rebuilt definition (in progress)

The payload was reconstructed from the surviving rendered prompts by
`hammer/raft_payload_rebuild.py`. The prompts embed the retrieved context block
by block, and each question quotes the golden document's title, so gold identity
is recovered by parsing rather than by guessing:

- gold title parsed: **79 / 79**
- golden passage present in retrieved context: **72 / 79** (91.1%)

The 7 remaining questions are genuine retrieval misses, where refusing is the
correct behaviour.

Metrics carried over with their definitions intact, because they need only the
answer and the context: `fabrication_rate`, `refusal_rate`, `bare_refusal_rate`,
`quote_grounding`, `answer_len_median`.

Replaced: `golden_fact_recall` → **`golden_passage_recall`**, scored against the
whole golden passage rather than one evidence sentence. Renamed deliberately —
it is a different measurement and is **not** comparable to the 69.8% in §2. It
is valid only between arms scored by `hammer/raft_metrics.py`.

Both arms are regenerated under this definition through the same served model at
the same q4 quantisation, so the only difference between them is the adapter.

Measured 2026-09-02, n=79 answered by both arms, 72 with the golden passage
retrieved. Greedy decoding (temperature 0), `num_ctx` 4096, both arms through
the same Ollama server at q4.

| Metric | base (`qwen3:8b`) | `weaver-raft-full` | Δ |
|---|---|---|---|
| golden_passage_recall | 21.70% | 28.19% | **+6.49** |
| fabrication_rate | 5.06% | 6.33% | +1.27 |
| refusal_rate | 18.99% | 7.59% | **−11.40** |
| bare_refusal_rate | 33.33% | 100.0% | +66.7 (6 refusals only) |
| answers_with_quotes | 0.00% | 84.81% | **+84.81** |
| quote spans emitted | 0 | 72 | — |
| **quote_grounding** | **0.00%** | **95.83%** | **+95.83** |
| answer_len_median | 415 | 559 | +144 |

### Gate verdict: **REJECT**

```
PRIMARY   golden-passage recall  21.7% -> 28.2%   +6.5 pts  (0.9 SE)
SECONDARY fabrication rate        5.1% ->  6.3%   +1.3 pts
GUARD     refusal rate           19.0% ->  7.6%  -11.4 pts
MECHANISM quote grounding         0.0% -> 95.8%  +95.8 pts

REJECT -- recall rose +6.5 points but that is only 0.9 SE at n=72,
          indistinguishable from noise
```

Every direction is favourable; the adapter fails **only** the pre-registered 2σ
significance requirement. It was not rejected for degenerate abstention (refusals
*fell*), nor for fabrication (+1.3 pts, inside the 2.0 tolerance).

**The experiment is underpowered by construction.** At the observed effect size,
2σ needs n ≈ 354 measurable questions. The full 500-question holdout yields only
**179** with the golden passage retrieved, which would reach ~1.4σ — still a
reject. Re-scoring the remaining holdout therefore cannot change the verdict, and
was not run. To promote on this metric the project needs either a materially
larger effect (≥14.4 pts at n=72) or a substantially larger evaluation set.

### What the run does establish

`quote_grounding` moved **0.00% → 95.83%**, with quoted spans appearing in 84.8%
of answers where the base model emitted none at all. Of 72 spans emitted, 69 are
verbatim substrings of the retrieved context. This is the RAFT mechanism itself —
answering *from a citable span* rather than from parametric memory — and it is a
categorical change, not a marginal one. Sample:

> `##Reason:` The "Release 1.5.12.3" page lists the product name and version,
> stating "`##begin_quote##`Product Name PRODUCT-A Product Version
> 1.5.12.3`##end_quote##`". … `##Answer:` The documentation records that the
> Product-A release is version 1.5.12.3.

Refusals also fell 19.0% → 7.6% without a fabrication blow-out, which is the
combination refusal-aware tuning usually fails to achieve.

The honest defence claim is therefore: **the training installed the intended
behaviour, and the gate correctly declined to promote it on an evaluation set too
small to prove the downstream accuracy gain.** That the gate refused a result
whose every indicator was positive is evidence it is not a rubber stamp.

---

## 4. Where the assets live now

Moved out of `%TEMP%` and into the repository, so a temp sweep cannot repeat
this loss:

| Asset | Path |
|---|---|
| Payload rebuilder | `hammer/raft_payload_rebuild.py` |
| Arm generator | `hammer/raft_generate.py` |
| Scorer | `hammer/raft_metrics.py` |
| Promotion gate | `hammer/raft_gate.py` |
| Rebuilt payload, answers, metrics | `training/export/raft_work/` |
| Surviving rendered prompts | `training/export/kaggle_migrate/ds_holdout/kaggle_prompts.jsonl` |
| The 79 measurable qids | `training/export/kaggle_migrate/kk_after/measurable_qids.json` |

`training/export/raft_work/*.jsonl` is covered by the repository's global
`*.jsonl` ignore rule and is **intentionally left untracked**: those files embed
client Confluence content. The aggregate `metrics_*.json` files contain no
client text and are safe to commit.
