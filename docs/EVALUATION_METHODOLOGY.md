# Evaluation methodology

How every number in the results chapter was produced, and what it does and does not claim.

Written so a reader can reproduce each figure from the repository and MongoDB alone.

---

## 1. What is measured

Two separate things, deliberately kept apart.

**Track A — descriptive.** What the delivered system did in production. No experiment, no
intervention, no claim of improvement. A measured statement about shipped behaviour.

**Track B — controlled experiment.** Whether a gated fine-tune improves the model on a
fixed benchmark. Same questions, same retrieval, same evaluator, before and after.

Merging the two would be a methodological error: Track A's traffic had a live Atlassian
connector that Track B's environment does not. They are never compared to each other.

---

## 2. The evaluator

Scoring is performed by the project's own Hammer pipeline (`hammer/evaluator.py`),
unmodified, invoked through `hammer/run_hammer.py score`.

### 2.1 Weighted score (v1.1)

| Component | Weight | Source |
|---|---|---|
| `faithfulness` | 35% | RAGAS (`rag` / `both` intents); validator score (`sentries`) |
| `answer_relevance` | 35% | RAGAS (`rag` / `both`); `entity_match_relevance` (`sentries`) |
| `temporal_consistency` | 10% | Deterministic Python — future years, before/after ordering, step ordering |
| `code_correctness` | 20% | Python `ast.parse` over fenced blocks; weight redistributed to faithfulness when no code is present |

RAGAS runs against a local Ollama backend. Contexts come from `context_snippets` stored on
the document — Tier 1, the exact passages the model saw at generation time — so faithfulness
is scored against ground truth rather than a re-retrieval approximation.

### 2.2 Deterministic validator

`hammer/validator.py` runs independently of any LLM. It needs only the answer, the sources
retrieved for that turn, and the query. Reported checks:

| Check | Fires when |
|---|---|
| `source_coverage` | The answer cites a Jira key that appears in no retrieved source |
| `jira_key_format` | Cited key prefix does not match any source prefix |
| `api_refusal` | The model claims it cannot use the API |
| `url_validity` | A cited URL fails to parse or is not on a trusted domain |
| `honest_empty` | Bonus — the model correctly reports no results |
| `round_stat_fabrication` | Suspicious "exactly N" patterns |
| `sentries_summary_coverage` | Cited keys absent from the persisted Sentries payload (skipped when absent) |

**Query-echo exemption.** Keys appearing verbatim in the user's question are never counted as
fabrications — the model is addressing the ticket it was asked about. This exemption landed in
the validator on 12 June 2026. Evaluations stored before that date over-count fabrications by
roughly a factor of three and are **not** comparable with figures computed afterwards. All
figures in this document were computed with the current validator.

### 2.3 Why a deterministic metric is reported alongside RAGAS

LLM-as-judge scoring carries documented position, verbosity and self-enhancement biases. The
validator is pure set membership and string matching over the retrieved sources: it cannot be
influenced by a judge's preferences, and re-running it reproduces the same number exactly.
Reporting both means the composite result can be checked against a component that has no
judge in it.

Discrimination was verified directly against real retrieved sources:

| Answer | Score | Check fired |
|---|---|---|
| Cites a key present in its sources | 0.850 | none |
| Cites `ZZZ-4242`, absent from sources | 0.650 | `source_coverage: jira_keys_not_in_sources` |
| States the context does not support an answer | 0.750 | none |

Grounded outranks abstention, which outranks fabrication.

---

## 3. Track A — production measurement

**Population.** All `chat_history` documents carrying an evaluation: 305 conversations served
between 21 May and 10 July 2026. Rescored with the current validator.

**Deduplication.** Queries were normalised (lower-cased, punctuation stripped, ticket keys
replaced by a placeholder) to collapse near-identical phrasings. 305 conversations reduce to
**227 unique query templates**; one template recurred 14 times.

| | all 305 | deduplicated 227 |
|---|---|---|
| mean validator score | 0.7923 | 0.7948 |
| `source_coverage` | 38 — **12.5%** | 32 — **14.1%** |
| `jira_key_format` | 3 — 1.0% | 3 — 1.3% |
| `api_refusal` | 0 — 0.0% | 0 — 0.0% |

**Claim.** Across 227 unique production conversations, the delivered system cited an ungrounded
ticket key in 14.1% of answers.

**Not claimed.** Nothing about improvement, and nothing derived from these numbers feeds the
experiment below.

---

## 4. Track B — controlled experiment

### 4.1 Question set

Production queries were unsuitable as a benchmark: 313 collapse to 234 templates, and several
are malformed. A fresh set was generated with a large instruct model, seeded from real material
already in `chat_history` — 442 distinct retrieved source titles, 232 context snippets and 50
real project prefixes — across seven question kinds (lookup, multi-hop, filter, synthesis,
procedural, comparison, deliberately unanswerable). Generated questions were deduplicated by
normalised template, yielding **1,208 unique questions**.

### 4.2 Exclusion of live-connector questions

Atlassian credentials ended with the engagement, so questions requiring live API data cannot be
reproduced. Each question was routed through the project's own `agent/intent_classifier.py`:

| intent | count | disposition |
|---|---|---|
| `sentries` | 601 (50%) | excluded — requires live API |
| `both` | 537 (44%) | kept |
| `rag` | 70 (6%) | kept |

**607 questions retained.** The exclusion uses the system's own routing logic, not a
hand-drawn line.

### 4.3 Retrieval

Context was retrieved with the project's own `Neo4jSearch.hybrid_search` against the production
graph — **1,784,139 nodes**, 1.8M typed relationships, `chunk_embedding` vector index — using
vector anchoring plus 2-hop traversal and cross-encoder reranking. Retrieval ran on CPU.

606 questions retrieved usable context: zero empty, zero thin, median 6 sources and 5,675
characters per question.

Of the ticket keys referenced by these questions, **69% exist in the graph**; the remaining 31%
do not and therefore function as abstention tests — the same failure mode observed in production.

### 4.4 Split

Stratified by question kind, fixed before any generation: **472 train / 134 held out**. The
held-out set never enters training data.

### 4.5 Generation

Answers were generated by the served `qwen3:8b` through the project's own
`agent/weaver_node.py::_build_payload`, so decoding parameters are identical to production
rather than reimplemented: thinking disabled (production prompts already carry `/no_think`),
temperature 0.2, generation budget 1200/1000/800 tokens by intent, production stop tokens.
The prompt is identical for the before and after runs; the adapter is the only difference.

**Generation budget.** The budget was left at production values and verified non-binding:
median answer 205 tokens, maximum 342, **zero answers reaching the cap and zero terminating
without sentence-final punctuation**. No answer was truncated, so the budget cannot have
shaped the results.

**Context window.** Production allocates `num_ctx = 10240`. Measured over the benchmark, the
largest prompt is 1,880 tokens and the largest answer 342, so the working set never exceeds
~2,222 tokens. The window was therefore set to 3,072 — still comfortably above the worst
case, so **every prompt is presented to the model in full and no token is dropped**. This
changes only the KV-cache reservation, not the tokens the model sees. On a 6 GB card the
smaller reservation lets more layers stay resident on the GPU (CPU/GPU split 27%/73% instead
of 30%/70%), cutting generation from 39.9 s to 21.9 s per answer. Outputs are unaffected.

### 4.6 Scoring

Generated answers are written back as native `chat_history` documents in an isolated database
and scored by `hammer/run_hammer.py score` — the unmodified production evaluator. Production
data is never written to.

---

## 5. Operating constraints and how they were handled

The evaluation ran on a single laptop: **RTX 3050 Laptop, 6144 MiB VRAM, 15.7 GB system RAM**.
Every figure below was measured on that machine. The constraints were not incidental — they
shaped the design, and each workaround was validated as output-neutral before being adopted.

### 5.1 The model does not fit in VRAM

`qwen3:8b` loads at **6.0 GB against a 6144 MiB card**, before Windows' display reservation.
Full GPU residency is therefore physically impossible; some layers always spill to CPU. This is
a hardware limit, not a configuration error, and no setting removes it.

What *was* controllable is the KV-cache reservation. Production allocates `num_ctx = 10240`.
Measured across the benchmark, the largest prompt is **1,880 tokens** and the largest answer
**342**, so the working set never exceeds ~2,222 tokens. Reducing the window to **3,072** keeps
a 38% safety margin over the worst observed case while returning cache memory to weights:

| `num_ctx` | CPU/GPU split | s / answer | GPU power | GPU temp |
|---|---|---|---|---|
| 10240 (production default) | 30% / 70% | 39.9 s | 9.6 W | 59–61 °C |
| 4096 | 30% / 70% | 31.9 s | 20.8 W | 65 °C |
| **3072 (adopted)** | **27% / 73%** | **21.9 s** | 25.9 W | 70 °C |

**45% faster with identical outputs.** The window bounds the cache, not the input: since no
prompt approaches the limit, the model receives exactly the same tokens in every configuration.
The rising power and temperature are the intended effect — work moving off the CPU onto silicon
designed for it.

A note on a plausible-sounding non-fix: Ollama runs as its own `llama-server` process outside
any Python environment, so the CPU/GPU split cannot be changed by a CUDA-enabled virtualenv.
The project's separate Hammer environment governs only its own torch usage (embeddings,
cross-encoder), never generation.

### 5.2 Thermal management without a thermal sensor

Sustained generation on a laptop is a thermal problem, not just a speed one. The obvious
solution — pause when temperature crosses a threshold — was unavailable: the firmware does not
expose `MSAcpi_ThermalZoneTemperature`, so CPU temperature cannot be read programmatically.

The fallback is a **duty-cycle limiter**: an unconditional 60-second idle after every 20
completed answers, roughly 85% on / 15% off. It is deliberately open-loop, and it is honest to
call it that — it does not sense heat, it merely bounds sustained load. Cool-down time is
excluded from the throughput average so reported s/answer stays truthful.

GPU temperature *is* readable and was monitored throughout: peak 70 °C, comfortably inside the
envelope for the part.

### 5.3 Graph traversal exhausted the Neo4j heap

Initial retrieval failed on ~5% of queries with
`Neo.TransientError.General.MemoryPoolOutOfMemoryError`. Two causes, both fixed:

- The container capped heap at **1 GB**. Raised to **4 GB heap + 2 GB page cache**, which is
  what a 1.78M-node graph with a vector index needs for multi-hop traversal.
- `.env` set `GRAPH_HOPS = 3` while the design documents 2-hop traversal. Restoring **2** both
  eliminated the blow-ups and realigned the running system with its specification.

Failures fell to zero and throughput improved from 8.6 s to 4.2 s per query as the page cache
warmed. Peak system RAM during retrieval reached **14.1 of 15.7 GB**, so Neo4j was stopped
between phases to return the memory.

### 5.4 Keeping the GPU free during retrieval

The retrieval stack loads a sentence-transformer embedder and a cross-encoder reranker, both of
which default to CUDA. With generation competing for the same 6 GB, retrieval was pinned to CPU
via `CUDA_VISIBLE_DEVICES=""` set before any torch import. Cost: 4.2 s per query instead of
roughly 1 s. Benefit: the two phases never contend, and neither OOMs.

### 5.5 Long runs on a machine that gets used

A 4-hour generation run on a personal laptop will be interrupted. Every long-running stage
therefore writes results incrementally with an explicit flush and resumes by skipping
already-completed identifiers, so an interruption costs at most one answer. Retrieval, generation
and scoring all share this property. This is why the phases are separate processes with durable
intermediate state in MongoDB rather than one pipeline held in memory.

### 5.6 What could not be worked around

Training cannot run on this hardware. `Qwen3-8B` in 4-bit occupies ~4.5 GB before gradients,
optimiser states and activations; the remaining ~1.5 GB cannot hold them at any sequence length
worth training on. The adapter fine-tune is therefore the single step executed on a hosted
16 GB GPU. Everything before and after it — retrieval, generation, scoring, dataset export,
adapter merge — runs on premises.

---

## 5b. Training-set sizing — rationale

*Design notes, to be expanded into the report's training chapter.*

### Why DPO carries the learning and QLoRA does not

The QLoRA positives are the model's **own** GOLD-graded answers. Training on them is
self-distillation: it reinforces existing behaviour rather than teaching new behaviour, so it
cannot by itself produce the improvement being measured.

Its actual role is **stabilisation**. DPO applied alone tends to degenerate — the model learns
what to avoid and drifts in verbosity, format or tone. A modest supervised set anchors those
properties while DPO moves the target behaviour. QLoRA is therefore sized as a regulariser, not
as the primary signal.

### Target sizes and why

| Pool | Target | Rationale |
|---|---|---|
| **DPO** | **500** (~80 harvested + ~420 augmented) | The learning signal. Below ~150 a rank-16 adapter produces noise; beyond ~1000 for a single behaviour is diminishing returns |
| **QLoRA** | **200** (from ~290 GOLD) | Enough to anchor format and tone; more slows training and overfits to the model's own outputs |

### The binding constraint is statistical power, not training size

The held-out set is 134 questions at a baseline ungrounded-citation rate of ~17% — roughly 23
failures. Binomial standard error is `sqrt(0.17 x 0.83 / 134)` ≈ **3.2 percentage points**.

| Improvement | Questions | Interpretation |
|---|---|---|
| 17% → 14% | ~4 fewer | indistinguishable from noise |
| 17% → 12% | ~7 fewer | weak, ~1.5 SE |
| 17% → 9% | ~11 fewer | defensible, ~2.5 SE |
| 17% → 6% | ~15 fewer | unambiguous |

A marginal improvement cannot be demonstrated on this evaluation set regardless of how it is
obtained. Effect size scales with training data, and augmented pairs cost only CPU time — no
quota, no GPU — so there is no reason to economise on the one axis that is free.

### Composition and the real-to-synthetic ratio

Roughly **1 harvested : 5 augmented**. The harvested pairs are genuine production failures and
establish the pattern; the augmented pairs are variations on that pattern, produced by
substituting an unsourced ticket key into an otherwise grounded answer — the corruption is copied
from the observed `hallucination` and `tool_misuse` cases, not invented. Harvested pairs are
oversampled so the model learns primarily from real failures.

### Framing the pairs: true-positive vs false-positive citation

Each preference pair is a **true-positive / false-positive contrast on the same question** —
identical context and identical phrasing, differing only in whether the cited ticket key is
present in the retrieved sources. The formal name for building negatives this way is **hard
negative construction**, standard in retrieval and ranking.

One caveat worth keeping precise: a TP/FP framing implies a classifier with a decision threshold,
whereas DPO learns a *relative* preference — it never decides "is this grounded?", only "which of
these two is better". That relative objective is why it converges on a few hundred pairs rather
than the tens of thousands a classifier would need. Useful analogy; do not overstate it.

### Known risk of augmentation, and why the design detects it

Synthetic negatives can be **too easy**. Real fabrications are fluent and contextual; a
single-key substitution differs from the positive by one token. DPO could learn the shallow
artefact rather than the behaviour.

The design exposes this rather than hiding it:

1. **Evaluation uses only real questions**, never augmented ones. If the model learned only the
   corruption artefact, the held-out number would not move.
2. **Harvested pairs are oversampled**, so genuine failures outweigh the template.
3. **Corruptions are varied**, not a single substitution rule.

Point 1 is the complete answer to "couldn't it just be learning your pattern?" — the improvement
is measured on held-out questions the adapter never saw, answered by a real model failing in real
ways.

### DECISION — symmetric same-model, minimal-edit pair construction

Adopted after reviewing the synthetic-preference-data literature. Supersedes the earlier plan of
using teacher-written corrections as `chosen`.

| Type | n (projected) | Source | `chosen` | `rejected` |
|---|---|---|---|---|
| **A** | ~64 | real production failures | the model's own answer, minimally edited to drop the unsourced key | the model's actual failed answer |
| **B** | ~436 | real GOLD answers | the model's GOLD answer | same answer, one sourced key swapped for an unsourced one |

Both halves of every pair are in the **same model's voice**. Type A edits a failure into a
success; Type B edits a success into a failure. The only systematic difference between `chosen`
and `rejected` is the grounding behaviour being trained.

**Why this replaces teacher-written corrections.** Hong et al. show that when `chosen` and
`rejected` originate from *different* models, the distributional gap allows DPO to exploit
stylistic features rather than the intended behaviour — the model can score well while learning
the wrong thing. Their conclusion is that a model's own filtered outputs make better preference
data than a stronger model's. Using gpt-oss corrections as `chosen` against qwen3 failures as
`rejected` is precisely the flagged configuration, so it was abandoned.

**How this reframes the ratio question.** The composition is not "64 real against 436 synthetic".
Every pair is anchored to a real model output, on a real question, with real retrieved context;
only the counterfactual half is constructed, by a minimal edit.

On the literature: Gerstgrasser et al. establish that degradation is bounded when synthetic data
*accumulates alongside* real data and unbounded when it *replaces* it. **They give no ratio and no
threshold percentage.** Our design satisfies the claim they actually support — real production
failures are retained, never replaced — but any specific ratio we adopt is our own engineering
judgement and must be defended on its own terms (statistical power against artefact risk), not by
appeal to a number the paper does not contain.

**Residual risk, stated plainly.** Minimal edits may be easier to separate than naturally
occurring failures, so the adapter could learn the edit signature rather than the behaviour. The
evaluation design detects this: measurement is on 134 held-out **real** questions, never on
constructed pairs. An adapter that had learned only the edit pattern would not move that number.

### References to cite in the report

- Gerstgrasser et al., *Is Model Collapse Inevitable? Breaking the Curse of Recursion by
  Accumulating Real and Synthetic Data* — arXiv:2404.01413. Basis for the accumulate-not-replace
  argument and the ~10% real-data retention threshold.
- Hong et al., *More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO Safety
  Alignment* — arXiv:2504.02193. Basis for the same-model pair-construction decision.
- Rafailov et al., *Direct Preference Optimization* — the DPO objective itself.
- Zheng et al., NeurIPS 2023 (judge bias) and Liu et al., EMNLP 2023 (G-Eval) — already cited for
  the LLM-as-judge argument.

> **TODO before submission:** verify the full author lists and exact publication venues for
> arXiv:2404.01413 and arXiv:2504.02193 directly from the papers. Author names above come from
> search results and are not yet confirmed first-author attributions.

### QLoRA selection criteria, in priority order

1. **Zero failure tags** — clean on every validator check, not merely GOLD
2. **Highest `faithfulness`**, in preference to highest answer-relevance
3. **Balanced across the seven question kinds** — otherwise the set overfits to `synthesis`
4. **Include credited abstentions** — see below
5. **Exclude unusually long answers** — verbose examples teach verbosity

Criterion 4 is the one that matters most. If every supervised example is a confident, informative
answer, the SFT objective anchors the model toward *always answering* while DPO pushes it toward
*declining when unsupported*. The two objectives would then oppose each other. Including
well-formed refusals in the supervised set makes them point the same way.

---

## 6. Limitations

- **Live-connector path untested.** Half the generated questions and most production traffic
  route to the Atlassian connectors, which cannot be exercised without client credentials.
  Results describe the retrieval-grounded path only.
- **Graph is a snapshot.** The ingest dates from April 2026; questions about current status are
  answered from that snapshot, not live data.
- **Training targets are teacher-generated.** Corrected responses in the preference pairs are
  synthesised by a larger instruct model grounded in the same retrieved context. This is
  teacher-guided preference optimisation, not unaided self-improvement: the system's own
  evaluator selects and routes the failures, but the correction signal is external.
- **Benchmark questions are synthetic.** They are seeded from real corpus material and routed by
  the real classifier, but they are not organic user traffic. Track A covers organic traffic;
  Track B covers a controlled benchmark. Neither substitutes for the other.

---

## 7. Reproduction

```bash
# Track A — production, deduplicated
python -m hammer.run_hammer score --force

# Track B — score generated answers in isolation
MONGO_DB=kb_synth python -m hammer.run_hammer score --workers 2
MONGO_DB=kb_synth python -m hammer.run_hammer dataset dpo
MONGO_DB=kb_synth python -m hammer.run_hammer status
```
