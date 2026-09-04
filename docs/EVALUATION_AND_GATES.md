# Evaluations and gates: what exists, what decides, what to talk about

Inventory written after auditing the code on 2026-09-05, because the report and
the presentation both say "the gate" while the codebase contains three gate
implementations and five evaluation paths, and nothing reconciled them.

Every row below was read out of the source. The last section is a judgement
call about which ones are worth defending and which should stop being called
gates.

---

## 1. Evaluations — things that produce a score

| # | Name | Where | LLM judge? | Unit | Purpose |
|---|---|---|---|---|---|
| E1 | Deterministic validator | `hammer/validator.py` | no | one answer | 16 fabrication/format checks; holds a veto inside E2 |
| E2 | Hammer answer evaluator | `hammer/evaluator.py` | **yes** (RAGAS) | one answer | `GOLD/SILVER/BRONZE/FAILED` + failure tags; **selects the training data** |
| E3 | Capability benchmark | `hammer/benchmark.py` | **yes** (via E2) | one model | Fixed eval set re-scored before/after; regression detection |
| E4 | RAFT holdout scorer | `hammer/raft_metrics.py` | no | one model | 500-question holdout (qids 500003+); measures one training effect |
| E5 | Rolling quality check | `run_hammer check` | **yes** (via E2) | a time window | Alerts on regression (default 0.05); monitoring only |

### E1 — the 16 checks

Two families, both zero-LLM.

**Ticket traffic (7):** `api_refusal`, `jira_key_format`, `url_validity`,
`honest_empty`, `round_stat_fabrication`, `source_coverage`,
`sentries_summary_coverage`

**Document fabrication (9, v1.2):** `doc_version_fabrication`,
`doc_identifier_fabrication`, `doc_jira_fabrication`,
`confluence_title_fabrication`, `gitlab_ref_fabrication`,
`quantity_fabrication`, `url_fabrication`, `uncited_answer`,
`instruction_violation`

The second family exists because the first only matches ticket answers — on
document-grounded answers the `hallucination` tag could never fire.

### E2 — weights and grade rule

`faithfulness 35% · answer_relevance 35% · temporal 10% · code 20%`
(with no code block present, the code weight moves to faithfulness)

`GOLD` requires score >= 0.85 **and** faithfulness >= 0.70 **and** zero tags.
Any critical tag (`hallucination`, `tool_misuse`, `retrieval_miss`) forces
`FAILED` regardless of score. That override is E1's veto.

---

## 2. Gates — things that decide

| # | Name | Where | LLM judge? | Wired to a promote path? |
|---|---|---|---|---|
| G1 | Benchmark "deployment gate" | `hammer/benchmark.py` | **yes** | **no** |
| G2 | Adapter gate, Mongo benchmark | `hammer/adapter_eval.py` | no | no (CLI only) |
| G3 | RAFT promotion gate | `hammer/raft_gate.py` | no | **yes — the Promote button** |
| G4 | Export-time dataset guards | `hammer/dataset_builder.py` | no | blocks bad training data |
| G5 | Adapter/serving compatibility | `training/pipeline.py` | no | **warns, never blocks** |

### G1 — four conditions, judge-based

| Condition | Bound | Env var |
|---|---|---|
| Overall faithfulness | >= 0.80 | `GATE_MIN_FAITHFULNESS` |
| Sentries faithfulness | >= 0.75 | `GATE_MIN_SENTRIES_FAITH` |
| Weighted-score delta | > +0.005 | `GATE_MIN_SCORE_DELTA` |
| Critical-tag counts | must decrease | — |

This is what the report calls FR-7 and Table `tab:gate`. Its inputs are E2's
metrics, so the RAGAS judge sits inside this decision.

### G2 — judge-free, on the Mongo benchmark

`source_coverage` primary, `refusal_rate` as the degenerate-abstention guard.
`MIN_SIGMA = 2.0`, `REFUSAL_DEGENERATE_PTS = 15.0`.

Cannot rule on the RAFT experiment at all: the RAFT holdout has zero qid overlap
with the 134-question Mongo benchmark, so it exits with `no held-out questions
answered by both models`. That is why G3 was written.

### G3 — the gate of record

| Role | Metric | Rule | Constant |
|---|---|---|---|
| PRIMARY | golden-passage recall | must rise >= 2 standard errors | `MIN_SIGMA = 2.0` |
| GUARD | refusal rate | must not rise > 15 points | `REFUSAL_DEGENERATE_PTS = 15.0` |
| SECONDARY | fabrication rate | must not rise > 2 points | `FABRICATION_TOLERANCE_PTS = 2.0` |
| MECHANISM | quote grounding | reported, not binding | — |

Order of evaluation: degenerate -> fabrication -> significance. Thresholds fixed
before any adapter existed (`docs/EXPERIMENT_PREREGISTRATION.md`).

`api/training_pipeline_router.py:337` imports it; the Promote button runs it; it
rejected this project's own adapter.

### G4 — dataset validation at export

Abstentions must be derived from their own context; a set whose targets are more
than `RAFT_MAX_DUP_RATE = 0.05` identical is refused; an empty golden set raises.

These exist because the first RAFT set shared one hardcoded abstention string
across 20% of its examples, and G3 caught the trained result as degenerate
abstention. The guards move that catch upstream, to export time.

### G5 — compatibility warning

Compares training base against serving model (family, parameter size). A LoRA
adapter only loads into the exact architecture it was trained on. It **warns and
continues**; `TRAIN_COMPAT_ACK=1` silences it. Not a gate.

---

## 3. Findings

### 3.1 One promote path is ungated — this is a defect

The API runs G3 before promoting. The CLI does not:

```
python training/pipeline.py promote <run_id>
```

calls `promote_model()` directly (`training/pipeline.py:1296`) and transitions
the MLflow registry to Production with no evaluation of any kind.

This contradicts the report's chapter 4 keybox: *"No adapter is promoted to
Production without passing an eval_gate."*

### 3.2 `eval_gate` does not exist

The identifier named in chapter 4 appears nowhere in the codebase. The behaviour
it describes is real but split across G1 and G3 under other names.

### 3.3 The report calls two different gates "the gate"

- Chapter 4 / FR-7 / Table `tab:gate` -> **G1** (judge-based)
- Chapter 5 `sec:raft` -> **G3** (judge-free)

Chapter 5 (~line 175) says the deployment gate "re-scores its small benchmark
with the same local judge under multi-sample self-consistency" — that is G1, and
it sits directly after the passage admitting the local judge is miscalibrated
(~0.25 vs ~0.80 on the same answers). As written, the calibration limitation
reads as applying to the promotion decision. It does not apply to G3, which asks
the judge nothing.

### 3.4 The separation rationale is documented nowhere

E2 selects the training data. A gate that also scored with E2 would judge the
adapter with the instrument that taught it, and any apparent improvement would be
unfalsifiable. G3 avoids this on three axes at once: different scorer
(deterministic), different data (zero qid overlap), thresholds fixed in advance.

G1 does not avoid it. Its inputs are E2's metrics on E3's benchmark.

This distinction appears in no chapter, slide, README section or demo scene.

### 3.5 DPO training code is still present

`train_dpo()` remains in `training/pipeline.py`. It is unreachable from the UI
(the method list is `['raft']`) but still callable.

---

## 4. Judgement: what to defend, what to stop calling a gate

Standard MLOps practice is one promotion decision that composes its checks, not
several parallel gates that different entrypoints happen to call. Three gate
implementations with no designated owner is the actual problem — not that two of
them measure different things.

**Defend as the gate — G3 only.** It is pre-registered, judge-free, wired into
the product, and it refused its own author's adapter. One gate, one decision,
one story.

**Keep, rename, mention briefly — G4.** These are dataset validation checks, and
that is the correct name for them. Worth one paragraph because they are the fix
for a real defect the gate caught. They are not a gate.

**Keep the code, drop the word "gate" — G1.** As a before/after regression check
over the capability benchmark it is useful and honest. As a *gate* it is not: it
promotes nothing, and its judge-based inputs are the circularity G3 exists to
avoid. Rename to "benchmark regression check", stop attaching FR-7 to it.

**Do not mention — G2.** It is a leftover from the DPO era. It cannot run on the
current holdout (zero qid overlap) and has never ruled on anything in the
delivered experiment. Mentioning it adds a third gate to the story and answers no
question a reader will ask.

**Mention as an engineering safeguard, not a gate — G5.** One line in the
implementation chapter.

### Resulting structure

This maps onto the four standard stages, which is what makes it defensible:

| Stage | Component |
|---|---|
| Data validation | G4 — export-time guards |
| Offline evaluation | E4 — RAFT holdout scorer |
| Promotion gate | **G3 — the only gate** |
| Monitoring / regression | E2, E3, E5 — advisory, never promote |

### Work this implies

1. Close the ungated CLI path (3.1) so `pipeline.py promote` runs G3. This is the
   only item that is a defect rather than a naming problem.
2. Correct the chapter 4 keybox: `eval_gate` does not exist, and FR-7 should
   point at G3.
3. Rename G1 throughout the report to a regression check.
4. Remove G2 from the narrative.
5. Add the separation rationale (3.4) to the report and one slide, so the
   judge-calibration admission is scoped to the check it actually constrains.
