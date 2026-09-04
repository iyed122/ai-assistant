# Pre-registration — adapter experiment

**Fixed 13 August 2026, before the adapter is trained and before the full baseline is known.**

Every threshold and decision rule below is committed in advance. The purpose is that no choice
made after seeing the outcome can be accused of having been chosen *because* of the outcome.

At the time of writing: 133 of 606 documents are genuinely scored; no adapter exists; no
after-measurement has been taken.

---

## D1 — Which metric is primary

**Rule, fixed now:** the primary outcome is **`source_coverage` on the held-out split** — the
deterministic rate at which the model cites a ticket key its retrieved sources do not contain.

The composite `weighted_score` is **secondary** and reported alongside.

**Justification.** `source_coverage` is computed by set membership, needs no LLM, predates this
experiment in the codebase, and is unaffected by abstention. The composite contains RAGAS
answer-relevance, which scores correct refusals near zero — the exact behaviour being trained.
A composite that can fall while the model improves cannot be the primary outcome.

**This holds even if the composite looks better.** Committing now removes the option of choosing
later.

---

## D2 — Pool composition and the synthetic ratio

**Rule:** all harvested failures become Type A pairs. Type B augmented pairs are added to reach a
target of 500 total, **subject to a hard cap: total pairs ≤ 10 × harvested pairs.**

| Harvested | Cap | Target actually used |
|---|---|---|
| ≥ 50 | ≥ 500 | 500 |
| 40 | 400 | 400 |
| 30 | 300 | 300 |
| < 20 | 200 | 200, and the limitation is stated prominently |

**Justification.** Gerstgrasser et al. show that degradation is bounded when synthetic data
*accumulates alongside* real data, and unbounded when synthetic *replaces* it. **They specify no
ratio or threshold** — the finding is about the mechanism, not a proportion. Our design satisfies
the supported claim (real production failures remain in the set; they are never replaced), but the
specific cap below is **our engineering judgement, not a literature threshold**, and is presented
as such.

The cap ties the synthetic volume to the real harvest so the real share cannot silently vanish as
the synthetic side scales. If the harvest is small, the training set shrinks with it — we do not
hold the target fixed and let the real share fall.

**Both pair types are same-model, minimal-edit** (see methodology §5b). No teacher-written
response is used as `chosen`, per Hong et al. on cross-model preference gaps.

---

## D3 — Whether the experiment is powered at all

**Rule:** compute the held-out baseline `source_coverage` rate *before* training. If it is
**below 10%** (fewer than ~13 failures in 134), declare the experiment **underpowered** and report
it as such rather than training a claim the data cannot support.

**Justification.** With n=134 the binomial standard error at p=0.10 is ~2.6 points. A defect rate
that low leaves no room for a detectable improvement, and any movement would be noise. Reporting
"the baseline was already too clean on this axis to demonstrate improvement" is a legitimate
finding; manufacturing a claim from it is not.

---

## D4 — Held-out integrity

**Rules, absolute:**

1. No held-out document contributes to any training pair, of either type.
2. The split is fixed before pair construction and never redrawn afterwards.
3. If the split is redrawn (e.g. 472/134 → 350/256 for power), it is redrawn **before** any pair
   is built, and the change is recorded here with its timestamp.
4. The after-measurement uses the identical prompt, identical decoding parameters and identical
   scoring path as the before-measurement. The adapter is the only difference.

---

## D5 — The promotion decision

**Rule:** `GATE_MIN_FAITHFULNESS` is set to *(measured baseline faithfulness − 0.02)*, computed
from the baseline and fixed before training. `GATE_MIN_SCORE_DELTA` stays at its shipped 0.005.

The gate's verdict is reported **whichever way it falls.**

**Justification.** The shipped default of 0.80 was calibrated for the validator-scored `sentries`
path (baseline 0.83) and does not transfer to the RAGAS-scored `rag`/`both` path (baseline ~0.42
in production). Applying it unchanged would reject every adapter regardless of merit. Deriving the
floor from the measured baseline preserves the gate's documented intent — *improve the score, do
not regress faithfulness* — which is what its own docstring states.

**A rejection is a result.** A gate that refuses to promote a weak adapter is the safety mechanism
functioning as designed, and will be reported in those terms, without apology.

---

## D6 — Reporting all runs

**Rule:** every training run is recorded in MLflow and **every run is reported**, including failed
and abandoned ones. If hyperparameters are changed and training repeated, all attempts appear in
the report, not only the best.

**Justification.** Selecting the best of several runs and presenting it as *the* result is the
most common way an honest experiment becomes a dishonest claim. Committing in advance removes the
temptation.

---

## D7 — If the composite and the primary metric disagree

**Anticipated case:** `source_coverage` improves while `weighted_score` falls, because the adapter
abstains more and RAGAS answer-relevance penalises abstention.

**Rule:** report both, lead with the primary (D1), and state the mechanism explicitly. Do not
suppress the composite, and do not re-weight the composite after the fact to make it agree.

**Justification.** This divergence is predicted in advance here, with its cause. A predicted
divergence that then occurs is evidence the model is understood. Discovering it afterwards and
explaining it away is not.

---

## D8 — The metric must not be satisfiable by silence

**Added 13 August 2026, before training, after pool construction exposed the risk.**

`source_coverage` counts answers citing a key their sources do not support. **An adapter that
refuses every question scores a perfect 0%** while being useless. The primary metric is therefore
gameable by degenerate refusal, and training data can induce exactly that: an uncapped selection
rule produced 164 of 206 supervised candidates (80%) as refusals.

**Rules, fixed now:**

1. **Refusals are capped at 30% of the supervised set.** Retained, because SFT and DPO must not
   pull in opposite directions — but held to a minority so the dominant supervised signal remains
   "answer, and cite what your sources support".
2. **Refusal rate is reported alongside the primary metric, before and after.** A drop in
   `source_coverage` accompanied by a large rise in refusal rate is **not** an improvement and
   will not be reported as one.
3. **Pre-committed interpretation.** If refusal rate on the held-out set rises by more than
   **15 percentage points**, the result is declared *degenerate abstention* rather than improved
   grounding, regardless of what `source_coverage` shows.

**Justification.** Any single metric can be satisfied by a degenerate policy. Naming the
degenerate policy in advance, and fixing the threshold that identifies it, is what separates a
measured improvement from a gamed one. This was written before the adapter existed.

---

## Recorded state at pre-registration

- Genuinely scored: 133 / 606
- Observed rates: `dpo_rejected` 10.5%, `qlora_positive` 54.1%
- Projected pools: ~64 harvested DPO, ~328 QLoRA
- Baseline `source_coverage` (full 606, prior scoring pass): 17.4%
- Production reference (227 deduplicated real conversations): 14.1%
- No adapter trained; no after-measurement taken

---

## Recorded baseline — 13 August 2026, 17:05, before any adapter exists

The cloud judge's daily quota was exhausted at 167/606 (1,790 rate-limit errors). The primary
metric does not depend on it: `source_coverage` is deterministic, so the **complete** held-out
baseline was measured on all 134 documents.

| | value |
|---|---|
| **`source_coverage` (PRIMARY, D1)** | **27 / 134 = 20.1%** |
| `jira_key_format` | 13 / 134 = 9.7% |
| binomial SE at n=134 | 3.5 percentage points |
| 2-SE detectable improvement | 6.9 points, i.e. down to ~13.2% |

**D3 verdict: powered.** 20.1% is comfortably above the 10% floor at which the experiment would
have been declared underpowered.

Failure distribution by question kind (`source_coverage` fired / total):
`synthesis` 9/26 · `comparison` 6/23 · `procedural` 5/27 · `unanswerable` 4/16 ·
`multi_hop` 2/20 · `lookup` 1/19 · `filter` 0/3.

Multi-document reasoning fabricates most; single-fact lookup least.

**Harvested DPO pool: 82** of 472 training documents (17.4%), identified by deterministic
validator checks alone — no judge involvement, therefore unaffected by the quota.

**Consequence for the critical path:** baseline, pair harvest, and the after-measurement are all
deterministic. The cloud judge gates only the secondary composite metric, which D1 already
designates as secondary. The experiment can complete without further quota.
