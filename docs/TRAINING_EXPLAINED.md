# Fine-tuning, explained from scratch

*For defending the training chapter. Assumes no prior fine-tuning experience.*

Everything here describes what your notebook actually does, in the order it does it, with the
reasoning a jury will ask for.

---

## 1. The problem, and why we are not "retraining the model"

Qwen3-8B has **8 billion** parameters. Training all of them requires holding the weights, their
gradients, and the optimiser's running statistics in memory simultaneously — roughly **16× the
model size**, or well over 100 GB. That is data-centre hardware.

You have a 6 GB laptop GPU and a free 16 GB cloud card.

More importantly, you do not *want* to retrain it. The model already knows English, Jira
terminology, and how to summarise. It has one specific defect: **it cites ticket keys that its
retrieved sources do not contain.** Retraining everything to fix one behaviour would be like
rebuilding an engine to fix a rattle — expensive, and likely to break things that worked.

So the goal is: **change one behaviour, touch as little as possible.**

---

## 2. LoRA — the core idea

A neural network layer is a large matrix of numbers, `W`. Fine-tuning normally means changing
every number in `W`. LoRA (Low-Rank Adaptation) says: **leave `W` frozen, and learn a small
correction beside it.**

```
output  =  W · x   +   (B · A) · x
           ^frozen      ^^^^^^ trainable, tiny
```

`A` and `B` are two thin matrices. If `W` is 4096×4096 (~16.8 M numbers), then with **rank 16**,
`A` is 16×4096 and `B` is 4096×16 — about 131 K numbers, **0.8% of the original**.

The insight behind it: the *change* a model needs in order to learn one new behaviour is
low-rank — it lives in a small number of directions, even though the model itself is huge. So
you only need to learn those directions.

**In your run:** ~40 M trainable parameters against 8 B total — about **0.5%**. The notebook
prints the exact figure, and it's worth quoting.

### Why this matters practically

- The frozen original cannot be damaged. Delete the adapter and you have the base model back.
- The adapter file is tens of megabytes, not sixteen gigabytes.
- You can serve one base model with several adapters swapped in and out.

**`r` (rank)** is the capacity dial. Too low and it cannot represent the change; too high and it
overfits and stops being a small correction. **16 is the standard default** for behavioural work.

**`lora_alpha`** scales the adapter's contribution. Convention is `alpha = 2 × r`, hence **32**.
The effective strength is `alpha / r = 2`.

---

## 3. QLoRA — the "Q" is quantisation

Even frozen, 8 B parameters at 16-bit precision is ~16 GB. QLoRA stores the **frozen** weights at
**4-bit** precision (~4.5 GB), while the small adapter trains in full precision.

The trick is that quantisation error is tolerable in weights you are *not* updating. You lose
some fidelity in the frozen base; you lose none in the part actually learning.

That is what puts an 8 B model on a 16 GB card at all.

---

## 4. Stage 1 — supervised fine-tuning (SFT), the stabiliser

**What it does:** shows the model input/output pairs and trains it to reproduce the outputs.
Standard next-token prediction — for each token, predict it, compare, adjust.

**Why yours is small (60 examples), and why that's deliberate:**

Your SFT examples are the model's **own** best answers. Training a model on its own correct output
teaches it nothing new — that's **self-distillation**. So SFT is not where the improvement comes
from.

Its real job is **stabilisation**. DPO alone tends to drift: the model learns what to avoid and
starts producing shorter, stranger, or oddly-formatted text. A small supervised set anchors tone
and format while DPO moves the target behaviour.

[LIMA (Zhou et al., NeurIPS 2023)](https://arxiv.org/abs/2305.11206) is the citation: 1,000
curated examples fine-tuned a 65 B model to match GPT-4 in 43% of comparisons. Their conclusion —
*almost all knowledge comes from pretraining; instruction tuning teaches format and style* — is
exactly what this stage is for.

---

## 5. Stage 2 — DPO, where the learning happens

### The problem DPO solves

You cannot write down "the correct answer" for grounding — there are many good answers. But you
*can* say **which of two answers is better**. That is a preference, and it is much easier to
produce.

Historically this needed **RLHF**: train a separate reward model on preferences, then use
reinforcement learning against it. Three models, unstable, hard to tune.

**DPO (Direct Preference Optimization)** showed the reward model is unnecessary — you can
optimise the preference directly with a simple loss. One training run, no RL.

### What it actually does

Each training example is a triple: a **prompt**, a **chosen** answer, a **rejected** answer.

DPO adjusts the model so that `chosen` becomes *relatively* more likely than `rejected` — while a
frozen reference copy keeps it from drifting too far from where it started.

```
loss  =  -log σ( β · [ (log π(chosen) - log π_ref(chosen))
                     - (log π(rejected) - log π_ref(rejected)) ] )
```

In words: **increase the margin between chosen and rejected, but stay anchored to the original.**

**`beta` (0.1)** controls that anchor. Higher = stay closer to the base model, learn less. Lower =
move further, risk degradation. 0.1 is the standard starting value.

**Learning rate 5e-6** — far smaller than SFT's 2e-4. Preference optimisation is delicate; too
large a step and the model collapses into degenerate text.

### Why it fits your problem exactly

Your pairs differ in **one respect**: whether a cited key is supported by the retrieved sources.
Same question, same context, same voice, one changed fact. That isolates the behaviour — the model
cannot learn "prefer the longer one" or "prefer the politer one", because those are held constant.

---

## 6. The hyperparameters, and what each one does

| Setting | Value | What it controls |
|---|---|---|
| `r` | 16 | Adapter capacity. Low-rank correction size |
| `lora_alpha` | 32 | Adapter strength, conventionally 2×r |
| `lora_dropout` | 0.05 | Randomly ignores 5% of adapter units per step; mild overfitting guard |
| `target_modules` | 7 projections | Which layers get adapters — attention (q,k,v,o) and MLP (gate,up,down) |
| `max_seq_length` | 2048 | Longest sequence. Your prompts peak near 1,880 tokens |
| SFT `learning_rate` | 2e-4 | Step size for supervised stage |
| DPO `learning_rate` | 5e-6 | 40× smaller — preference training is fragile |
| `beta` | 0.1 | How tightly DPO stays anchored to the base model |
| `num_train_epochs` | 1 SFT / 2 DPO | Passes over the data |
| `gradient_accumulation_steps` | 8 | Accumulate 8 samples before updating — simulates batch size 8 in limited memory |
| `warmup_ratio` | 0.05–0.1 | Ramp the learning rate up gradually to avoid an early destructive step |
| `optim` | `adamw_8bit` | Optimiser state in 8-bit to save memory |
| `seed` | 20260813 | Reproducibility |

**Why gradient accumulation matters:** you cannot fit 8 sequences of 2048 tokens in memory at once.
So you process one at a time, add up the gradients, and update once every 8. Mathematically close
to a batch of 8, at the memory cost of 1.

**How many actual updates you get:** 188 DPO pairs × 2 epochs ÷ 8 accumulation ≈ **47 optimiser
steps**. That is a small amount of training — which is exactly why the effect may be modest, and
why the pre-registration allows for a null result.

---

## 7. Reading the output — what tells you it worked

DPO logs three numbers that matter more than the loss.

| Metric | Healthy | Meaning |
|---|---|---|
| `rewards/margins` | **> 0, rising** | How much more the model prefers chosen over rejected. The core signal |
| `rewards/accuracies` | **> 0.5, rising toward 0.8+** | Fraction of pairs ranked correctly |
| `rewards/chosen` and `/rejected` | chosen above rejected | The two should separate |

### The failure signature to recognise

```
eval_loss           = 0.6931471805...      ← ln(2)
rewards/margins     = 0.0
rewards/accuracies  = 0.0
grad_norm           = 0.0
```

**`ln(2) = 0.693` is DPO's loss when the model is exactly indifferent** between chosen and
rejected — it has learned nothing. Combined with zero gradient norm it means no gradient reached
the adapter at all.

This is not hypothetical: **your earlier 15 DPO runs all show exactly this**, while reporting
status `FINISHED`. The notebook now checks for it explicitly and prints
`*** DEGENERATE — do not promote ***` rather than letting it pass.

---

## 8. What can go wrong

| Symptom | Cause | Response |
|---|---|---|
| ln(2) loss, zero margins | No gradient reaching the adapter | Run is void. Do not promote |
| Loss → 0, margins huge | Overfitting on a small set | Fewer epochs, or more data |
| Model refuses everything afterwards | Too many refusals in training | Refusal cap (D8); check refusal rate |
| Fluent but format collapses | DPO drift without enough SFT anchoring | Raise SFT weight |
| OOM | Sequence or batch too large | Lower `max_seq_length`, raise accumulation |

---

## 9. From adapter to running system

The adapter comes back as `adapter.zip` — a directory of LoRA weights plus config, tens of MB.

```bash
python -m training.export_to_ollama --adapter training/export/final_adapter --name weaver-ft
```

That converts the PEFT adapter to GGUF and builds an Ollama model:

```
FROM qwen3:8b
ADAPTER weaver-ft-lora.gguf
```

**The one hard rule:** an adapter only loads into the architecture it was trained on. Trained on
Qwen3-8B → serves on `qwen3:8b`. Train on a different base and it will not load.

Then set `OLLAMA_MODEL=weaver-ft`, re-run the 134 held-out questions, score with the same
deterministic validator, and compare to the 20.1% baseline.

---

## 10. The three sentences that carry this chapter

> **LoRA:** I froze the 8-billion-parameter model and trained a low-rank correction beside it —
> about 0.5% of the parameters — because the change needed to fix one behaviour is small, and the
> rest of the model was already correct.

> **DPO:** I never had to define the right answer. I only had to say which of two answers was
> better, and the pairs differ in exactly one respect — whether a cited ticket key is supported by
> the retrieved sources.

> **The gate:** whichever way the numbers fall, the promotion decision is made by a threshold set
> from the measured baseline before training, not by me looking at the result.
