#!/usr/bin/env python3
"""
Adapter gate — evaluate a candidate model against the held-out benchmark.

Answers the only question that matters about a fine-tune: did it actually get
better, and did it get better for the right reason?

Two metrics, both deterministic (no LLM judge, so no quota and no judge bias):

  source_coverage  — how often an answer cites a ticket key none of its
                     retrieved sources support. The behaviour being trained.

  refusal_rate     — how often the model declines. Reported because
                     source_coverage is gameable by silence: a model that
                     refuses everything has a perfect fabrication rate and is
                     useless. A large rise here alongside a fall there is
                     degenerate abstention, not improvement.

Results are persisted so the UI can show the gate's reasoning rather than a
bare verdict.

usage:
    python -m hammer.adapter_eval --answers after.jsonl --model weaver-ft --label "DPO run 1"
    python -m hammer.adapter_eval --list
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pymongo import MongoClient                      # noqa: E402
from hammer.validator import validate_sentry_answer  # noqa: E402

MONGO_URI = "mongodb://localhost:27017/"
SYNTH_DB  = "kb_synth"
EVAL_COLL = "adapter_evaluations"

# Pre-registered guard (docs/EXPERIMENT_PREREGISTRATION.md D8): a rise of more
# than this many points in refusal rate is degenerate abstention, whatever the
# grounding number does.
REFUSAL_DEGENERATE_PTS = 15.0
# Minimum improvement to call a result more than noise (2 standard errors).
MIN_SIGMA = 2.0

REFUSAL_RE = re.compile(
    r"does not (contain|mention|support|provide|include|specify|cover)"
    r"|no (relevant |specific |such )?(information|ticket|details?|documentation)"
    r"|not (possible|enough information) to (answer|determine)"
    r"|cannot be (answered|determined|verified)"
    r"|unable to (answer|determine|find)"
    r"|is not (available|present|found) in the", re.IGNORECASE)


def _fired(answer: str, sources, query: str) -> set[str]:
    _, det = validate_sentry_answer(answer or "", sources or [], None, query or "")
    return {c["name"] for c in det.get("checks", []) if c.get("penalty", 0) > 0}


def evaluate(answers_path: Path, model: str, label: str, persist: bool = True) -> dict:
    db = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)[SYNTH_DB]
    baseline = {d["qid"]: d for d in db["chat_history"].find({"split": "holdout"})}

    after = {}
    for line in answers_path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            after[r["qid"]] = r

    common = sorted(set(baseline) & set(after))
    if not common:
        raise SystemExit("no held-out questions answered by both models")

    n = len(common)
    b_sc = b_ref = a_sc = a_ref = 0
    fixed = broke = 0
    per_kind: dict[str, dict] = {}

    for q in common:
        b, a = baseline[q], after[q]
        src, qry = b.get("sources") or [], b.get("query", "")
        bf = "source_coverage" in _fired(b.get("answer", ""), src, qry)
        af = "source_coverage" in _fired(a.get("answer", ""), src, qry)
        b_sc += bf; a_sc += af
        fixed += (bf and not af)
        broke += (not bf and af)
        b_ref += bool(REFUSAL_RE.search(b.get("answer", "")))
        a_ref += bool(REFUSAL_RE.search(a.get("answer", "")))
        k = b.get("kind", "?")
        d = per_kind.setdefault(k, {"n": 0, "before": 0, "after": 0})
        d["n"] += 1; d["before"] += bf; d["after"] += af

    bp, ap = 100 * b_sc / n, 100 * a_sc / n
    brp, arp = 100 * b_ref / n, 100 * a_ref / n
    p = b_sc / n
    se = 100 * math.sqrt(p * (1 - p) / n) if n else 0.0
    delta = ap - bp
    sigma = abs(delta) / se if se else 0.0

    degenerate = (arp - brp) > REFUSAL_DEGENERATE_PTS
    improved = delta < 0                       # lower is better
    significant = sigma >= MIN_SIGMA

    if degenerate:
        verdict, reason = "REJECT", (
            f"degenerate abstention — refusal rate rose {arp-brp:+.1f} points "
            f"(threshold {REFUSAL_DEGENERATE_PTS:+.0f}); the metric was satisfied by silence")
    elif improved and significant:
        verdict, reason = "PROMOTE", (
            f"ungrounded citations fell {abs(delta):.1f} points ({sigma:.1f} SE) "
            f"without a degenerate rise in refusals")
    elif improved:
        verdict, reason = "REJECT", (
            f"improvement of {abs(delta):.1f} points is only {sigma:.1f} SE — "
            f"indistinguishable from noise at n={n}")
    else:
        verdict, reason = "REJECT", (
            f"ungrounded citations rose {delta:+.1f} points — worse than baseline")

    result = {
        "label": label, "model": model, "evaluated_at": datetime.now(timezone.utc),
        "n": n, "n_holdout_total": len(baseline),
        "source_coverage": {"before": b_sc, "after": a_sc,
                            "before_pct": round(bp, 2), "after_pct": round(ap, 2),
                            "delta_pts": round(delta, 2), "sigma": round(sigma, 2),
                            "se_pts": round(se, 2)},
        "refusal": {"before": b_ref, "after": a_ref,
                    "before_pct": round(brp, 2), "after_pct": round(arp, 2),
                    "delta_pts": round(arp - brp, 2)},
        "fixed": fixed, "broke": broke,
        "by_kind": per_kind,
        "verdict": verdict, "reason": reason,
        "thresholds": {"refusal_degenerate_pts": REFUSAL_DEGENERATE_PTS,
                       "min_sigma": MIN_SIGMA},
    }

    if persist:
        db[EVAL_COLL].replace_one({"label": label}, result, upsert=True)
    return result


def _print(r: dict) -> None:
    sc, rf = r["source_coverage"], r["refusal"]
    print(f"\n{r['label']}   model={r['model']}   n={r['n']}/{r['n_holdout_total']}")
    print("-" * 62)
    print(f"  source_coverage  {sc['before_pct']:5.1f}%  ->  {sc['after_pct']:5.1f}%   "
          f"{sc['delta_pts']:+.1f} pts  ({sc['sigma']:.1f} SE)")
    print(f"  refusal rate     {rf['before_pct']:5.1f}%  ->  {rf['after_pct']:5.1f}%   "
          f"{rf['delta_pts']:+.1f} pts")
    print(f"  fixed {r['fixed']}   broke {r['broke']}")
    print(f"\n  VERDICT: {r['verdict']}")
    print(f"  {r['reason']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers")
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--label", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        db = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)[SYNTH_DB]
        for r in db[EVAL_COLL].find().sort("evaluated_at", -1):
            _print(r)
        return

    if not a.answers:
        raise SystemExit("--answers is required (or use --list)")
    p = Path(a.answers)
    if not p.is_absolute():
        p = Path.cwd() / p
    _print(evaluate(p, a.model, a.label or p.stem))


if __name__ == "__main__":
    main()
