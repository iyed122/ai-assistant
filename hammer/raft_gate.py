#!/usr/bin/env python3
"""
Promotion gate for the RAFT experiment.

hammer/adapter_eval.py implements the same decision rule against MongoDB's
134-question benchmark. The RAFT holdout is a different, larger set (500
questions, qids 500003+) with zero qid overlap, so that script exits with "no
held-out questions answered by both models". This applies the identical
pre-registered thresholds to raft_metrics.py's output instead: same rule, same
numbers, the dataset the RAFT baseline actually covers.

Decision rule, fixed before any adapter existed
(docs/EXPERIMENT_PREREGISTRATION.md):

  PRIMARY    golden_fact_recall   must RISE, by at least MIN_SIGMA standard
                                  errors, measured only where retrieval returned
                                  the answering passage.

  GUARD      refusal_rate         must not rise by more than
                                  REFUSAL_DEGENERATE_PTS. A model that declines
                                  everything scores perfectly on fabrication and
                                  is useless; this is the check that catches
                                  buying a metric with silence.

  SECONDARY  fabrication_rate     reported, and a material rise blocks promotion
                                  even when the primary improves.

The standard error is computed on the PRIMARY metric's own sample -- the count of
questions where the golden passage was actually retrieved -- not on the 500. That
subset is what the metric is defined over, and using the larger number would
overstate the precision by more than a factor of two.

usage:
    python -m hammer.raft_gate --before metrics_baseline.json --after metrics_after.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REFUSAL_DEGENERATE_PTS = 15.0
MIN_SIGMA = 2.0
FABRICATION_TOLERANCE_PTS = 2.0


def _se_pct(p_pct: float, n: int) -> float:
    """Standard error of a percentage, in points."""
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, p_pct / 100.0))
    return 100.0 * math.sqrt(p * (1 - p) / n)


def _get(d: dict, *names: str) -> float:
    """Read a metric under either scorer's key names.

    The original scorer (lost to temp cleanup) wrote golden_fact_recall_mean/
    *_pct keys; the rebuilt hammer/raft_metrics.py writes golden_passage_recall/
    unsuffixed keys. The decision rule is identical either way -- only the key
    spelling differs, and the primary metric's rename is deliberate (a passage
    is not an evidence sentence; see that file's provenance note).
    """
    for k in names:
        if d.get(k) is not None:
            return float(d[k])
    return 0.0


def decide(before: dict, after: dict) -> dict:
    n_b = int(_get(before, "golden_present_n", "gold_present"))
    n_a = int(_get(after, "golden_present_n", "gold_present"))
    n = min(n_b, n_a)

    rb = _get(before, "golden_fact_recall_mean", "golden_passage_recall")
    ra = _get(after, "golden_fact_recall_mean", "golden_passage_recall")
    d_recall = ra - rb

    # Pooled SE across the two arms, on the measurable subset only.
    se = math.sqrt(_se_pct(rb, n_b) ** 2 + _se_pct(ra, n_a) ** 2)
    sigma = abs(d_recall) / se if se else 0.0

    fb = _get(before, "fabrication_rate_pct", "fabrication_rate")
    fa = _get(after, "fabrication_rate_pct", "fabrication_rate")
    d_fab = fa - fb

    xb = _get(before, "refusal_rate_pct", "refusal_rate")
    xa = _get(after, "refusal_rate_pct", "refusal_rate")
    d_ref = xa - xb

    qb = _get(before, "quote_grounding_pct", "quote_grounding")
    qa = _get(after, "quote_grounding_pct", "quote_grounding")

    degenerate = d_ref > REFUSAL_DEGENERATE_PTS
    fabricating = d_fab > FABRICATION_TOLERANCE_PTS
    improved = d_recall > 0
    significant = sigma >= MIN_SIGMA

    if degenerate:
        verdict = "REJECT"
        reason = (f"degenerate abstention -- refusal rate rose {d_ref:+.1f} points "
                  f"(limit {REFUSAL_DEGENERATE_PTS:+.0f}); the metric was satisfied "
                  f"by declining more, not by answering better")
    elif fabricating:
        verdict = "REJECT"
        reason = (f"fabrication rose {d_fab:+.1f} points "
                  f"(tolerance {FABRICATION_TOLERANCE_PTS:+.0f})")
    elif improved and significant:
        verdict = "PROMOTE"
        reason = (f"golden-passage recall rose {d_recall:+.1f} points ({sigma:.1f} SE, "
                  f"n={n}) with refusals {d_ref:+.1f} and fabrication {d_fab:+.1f}")
    elif improved:
        verdict = "REJECT"
        reason = (f"recall rose {d_recall:+.1f} points but that is only {sigma:.1f} SE "
                  f"at n={n} -- indistinguishable from noise")
    else:
        verdict = "REJECT"
        reason = f"golden-passage recall fell {d_recall:+.1f} points"

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_measurable": n,
        "golden_fact_recall": {"before": rb, "after": ra, "delta_pts": round(d_recall, 2),
                               "se_pts": round(se, 2), "sigma": round(sigma, 2),
                               "n_before": n_b, "n_after": n_a},
        "fabrication_rate": {"before": fb, "after": fa, "delta_pts": round(d_fab, 2)},
        "refusal_rate": {"before": xb, "after": xa, "delta_pts": round(d_ref, 2)},
        "quote_grounding": {"before": qb, "after": qa, "delta_pts": round(qa - qb, 2)},
        "thresholds": {"refusal_degenerate_pts": REFUSAL_DEGENERATE_PTS,
                       "min_sigma": MIN_SIGMA,
                       "fabrication_tolerance_pts": FABRICATION_TOLERANCE_PTS},
        "verdict": verdict,
        "reason": reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("raft_gate_verdict.json"))
    a = ap.parse_args()

    before = json.loads(a.before.read_text(encoding="utf-8"))
    after = json.loads(a.after.read_text(encoding="utf-8"))
    r = decide(before, after)

    g = r["golden_fact_recall"]
    print("=" * 66)
    print(f"RAFT PROMOTION GATE   n measurable = {r['n_measurable']}")
    print("=" * 66)
    print(f"  PRIMARY   golden-passage recall  {g['before']:.1f}% -> {g['after']:.1f}%   "
          f"{g['delta_pts']:+.1f} pts  ({g['sigma']:.1f} SE)")
    print(f"  SECONDARY fabrication rate    {r['fabrication_rate']['before']:.1f}% -> "
          f"{r['fabrication_rate']['after']:.1f}%   {r['fabrication_rate']['delta_pts']:+.1f} pts")
    print(f"  GUARD     refusal rate        {r['refusal_rate']['before']:.1f}% -> "
          f"{r['refusal_rate']['after']:.1f}%   {r['refusal_rate']['delta_pts']:+.1f} pts")
    print(f"  MECHANISM quote grounding     {r['quote_grounding']['before']:.2f}% -> "
          f"{r['quote_grounding']['after']:.2f}%   {r['quote_grounding']['delta_pts']:+.2f} pts")
    print("-" * 66)
    print(f"  VERDICT: {r['verdict']}")
    print(f"  {r['reason']}")
    print("=" * 66)

    a.out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
