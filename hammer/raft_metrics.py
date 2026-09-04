#!/usr/bin/env python3
"""
raft_metrics.py
═══════════════
Score one evaluation arm against the rebuilt RAFT holdout payload.

METRIC PROVENANCE -- READ BEFORE COMPARING NUMBERS
The original scorer and its payload were lost with the session scratch
directory (see hammer/raft_payload_rebuild.py). Of the metrics it produced,
these are reproduced here with their definitions intact, because they need only
the answer and the retrieved context, both of which survive inside the rendered
prompts:

    fabrication_rate      invented identifiers -- ticket keys, versions, dates
                          that appear in the answer but nowhere in the context
    refusal_rate          the answer declines to answer
    bare_refusal_rate     it declines without saying what the context DOES hold
    quote_grounding       share of ##begin_quote## spans that really occur in
                          the context. This is the RAFT mechanism itself, and
                          the metric that moved at checkpoint 60.
    answer_len_median

One metric cannot be reproduced. The original golden_fact_recall scored against
a single extracted evidence sentence per question; that sentence is gone, and
only the whole golden passage is recoverable. So this module computes

    golden_passage_recall

which is a DIFFERENT measurement under a deliberately different name. It is not
comparable to the 69.8% recorded for the base model under the old definition,
and no table should place the two in one column. It is valid only between arms
scored by this file.

Usage
    python -m hammer.raft_metrics --answers local_before.jsonl --tag before
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

# Console output is UTF-8 regardless of the terminal's own encoding.
# Windows consoles default to cp1252, which cannot encode the box-drawing and
# check characters used in this project's banners. An unguarded print then
# raises UnicodeEncodeError from inside module import or setup, and the caller
# sees an unrelated failure -- in one case retrieval silently returned zero
# sources and every answer became a refusal.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ── identifier shapes used for both fabrication and fact recall ──────────────
TICKET_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,6}\b")
VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+){0,3}\b")
DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")

QUOTE_RE = re.compile(r"##begin_quote##(.*?)##end_quote##", re.DOTALL)

REFUSAL_PAT = re.compile(
    r"(?:does not (?:contain|support|include|provide|mention)"
    r"|no (?:information|mention|details?|reference)"
    r"|not (?:present|available|found|specified|stated) in the (?:context|provided)"
    r"|cannot (?:be )?(?:answer|determine|find)"
    r"|unable to (?:answer|determine|find)"
    r"|i (?:don'?t|do not) have (?:enough|sufficient|any))",
    re.IGNORECASE)

# A refusal that also points at what IS in the context -- naming a document or
# listing available material -- is the behaviour we want. A bare refusal is not.
HELPFUL_PAT = re.compile(
    r"(?:what is available|however|the context does (?:contain|include)"
    r"|does (?:contain|include|mention)|available information"
    r"|related (?:pages?|documents?|tickets?)|you (?:may|might|could)"
    r"|instead|closest|the context covers)",
    re.IGNORECASE)


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def identifiers(text: str) -> set[str]:
    """Checkable atoms: things that are either in the source or invented."""
    return (set(TICKET_RE.findall(text or ""))
            | set(VERSION_RE.findall(text or ""))
            | set(DATE_RE.findall(text or "")))


def content_terms(text: str) -> set[str]:
    """Distinctive content tokens, for passage-level recall.

    Stopwords and short tokens carry no evidence of grounding, so they are
    dropped; what remains is identifiers plus longer domain words.
    """
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{4,}", text or "")
    stop = {"which", "there", "these", "those", "their", "about", "would",
            "could", "should", "other", "after", "before", "where", "while",
            "being", "using", "please", "refer", "context", "documentation"}
    return {t.lower() for t in toks if t.lower() not in stop} | identifiers(text)


def score(answers: list[dict], payload: dict[str, dict]) -> dict:
    n = 0
    recalls: list[float] = []
    fabricated = refusals = bare = 0
    quotes_total = quotes_grounded = 0
    answers_with_quotes = 0
    lengths: list[int] = []
    gold_present = 0

    for a in answers:
        p = payload.get(a["qid"])
        if not p:
            continue
        n += 1
        ans = a.get("answer") or ""
        ctx = p["context"]
        lengths.append(len(ans))

        ctx_ids = identifiers(ctx)
        ans_ids = identifiers(ans)
        if ans_ids - ctx_ids:
            fabricated += 1

        is_ref = bool(REFUSAL_PAT.search(ans))
        if is_ref:
            refusals += 1
            if not HELPFUL_PAT.search(ans):
                bare += 1

        spans = [s.strip() for s in QUOTE_RE.findall(ans) if s.strip()]
        if spans:
            answers_with_quotes += 1
        nctx = norm_ws(ctx)
        for s in spans:
            quotes_total += 1
            if norm_ws(s) and norm_ws(s) in nctx:
                quotes_grounded += 1

        if p.get("gold_retrieved") and p.get("golden_text"):
            gold_present += 1
            gold_terms = content_terms(p["golden_text"])
            if gold_terms:
                hit = gold_terms & content_terms(ans)
                recalls.append(len(hit) / len(gold_terms))

    pct = lambda x, d: (100.0 * x / d) if d else 0.0
    return {
        "n": n,
        "gold_present": gold_present,
        "golden_passage_recall": round(100 * statistics.mean(recalls), 2) if recalls else 0.0,
        "fabrication_rate": round(pct(fabricated, n), 2),
        "refusal_rate": round(pct(refusals, n), 2),
        "bare_refusal_rate": round(pct(bare, refusals), 2),
        "answers_with_quotes": round(pct(answers_with_quotes, n), 2),
        "quote_spans": quotes_total,
        "quote_grounding": round(pct(quotes_grounded, quotes_total), 2),
        "answer_len_median": int(statistics.median(lengths)) if lengths else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--payload", default="training/export/raft_work/measurable_payload.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    payload = {}
    for line in Path(args.payload).open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            payload[r["qid"]] = r

    answers = [json.loads(l) for l in Path(args.answers).open(encoding="utf-8") if l.strip()]
    m = score(answers, payload)
    m["tag"] = args.tag
    m["model"] = answers[0].get("model") if answers else None

    out = Path(args.out) if args.out else Path(args.answers).parent / f"metrics_{args.tag}.json"
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")

    # ASCII only: this runs under cp1252 consoles on Windows.
    print(f"== {args.tag}  ({m['model']})  n={m['n']} ==")
    for k in ("golden_passage_recall", "fabrication_rate", "refusal_rate",
              "bare_refusal_rate", "answers_with_quotes", "quote_spans",
              "quote_grounding", "answer_len_median"):
        print(f"  {k:24s} {m[k]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
