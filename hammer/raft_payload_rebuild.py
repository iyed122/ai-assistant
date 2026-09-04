#!/usr/bin/env python3
"""
raft_payload_rebuild.py
═══════════════════════
Reconstruct the RAFT holdout evaluation payload from the pre-rendered prompts.

WHY THIS EXISTS
The original payload (holdout_payload.jsonl, carrying per-question `context`,
`evidence` and `golden` fields) lived only in a session scratch directory under
%TEMP% and was destroyed by Windows temp cleanup. The generator that produced it
(raft_pool.jsonl) went with it, and the pool was never persisted to Mongo -- the
sibling collections kb_synth.doc_bench and knowledge_base.synth_eval share the
question *style* but have zero qid and zero question-text overlap with this
holdout, so they cannot stand in for it.

What did survive is ds_holdout/kaggle_prompts.jsonl: 500 fully rendered prompts.
Those prompts are sufficient, because the generator embedded everything the
metrics need directly in the prompt text:

  * the retrieved context, block by block, each headed "[source] Title"
  * the question, which quotes the golden document's title verbatim
    e.g.  What does the "Default Passwords" documentation state about ...?

So gold identity is recoverable by parsing, not by guessing.

WHAT IS FAITHFUL AND WHAT IS NOT
  faithful      context blocks, block titles, the question, gold_title,
                gold_retrieved (whether the golden block is in the context)
  reconstructed golden_text -- the full golden passage, where the original
                payload held a single extracted evidence sentence.

That difference matters and is not papered over. Recall scored against a whole
passage is a different measurement from recall scored against one sentence, so
the previously recorded before-arm figure (69.8%) is NOT comparable to anything
computed here. The only sound use of this payload is to score both arms afresh
under one definition; see docs/RAFT_EVALUATION_RUNBOOK.md.

Usage
    python -m hammer.raft_payload_rebuild \
        --prompts training/export/kaggle_migrate/ds_holdout/kaggle_prompts.jsonl \
        --qids    training/export/kaggle_migrate/kk_after/measurable_qids.json \
        --out     training/export/raft_work/measurable_payload.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

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


# A context block starts at a line like "[confluence] Release 4.1.20.8" and runs
# until the "---" separator that the renderer writes between blocks.
BLOCK_RE = re.compile(r"^\[([a-z_]+)\]\s+(.+?)\s*$", re.MULTILINE)

# The question always quotes the golden document's title. Both straight and
# typographic quotes appear in the corpus, so accept either.
TITLE_RE = re.compile(r"[\"“]([^\"”]{2,200})[\"”]")

CONTEXT_START = "CONTEXT (retrieved from the knowledge base):"
QUESTION_MARK = "\nQUESTION: "


def _norm(s: str) -> str:
    """Loose title comparison: case, punctuation and spacing are all noise here."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def parse_prompt(prompt: str) -> dict:
    """Split a rendered prompt back into its context blocks and its question."""
    ctx_start = prompt.find(CONTEXT_START)
    q_start = prompt.find(QUESTION_MARK)
    if ctx_start < 0 or q_start < 0:
        raise ValueError("prompt does not have the expected context/question shape")

    context = prompt[ctx_start + len(CONTEXT_START):q_start].strip()
    question = prompt[q_start + len(QUESTION_MARK):].split("\n", 1)[0].strip()

    blocks = []
    matches = list(BLOCK_RE.finditer(context))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        body = context[m.end():end]
        # Drop the "---" separator and the trailing "Tags:" line the renderer adds.
        body = re.sub(r"\n-{3,}\s*$", "", body).strip()
        blocks.append({"source": m.group(1), "title": m.group(2).strip(), "text": body})

    return {"context": context, "question": question, "blocks": blocks}


def find_golden(question: str, blocks: list[dict]) -> tuple[Optional[str], Optional[str]]:
    """Return (gold_title, golden_text) by matching the question's quoted title.

    The question may quote several strings; the golden one is whichever quoted
    string names a block actually present in the context. Preference goes to an
    exact normalised match, then to a containment match, because some titles are
    quoted in shortened form.
    """
    quoted = TITLE_RE.findall(question)
    if not quoted:
        return None, None

    by_norm = {_norm(b["title"]): b for b in blocks}

    for q in quoted:                                   # exact title match
        b = by_norm.get(_norm(q))
        if b:
            return b["title"], b["text"]

    for q in quoted:                                   # containment either way
        nq = _norm(q)
        if not nq:
            continue
        for nb, b in by_norm.items():
            if nq and (nq in nb or nb in nq):
                return b["title"], b["text"]

    # Quoted but absent: the question names a document that retrieval did not
    # return. That is a real and important case -- the honest answer is a
    # refusal -- so record the title with no text rather than dropping the row.
    return quoted[0], None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--qids", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.prompts).open(encoding="utf-8") if l.strip()]
    if args.qids:
        keep = set(json.loads(Path(args.qids).read_text(encoding="utf-8")))
        rows = [r for r in rows if r["qid"] in keep]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_gold = n_present = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            p = parse_prompt(r["prompt"])
            question = r.get("question") or p["question"]
            gold_title, golden_text = find_golden(question, p["blocks"])
            n_gold += gold_title is not None
            n_present += golden_text is not None
            fh.write(json.dumps({
                "qid": r["qid"],
                "uid": r.get("uid"),
                "question": question,
                "prompt": r["prompt"],
                "context": p["context"],
                "blocks": [{"source": b["source"], "title": b["title"]} for b in p["blocks"]],
                "n_blocks": len(p["blocks"]),
                "gold_title": gold_title,
                "gold_retrieved": golden_text is not None,
                "golden_text": golden_text or "",
            }, ensure_ascii=False) + "\n")

    print(f"wrote {out_path}  rows={len(rows)}")
    print(f"  gold_title parsed : {n_gold}/{len(rows)}")
    print(f"  golden in context : {n_present}/{len(rows)}")


if __name__ == "__main__":
    main()
