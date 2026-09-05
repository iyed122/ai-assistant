#!/usr/bin/env python3
"""
shoot_data_panel.py
═══════════════════
Capture docs/images/data_panel.png -- the RAFT candidate pool as the running
application shows it.

The previous capture of this view is unusable in a public document twice over:
its tabs read "DPO Candidates" and "QLoRA Candidates", naming objectives this
project no longer trains, and the rows carried the client's real document
titles. This retakes it against the corrected panel and neutralises the
identifiers in the DOM before the shutter, so the figure still demonstrates that
each candidate carries a score, a grade and an intent without publishing what
the client's pages are called.

Usage
    python docs/shoot_data_panel.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Client vocabulary to neutralise, as an alternating regex. It is read from the
# environment rather than hardcoded, and from the SAME variable the deterministic
# validator uses (hammer/validator.py), so the client's product names live in one
# place and a term added for one purpose is covered by the other. Anything left
# here in source would itself be a disclosure.
PRODUCT_TERMS = os.getenv(
    "VALIDATOR_PRODUCT_TERMS",
    # Fallback shapes only -- generic enough to name no one.
    r"Product-[A-Z]|Platform[- ]?(?:One|Two)")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

OUT = Path(__file__).resolve().parent / "images" / "data_panel.png"
APP = "http://localhost:3000"

HIDE_SIDEBAR = """
() => {
  const st = document.createElement('style');
  st.textContent = '.sidebar { display: none !important; }';
  document.head.appendChild(st);
}
"""

# Same redaction the answer capture uses: ticket keys keep their shape so a
# citation still reads as a citation, and each distinct real key gets its own
# placeholder rather than collapsing to one.
REDACT = """
(terms) => {
  const KEY = /\\b[A-Z][A-Z0-9]{1,6}-\\d{2,6}\\b/g;
  // Any all-caps acronym of 3+ letters is client vocabulary until proven
  // otherwise; a handful of protocol words are kept so the rows still read as
  // engineering questions.
  // This project's own vocabulary, plus ordinary technical words. Everything
  // outside it is treated as the client's and replaced.
  const KEEP = new Set([
    'RAFT','QLORA','LORA','GRPO','SFT','RAG','GOLD','SILVER','BRONZE','FAILED',
    'JSONL','JSON','CSV','PDF','ALL','API','URL','HTTP','HTTPS','SQL','SSE',
    'LLM','GPU','CPU','RAM','VRAM','UI','ID','OK','MLOPS','NF4','SE','AI']);
  const ACRONYM = /\\b[A-Z]{3,10}\\b/g;
  const PRODUCTS = new RegExp('\\\\b(?:' + terms + ')\\\\w*', 'gi');
  const seen = new Map();
  const nextKey = (o) => {
    if (!seen.has(o)) seen.set(o, 'PRJ-' + (1040 + seen.size * 7));
    return seen.get(o);
  };
  const prod = new Map();
  const nextProd = (o) => {
    if (!prod.has(o)) prod.set(o, 'Product-' + 'ABCDEFGH'[prod.size % 8]);
    return prod.get(o);
  };
  let n = 0;
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walk.nextNode()) nodes.push(walk.currentNode);
  for (const t of nodes) {
    const before = t.nodeValue;
    let after = before.replace(KEY, (m) => nextKey(m));
    after = after.replace(PRODUCTS, (m) => nextProd(m));
    after = after.replace(ACRONYM, (m) => KEEP.has(m) ? m : nextProd(m));
    if (after !== before) { t.nodeValue = after; n++; }
  }
  return n;
}
"""


def main() -> None:
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--force-device-scale-factor=1"])
        ctx = b.new_context(viewport={"width": 1918, "height": 884},
                            device_scale_factor=1)
        pg = ctx.new_page()
        pg.goto(APP, wait_until="networkidle", timeout=90000)
        time.sleep(1.5)
        pg.evaluate(HIDE_SIDEBAR)

        pg.evaluate("""() => {
          const b = [...document.querySelectorAll('button')]
            .find(x => /^\\s*Data\\s*$/.test(x.textContent || ''));
          if (b) b.click();
        }""")
        time.sleep(3.0)

        n = pg.evaluate(REDACT, PRODUCT_TERMS)
        sys.stdout.write(f"redacted {n} text node(s)\n")
        time.sleep(0.4)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        pg.screenshot(path=str(OUT))
        sys.stdout.write(f"wrote {OUT}\n")
        ctx.close()
        b.close()


if __name__ == "__main__":
    main()
