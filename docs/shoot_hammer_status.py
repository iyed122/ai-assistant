#!/usr/bin/env python3
"""
shoot_hammer_status.py
══════════════════════
Re-capture docs/images/hammer_status.png from a live run of the evaluator's
status command.

WHY THIS EXISTS
The previous capture was taken by hand and had gone stale in a way that mattered:
its training-pool block read

    QLoRA (GOLD)    : 59
    DPO candidates  : 0
    qlora_positive  : 59
    dpo_rejected    : 74

naming two training objectives this project no longer uses -- and that image is
reproduced both in the report (Figure "hammer_status") and on a slide, so the
stale words were on the two surfaces a reader actually sees. The CLI itself has
since been corrected; this script re-shoots the figure from its real output so
the two cannot drift apart again.

Nothing here is typed by hand. The console text is whatever the command printed
on this machine, rendered in a terminal frame at the size the figure pipeline
expects (1577x1007, with a 42px title bar that make_ui_figures.py crops).

Usage
    python docs/shoot_hammer_status.py
"""

from __future__ import annotations

import html
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "venv_hammer" / "Scripts" / "python.exe"
OUT = ROOT / "docs" / "images" / "hammer_status.png"

W, H, BAR = 1577, 1007, 42

CSS = """
:root { --term:#141414; --bar:#2d2d2d; --txt:#cccccc; --dim:#6a6a6a;
        --grn:#6a9955; --cyn:#9cdcfe; --yel:#c8a24a; --red:#c05c5c; }
html,body{margin:0;padding:0;width:%(w)dpx;height:%(h)dpx;overflow:hidden;
          background:var(--term);}
.win{position:absolute;left:0;top:0;width:%(w)dpx;height:%(h)dpx;overflow:hidden;}
.bar{height:%(bar)dpx;background:var(--bar);display:flex;align-items:center;
     padding:0 14px;gap:9px;border-bottom:1px solid #2a2a2a;box-sizing:border-box;}
.dot{width:11px;height:11px;border-radius:50%%;background:#4a4a4a}
.title{color:var(--dim);font:12.5px/1 'Segoe UI',system-ui;margin-left:12px;}
.body{background:var(--term);padding:14px 26px;height:%(body)dpx;box-sizing:border-box;
      font:13.5px/1.48 'Cascadia Mono','Consolas',monospace;color:var(--txt);
      white-space:pre;overflow:hidden;}
.rule{color:#454545}.dimc{color:var(--dim)}.hi{color:var(--yel)}
.crit{color:var(--red)}.ok{color:var(--grn)}.lab{color:var(--cyn)}
"""


def colourise(line: str) -> str:
    """Colour the console line the way a terminal would. Text is never altered."""
    e = html.escape(line)
    if set(line.strip()) and set(line.strip()) <= set("═─"):
        return f'<span class="rule">{e}</span>'
    if "WARNING" in line or "REGRESSION DETECTED" in line:
        return f'<span class="hi">{e}</span>'
    if "[CRITICAL]" in line:
        return f'<span class="crit">{e}</span>'
    if "DEPLOYED" in line or line.strip().startswith("✓"):
        return f'<span class="ok">{e}</span>'
    if line.strip().endswith(":") or "HAMMER STATUS" in line:
        return f'<span class="lab">{e}</span>'
    return e


def main() -> None:
    proc = subprocess.run(
        [str(PY), str(ROOT / "hammer" / "run_hammer.py"), "status"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900)
    text = (proc.stdout or "") + (proc.stderr or "")
    lines = [l.rstrip() for l in text.splitlines()]
    # Drop the logger preamble; the figure is the status report itself.
    start = next((i for i, l in enumerate(lines) if "HAMMER STATUS" in l), 0)
    lines = lines[max(0, start - 1):]
    if not lines:
        raise SystemExit("status produced no output")

    body = "\n".join(colourise(l) for l in lines)
    doc = (f"<style>{CSS % {'w': W, 'h': H, 'bar': BAR, 'body': H - BAR}}</style>"
           '<div class="win"><div class="bar">'
           '<div class="dot"></div><div class="dot"></div><div class="dot"></div>'
           '<div class="title">Windows PowerShell &nbsp;—&nbsp; AI Assistant '
           '&nbsp;·&nbsp; hammer status</div></div>'
           f'<div class="body">{body}</div></div>')

    tmp = Path(tempfile.gettempdir()) / "hammer_status_frame.html"
    tmp.write_text(doc, encoding="utf-8")

    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--force-device-scale-factor=1"])
        pg = b.new_context(viewport={"width": W, "height": H},
                           device_scale_factor=1).new_page()
        pg.goto(tmp.as_uri(), wait_until="load")
        pg.wait_for_timeout(400)
        pg.screenshot(path=str(OUT))
        b.close()

    print(f"wrote {OUT}  ({len(lines)} console lines)")


if __name__ == "__main__":
    main()
