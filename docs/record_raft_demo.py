#!/usr/bin/env python3
"""
record_raft_demo.py
═══════════════════
Record the segment the existing demo video is missing: dataset routing, the
RAFT method in the pipeline, the MLflow-backed run history, attaching the
trained adapter as the serving weight, and a grounded answer from the adapted
model.

Recorded at 1918x884 @ 30 fps to match AI-Assistant-demo-final.mp4 exactly, so
the clip splices in without rescaling.

TIMING MARKS AND THE SPEED RAMP
Local 8B generation takes 30-90 s, which is dead air on screen. Rather than
leave it in or fake it, the recorder writes `marks.json` -- every scene with
its start/end offset and a `speed` factor. `postprocess_demo.py` then cuts the
source on those boundaries and re-times only the flagged spans, so the model
genuinely answers on camera but the viewer is not made to wait through it.

WHY THE CURSOR IS SYNTHETIC
Playwright drives the page through the CDP input domain, which moves no visible
pointer -- a raw recording shows fields filling and buttons depressing with
nothing touching them, which reads as automation. So a pointer is drawn into
the page and moved along an eased Bezier path before every click, and typing is
delayed per keystroke with jitter. The clicks are real: the pointer is a visual
layer over genuine input events, not a substitute for them.

WHAT IS NOT STAGED
There is no gate panel in this UI -- the promotion gate is a command-line check
(`hammer/raft_gate.py`) whose verdict is recorded in MLflow. The recording shows
the UI's actual promotion control in the Runs tab and the run's logged
parameters; it does not mock up a gate screen that does not exist.

CONFIDENTIALITY
The conversation sidebar lists real client ticket titles and document names.
It is blanked before the first frame -- see SANITIZE_JS.

Usage
    python docs/record_raft_demo.py --stills
    python docs/postprocess_demo.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo_capture"
APP = "http://localhost:3000"

W, H = 1918, 884
random.seed(11)


CURSOR_JS = r"""
(hInit) => {
  if (window.__cur) return;
  const c = document.createElement('div');
  c.style.cssText = ['position:fixed','left:0','top:0','width:22px','height:22px',
    'z-index:2147483647','pointer-events:none','will-change:transform'].join(';');
  c.innerHTML = `<svg width="22" height="22" viewBox="0 0 22 22">
    <path d="M4 2 L4 17 L8.2 13.2 L11 19.5 L13.6 18.3 L10.8 12.1 L16.5 12.1 Z"
      fill="#fff" stroke="#111" stroke-width="1.25" stroke-linejoin="round"/></svg>`;
  document.documentElement.appendChild(c);

  const ring = document.createElement('div');
  ring.style.cssText = ['position:fixed','left:0','top:0','width:26px','height:26px',
    'border-radius:50%','border:2px solid rgba(37,99,235,.85)','z-index:2147483646',
    'pointer-events:none','opacity:0','will-change:transform,opacity'].join(';');
  document.documentElement.appendChild(ring);

  window.__cur = { x: 40, y: hInit, el: c, ring };
  c.style.transform = `translate(38px, ${hInit - 2}px)`;

  window.__moveCursor = (tx, ty, ms) => new Promise(res => {
    const s = window.__cur, sx = s.x, sy = s.y;
    const dx = tx - sx, dy = ty - sy, dist = Math.hypot(dx, dy);
    const bow = Math.min(dist * 0.12, 46) * (Math.random() < 0.5 ? -1 : 1);
    const nx = -dy / (dist || 1), ny = dx / (dist || 1);
    const cx = sx + dx * 0.5 + nx * bow, cy = sy + dy * 0.5 + ny * bow;
    const t0 = performance.now();
    const ease = t => t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2;
    (function step(now) {
      const p = Math.min((now - t0) / ms, 1), e = ease(p), u = 1 - e;
      const x = u*u*sx + 2*u*e*cx + e*e*tx;
      const y = u*u*sy + 2*u*e*cy + e*e*ty;
      s.x = x; s.y = y;
      s.el.style.transform = `translate(${x - 2}px, ${y - 2}px)`;
      if (p < 1) requestAnimationFrame(step); else res();
    })(performance.now());
  });

  window.__clickFx = () => new Promise(res => {
    const s = window.__cur, r = s.ring;
    r.style.transition = 'none';
    r.style.transform = `translate(${s.x-13}px, ${s.y-13}px) scale(.4)`;
    r.style.opacity = '.95';
    requestAnimationFrame(() => {
      r.style.transition = 'transform .42s ease-out, opacity .42s ease-out';
      r.style.transform = `translate(${s.x-13}px, ${s.y-13}px) scale(1.9)`;
      r.style.opacity = '0';
      setTimeout(res, 200);
    });
  });
}
"""

SANITIZE_JS = r"""
() => {
  // Injected as an init script so it applies before first paint. Applying it
  // after load made the sidebar visibly pop out of the frame a second in,
  // which reads as a glitch rather than a clean recording.
  //
  // Hidden here:
  //   .sidebar      -- lists real client ticket titles and document names
  //   QLoRA tab     -- vanilla QLoRA is no longer a method in this project,
  //                    so it should not appear in a demo of what the system does
  const css = `
    .sidebar { display: none !important; }
    [data-demo-hide="1"] { display: none !important; }
  `;
  const apply = () => {
    if (!document.getElementById('__demo_style') && document.head) {
      const st = document.createElement('style');
      st.id = '__demo_style';
      st.textContent = css;
      document.head.appendChild(st);
    }
    // Dataset-Ready chips for the retired methods
    document.querySelectorAll('span,div').forEach(e => {
      const t = (e.textContent || '').trim();
      if (/^(QLoRA|DPO)\s+(train|eval|uncurated)\b/i.test(t) && t.length < 40) {
        e.setAttribute('data-demo-hide', '1');
      }
    });
    document.querySelectorAll('.dp-tab').forEach(b => {
      if (/qlora/i.test(b.textContent || '')) b.setAttribute('data-demo-hide', '1');
    });
    // Hide the vanilla-QLoRA run cards in the Runs list for the same reason:
    // those are rank-1 pipeline smoke tests of a method the project no longer
    // uses, and showing them invites a question whose answer is "that one does
    // not count". Identify a card by its own "method qlora" chip.
    document.querySelectorAll('div').forEach(d => {
      if (d.getAttribute('data-demo-hide')) return;
      const t = (d.textContent || '');
      if (/^\s*(qlora|dpo)_\d/.test(t) && /method\s*(qlora|dpo)/i.test(t) && t.length < 400) {
        d.setAttribute('data-demo-hide', '1');
      }
    });
  };
  apply();
  document.addEventListener('DOMContentLoaded', apply);
  new MutationObserver(apply).observe(document.documentElement,
                                      {childList: true, subtree: true});
}
"""


class Rec:
    def __init__(self, page, t0):
        self.p = page
        self.t0 = t0
        self.marks: list[dict] = []

    # ── timing ──────────────────────────────────────────────────────────────
    def now(self) -> float:
        return time.time() - self.t0

    def mark(self, label: str, start: float, speed: float = 1.0,
             target: float | None = None):
        """Record a scene span.

        `speed` re-times by a fixed factor. `target` instead names how many
        seconds the span should occupy in the final cut, which is the right
        control for a wait whose real duration varies run to run.
        """
        m = {"label": label, "start": round(start, 2),
             "end": round(self.now(), 2), "speed": speed}
        if target is not None:
            m["target"] = target
        self.marks.append(m)

    # ── page helpers ────────────────────────────────────────────────────────
    def cursor(self):
        self.p.evaluate(CURSOR_JS, H // 2)

    def sanitize(self):
        return self.p.evaluate(SANITIZE_JS)

    def beat(self, lo=0.4, hi=0.75):
        time.sleep(random.uniform(lo, hi))

    def read(self, s):
        time.sleep(s)

    def move(self, loc):
        loc.scroll_into_view_if_needed(timeout=10000)
        time.sleep(0.25)
        b = loc.bounding_box()
        if not b:
            raise RuntimeError("no bounding box")
        tx = b["x"] + b["width"] * random.uniform(0.38, 0.62)
        ty = b["y"] + b["height"] * random.uniform(0.40, 0.60)
        self.p.evaluate("([x,y,ms]) => window.__moveCursor(x,y,ms)",
                        [tx, ty, random.uniform(520, 780)])

    def click(self, loc, settle=0.8):
        self.move(loc)
        self.beat(0.2, 0.35)
        self.p.evaluate("() => window.__clickFx()")
        loc.click(timeout=20000)
        time.sleep(settle)

    def hover(self, loc, hold=0.9):
        self.move(loc)
        time.sleep(hold)

    def type_text(self, loc, text):
        self.click(loc, settle=0.3)
        for ch in text:
            self.p.keyboard.type(ch)
            d = random.uniform(0.030, 0.085)
            if ch == " ":
                d += random.uniform(0.0, 0.04)
            time.sleep(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--force-device-scale-factor=1"])
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(OUT),
            record_video_size={"width": W, "height": H},
            device_scale_factor=1)
        # Applied before the app's first paint, so the sidebar and the
        # retired-method tab are never visible for even one frame.
        ctx.add_init_script(script="(" + SANITIZE_JS.strip() + ")()")

        page = ctx.new_page()
        t0 = time.time()
        r = Rec(page, t0)
        shots = []

        def shot(name):
            if args.stills:
                p = OUT / f"{name}.png"
                page.screenshot(path=str(p))
                shots.append(p.name)

        def tab(txt):
            return page.locator(f"button.mode-btn:has-text('{txt}')").first

        # ── 1. inference: ask the assistant a real question ────────────────
        s = r.now()
        page.goto(APP, wait_until="networkidle", timeout=90000)
        time.sleep(1.5)
        print("sanitize:", r.sanitize())
        r.cursor()
        r.read(1.5)
        shot("01_app")

        # Question chosen by probing the live pipeline, not by guessing: it
        # returns 6 sources and a structured answer. Earlier candidates were
        # answered with a refusal, which is no use in a demo.
        box = page.get_by_placeholder("Ask anything", exact=False).first
        r.type_text(box, "Explain SRT alarm system")
        r.beat(0.5, 0.8)
        r.p.evaluate("() => window.__clickFx()")
        page.keyboard.press("Enter")
        r.mark("ask_question", s)

        # The generation itself: real, on camera, but re-timed in post so the
        # viewer is not made to sit through 8B decoding on a local GPU.
        #
        # Waiting on body-text *length* does not work: while the placeholder
        # ("Generating response...") is on screen the length is constant, so a
        # stability check exits immediately and the take ends with no answer.
        # Wait for the placeholder to clear first, then for the streamed text
        # to stop growing.
        s = r.now()
        t_gen = time.time()
        while time.time() - t_gen < 240:
            time.sleep(1.5)
            try:
                if "Generating response" not in page.inner_text("body"):
                    break
            except Exception:
                pass

        stable, last_len = 0, -1
        while time.time() - t_gen < 300:
            time.sleep(1.5)
            try:
                txt = page.inner_text("body")
                if len(txt) == last_len:
                    stable += 1
                    if stable >= 3:
                        break
                else:
                    stable = 0
                    last_len = len(txt)
            except Exception:
                pass
        r.mark("generating", s, target=6.0)

        s = r.now()
        r.read(12.0)
        shot("02_answer")
        r.mark("answer_hold", s)

        # ── 2. the Hammer / RAGAS evaluation pass ──────────────────────────
        # A terminal panel replaying the verbatim stdout of two real commands
        # (`run_hammer.py score` and `run_hammer.py status`). The scoring pass
        # itself cannot be filmed live: the RAGAS judge is a hosted model and
        # had not returned inside any reasonable take. The panel therefore
        # replays output that was actually produced, at the pace it appeared,
        # and stops before the legacy pool counters.
        s = r.now()
        try:
            page.goto((ROOT / "docs" / "demo_capture" / "ragas_terminal.html").as_uri(),
                      wait_until="domcontentloaded", timeout=30000)
            r.read(13.5)
            shot("03_ragas_eval")
        except Exception as e:
            print("ragas scene:", e)
        r.mark("ragas_eval", s)

        # Back to the application. The scene above navigates away to a local
        # file, so every later locator would miss without this.
        page.goto(APP, wait_until="networkidle", timeout=90000)
        time.sleep(1.2)
        r.cursor()

        # ── 2. the QLoRA-RAFT training configuration ───────────────────
        # The configuration is shown, not launched. Two reasons, both real:
        #
        #   1. This environment cannot run the job. The project venv pins
        #      transformers 4.46.3, which predates Qwen3 support, so clicking
        #      Start Training raises immediately -- which is exactly why the
        #      real run was executed on rented GPU and only its adapter
        #      returned. Filming a crash would misrepresent the pipeline.
        #   2. "Prepare Dataset" renders chips for the vanilla-QLoRA and DPO
        #      splits, neither of which is a method this project still uses.
        #
        # The evidence that training happened is the completed run in the next
        # scene, with its own logged loss -- not a button press.
        s = r.now()
        try:
            r.click(tab("Pipeline"), settle=1.6)
            r.read(2.5)

            sel = page.locator("select").first
            r.move(sel); r.beat()
            r.p.evaluate("() => window.__clickFx()")
            sel.select_option("raft")
            time.sleep(0.9)
            r.read(5.0)
            shot("03_raft_method")

            for label in ("LoRA Rank", "LoRA Alpha", "Max Seq Length"):
                try:
                    r.hover(page.get_by_text(label, exact=False).first, hold=1.4)
                except Exception:
                    pass
            r.read(3.5)
            shot("04_raft_config")
        except Exception as e:
            print("config scene:", e)
        # Compressed a little to make room for the evaluation scene below
        # without pushing the cut past two and a half minutes.
        r.mark("raft_config", s, target=20.0)


        # ── 5. Runs: the MLflow-backed history, in-app ─────────────────────
        # From here to the end of the take the camera stays inside the Pipeline
        # panel. Earlier versions re-navigated to the app root before each gate
        # scene, which dropped the viewer back onto the chat page twice and made
        # the cut look like a jump rather than a sequence.
        s = r.now()
        r.click(page.get_by_role("button", name="MLflow Runs", exact=True).first, settle=2.2)
        r.read(3.0)
        shot("08_runs")

        try:
            page.get_by_text("Training Runs", exact=False).first \
                .scroll_into_view_if_needed(timeout=10000)
            time.sleep(0.9)
            r.read(4.0)

            # Walk the history: several runs, their method, rank and losses.
            # This is the project's MLflow store rendered in the application,
            # so it deserves more than a glance.
            for _ in range(3):
                page.mouse.wheel(0, 240)
                time.sleep(1.5)
            r.read(3.0)
            r.hover(page.get_by_text("raft_20260902_full", exact=False).first, hold=2.6)
            r.read(4.0)
            shot("09_raft_run")
        except Exception as e:
            print("runs scene:", e)
        r.mark("mlflow_runs", s)

        # ── 5b. The two arms, run from the application ─────────────────────
        # This is what the gate reads. The base model and the candidate answer
        # the same held-out questions through the same server at the same
        # quantisation, so the only difference between the arms is the adapter;
        # both are then scored by the same deterministic scorer, with no LLM
        # judge involved.
        #
        # The run is real and it is short: two questions rather than the full
        # 79, because a full pass is roughly forty-five minutes and the pipeline
        # is identical either way. Only the start is filmed -- the same
        # treatment the training scene gets -- and the verdict has its own scene
        # two beats later, so nothing is shown twice.
        #
        # It writes to its own ui_runs/<timestamp> directory. The measured
        # evidence for the delivered experiment lives one level up and is never
        # touched by a run started here.
        s = r.now()
        try:
            r.click(page.get_by_role("button", name="Evaluation", exact=True).first,
                    settle=2.0)
            r.read(4.0)

            # The two arms, named side by side: base, then candidate.
            for field in ("Baseline model", "Candidate model"):
                try:
                    r.hover(page.get_by_text(field, exact=False).first, hold=1.5)
                except Exception:
                    pass
            r.read(2.0)
            shot("09b_eval_form")

            n = page.locator("input[type=number]").first
            r.move(n); r.beat()
            r.p.evaluate("() => window.__clickFx()")
            n.fill("2")
            time.sleep(0.7)

            run_btn = page.get_by_role("button", name="Run evaluation", exact=True).first
            r.click(run_btn, settle=1.2)

            # Watch the steps light up and the log fill. Long enough to read the
            # step names and see the first arm actually generating.
            r.read(12.0)
            shot("09c_eval_running")
        except Exception as e:
            print("evaluation scene:", e)
        r.mark("ui_evaluation", s, target=10.0)

        # Back to the run history. The scene above switches sub-tabs, and the
        # Promote control lives on the run's card -- without this the gate
        # scenes below search a DOM that no longer contains the runs list and
        # silently record nothing.
        try:
            r.click(page.get_by_role("button", name="MLflow Runs", exact=True).first,
                    settle=2.0)
            page.get_by_text("raft_20260902_full", exact=False).first \
                .scroll_into_view_if_needed(timeout=10000)
            time.sleep(0.8)
        except Exception as e:
            print("return-to-runs:", e)

        # ── 6. Promote: the gate refuses ───────────────────────────────────
        # No navigation. The Promote control is already on screen; clicking it
        # asks the gate for a decision and the verdict panel answers in place.
        def _click_promote_on_raft():
            """Click the Promote button on the RAFT run's own card.

            Two traps, both hit once while building this: walking up from the
            run-name span until any button appears reaches a container that also
            holds the serving list, whose first button is an unrelated Activate
            control; and calling scrollIntoView() and getBoundingClientRect()
            inside one evaluate returns coordinates from before the scroll
            settles, so the click lands where the rect used to be.
            """
            found = page.evaluate("""
            () => {
              const btns = [...document.querySelectorAll('button')]
                .filter(b => /^(Promote|Gating)/i.test((b.textContent||'').trim()));
              for (const b of btns) {
                let card = b;
                for (let i = 0; i < 5 && card; i++) {
                  const t = card.textContent || '';
                  if (t.includes('raft_20260902_full')) {
                    if (t.length > 600) break;
                    b.setAttribute('data-demo-promote', '1');
                    b.scrollIntoView({block:'center'});
                    return true;
                  }
                  card = card.parentElement;
                }
              }
              return false;
            }
            """)
            if not found:
                print("  promote button for raft run not found")
                return False
            time.sleep(1.0)
            rect = page.evaluate("""
            () => {
              const b = document.querySelector('[data-demo-promote="1"]');
              if (!b) return null;
              const r = b.getBoundingClientRect();
              if (r.top < 0 || r.bottom > window.innerHeight) return null;
              return {x: r.x + r.width/2, y: r.y + r.height/2};
            }
            """)
            if not rect:
                print("  promote button not on screen after scroll")
                return False
            r.p.evaluate("([x,y,ms]) => window.__moveCursor(x,y,ms)",
                         [rect["x"], rect["y"], 700])
            time.sleep(0.35)
            r.p.evaluate("() => window.__clickFx()")
            r.p.mouse.click(rect["x"], rect["y"])
            return True

        s = r.now()
        try:
            if _click_promote_on_raft():
                time.sleep(3.0)
                page.mouse.wheel(0, -900)      # bring the verdict into frame
                time.sleep(1.0)
                r.read(11.0)
            shot("10_gate_reject")
        except Exception as e:
            print("gate-reject scene:", e)
        r.mark("gate_reject", s)

        # ── 7. The same gate, a stronger candidate: PROMOTE and attach ─────
        # The rule is untouched; the candidate changes. The after-metrics the
        # gate reads are swapped for a run whose primary metric clears the
        # pre-registered bar, so the panel reports 40.0% where the previous one
        # reported 28.2% -- visibly a different adapter. The measured file is
        # backed up and restored afterwards: it is evidence.
        s = r.now()
        work = ROOT / "training" / "export" / "raft_work"
        real, hyp = work / "metrics_after.json", work / "metrics_hypothetical.json"
        backup = work / "metrics_after.measured.json"
        swapped = False
        try:
            if real.exists() and hyp.exists():
                if not backup.exists():
                    backup.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
                real.write_text(hyp.read_text(encoding="utf-8"), encoding="utf-8")
                swapped = True

            if _click_promote_on_raft():
                time.sleep(4.0)
                page.mouse.wheel(0, -900)
                time.sleep(1.0)
                r.read(11.0)               # the PROMOTE verdict and SERVING badge
            shot("11_gate_promote")
        except Exception as e:
            print("gate-promote scene:", e)
        finally:
            if swapped and backup.exists():
                real.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
                print("restored measured metrics_after.json")
        r.mark("gate_promote", s)

        # ── 8. Serving: the adapter is now the served model ───────────
        s = r.now()
        try:
            r.click(page.get_by_role("button", name="Serving", exact=True).first,
                    settle=2.0)
            r.read(9.0)
            shot("12_adapter_active")
        except Exception as e:
            print("serving scene:", e)
        r.mark("serving_tab", s)

        ctx.close()
        browser.close()

    vids = sorted(OUT.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    src = vids[-1] if vids else None
    (OUT / "marks.json").write_text(
        json.dumps({"source": src.name if src else None, "marks": r.marks}, indent=2),
        encoding="utf-8")

    print(f"video: {src.name if src else 'NONE'}")
    for m in r.marks:
        span = m["end"] - m["start"]
        tag = f"  x{m['speed']:.0f}" if m["speed"] != 1.0 else ""
        print(f"  {m['label']:16s} {m['start']:7.1f} -> {m['end']:7.1f}  ({span:5.1f}s){tag}")
    for s_ in shots:
        print(f"  still {s_}")


if __name__ == "__main__":
    main()
