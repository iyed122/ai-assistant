#!/usr/bin/env python3
"""
postprocess_demo.py
═══════════════════
Turn the raw Playwright capture into the deliverable clip.

The recorder writes `marks.json`: one entry per scene with its start and end
offset in the source, plus a `speed` factor. Everything is 1.0 except the span
where the local 8B model is decoding, which is real but far too slow to watch.

Rather than cut the generation out -- which would misrepresent the system as
faster than it is -- the span is re-timed. The model genuinely answers on
camera; the viewer is simply not made to sit through 90 s of it.

Segments are cut and re-encoded individually, then concatenated. Doing it that
way (instead of one large filter_complex) keeps each `setpts` independent and
avoids the timestamp drift that shows up when several trims share a graph.

Output: docs/demo_capture/raft_segment.mp4, H.264, 1918x884, 30 fps -- matching
AI-Assistant-demo-final.mp4 so the clip splices without rescaling.

Usage
    python docs/postprocess_demo.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo_capture"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

W, H, FPS = 1918, 884, 30

# Window in AI-Assistant-demo-final.mp4 holding the RAGAS/Hammer scoring run:
# the command, "RAGAS Evaluation Complete", and the dataset-readiness block
# with quality grades and critical failure tags. Set to None to omit.
# The RAGAS beat is now recorded directly (see record_raft_demo.py),
# so no footage is spliced from the earlier demo.
RAGAS_WINDOW = None


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-1200:])


def main() -> None:
    meta = json.loads((OUT / "marks.json").read_text(encoding="utf-8"))
    src = OUT / meta["source"]
    marks = meta["marks"]
    if not src.exists():
        raise SystemExit(f"source video missing: {src}")

    parts_dir = OUT / "_parts"
    parts_dir.mkdir(exist_ok=True)
    for old in parts_dir.glob("*.mp4"):
        old.unlink()

    listing = []
    for i, m in enumerate(marks):
        start, end, speed = m["start"], m["end"], m.get("speed", 1.0)
        dur = end - start
        if dur <= 0.15:
            continue

        # A span may name a target duration instead of a speed factor. That is
        # the right control for the generation wait, whose real length swings
        # from 8 s to 250 s depending on cache state: the viewer should always
        # see the same short beat. The UI shows a static "Generating
        # response..." placeholder rather than streaming tokens, so there is
        # nothing gained by leaving more of it on screen.
        target = m.get("target")
        if target:
            speed = max(1.0, dur / float(target))
        elif dur < 20.0:
            speed = 1.0
        elif speed > 1.0:
            # keep the result watchable: never below ~8 s on screen
            speed = min(speed, max(1.0, dur / 8.0))
        part = parts_dir / f"{i:02d}_{m['label']}.mp4"

        # setpts scales presentation timestamps: 1/speed compresses the span.
        vf = f"setpts={1.0/speed:.6f}*PTS,fps={FPS}"
        run([FFMPEG, "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
             "-vf", vf, "-an",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             "-preset", "medium", str(part)])
        listing.append(part)
        tag = f"  x{speed:g}" if speed != 1.0 else ""
        print(f"  {m['label']:16s} {dur:6.1f}s -> {dur/speed:6.1f}s{tag}")

    # ── splice in the RAGAS evaluation from the existing demo ─────────────
    # The Hammer/RAGAS scoring runs in a terminal, which a browser recorder
    # cannot capture. Rather than restage it, the real footage from
    # AI-Assistant-demo-final.mp4 is cut in directly -- same 1918x884 @ 30 fps,
    # so it concatenates without rescaling. It lands immediately after the
    # inference block, which is where the evaluation belongs in the narrative.
    ragas_src = ROOT / "AI-Assistant-demo-final.mp4"
    if ragas_src.exists() and RAGAS_WINDOW:
        rs, re_ = RAGAS_WINDOW
        ragas_part = parts_dir / "__ragas.mp4"
        run([FFMPEG, "-y", "-ss", f"{rs:.3f}", "-t", f"{re_ - rs:.3f}",
             "-i", str(ragas_src), "-vf", f"fps={FPS}", "-an",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             "-preset", "medium", str(ragas_part)])
        # insert after the last inference-side scene
        anchor = next((k for k, pp in enumerate(listing)
                       if "answer_hold" in pp.name), None)
        if anchor is not None:
            listing.insert(anchor + 1, ragas_part)
        else:
            listing.append(ragas_part)
        print(f"  {'ragas (spliced)':16s} {re_ - rs:6.1f}s -> {re_ - rs:6.1f}s")

    concat = OUT / "_concat.txt"
    concat.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in listing), encoding="utf-8")

    final = OUT / "raft_segment.mp4"
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-c", "copy", str(final)])

    probe = subprocess.run([FFMPEG, "-i", str(final)],
                           capture_output=True, text=True, errors="replace")
    for line in (probe.stderr or "").splitlines():
        if "Duration" in line or "Video:" in line:
            print(" ", line.strip())
    print(f"\nwrote {final}")


if __name__ == "__main__":
    main()
