#!/usr/bin/env python3
"""
make_architecture.py
════════════════════
Draw the system architecture diagram.

WHY THIS EXISTS
The previous architecture image had no generator in the repository. Its labels
were baked in as glyph outlines, so the only way to correct a word was to redraw
the whole thing by hand -- and it had gone stale in three ways that mattered:

  * the training stage read "QLoRA / DPO", naming an objective this project no
    longer uses and a dataset shape it no longer builds
  * it put "6 GB" on the training box, but the delivered 8B run did not fit on
    the 6 GB card and executed on rented GPU; only its adapter came back
  * the promotion arrow went straight from the registry to the served model.
    The gate -- the component the whole experiment turns on -- was not on the
    path at all, which is precisely the confusion this diagram should prevent.

Everything drawn here is the system as built. The gate has two outcomes and both
are shown, because a gate drawn with only its success branch is a pipeline.

Usage
    python docs/make_architecture.py
writes docs/images/architecture.png and docs/images/architecture.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

OUT = Path(__file__).resolve().parent / "images"

# Palette -- carried over from the previous diagram so the figure stays
# recognisable to anyone who has seen the earlier version.
INK = "#0f172a"
GREY_F, GREY_E = "#eef1f5", "#94a3b8"
BLUE_F, BLUE_E = "#e8f0fe", "#3b82f6"
TEAL_F, TEAL_E = "#dcf5f0", "#0d9488"
GRN_F,  GRN_E = "#e6f7ec", "#16a34a"
AMB_F,  AMB_E = "#fef3d7", "#f59e0b"
PUR_F,  PUR_E = "#efe9fd", "#7c3aed"
RED_F,  RED_E = "#fdeaea", "#ef4444"

BAND_ON = "#f2fbf5"
BAND_OFF = "#fdf6ec"


def box(ax, x, y, w, h, text, fc, ec, *, fs=10.5, weight="bold", tc=None, r=0.9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc or INK, zorder=4,
            linespacing=1.45)
    return (x, y, w, h)


def band(ax, x, y, w, h, fc):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
        facecolor=fc, edgecolor="none", zorder=1))


def arrow(ax, p0, p1, color, *, style="-|>", lw=1.9, rad=0.0, ls="-", z=5):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=15, linewidth=lw,
        color=color, linestyle=ls, zorder=z,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def label(ax, x, y, s, color, *, fs=9, style="italic", weight="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, style=style,
            color=color, fontweight=weight, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9))


def main() -> None:
    fig, ax = plt.subplots(figsize=(16.0, 10.4), dpi=160)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(50, 96.5, "AI Assistant — System Architecture", ha="center",
            va="center", fontsize=22, fontweight="bold", color=INK)
    ax.text(50, 93.2, "Local-first Graph-RAG + live-API agent, with a gated "
            "self-improving training loop", ha="center", va="center",
            fontsize=11.5, color="#475569")

    # ── ONLINE band ──────────────────────────────────────────────────────────
    band(ax, 1.5, 57.5, 97, 33, BAND_ON)
    ax.text(3.5, 88.2, "ONLINE  ·  ingest → store → serve", ha="left",
            va="center", fontsize=12, fontweight="bold", color="#15803d")

    ax.text(8.0, 85.0, "Sources", ha="center", fontsize=10.5,
            fontweight="bold", color=INK)
    jira = box(ax, 2.5, 79.5, 11, 4.0, "Jira", GREY_F, GREY_E)
    conf = box(ax, 2.5, 74.7, 11, 4.0, "Confluence", GREY_F, GREY_E)
    gitl = box(ax, 2.5, 69.9, 11, 4.0, "GitLab", GREY_F, GREY_E)

    ax.text(25.0, 85.0, "Ingestion pipeline", ha="center", fontsize=10.5,
            fontweight="bold", color=INK)
    ing = [
        box(ax, 17.0, 79.9, 16, 3.6, "1 · Ingest", BLUE_F, BLUE_E),
        box(ax, 17.0, 76.0, 16, 3.6, "2 · Normalize", BLUE_F, BLUE_E),
        box(ax, 17.0, 72.1, 16, 3.6, "3 · Chunk", BLUE_F, BLUE_E),
        box(ax, 17.0, 68.2, 16, 3.6, "4 · Embed\narctic-embed-m · 768d",
            BLUE_F, BLUE_E, fs=8.2),
    ]
    for a, b in zip(ing, ing[1:]):
        arrow(ax, (a[0] + a[2] / 2, a[1]), (b[0] + b[2] / 2, b[1] + b[3]), BLUE_E, lw=1.5)
    arrow(ax, (13.5, 76.7), (17.0, 78.5), GREY_E, lw=1.5)

    ax.text(44.0, 85.0, "Knowledge stores", ha="center", fontsize=10.5,
            fontweight="bold", color=INK)
    mongo = box(ax, 36.0, 78.4, 16, 5.2, "MongoDB\nsource of truth", TEAL_F, TEAL_E)
    neo = box(ax, 36.0, 70.0, 16, 5.2, "Neo4j\ngraph + vector index", TEAL_F, TEAL_E)
    arrow(ax, (33.0, 78.0), (36.0, 80.0), BLUE_E, rad=-0.15)
    arrow(ax, (44.0, 78.4), (44.0, 75.2), TEAL_E, lw=1.5)
    label(ax, 47.6, 76.8, "neo4j_import", TEAL_E, fs=8)

    ax.add_patch(FancyBboxPatch(
        (55.0, 62.5), 43.0, 24.0, boxstyle="round,pad=0,rounding_size=1.2",
        facecolor="#f4fdf7", edgecolor=GRN_E, linewidth=1.6, zorder=2))
    ax.text(76.5, 84.6, "Inference — LangGraph agent  ·  FastAPI + React (SSE)",
            ha="center", fontsize=10.5, fontweight="bold", color=INK, zorder=4)
    router = box(ax, 58.0, 79.0, 37.0, 4.0,
                 "Intent router   (rag | sentries | both)", GRN_F, GRN_E, fs=10)
    rag = box(ax, 58.0, 72.0, 17.5, 5.0, "RAG\nvector + graph + rerank", GRN_F, GRN_E, fs=9.5)
    sen = box(ax, 77.5, 72.0, 17.5, 5.0, "Sentries\nlive Jira / Confluence / GitLab",
              GRN_F, GRN_E, fs=9.5)
    weaver = box(ax, 58.0, 64.8, 37.0, 4.6,
                 "Weaver  ·  budgeted context  →  Ollama LLM", GRN_F, GRN_E, fs=10)
    arrow(ax, (66.0, 79.0), (66.0, 77.0), GRN_E, lw=1.5)
    arrow(ax, (88.0, 79.0), (88.0, 77.0), GRN_E, lw=1.5)
    arrow(ax, (66.0, 72.0), (66.0, 69.4), GRN_E, lw=1.5)
    arrow(ax, (86.0, 72.0), (80.0, 69.4), GRN_E, lw=1.5)
    arrow(ax, (52.0, 72.6), (58.0, 74.5), TEAL_E, rad=0.12)
    label(ax, 55.4, 74.6, "retrieval", TEAL_E, fs=8)

    answer = box(ax, 70.0, 52.0, 25.0, 4.4, "Answer  →  user   (streamed)",
                 GRN_F, GRN_E, fs=10.5)
    arrow(ax, (82.5, 64.8), (82.5, 56.4), GRN_E)

    # ── OFFLINE band ─────────────────────────────────────────────────────────
    band(ax, 1.5, 4.0, 97, 43.0, BAND_OFF)
    ax.text(3.5, 44.6, "OFFLINE  ·  score → build → train → gate", ha="left",
            va="center", fontsize=12, fontweight="bold", color="#b45309")

    chat = box(ax, 80.0, 36.0, 17.0, 5.4, "chat_history\nMongoDB", AMB_F, AMB_E)
    arrow(ax, (86.0, 52.0), (88.0, 41.4), AMB_E, rad=-0.18)
    label(ax, 91.5, 47.0, "log every turn", "#b45309", fs=8.5)

    hammer = box(ax, 57.0, 36.0, 20.0, 5.4,
                 "Hammer  ·  async evaluation\nscore · grade · failure tags",
                 AMB_F, AMB_E, fs=9.5)
    arrow(ax, (80.0, 38.7), (77.0, 38.7), AMB_E)

    dataset = box(ax, 34.0, 36.0, 20.0, 5.4,
                  "RAFT dataset\ngolden passage + distractors", AMB_F, AMB_E, fs=9.5)
    arrow(ax, (57.0, 38.7), (54.0, 38.7), AMB_E)
    label(ax, 55.5, 41.6, "GOLD only", "#b45309", fs=8)

    train = box(ax, 12.0, 36.0, 19.0, 5.4,
                "Training  ·  RAFT objective\nQLoRA 4-bit adapter", PUR_F, PUR_E, fs=9.5)
    arrow(ax, (34.0, 38.7), (31.0, 38.7), PUR_E)

    mlflow = box(ax, 12.0, 27.0, 19.0, 5.0, "MLflow  ·  tracking + registry",
                 PUR_F, PUR_E, fs=9.5)
    arrow(ax, (21.5, 36.0), (21.5, 32.0), PUR_E, lw=1.6)
    label(ax, 26.6, 34.0, "log runs", PUR_E, fs=8)

    # The two arms. This is what the gate reads, and the reason the comparison
    # means anything: identical server, identical quantisation, one difference.
    arms = box(ax, 34.0, 26.2, 22.0, 6.6,
               "Held-out evaluation  ·  both arms\n"
               "base vs adapter\nsame server, same quantisation",
               AMB_F, AMB_E, fs=8.6)
    arrow(ax, (31.0, 29.5), (34.0, 29.5), PUR_E)

    gate = box(ax, 62.0, 24.6, 30.0, 9.0,
               "PROMOTION GATE\n"
               "recall ≥ 2 SE\n"
               "refusal ≤ +15 pts  ·  fabrication ≤ +2 pts\n"
               "deterministic — no LLM judge",
               "#fff7ed", "#c2410c", fs=8.8)
    arrow(ax, (56.0, 29.5), (62.0, 29.5), AMB_E)
    ax.text(70.0, 22.9, "thresholds fixed before training", ha="center",
            fontsize=8.2, style="italic", color="#9a3412", zorder=6)

    # Both outcomes. A gate drawn with only its success branch is a pipeline.
    reject = box(ax, 66.0, 12.0, 22.0, 4.6,
                 "REJECT  →  incumbent stays live", RED_F, RED_E, fs=9.5, tc="#b91c1c")
    arrow(ax, (77.0, 24.6), (77.0, 16.6), RED_E, lw=1.7)
    label(ax, 83.6, 20.4, "verdict recorded", RED_E, fs=8)

    export = box(ax, 11.0, 12.0, 25.0, 5.6,
                 "PROMOTE  →  export_to_ollama\nadapter merged into the served model",
                 GRN_F, GRN_E, fs=9.5, tc="#15803d")
    arrow(ax, (62.0, 26.5), (36.0, 16.4), GRN_E, lw=2.0, rad=0.14)

    # The loop closes: the promoted adapter becomes the model that answers.
    # Routed down the left margin and across, so it crosses nothing -- an
    # earlier version cut straight through the ingestion column.
    lx, ly = 6.5, 60.0
    ax.plot([11.0, lx], [14.8, 14.8], color=GRN_E, lw=2.4, zorder=5,
            solid_capstyle="round")
    ax.plot([lx, lx], [14.8, ly], color=GRN_E, lw=2.4, zorder=5,
            solid_capstyle="round")
    ax.plot([lx, 52.0], [ly, ly], color=GRN_E, lw=2.4, zorder=5,
            solid_capstyle="round")
    arrow(ax, (52.0, ly), (59.5, 64.8), GRN_E, lw=2.4, rad=-0.18)
    ax.text(29.0, ly, "promoted adapter  →  served Ollama model",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#15803d", zorder=6,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRN_E, lw=1.2))

    # ── legend ───────────────────────────────────────────────────────────────
    keys = [("Sources", GREY_F, GREY_E), ("Ingestion", BLUE_F, BLUE_E),
            ("Storage", TEAL_F, TEAL_E), ("Inference", GRN_F, GRN_E),
            ("Evaluation", AMB_F, AMB_E), ("Training / MLOps", PUR_F, PUR_E),
            ("Gate", "#fff7ed", "#c2410c")]
    x = 14.0
    for name, fc, ec in keys:
        ax.add_patch(Rectangle((x, 0.8), 2.0, 1.6, facecolor=fc, edgecolor=ec,
                               linewidth=1.3, zorder=3))
        ax.text(x + 2.7, 1.6, name, ha="left", va="center", fontsize=9.5,
                color=INK, zorder=3)
        x += len(name) * 0.62 + 7.0

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        p = OUT / f"architecture.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
