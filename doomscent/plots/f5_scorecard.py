"""F5: eval scorecard for the best checkpoint, beside the scripted baseline."""

from __future__ import annotations

import json

import pandas as pd

from doomscent.checkpoints import best_pointer
from doomscent.plots.common import FigureSpec, fmt_steps
from doomscent.plots.data import load_baseline_for, load_eval, subtitle
from doomscent.runs import RUNS_ROOT, RunPaths


def build(paths: RunPaths) -> FigureSpec:
    ev = load_eval(paths).set_index("step")
    bp = best_pointer(paths)
    step = json.loads(bp.read_text())["step"] if bp.exists() else int(ev["kills"].idxmax())
    r = ev.loc[step]
    base = load_baseline_for(paths)
    bdf = None
    if base:
        bdf = pd.read_csv(RUNS_ROOT / "baseline" / base["scenario"] / "episodes.csv")

    def b(col: str, fmt: str) -> str:
        return format(bdf[col].mean(), fmt) if bdf is not None and col in bdf else "n/a"

    rows = [
        (
            "kills per episode",
            f"{r.kills:.2f}",
            f"[{r.kills_ci_lo:.2f}, {r.kills_ci_hi:.2f}]",
            b("kills", ".2f"),
        ),
        ("deaths per episode", f"{r.deaths:.2f}", "", b("deaths", ".2f")),
        (
            "accuracy (hits / shots)",
            f"{r.accuracy:.0%}",
            "",
            f"{bdf.hits.sum() / bdf.shots.sum():.0%}" if bdf is not None else "n/a",
        ),
        ("survival time (s)", f"{r.survival_time:.1f}", "", b("survival_time_s", ".1f")),
        ("coverage (64-unit cells)", f"{r.coverage:.1f}", "", b("coverage", ".1f")),
        ("damage dealt (hp)", f"{r.damage_dealt:.0f}", "", b("damage_dealt", ".0f")),
        ("damage taken (hp)", f"{r.damage_taken:.0f}", "", b("damage_taken", ".0f")),
    ]
    ratio = r.kills / base["kills_mean"] if base else float("nan")
    title = (
        f"The best checkpoint averages {r.kills:.1f} kills per episode, "
        f"{ratio:.1f}x the scripted baseline"
    )
    sub = subtitle(
        paths, 1, f"best: step {fmt_steps(step)}; 100 held-out episodes; 95% bootstrap interval"
    )

    def draw(fig, theme, size):
        left, bottom, right, top = fig.rect
        ax = fig.add_axes((left, bottom, right - left, top - bottom))
        ax.axis("off")
        cols = [0.0, 0.52, 0.70, 1.0]
        heads = ["", "best checkpoint", "95% interval", "scripted baseline"]
        n = len(rows) + 1
        dy = 1.0 / n
        fs = 12 if size != "square" else 11
        for j, h in enumerate(heads):
            ax.text(
                cols[j],
                1 - dy * 0.5,
                h,
                color=theme.subtext,
                fontsize=fs - 1.5,
                ha="left" if j == 0 else "right",
                va="center",
                transform=ax.transAxes,
            )
        for i, row in enumerate(rows):
            yy = 1 - dy * (i + 1.5)
            ax.axhline(1 - dy * (i + 1), color=theme.grid, lw=0.8)
            for j, val in enumerate(row):
                color = theme.accent if (i == 0 and j == 1) else theme.text
                if j in (2, 3):
                    color = theme.subtext
                weight = "semibold" if (i == 0 and j == 1) else "regular"
                ax.text(
                    cols[j],
                    yy,
                    val,
                    color=color,
                    fontsize=fs,
                    fontweight=weight,
                    ha="left" if j == 0 else "right",
                    va="center",
                    transform=ax.transAxes,
                )

    return FigureSpec("F5", "eval_scorecard", title, sub, draw, right_margin_in=0.6)
