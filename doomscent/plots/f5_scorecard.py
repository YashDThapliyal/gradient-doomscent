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

    conf_file = paths.metrics / "confirm" / f"step_{int(step):07d}.json"
    conf = json.loads(conf_file.read_text()) if conf_file.exists() else None
    head_kills = conf["kills"] if conf else r.kills
    rows = []
    if conf:
        rows.append(
            (
                "kills, fresh held-out games",
                f"{conf['kills']:.2f}",
                f"[{conf['kills_ci_lo']:.2f}, {conf['kills_ci_hi']:.2f}]",
                b("kills", ".2f"),
            )
        )
    rows += [
        (
            "kills, games used to pick best" if conf else "kills per episode",
            f"{r.kills:.2f}",
            f"[{r.kills_ci_lo:.2f}, {r.kills_ci_hi:.2f}]",
            "" if conf else b("kills", ".2f"),
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
    ratio = head_kills / base["kills_mean"] if base else float("nan")
    fresh = " on fresh held-out games" if conf else ""
    title = (
        f"The best checkpoint averages {head_kills:.1f} kills per episode{fresh}, "
        f"{ratio:.1f}x the scripted baseline"
    )
    note = (
        "best picked on held-out seeds 0 to 99, re-scored on 100 unseen seeds"
        if conf
        else "100 held-out episodes"
    )
    sub = subtitle(paths, 1, f"best: step {fmt_steps(step)}; {note}; 95% bootstrap interval")

    def draw(fig, theme, size):
        left, bottom, right, top = fig.rect
        ax = fig.add_axes((left, bottom, right - left, top - bottom))
        ax.axis("off")
        cols = [0.0, 0.50, 0.54, 1.0]
        aligns = ["left", "right", "left", "right"]
        heads = ["", "best checkpoint", "95% interval", "scripted baseline"]
        n = len(rows) + 1
        dy = 1.0 / n
        row_pt = (top - bottom) * fig.get_size_inches()[1] * 72 / n
        fs = min(13.0, row_pt * 0.5)
        for j, h in enumerate(heads):
            ax.text(
                cols[j],
                1 - dy * 0.5,
                h,
                color=theme.subtext,
                fontsize=fs * 0.85,
                ha=aligns[j],
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
                    ha=aligns[j],
                    va="center",
                    transform=ax.transAxes,
                )

    return FigureSpec("F5", "eval_scorecard", title, sub, draw, right_margin_in=0.6)
