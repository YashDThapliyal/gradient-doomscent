"""F1: held-out kills per episode vs env steps, seed mean with a band, baseline and gate."""

from __future__ import annotations

import numpy as np

from doomscent.plots.common import (
    FigureSpec,
    direct_label,
    fmt_steps,
    main_axes,
    ref_line,
    step_axis,
)
from doomscent.plots.data import load_baseline_for, load_eval, sibling_runs, subtitle
from doomscent.runs import RunPaths


def seed_matrix(runs: list[RunPaths], column: str) -> tuple[np.ndarray, np.ndarray]:
    """Steps evaluated in every seed, and a (seeds, steps) matrix of `column`."""
    evals = [load_eval(r).set_index("step") for r in runs]
    steps = sorted(set.intersection(*(set(e.index) for e in evals)))
    return np.array(steps), np.array([[e.loc[s, column] for s in steps] for e in evals])


def build(paths: RunPaths) -> FigureSpec:
    runs = sibling_runs(paths)
    steps, mat = seed_matrix(runs, "kills")
    mean, lo, hi = mat.mean(0), mat.min(0), mat.max(0)
    base = load_baseline_for(paths)
    gate = base["gate_kills"] if base else None
    who = "Mean held-out kills" if len(runs) > 1 else "Held-out kills"
    crossed = np.nonzero(mean >= gate)[0] if gate else []
    if gate and len(crossed):
        title = (
            f"{who} per episode pass the 1.5x baseline gate by {fmt_steps(steps[crossed[0]])} steps"
        )
    elif gate:
        title = f"{who} per episode peak at {mean.max():.1f}, short of the {gate:.1f} gate"
    else:
        title = f"{who} per episode rise to {mean.max():.1f}"
    band = "band: min to max across seeds; " if len(runs) > 1 else ""
    sub = subtitle(paths, len(runs), f"{band}100 held-out episodes per point")

    def draw(fig, theme, size):
        ax = main_axes(fig)
        if len(runs) > 1:
            for row in mat:
                ax.plot(steps, row, color=theme.muted, lw=0.9, alpha=0.55)
            ax.fill_between(steps, lo, hi, color=theme.accent, alpha=0.13, lw=0)
        ax.plot(steps, mean, color=theme.accent, lw=2.2)
        label = f"mean of {len(runs)} seeds" if len(runs) > 1 else "held-out kills"
        direct_label(ax, steps[-1], mean[-1], label, theme.accent)
        if base:
            ref_line(ax, base["kills_mean"], f"scripted baseline {base['kills_mean']:.2f}", theme)
            ref_line(ax, gate, f"1.5x gate {gate:.2f}", theme, dashes=(2, 2))
        top = max(float(hi.max()), gate or 0)
        ax.set_ylim(0, top * 1.12 + 0.5)
        ax.set_xlim(0, steps[-1])
        step_axis(ax)
        ax.set_ylabel("kills per episode")

    return FigureSpec("F1", "learning_curve", title, sub, draw, right_margin_in=1.9)
