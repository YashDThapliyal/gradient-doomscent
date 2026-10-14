"""F2: damage dealt and taken per held-out episode vs env steps."""

from __future__ import annotations

import numpy as np

from doomscent.plots.common import FigureSpec, direct_labels, fmt_steps, main_axes, step_axis
from doomscent.plots.data import sibling_runs, subtitle
from doomscent.plots.f1_learning_curve import seed_matrix
from doomscent.runs import RunPaths


def build(paths: RunPaths) -> FigureSpec:
    runs = sibling_runs(paths)
    steps, dealt = seed_matrix(runs, "damage_dealt")
    _, taken = seed_matrix(runs, "damage_taken")
    d, t = dealt.mean(0), taken.mean(0)
    over = np.nonzero(d > t)[0]
    if len(over) and over[0] > 0:
        title = f"Damage dealt overtakes damage taken by {fmt_steps(steps[over[0]])} steps"
    elif len(over):
        title = (
            f"The agent deals more damage than it takes from the start ({d[-1]:.0f} vs {t[-1]:.0f})"
        )
    else:
        title = f"Damage taken still exceeds damage dealt at {fmt_steps(steps[-1])} steps"
    seeds = "mean of seeds; " if len(runs) > 1 else ""
    sub = subtitle(paths, len(runs), f"{seeds}health points per held-out episode")

    def draw(fig, theme, size):
        ax = main_axes(fig)
        ax.plot(steps, d, color=theme.accent, lw=2.2)
        ax.plot(steps, t, color=theme.muted, lw=2.0)
        ax.set_ylim(0, max(d.max(), t.max()) * 1.12)
        direct_labels(
            ax,
            [
                (steps[-1], d[-1], "damage dealt", theme.accent),
                (steps[-1], t[-1], "damage taken", theme.muted),
            ],
        )
        ax.set_xlim(0, steps[-1])
        step_axis(ax)
        ax.set_ylabel("health points per episode")

    return FigureSpec("F2", "kill_death_exchange", title, sub, draw, right_margin_in=1.6)
