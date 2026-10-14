"""F4: policy entropy per head vs env steps, with flagged-behavior events as x-axis ticks."""

from __future__ import annotations

import json

import numpy as np

from doomscent.plots.common import FigureSpec, direct_labels, main_axes, step_axis
from doomscent.plots.data import load_flagged, load_train, subtitle
from doomscent.runs import RunPaths
from doomscent.stats import ema


def build(paths: RunPaths) -> FigureSpec:
    tr = load_train(paths)
    heads = [h["name"] for h in json.loads((paths.meta / "action_space.json").read_text())["heads"]]
    series = {}
    for h in heads:
        s = tr[["env_steps", f"policy_entropy_{h}"]].dropna()
        series[h] = (s["env_steps"].to_numpy(), s[f"policy_entropy_{h}"].to_numpy())
    flags = load_flagged(paths)
    first = heads[0]
    _x, y = series[first]
    sm = ema(y, 0.05)
    drop = 1 - sm[-1] / sm[0] if sm[0] else 0.0
    title = (
        f"{first.capitalize()} entropy falls {drop:.0%}, from {sm[0]:.2f} to {sm[-1]:.2f} nats"
        if drop > 0.05
        else f"{first.capitalize()} entropy stays near {sm[-1]:.2f} nats"
    )
    n_flag = len(flags)
    sub = subtitle(paths, 1, f"thin: raw, bold: moving average; ticks: {n_flag} flagged episodes")

    def draw(fig, theme, size):
        ax = main_axes(fig)
        colors = [theme.accent, theme.muted, theme.second, theme.muted2]
        ends = []
        for h, c in zip(heads, colors, strict=False):
            xs, ys = series[h]
            ax.plot(xs, ys, color=c, lw=0.8, alpha=0.35)
            sm_h = ema(ys, 0.05)
            ax.plot(xs, sm_h, color=c, lw=2.0)
            ends.append((xs[-1], sm_h[-1], f"{h} head", c))
        if len(flags):
            fx = flags["step"].to_numpy(dtype=float)
            ax.plot(
                fx,
                np.zeros_like(fx),
                "|",
                color=theme.subtext,
                ms=9,
                mew=1.0,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
        ax.set_ylim(0, max(np.max(series[h][1]) for h in heads) * 1.1)
        ax.set_xlim(0, max(series[h][0][-1] for h in heads))
        step_axis(ax)
        ax.set_ylabel("entropy (nats)")
        direct_labels(ax, ends)

    return FigureSpec("F4", "entropy_and_exploits", title, sub, draw, right_margin_in=1.4)
