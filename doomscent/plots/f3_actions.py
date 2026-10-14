"""F3: action evolution, one heatmap per head: choices by checkpoint, shade is usage share."""

from __future__ import annotations

import json

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from doomscent.plots.common import FigureSpec, fmt_steps
from doomscent.plots.data import load_eval, subtitle
from doomscent.runs import RunPaths


def build(paths: RunPaths) -> FigureSpec:
    ev = load_eval(paths)
    heads = json.loads((paths.meta / "action_space.json").read_text())["heads"]
    steps = ev["step"].to_numpy()
    mats = {
        h["name"]: np.array([ev[f"share_{h['name']}_{c}"].to_numpy() for c in h["choices"]])
        for h in heads
    }
    last = {n: m[:, -1] for n, m in mats.items()}
    bits = []
    for h in heads:
        i = int(np.argmax(last[h["name"]]))
        bits.append(f"{h['name']} '{h['choices'][i].replace('_', ' ')}' {last[h['name']][i]:.0%}")
    title = f"By {fmt_steps(steps[-1])} steps the top choices are " + ", ".join(bits)
    sub = subtitle(paths, 1, "share of held-out steps per choice, deterministic policy")

    def draw(fig, theme, size):
        left, bottom, right, top = fig.rect
        cmap = LinearSegmentedColormap.from_list("acc", [theme.background, theme.accent])
        n_rows = [len(h["choices"]) for h in heads]
        gap = 0.06
        total = sum(n_rows)
        h_avail = top - bottom - gap * (len(heads) - 1)
        y = top
        for idx, (h, n) in enumerate(zip(heads, n_rows, strict=True)):
            hh = h_avail * n / total
            ax = fig.add_axes((left + 0.08, y - hh, right - left - 0.08, hh))
            ax.imshow(
                mats[h["name"]], aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
            )
            ax.set_yticks(range(n), [c.replace("_", " ") for c in h["choices"]])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(
                h["name"], loc="left", fontsize=11, fontweight="semibold", color=theme.text, pad=4
            )
            k = max(1, len(steps) // 8)
            labels = [fmt_steps(s) for s in steps[::k]] if idx == len(heads) - 1 else []
            ax.set_xticks(range(0, len(steps), k), labels)
            y -= hh + gap
        ax.set_xlabel("checkpoint (environment steps)")

    return FigureSpec("F3", "action_evolution", title, sub, draw, right_margin_in=0.5)
