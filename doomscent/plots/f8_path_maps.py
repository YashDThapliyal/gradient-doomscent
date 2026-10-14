"""F8: top-down path maps over many held-out episodes, early checkpoint vs best.

In Defend the Center the agent cannot move, so the path is a single point; each panel
therefore also shows where the agent aims: the share of steps spent facing each 10-degree
direction, drawn as rays from its position.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from doomscent.checkpoints import best_pointer
from doomscent.minimap import MapGeometry
from doomscent.plots.common import FigureSpec, fmt_steps
from doomscent.plots.data import load_eval, subtitle
from doomscent.recording import read_sidecar
from doomscent.runs import RunPaths, write_map_geometry

N_BINS = 36


def episode_tracks(paths: RunPaths, label: str, limit: int = 100) -> list[dict]:
    df = pd.read_csv(paths.metrics / "eval_episodes" / f"{label}.csv").head(limit)
    tracks = []
    for sc in df["sidecar"]:
        s = read_sidecar(sc)
        v = [st["vars"] for st in s["steps"]]
        tracks.append(
            {
                "x": np.array([p["POSITION_X"] for p in v]),
                "y": np.array([p["POSITION_Y"] for p in v]),
                "angle": np.array([p["ANGLE"] for p in v]),
            }
        )
    return tracks


def aim_shares(tracks: list[dict]) -> np.ndarray:
    ang = np.concatenate([t["angle"] for t in tracks]) % 360
    h, _ = np.histogram(ang, bins=N_BINS, range=(0, 360))
    return h / max(1, h.sum())


def directions_covered(shares: np.ndarray) -> float:
    """Fraction of 10-degree directions holding at least half a uniform share."""
    return float((shares >= 0.5 / N_BINS).mean())


def build(paths: RunPaths) -> FigureSpec:
    cfg = paths.load_config()
    geom = MapGeometry.from_dict(json.loads(write_map_geometry(cfg, paths).read_text()))
    ev = load_eval(paths)
    nonzero = ev[ev.step > 0]
    early_step = int(nonzero.step.iloc[0]) if len(nonzero) else int(ev.step.iloc[0])
    bp = best_pointer(paths)
    best_step = json.loads(bp.read_text())["step"] if bp.exists() else int(ev.step.iloc[-1])
    panels = []
    for step, name in ((early_step, "early"), (best_step, "best")):
        tracks = episode_tracks(paths, f"step_{step:07d}")
        panels.append((step, name, tracks, aim_shares(tracks)))
    disp = max(
        float(np.hypot(t["x"] - t["x"][0], t["y"] - t["y"][0]).max()) for p in panels for t in p[2]
    )
    c0, c1 = directions_covered(panels[0][3]), directions_covered(panels[1][3])
    stay = "never leaves the center" if disp < 32 else f"moves up to {disp:.0f} units"
    title = f"The agent {stay}; its aim covers {c1:.0%} of directions at its best vs {c0:.0%} early"
    sub = subtitle(
        paths,
        1,
        f"{len(panels[1][2])} held-out episodes per panel; rays: share of steps facing each 10° direction",
    )

    def draw(fig, theme, size):
        left, bottom, right, top = fig.rect
        w = (right - left) / 2
        xmin, ymin, xmax, ymax = geom.bounds
        pad = 0.06 * max(xmax - xmin, ymax - ymin)
        for i, (step, name, tracks, shares) in enumerate(panels):
            ax = fig.add_axes((left + i * w, bottom, w * 0.94, (top - bottom) * 0.86))
            ax.set_aspect("equal")
            ax.axis("off")
            for x1, y1, x2, y2 in geom.walls:
                ax.plot([x1, x2], [y1, y2], color=theme.wall, lw=1.0)
            cx = np.mean([t["x"].mean() for t in tracks])
            cy = np.mean([t["y"].mean() for t in tracks])
            radius = 0.85 * min(xmax - cx, cx - xmin, ymax - cy, cy - ymin)
            mids = np.deg2rad((np.arange(N_BINS) + 0.5) * 360 / N_BINS)
            scale = radius / max(shares.max(), 1e-9)
            for a, s in zip(mids, shares, strict=True):
                if s > 0:
                    ax.plot(
                        [cx, cx + s * scale * np.cos(a)],
                        [cy, cy + s * scale * np.sin(a)],
                        color=theme.accent,
                        lw=2.4,
                        alpha=0.85,
                        solid_capstyle="round",
                    )
            for t in tracks:
                ax.plot(t["x"], t["y"], color=theme.accent, lw=1.0, alpha=0.3)
            ax.plot([cx], [cy], "o", color=theme.text, ms=4)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)
            lbl = f"{name}: step {fmt_steps(step)}"
            ax.text(
                0.0,
                1.0,
                lbl,
                transform=ax.transAxes,
                color=theme.text,
                fontsize=11,
                fontweight="semibold",
                va="top",
            )
            cov = directions_covered(shares)
            ax.text(
                0.0,
                0.93,
                f"aim covers {cov:.0%} of directions",
                transform=ax.transAxes,
                color=theme.subtext,
                fontsize=9.5,
                va="top",
            )

    return FigureSpec("F8", "path_maps", title, sub, draw, right_margin_in=0.4)
