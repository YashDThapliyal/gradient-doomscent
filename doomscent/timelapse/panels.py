"""Drawing the timelapse panels from cached brain data. Pure drawing: no model, no game."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw

from doomscent.plots.common import fmt_steps
from doomscent.style import Theme, font, hex_to_rgb
from doomscent.timelapse.brain import LAYERS

PAIRS = tuple(f"{a}->{b}" for a, b in itertools.pairwise(LAYERS))
LAYER_LABELS = {
    "input": "input",
    "conv1": "conv 1",
    "conv2": "conv 2",
    "conv3": "conv 3",
    "fc": "dense",
    "gru": "memory (GRU)",
    "out": "actions + value",
}


def lerp_color(a: str, b: str, t: float) -> tuple[int, int, int]:
    ca, cb = np.array(hex_to_rgb(a), float), np.array(hex_to_rgb(b), float)
    t = float(np.clip(t, 0, 1))
    return tuple(int(v) for v in ca + (cb - ca) * t)  # type: ignore[return-value]


def text(
    img: np.ndarray,
    xy: tuple[int, int],
    s: str,
    size: int,
    color: str,
    weight: str = "regular",
    anchor: str = "la",
) -> np.ndarray:
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text(xy, s, font=font(size, weight), fill=hex_to_rgb(color), anchor=anchor)
    return np.asarray(pil).copy()


@dataclass
class BrainLayout:
    width: int
    height: int
    units: dict[str, list[int]]
    sizes: dict[str, int]

    def node_positions(self) -> dict[str, np.ndarray]:
        top, bottom = 70, self.height - 70
        xs = np.linspace(60, self.width - 60, len(LAYERS))
        pos = {}
        for x, layer in zip(xs, LAYERS, strict=True):
            n = len(self.units[layer])
            ys = np.linspace(top, bottom, n) if n > 1 else np.array([(top + bottom) / 2])
            if n < 8:  # small layers sit centered rather than spread over the full height
                mid, span = (top + bottom) / 2, (bottom - top) * n / 24
                ys = np.linspace(mid - span / 2, mid + span / 2, n)
            pos[layer] = np.stack([np.full(n, x), ys], axis=1)
        return pos


def draw_brain(
    layout: BrainLayout,
    change: dict[str, np.ndarray],
    change_scale: dict[str, float],
    acts: np.ndarray,
    act_scale: dict[str, float],
    theme: Theme,
    title: str = "weight change since last checkpoint · activity on the current frame",
) -> np.ndarray:
    """Views 1 and 2: edges lit by weight change, nodes lit by activity."""
    img = np.full((layout.height, layout.width, 3), hex_to_rgb(theme.background), np.uint8)
    pos = layout.node_positions()
    edges = []
    for pair in PAIRS:
        src, dst = pair.split("->")
        b = np.clip(change[pair] / max(change_scale[pair], 1e-12), 0, 1)
        for j in range(b.shape[0]):
            for i in range(b.shape[1]):
                edges.append((float(b[j, i]), pos[src][i], pos[dst][j]))
    edges.sort(key=lambda e: e[0])
    for bright, p0, p1 in edges:
        if bright < 0.04:
            continue
        if bright > 0.8:
            color = lerp_color(theme.text, theme.accent, (bright - 0.8) / 0.2)
        else:
            color = lerp_color(theme.grid, theme.subtext, bright / 0.8)
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, 1, cv2.LINE_AA)
    start = 0
    for layer in LAYERS:
        n = len(layout.units[layer])
        a = np.abs(acts[start : start + n]) / max(act_scale[layer], 1e-12)
        start += n
        for k, (x, y) in enumerate(pos[layer]):
            c = lerp_color(theme.muted2, theme.text, float(a[k]))
            cv2.circle(img, (int(x), int(y)), 5, c, -1, cv2.LINE_AA)
    img = text(img, (24, 22), title, 15, theme.subtext)
    for layer in LAYERS:
        x = int(pos[layer][0][0])
        img = text(
            img, (x, layout.height - 46), LAYER_LABELS[layer], 13, theme.text, "medium", "ma"
        )
        img = text(
            img,
            (x, layout.height - 26),
            f"{len(layout.units[layer])} of {layout.sizes[layer]}",
            12,
            theme.subtext,
            anchor="ma",
        )
    shown = sum(len(v) for v in layout.units.values())
    total = sum(layout.sizes.values())
    img = text(
        img,
        (layout.width - 24, 22),
        f"simplified view, {shown} of {total:,} units shown",
        13,
        theme.subtext,
        anchor="ra",
    )
    return img


@dataclass
class CurveData:
    steps: np.ndarray
    kills: np.ndarray
    baseline: float | None
    gate: float | None
    x_max: float


class CurvePanel:
    """Learning curve drawn progressively, with a cursor at the current checkpoint."""

    def __init__(self, data: CurveData, width: int, height: int, theme: Theme):
        self.d, self.w, self.h, self.theme = data, width, height, theme
        self.left, self.right, self.top, self.bottom = 74, width - 150, 64, height - 54
        hi = max(float(np.nanmax(data.kills)), data.gate or 0) * 1.15 + 0.5
        self.y_max = hi
        self.base = self._axes()

    def px(self, x: float, y: float) -> tuple[int, int]:
        fx = x / max(self.d.x_max, 1)
        fy = y / self.y_max
        return (
            int(self.left + fx * (self.right - self.left)),
            int(self.bottom - fy * (self.bottom - self.top)),
        )

    def _axes(self) -> np.ndarray:
        t = self.theme
        img = np.full((self.h, self.w, 3), hex_to_rgb(t.background), np.uint8)
        y_ticks = MaxNLocator(nbins=5, integer=True).tick_values(0, self.y_max)
        for yv in [v for v in y_ticks if 0 <= v <= self.y_max]:
            _, y = self.px(0, yv)
            cv2.line(img, (self.left, y), (self.right, y), hex_to_rgb(t.grid), 1, cv2.LINE_AA)
            img = text(img, (self.left - 10, y), f"{yv:.0f}", 13, t.subtext, anchor="rm")
        x_ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, self.d.x_max)
        for xv in [v for v in x_ticks if 0 <= v <= self.d.x_max]:
            x, _ = self.px(xv, 0)
            img = text(img, (x, self.bottom + 12), fmt_steps(xv), 13, t.subtext, anchor="ma")
        for val, label in ((self.d.baseline, "scripted baseline"), (self.d.gate, "1.5x gate")):
            if val is None:
                continue
            _, y = self.px(0, val)
            for x0 in range(self.left, self.right, 12):
                cv2.line(
                    img, (x0, y), (min(x0 + 6, self.right), y), hex_to_rgb(t.muted), 1, cv2.LINE_AA
                )
            img = text(img, (self.right + 8, y), f"{label} {val:.1f}", 12, t.subtext, anchor="lm")
        img = text(img, (24, 22), "held-out kills per episode", 15, t.subtext)
        return img

    def value_at(self, step: float) -> float:
        return float(np.interp(step, self.d.steps, self.d.kills))

    def draw(self, step: float) -> np.ndarray:
        t = self.theme
        img = self.base.copy()
        s = self.d.steps
        mask = s <= step
        xs = np.append(s[mask], step)
        ys = np.append(self.d.kills[mask], self.value_at(step))
        pts = np.array([self.px(x, y) for x, y in zip(xs, ys, strict=True)], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, hex_to_rgb(t.accent), 3, cv2.LINE_AA)
        cx, cy = pts[-1]
        cv2.line(
            img, (int(cx), self.top), (int(cx), self.bottom), hex_to_rgb(t.muted2), 1, cv2.LINE_AA
        )
        cv2.circle(img, (int(cx), int(cy)), 6, hex_to_rgb(t.accent), -1, cv2.LINE_AA)
        return img


def saliency_overlay(frame: np.ndarray, sal_u8: np.ndarray, theme: Theme, blur: int) -> np.ndarray:
    """Attention heatmap: the accent hue blended where gradients are largest."""
    h, w = frame.shape[:2]
    m = cv2.resize(sal_u8.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_LINEAR)
    k = blur * max(1, w // 128) | 1
    m = cv2.GaussianBlur(m, (k, k), 0)
    m = np.clip(m / max(float(m.max()), 1e-6), 0, 1)[..., None] * 0.65
    acc = np.array(hex_to_rgb(theme.accent), np.float32)
    return (frame.astype(np.float32) * (1 - m) + acc * m).astype(np.uint8)


def draw_decision(
    width: int,
    height: int,
    probs: np.ndarray,
    heads: Sequence[dict],
    value: float,
    value_range: tuple[float, float],
    theme: Theme,
) -> np.ndarray:
    """View 4 in miniature: action probabilities per head and a value meter."""
    img = np.full((height, width, 3), hex_to_rgb(theme.background), np.uint8)
    x0, y = 28, 24
    img = text(img, (x0, y), "decision", 15, theme.subtext)
    y += 34
    bw = width - x0 - 30
    start = 0
    for h in heads:
        n = len(h["choices"])
        p = probs[start : start + n]
        start += n
        img = text(img, (x0, y), h["name"], 14, theme.text, "semibold")
        y += 22
        top = int(np.argmax(p))
        for k, c in enumerate(h["choices"]):
            img = text(
                img, (x0, y), c.replace("_", " "), 12, theme.text if k == top else theme.subtext
            )
            bx = x0 + 100
            cv2.rectangle(img, (bx, y + 4), (bx + bw - 100, y + 12), hex_to_rgb(theme.grid), -1)
            fill = int((bw - 100) * float(p[k]))
            if fill > 0:
                col = theme.accent if k == top else theme.muted
                cv2.rectangle(img, (bx, y + 4), (bx + fill, y + 12), hex_to_rgb(col), -1)
            y += 18
        y += 10
    lo, hi = value_range
    frac = 0.0 if hi <= lo else float(np.clip((value - lo) / (hi - lo), 0, 1))
    img = text(img, (x0, y), f"value estimate {value:+.2f}", 13, theme.subtext)
    y += 22
    cv2.rectangle(img, (x0, y), (x0 + bw, y + 10), hex_to_rgb(theme.grid), -1)
    cv2.rectangle(img, (x0, y), (x0 + max(2, int(bw * frac)), y + 10), hex_to_rgb(theme.accent), -1)
    return img


def draw_tag_block(
    width: int, height: int, step: float, score: float, fight_kills: float, theme: Theme
) -> np.ndarray:
    img = np.full((height, width, 3), hex_to_rgb(theme.background), np.uint8)
    img = text(img, (28, 30), "environment steps", 14, theme.subtext)
    img = text(img, (28, 52), fmt_steps(step), 54, theme.text, "semibold")
    img = text(img, (28, 140), "held-out kills per episode", 14, theme.subtext)
    img = text(img, (28, 162), f"{score:.1f}", 54, theme.accent, "semibold")
    img = text(img, (28, 250), f"this fight: {fight_kills:.0f} kills so far", 14, theme.subtext)
    return img
