"""Shared figure machinery: theme, header, direct labels, step axis, and multi-size export.

Every figure is a function `draw(fig, data, theme) -> None` plus a title computed from the
data (the finding, not the topic). `export()` renders it at 16:9, square, and paper sizes,
in light and dark themes, as 2x PNG and vector PDF:
runs/<run>/figures/<ID>_<name>_{16x9,square,paper}_{light,dark}.{png,pdf}
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from doomscent.style import FONT_DIR, THEMES, Theme

STYLE_FILE = Path(__file__).with_name("style.mplstyle")
BASE_DPI = 200  # figure inches = pixels / BASE_DPI; PNGs are saved at 2x

SIZES: dict[str, tuple[int, int]] = {
    "16x9": (1600, 900),
    "square": (1200, 1200),
    "paper": (1300, 760),
}

for _f in FONT_DIR.glob("Inter-*.ttf"):
    font_manager.fontManager.addfont(str(_f))


def apply_theme(theme: Theme) -> None:
    plt.style.use(str(STYLE_FILE))
    plt.rcParams.update(
        {
            "figure.facecolor": theme.background,
            "axes.facecolor": theme.background,
            "savefig.facecolor": theme.background,
            "axes.edgecolor": theme.grid,
            "axes.labelcolor": theme.subtext,
            "text.color": theme.text,
            "xtick.color": theme.subtext,
            "ytick.color": theme.subtext,
            "grid.color": theme.grid,
        }
    )


def fmt_steps(x: float, _pos: Any = None) -> str:
    """0, 500K, 10M."""
    if x == 0:
        return "0"
    if abs(x) >= 1e6:
        v = x / 1e6
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def step_axis(ax: Axes, label: str = "environment steps") -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_steps))
    ax.set_xlabel(label)


TITLE_LINE_IN = 0.30
SUBTITLE_IN = 0.28


def header(fig: Figure, title: str, subtitle: str, theme: Theme) -> None:
    """Left-aligned finding title (may span lines) with a smaller gray subtitle below."""
    w_in, h_in = fig.get_size_inches()
    left = 0.42 / w_in
    y = 1 - 0.28 / h_in
    fig.text(
        left,
        y,
        title,
        fontsize=15,
        fontweight="semibold",
        color=theme.text,
        va="top",
        ha="left",
        linespacing=1.25,
    )
    y -= (title.count("\n") + 1) * TITLE_LINE_IN / h_in + 0.06 / h_in
    fig.text(
        left, y, subtitle, fontsize=10.5, color=theme.subtext, va="top", ha="left", linespacing=1.3
    )


def direct_label(ax: Axes, x: float, y: float, text: str, color: str, dx: float = 6) -> None:
    """Series name at the right end of its line, in the line's color."""
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        color=color,
        fontsize=10.5,
        fontweight="medium",
        va="center",
        ha="left",
        annotation_clip=False,
    )


def direct_labels(ax: Axes, items: Sequence[tuple[float, float, str, str]], min_gap: float = 0.05):
    """Label several line ends, nudging labels apart so they never overlap.

    `items` are (x, y, text, color); `min_gap` is a fraction of the y-axis range.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    ys = [items[i][1] for i in order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    for k, i in enumerate(order):
        x, _, label, color = items[i]
        direct_label(ax, x, ys[k], label, color)


def ref_line(ax: Axes, y: float, text: str, theme: Theme, dashes: Sequence[float] = (4, 3)):
    """Faint dashed reference line labeled at its right end."""
    ax.axhline(y, color=theme.muted, lw=1.0, dashes=dashes, zorder=1)
    ax.annotate(
        text,
        (1.0, y),
        xycoords=("axes fraction", "data"),
        xytext=(6, 0),
        textcoords="offset points",
        color=theme.subtext,
        fontsize=9.5,
        va="center",
        ha="left",
        annotation_clip=False,
    )


def note(
    ax: Axes, xy: tuple[float, float], xytext: tuple[float, float], text: str, theme: Theme
) -> None:
    """Small gray annotation with a thin leader line (at most two per figure)."""
    ax.annotate(
        text,
        xy,
        xytext=xytext,
        textcoords="offset points",
        fontsize=9.5,
        color=theme.subtext,
        arrowprops={"arrowstyle": "-", "color": theme.muted, "lw": 0.8},
    )


@dataclass(frozen=True)
class FigureSpec:
    fig_id: str
    name: str
    title: str
    subtitle: str
    draw: Callable[[Figure, Theme, str], None]  # (fig, theme, size_name)
    right_margin_in: float = 1.3  # room for direct labels


def layout_rect(
    w_in: float, h_in: float, right_in: float, title_lines: int, sub_lines: int = 1
) -> tuple[float, ...]:
    """(left, bottom, right, top) of the plotting area in figure fractions."""
    top_in = 0.28 + title_lines * TITLE_LINE_IN + 0.06 + sub_lines * SUBTITLE_IN + 0.32
    return (0.85 / w_in, 0.75 / h_in, 1 - right_in / w_in, 1 - top_in / h_in)


WRAP = {"16x9": 68, "square": 44, "paper": 54}


def export(
    spec: FigureSpec, out_dir: Path, themes: Sequence[str] = ("light", "dark")
) -> list[Path]:
    """Render every size and theme; returns the written files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for theme_name in themes:
        theme = THEMES[theme_name]
        for size, (w_px, h_px) in SIZES.items():
            apply_theme(theme)
            w_in, h_in = w_px / BASE_DPI, h_px / BASE_DPI
            fig = plt.figure(figsize=(w_in, h_in))
            title = _wrap(spec.title, WRAP[size])
            subtitle = _wrap(spec.subtitle, int(WRAP[size] * 1.45))
            rect = layout_rect(
                w_in, h_in, spec.right_margin_in, title.count("\n") + 1, subtitle.count("\n") + 1
            )
            fig.rect = rect  # type: ignore[attr-defined]
            spec.draw(fig, theme, size)
            header(fig, title, subtitle, theme)
            stem = f"{spec.fig_id}_{spec.name}_{size}_{theme_name}"
            for ext in ("png", "pdf"):
                p = out_dir / f"{stem}.{ext}"
                fig.savefig(p, dpi=BASE_DPI * 2 if ext == "png" else None)
                written.append(p)
            plt.close(fig)
    return written


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


def main_axes(fig: Figure) -> Axes:
    left, bottom, right, top = fig.rect  # type: ignore[attr-defined]
    return fig.add_axes((left, bottom, right - left, top - bottom))
