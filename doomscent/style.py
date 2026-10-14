"""Shared visual style: one accent hue, grays for everything else, Inter everywhere.

Used by clip overlays, figures, and the timelapse so every asset reads as one system.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from PIL import ImageFont

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FILES = {
    "regular": "Inter-Regular.ttf",
    "medium": "Inter-Medium.ttf",
    "semibold": "Inter-SemiBold.ttf",
    "bold": "Inter-Bold.ttf",
}


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    panel: str
    text: str
    subtext: str
    grid: str
    wall: str
    muted: str  # series that are not the point of the figure
    muted2: str
    accent: str  # the one thing that matters
    accent_soft: str
    second: str  # only when two series must be told apart


LIGHT = Theme(
    name="light",
    background="#FFFFFF",
    panel="#F4F4F2",
    text="#1A1A1A",
    subtext="#6B6B6B",
    grid="#E6E6E3",
    wall="#C9C9C4",
    muted="#9A9A95",
    muted2="#C4C4BF",
    accent="#E0461C",
    accent_soft="#F6C1AE",
    second="#3A5A8C",
)

DARK = Theme(
    name="dark",
    background="#111214",
    panel="#1B1C1F",
    text="#EDEDEA",
    subtext="#9B9B97",
    grid="#2A2B2F",
    wall="#5A5B60",
    muted="#7D7E83",
    muted2="#4A4B50",
    accent="#FF5A2C",
    accent_soft="#6B2A18",
    second="#7EA6E0",
)

THEMES = {"light": LIGHT, "dark": DARK}


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


@cache
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    """Bundled Inter at a pixel size, for Pillow drawing."""
    return ImageFont.truetype(str(FONT_DIR / FONT_FILES[weight]), size)
