"""Top-down minimap from ViZDoom's map geometry (sector lines) and object positions.

Walls are thin light-gray lines, the agent's recent path and position use the accent hue,
monsters are small gray dots. The same geometry feeds the path-map figure (F8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import vizdoom as vzd

from doomscent.env import MONSTER_CLASSES
from doomscent.style import DARK, Theme, hex_to_rgb


@dataclass(frozen=True)
class MapGeometry:
    walls: np.ndarray  # (N, 4) x1, y1, x2, y2 for blocking lines
    passable: np.ndarray  # (M, 4) non-blocking lines
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax

    @classmethod
    def from_state(cls, state: vzd.GameState) -> MapGeometry:
        walls, passable = [], []
        for sector in state.sectors:
            for ln in sector.lines:
                seg = (ln.x1, ln.y1, ln.x2, ln.y2)
                (walls if ln.is_blocking else passable).append(seg)
        w = np.array(walls or [(0, 0, 0, 0)], dtype=float)
        p = np.array(passable, dtype=float).reshape(-1, 4)
        allpts = np.concatenate([w, p]).reshape(-1, 2)
        xmin, ymin = allpts.min(axis=0)
        xmax, ymax = allpts.max(axis=0)
        return cls(w, p, (float(xmin), float(ymin), float(xmax), float(ymax)))

    def to_dict(self) -> dict:
        return {"walls": self.walls.tolist(), "passable": self.passable.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> MapGeometry:
        w = np.array(d["walls"], dtype=float).reshape(-1, 4)
        p = np.array(d["passable"], dtype=float).reshape(-1, 4)
        pts = np.concatenate([w, p]).reshape(-1, 2)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return cls(w, p, (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))


class Projector:
    """Map units to pixel coordinates in a square image, y flipped, with padding."""

    def __init__(self, bounds: tuple[float, float, float, float], size: int, pad: float = 0.08):
        xmin, ymin, xmax, ymax = bounds
        span = max(xmax - xmin, ymax - ymin, 1.0)
        self.scale = size * (1 - 2 * pad) / span
        self.cx, self.cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        self.size = size

    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> tuple:
        px = self.size / 2 + (np.asarray(x) - self.cx) * self.scale
        py = self.size / 2 - (np.asarray(y) - self.cy) * self.scale
        return px, py


def _pt(p: Projector, x: float, y: float) -> tuple[int, int]:
    px, py = p(x, y)
    return round(float(px)), round(float(py))


def draw_minimap(
    geom: MapGeometry,
    size: int,
    trail: Sequence[tuple[float, float]],
    player: tuple[float, float, float] | None,
    monsters: Sequence[tuple[float, float]] = (),
    theme: Theme = DARK,
) -> np.ndarray:
    """Render one minimap frame. `player` is (x, y, angle_degrees)."""
    img = np.full((size, size, 3), hex_to_rgb(theme.background), dtype=np.uint8)
    proj = Projector(geom.bounds, size)
    wall = hex_to_rgb(theme.wall)
    for x1, y1, x2, y2 in geom.passable:
        cv2.line(img, _pt(proj, x1, y1), _pt(proj, x2, y2), hex_to_rgb(theme.grid), 1, cv2.LINE_AA)
    for x1, y1, x2, y2 in geom.walls:
        cv2.line(img, _pt(proj, x1, y1), _pt(proj, x2, y2), wall, 2, cv2.LINE_AA)
    accent = hex_to_rgb(theme.accent)
    if len(trail) > 1:
        pts = np.array([_pt(proj, x, y) for x, y in trail], dtype=np.int32)
        cv2.polylines(img, [pts], False, hex_to_rgb(theme.accent_soft), 2, cv2.LINE_AA)
    r = max(3, size // 90)
    for mx, my in monsters:
        cv2.circle(img, _pt(proj, mx, my), r, hex_to_rgb(theme.muted), -1, cv2.LINE_AA)
    if player is not None:
        x, y, ang = player
        cx, cy = _pt(proj, x, y)
        a = np.deg2rad(ang)
        L = max(10, size // 22)
        tip = (cx + L * np.cos(a), cy - L * np.sin(a))
        left = (cx + 0.6 * L * np.cos(a + 2.5), cy - 0.6 * L * np.sin(a + 2.5))
        right = (cx + 0.6 * L * np.cos(a - 2.5), cy - 0.6 * L * np.sin(a - 2.5))
        tri = np.array([tip, left, right], dtype=np.int32)
        cv2.fillConvexPoly(img, tri, accent, cv2.LINE_AA)
    return img


def objects_from_state(state: vzd.GameState) -> tuple[tuple | None, list[tuple[float, float]]]:
    """Player (x, y, angle) and monster positions from the objects info."""
    player, monsters = None, []
    for o in state.objects or []:
        if o.name == "DoomPlayer" and player is None:
            player = (o.position_x, o.position_y, o.angle)
        elif o.name in MONSTER_CLASSES:
            monsters.append((o.position_x, o.position_y))
    return player, monsters
