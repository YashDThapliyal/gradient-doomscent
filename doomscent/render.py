"""Render a recorded demo into clips: replay at 1280x720 with the HUD on, 35 fps H.264.

One replay pass writes any of three versions:
- plain:   the game view.
- minimap: the game view with a top-down minimap inset in the top-right corner.
- overlay: the game view plus a side panel (action bars and a value meter) that never
           covers the action.
The replay is checked against the sidecar (tics and kills) so a clip always matches its data.
"""

from __future__ import annotations

import shutil
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import vizdoom as vzd
from PIL import Image, ImageDraw

from doomscent.env import check_demo_path, scenario_cfg_path, short_demo_path
from doomscent.minimap import MapGeometry, draw_minimap, objects_from_state
from doomscent.style import DARK, Theme, font, hex_to_rgb
from doomscent.video import VideoWriter, draw_tag, paste

VARIANTS = ("plain", "minimap", "overlay")
PANEL_WIDTH = 400


@dataclass
class RenderResult:
    outputs: dict[str, Path]
    replay_tics: int
    replay_kills: float
    matches_sidecar: bool
    trail: list[tuple[float, float]] = field(default_factory=list)
    geometry: MapGeometry | None = None


def _replay_game(
    scenario: str, width: int, height: int, buttons: Iterable[str], hud: bool = True
) -> vzd.DoomGame:
    g = vzd.DoomGame()
    g.load_config(str(scenario_cfg_path(scenario)))
    g.set_window_visible(False)
    g.set_mode(vzd.Mode.PLAYER)
    g.set_screen_resolution(getattr(vzd.ScreenResolution, f"RES_{width}X{height}"))
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_render_hud(hud)
    g.set_render_crosshair(False)
    g.set_render_weapon(True)
    g.set_sectors_info_enabled(True)
    g.set_objects_info_enabled(True)
    g.set_labels_buffer_enabled(False)
    g.set_available_buttons([getattr(vzd.Button, b) for b in buttons])
    g.set_available_game_variables([vzd.GameVariable.KILLCOUNT])
    g.init()
    return g


def _bar_rows(step: Mapping[str, Any], action_space: Mapping[str, Any]) -> list[tuple]:
    """(head, [(choice, share, chosen)]) using recorded probabilities when present."""
    rows = []
    probs = step.get("probs") or {}
    for i, head in enumerate(action_space["heads"]):
        chosen = step["action_idx"][i]
        p = probs.get(head["name"])
        if p is None:
            p = [1.0 if j == chosen else 0.0 for j in range(len(head["choices"]))]
        rows.append(
            (head["name"], [(c, float(p[j]), j == chosen) for j, c in enumerate(head["choices"])])
        )
    return rows


def draw_panel(
    height: int,
    step: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    value_range: tuple[float, float],
    theme: Theme = DARK,
) -> np.ndarray:
    """Side panel: header, kills and health, value meter, and one bar group per action head."""
    img = Image.new("RGB", (PANEL_WIDTH, height), hex_to_rgb(theme.panel))
    d = ImageDraw.Draw(img)
    x0, y = 28, 26
    text, sub = hex_to_rgb(theme.text), hex_to_rgb(theme.subtext)
    acc, muted = hex_to_rgb(theme.accent), hex_to_rgb(theme.muted2)
    d.text((x0, y), f"step {sidecar['step']:,}", font=font(26, "semibold"), fill=text)
    y += 36
    meta = f"seed {sidecar['seed']}  ·  ep {sidecar['episode']:02d}"
    d.text((x0, y), meta, font=font(17), fill=sub)
    y += 40
    gv = step["vars"]
    for label, val in (("kills", gv.get("KILLCOUNT", 0)), ("health", gv.get("HEALTH", 0))):
        d.text((x0, y), label, font=font(16), fill=sub)
        d.text((x0 + 90, y - 4), f"{val:.0f}", font=font(22, "semibold"), fill=text)
        y += 32
    y += 14
    d.text((x0, y), "value estimate", font=font(16), fill=sub)
    y += 26
    v = step.get("value")
    bw = PANEL_WIDTH - 2 * x0
    d.rounded_rectangle([x0, y, x0 + bw, y + 14], radius=7, fill=muted)
    if v is not None:
        lo, hi = value_range
        frac = 0.0 if hi <= lo else float(np.clip((v - lo) / (hi - lo), 0, 1))
        d.rounded_rectangle([x0, y, x0 + max(14, int(bw * frac)), y + 14], radius=7, fill=acc)
        d.text((x0, y + 20), f"{v:+.2f}", font=font(15), fill=sub)
    else:
        d.text((x0, y + 20), "not recorded", font=font(15), fill=sub)
    y += 56
    has_probs = bool(step.get("probs"))
    d.text(
        (x0, y), "action probabilities" if has_probs else "chosen action", font=font(16), fill=sub
    )
    y += 30
    for head, choices in _bar_rows(step, sidecar["action_space"]):
        d.text((x0, y), head, font=font(17, "semibold"), fill=text)
        y += 26
        for choice, share, chosen in choices:
            d.text((x0, y), choice.replace("_", " "), font=font(14), fill=text if chosen else sub)
            bx = x0 + 120
            d.rectangle([bx, y + 4, bx + bw - 120, y + 14], fill=hex_to_rgb(theme.grid))
            if share > 0:
                d.rectangle(
                    [bx, y + 4, bx + max(2, int((bw - 120) * share)), y + 14],
                    fill=acc if chosen else hex_to_rgb(theme.muted),
                )
            y += 21
        y += 12
    return np.asarray(img)


def render_demo(
    demo: str | Path,
    sidecar: Mapping[str, Any],
    variants: Iterable[str],
    out_dir: str | Path | None = None,
    record_cfg: Mapping[str, Any] | None = None,
    theme: Theme = DARK,
) -> RenderResult:
    """Replay a demo and write the requested clip versions next to it (or into out_dir)."""
    demo = Path(demo)
    variants = [v for v in variants]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants {unknown}")
    rc = {"width": 1280, "height": 720, "fps": 35, "crf": 20, "minimap_size": 300}
    rc["minimap_trail_tics"] = 350
    rc.update(record_cfg or {})
    W, H = int(rc["width"]), int(rc["height"])
    out_dir = Path(out_dir) if out_dir else demo.parent
    stem = demo.stem
    names = {"plain": f"{stem}.mp4", "minimap": f"{stem}__minimap.mp4"}
    names["overlay"] = f"{stem}__overlay.mp4"
    outputs = {v: out_dir / names[v] for v in variants}
    writers = {
        v: VideoWriter(
            p, W + (PANEL_WIDTH if v == "overlay" else 0), H, rc["fps"], rc["crf"], rc.get("gop")
        )
        for v, p in outputs.items()
    }
    steps = sidecar["steps"]
    fs = int(sidecar["frame_skip"])
    values = [s["value"] for s in steps if s.get("value") is not None]
    vrange = (min(values), max(values)) if values else (0.0, 1.0)
    buttons = sidecar["action_space"]["buttons"]
    game = _replay_game(sidecar["scenario"], W, H, buttons, bool(rc.get("hud", True)))
    trail: deque = deque(maxlen=int(rc["minimap_trail_tics"]))
    full_trail: list[tuple[float, float]] = []
    geom = None
    panels: dict[int, np.ndarray] = {}
    tics = 0
    short = short_demo_path("_replay")
    shutil.copyfile(demo, short)
    check_demo_path(short)
    try:
        game.replay_episode(str(short))
        while not game.is_episode_finished():
            state = game.get_state()
            frame = state.screen_buffer
            if geom is None:
                geom = MapGeometry.from_state(state)
            player, _ = objects_from_state(state)
            if player is not None:
                trail.append(player[:2])
                full_trail.append(player[:2])
            if "plain" in writers:
                writers["plain"].write(frame)
            if "minimap" in writers:
                mm = draw_minimap(geom, int(rc["minimap_size"]), list(trail), player, (), theme)
                x = W - mm.shape[1] - 20
                writers["minimap"].write(paste(frame, mm, x, 20, alpha=0.88))
            if "overlay" in writers and steps:
                si = min(tics // fs, len(steps) - 1)
                if si not in panels:
                    panels.clear()
                    panels[si] = draw_panel(H, steps[si], sidecar, vrange, theme)
                writers["overlay"].write(np.concatenate([frame, panels[si]], axis=1))
            game.advance_action()
            tics += 1
        kills = float(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        replay_time = game.get_episode_time()
    finally:
        game.close()
        short.unlink(missing_ok=True)
        for w in writers.values():
            w.close()
    expected_tics = sidecar.get("episode_tics")
    ok = kills == sidecar["stats"]["kills"] and (
        expected_tics is None or replay_time == expected_tics
    )
    return RenderResult(outputs, tics, kills, ok, full_trail, geom)


def tag_frames(frames: Iterable[np.ndarray], lines: list[str], theme: Theme = DARK):
    """Yield frames with a corner tag drawn on each (used by montages)."""
    for f in frames:
        yield draw_tag(f, lines, "tl", theme)
