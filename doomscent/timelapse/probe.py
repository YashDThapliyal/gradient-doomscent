"""The fixed probe set: a few hundred held-out frames covering distinct situations.

Frames come from replaying held-out evaluation demos with the policy's own view (128x72,
HUD off) and the labels buffer on. Each frame is tagged no_enemy, enemy_far, enemy_close,
several_enemies, low_health, or low_ammo, and keeps the frames before it so the recurrent
core can be given real context. Every checkpoint sees exactly this set.
"""

from __future__ import annotations

import shutil
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import vizdoom as vzd

from doomscent.env import MONSTER_CLASSES, scenario_cfg_path, short_demo_path

CATEGORIES = (
    "no_enemy",
    "enemy_far",
    "enemy_close",
    "several_enemies",
    "low_health",
    "low_ammo",
)


def categorize(enemy_heights: list[int], health: float, ammo: float, cfg: Mapping) -> str:
    if health <= cfg["probe_low_health"]:
        return "low_health"
    if ammo <= cfg["probe_low_ammo"]:
        return "low_ammo"
    if len(enemy_heights) >= 3:
        return "several_enemies"
    if not enemy_heights:
        return "no_enemy"
    if max(enemy_heights) >= cfg["probe_close_px"]:
        return "enemy_close"
    if max(enemy_heights) <= cfg["probe_far_px"]:
        return "enemy_far"
    return "enemy_mid"


def _replay_game(env_cfg: Mapping[str, Any], buttons: list[str]) -> vzd.DoomGame:
    g = vzd.DoomGame()
    g.load_config(str(scenario_cfg_path(env_cfg["scenario"])))
    g.set_window_visible(False)
    g.set_mode(vzd.Mode.PLAYER)
    g.set_screen_resolution(getattr(vzd.ScreenResolution, env_cfg["render_resolution"]))
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_render_hud(bool(env_cfg["hud"]))
    g.set_render_crosshair(False)
    g.set_labels_buffer_enabled(True)
    g.set_available_buttons([getattr(vzd.Button, b) for b in buttons])
    g.set_available_game_variables([vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2])
    g.init()
    return g


def demo_frames(
    demo: Path, env_cfg: Mapping[str, Any], buttons: list[str], frame_skip: int, context: int
):
    """Yield (obs, context_obs, enemy_heights, health, ammo, step) every agent step."""
    h, w = int(env_cfg["obs_height"]), int(env_cfg["obs_width"])
    g = _replay_game(env_cfg, buttons)
    short = short_demo_path("_probe")
    shutil.copyfile(demo, short)
    hist: deque = deque(maxlen=context)
    try:
        g.replay_episode(str(short))
        tic = 0
        while not g.is_episode_finished():
            if tic % frame_skip == 0:
                s = g.get_state()
                obs = np.ascontiguousarray(
                    cv2.resize(s.screen_buffer, (w, h), interpolation=cv2.INTER_AREA).transpose(
                        2, 0, 1
                    )
                )
                heights = [lb.height for lb in s.labels if lb.object_name in MONSTER_CLASSES]
                hp, ammo = s.game_variables
                ctx = list(hist)
                yield obs, ctx, heights, float(hp), float(ammo), tic // frame_skip
                hist.append(obs)
            g.advance_action()
            tic += 1
    finally:
        g.close()
        short.unlink(missing_ok=True)


def build_probe_set(
    demos: list[Path],
    env_cfg: Mapping[str, Any],
    tl_cfg: Mapping[str, Any],
    buttons: list[str],
    frame_skip: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Pick `probe_size` frames spread evenly over the categories, from the given demos."""
    context = int(tl_cfg["probe_context"])
    pools: dict[str, list[tuple]] = {c: [] for c in (*CATEGORIES, "enemy_mid")}
    for d in demos:
        for obs, ctx, heights, hp, ammo, step in demo_frames(
            d, env_cfg, buttons, frame_skip, context
        ):
            if len(ctx) < context:
                continue
            pools[categorize(heights, hp, ammo, tl_cfg)].append((obs, np.stack(ctx), d.name, step))
    rng = np.random.default_rng(seed)
    n = int(tl_cfg["probe_size"])
    per = n // len(CATEGORIES)
    chosen: list[tuple] = []
    cats: list[str] = []
    leftovers: list[tuple] = []
    for c in CATEGORIES:
        pool = pools[c]
        idx = rng.permutation(len(pool))
        chosen += [pool[i] for i in idx[:per]]
        cats += [c] * min(per, len(pool))
        leftovers += [(pool[i], c) for i in idx[per:]]
    leftovers += [(p, "enemy_mid") for p in pools["enemy_mid"]]
    for i in rng.permutation(len(leftovers))[: max(0, n - len(chosen))]:
        chosen.append(leftovers[i][0])
        cats.append(leftovers[i][1])
    return {
        "obs": np.stack([c[0] for c in chosen]),
        "context": np.stack([c[1] for c in chosen]),
        "category": np.array(cats),
        "source": np.array([f"{c[2]}#{c[3]}" for c in chosen]),
    }
