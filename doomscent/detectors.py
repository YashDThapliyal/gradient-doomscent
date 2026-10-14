"""Anomaly detectors that auto-flag episodes. Pure functions over an episode trace.

A trace is the list of per-step records stored in each episode's sidecar JSON, so the
detectors can be re-run on old recordings. Thresholds come from config (`detectors:`).
Each step record holds at least: `action` (head name -> choice label), `turn_delta`
(degrees per tic), `vars` (game variables after the step), `enemies_visible`, `reward`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Step = Mapping[str, Any]


@dataclass(frozen=True)
class Flag:
    detector: str
    detail: str


def _positions(steps: Sequence[Step]) -> np.ndarray:
    return np.array(
        [[s["vars"].get("POSITION_X", 0.0), s["vars"].get("POSITION_Y", 0.0)] for s in steps],
        dtype=float,
    )


def _max_displacement(pos: np.ndarray) -> float:
    if len(pos) == 0:
        return 0.0
    return float(np.max(np.linalg.norm(pos - pos[0], axis=1)))


def shots_per_step(steps: Sequence[Step], start_vars: Mapping[str, float]) -> np.ndarray:
    """Ammo spent on each step (weapon unchanged), a proxy for shots fired."""
    out = np.zeros(len(steps))
    prev = start_vars
    for i, s in enumerate(steps):
        cur = s["vars"]
        if cur.get("SELECTED_WEAPON") == prev.get("SELECTED_WEAPON"):
            spent = prev.get("SELECTED_WEAPON_AMMO", 0.0) - cur.get("SELECTED_WEAPON_AMMO", 0.0)
            out[i] = max(0.0, spent)
        prev = cur
    return out


def spinning(steps: Sequence[Step], cfg: Mapping[str, Any]) -> Flag | None:
    if not steps:
        return None
    turns = np.array([float(s.get("turn_delta", 0.0)) for s in steps])
    turning = turns != 0
    frac = turning.mean()
    if frac < cfg["min_turn_fraction"]:
        return None
    same = max((turns > 0).sum(), (turns < 0).sum()) / max(1, turning.sum())
    disp = _max_displacement(_positions(steps))
    if same >= cfg["same_direction_fraction"] and disp <= cfg["max_displacement"]:
        return Flag(
            "spinning",
            f"turning {frac:.0%} of steps, {same:.0%} one way, moved {disp:.0f} units",
        )
    return None


def stuck(steps: Sequence[Step], cfg: Mapping[str, Any]) -> Flag | None:
    pos = _positions(steps)
    run_start = None
    for i, s in enumerate(steps):
        forward = s["action"].get("move") == "forward"
        if forward and run_start is None:
            run_start = i
        if run_start is not None and (not forward or i == len(steps) - 1):
            end = i if forward else i - 1
            length = end - run_start + 1
            if length >= cfg["min_forward_steps"]:
                disp = _max_displacement(pos[run_start : end + 1])
                if disp <= cfg["max_displacement"]:
                    return Flag("stuck", f"{length} forward steps, moved {disp:.0f} units")
            run_start = None
    return None


def camping(steps: Sequence[Step], cfg: Mapping[str, Any]) -> Flag | None:
    pos = _positions(steps)
    if len(pos) == 0:
        return None
    center = np.median(pos, axis=0)
    inside = (np.linalg.norm(pos - center, axis=1) <= cfg["radius"]).mean()
    if inside >= cfg["min_fraction"]:
        return Flag("camping", f"{inside:.0%} of steps within {cfg['radius']:.0f} units")
    return None


def wasted_fire(steps: Sequence[Step], start_vars: Mapping[str, float], cfg: Mapping[str, Any]):
    shots = shots_per_step(steps, start_vars).sum()
    if not steps or shots < cfg["min_shots"]:
        return None
    hits = steps[-1]["vars"].get("HITCOUNT", 0.0) - start_vars.get("HITCOUNT", 0.0)
    acc = hits / shots
    if acc <= cfg["max_accuracy"]:
        return Flag("wasted_fire", f"{shots:.0f} shots, {hits:.0f} hits ({acc:.0%})")
    return None


def pacifist(steps: Sequence[Step], start_vars: Mapping[str, float], cfg: Mapping[str, Any]):
    shots = shots_per_step(steps, start_vars).sum()
    fired = shots > 0 or any(s["action"].get("attack") == "on" for s in steps)
    visible = sum(1 for s in steps if s.get("enemies_visible", 0) > 0)
    if not fired and visible >= cfg["min_enemy_visible_steps"]:
        return Flag("pacifist", f"never fired; enemies on screen for {visible} steps")
    return None


def suicide(
    steps: Sequence[Step], start_vars: Mapping[str, float], died: bool, cfg: Mapping[str, Any]
) -> Flag | None:
    """Death right after firing a rocket: the agent's own blast is the likely killer."""
    if not died or not steps:
        return None
    last = steps[-1]["vars"]
    if int(last.get("SELECTED_WEAPON", -1)) != int(cfg["rocket_weapon_slot"]):
        return None
    window = int(cfg["window_steps"])
    if shots_per_step(steps, start_vars)[-window:].sum() > 0:
        return Flag("suicide", f"died within {window} steps of firing a rocket")
    return None


def reward_spike(
    episode_reward: float, history: Sequence[float], cfg: Mapping[str, Any]
) -> Flag | None:
    recent = np.asarray(history[-int(cfg["window"]) :], dtype=float)
    if len(recent) < int(cfg["min_history"]):
        return None
    mean, std = recent.mean(), recent.std()
    z = (episode_reward - mean) / max(std, 1e-6)
    if z >= cfg["z"]:
        return Flag("reward_spike", f"reward {episode_reward:.2f} is {z:.1f} sd above {mean:.2f}")
    return None


def run_detectors(
    steps: Sequence[Step],
    start_vars: Mapping[str, float],
    died: bool,
    episode_reward: float,
    reward_history: Sequence[float],
    cfg: Mapping[str, Any],
) -> list[Flag]:
    """Run every enabled detector on one episode."""
    checks = {
        "spinning": lambda c: spinning(steps, c),
        "stuck": lambda c: stuck(steps, c),
        "camping": lambda c: camping(steps, c),
        "wasted_fire": lambda c: wasted_fire(steps, start_vars, c),
        "pacifist": lambda c: pacifist(steps, start_vars, c),
        "suicide": lambda c: suicide(steps, start_vars, died, c),
        "reward_spike": lambda c: reward_spike(episode_reward, reward_history, c),
    }
    flags = []
    for name, check in checks.items():
        c = cfg.get(name)
        if c and c.get("enabled", False):
            flag = check(c)
            if flag is not None:
                flags.append(flag)
    return flags
