"""make_env(): ViZDoom game setup, factored action space, reward, and seeding.

One Gymnasium env class serves Sample Factory training and every tool of ours (evaluate,
record, baseline), so the policy sees identical observations everywhere.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from omegaconf import DictConfig, OmegaConf

from doomscent.actions import FactoredActionSpace, build_action_space
from doomscent.reward import RewardWeights, step_reward

REPO_SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"

# ZDoom actor class names of monsters, for counting enemies in the labels buffer.
MONSTER_CLASSES = frozenset(
    {
        "Zombieman",
        "ShotgunGuy",
        "ChaingunGuy",
        "DoomImp",
        "Demon",
        "Spectre",
        "LostSoul",
        "Cacodemon",
        "HellKnight",
        "BaronOfHell",
        "Arachnotron",
        "PainElemental",
        "Revenant",
        "Fatso",
        "Archvile",
        "Cyberdemon",
        "SpiderMastermind",
        "WolfensteinSS",
        "MarineChainsawVzd",
    }
)

SPLITS = ("train", "heldout")


def scenario_cfg_path(scenario: str) -> Path:
    """Our own `scenarios/` folder first, then ViZDoom's bundled scenarios."""
    for root in (REPO_SCENARIOS, Path(vzd.scenarios_path)):
        p = root / f"{scenario}.cfg"
        if p.exists():
            return p
    raise FileNotFoundError(f"scenario {scenario!r} not found in {REPO_SCENARIOS} or ViZDoom")


def scenario_files(scenario: str) -> list[Path]:
    """The scenario's .cfg and the .wad it names, for hashing into the run folder."""
    cfg_path = scenario_cfg_path(scenario)
    files = [cfg_path]
    for line in cfg_path.read_text().splitlines():
        key, _, val = line.partition("=")
        if key.strip().lower().replace("_", "") == "doomscenariopath":
            files.append(cfg_path.parent / val.strip())
    return files


# ViZDoom passes demo paths to the engine as console arguments. On this build it hangs
# (no error) when the path is longer than 112 characters, has a space, or has a component
# starting with '-'. Demos are therefore recorded and replayed through short temp paths.
MAX_DEMO_PATH = 100
SHORT_DEMO_DIR = Path("/tmp") / f"doomscent_{os.getuid()}"


def check_demo_path(path: str | Path) -> None:
    """Raise instead of letting ViZDoom hang on a demo path it cannot handle."""
    s = str(path)
    bad = [part for part in Path(path).parts if part.startswith("-")]
    if bad or " " in s or len(s) > MAX_DEMO_PATH:
        raise ValueError(f"demo path unusable by ViZDoom (len {len(s)}, '-' or space): {s}")


def short_demo_path(tag: str = "") -> Path:
    """A fresh short path for ViZDoom to read or write a demo through."""
    SHORT_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    return SHORT_DEMO_DIR / f"{os.getpid()}_{uuid.uuid4().hex[:12]}{tag}.lmp"


def _plain(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        out = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(out, dict)
        return out
    return dict(cfg)


class DoomEnv(gym.Env):
    """Defend-the-Center-style single-player env with factored actions and shaped reward.

    Observations are uint8 CHW RGB (3, obs_height, obs_width), HUD off. Game variables are
    returned in `info` for reward, metrics, and detectors, and never in the observation.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 35}

    def __init__(
        self,
        cfg: Mapping[str, Any] | DictConfig,
        split: str = "train",
        rng_seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        full = _plain(cfg)
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.cfg = full
        self.env_cfg = full["env"]
        self.split = split
        self.render_mode = render_mode
        seeds = full["seeds"][split]
        self.seed_range = (int(seeds["start"]), int(seeds["stop"]))
        self.actions: FactoredActionSpace = build_action_space(
            full["actions"]["heads"], full["actions"]["turn_speeds_deg_per_tic"]
        )
        self.weights = RewardWeights.from_config(full["reward"])
        self.var_names: list[str] = list(self.env_cfg["game_variables"])
        self.ammo_vars: list[str] = list(self.env_cfg.get("ammo_variables", []))
        self.weapon_vars: list[str] = list(self.env_cfg.get("weapon_variables", []))
        self.frame_skip = int(self.env_cfg["frame_skip"])
        self.obs_hw = (int(self.env_cfg["obs_height"]), int(self.env_cfg["obs_width"]))
        self.coverage_cell = float(full.get("coverage_cell", 64))

        self.action_space = self.actions.gym_space()
        self.observation_space = gym.spaces.Box(0, 255, (3, *self.obs_hw), dtype=np.uint8)

        self._rng = np.random.default_rng(rng_seed)
        self.game: vzd.DoomGame | None = None
        self._last_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self._prev_vars: dict[str, float] = {}
        self._ep: dict[str, Any] = {}

    # ----- game setup -------------------------------------------------------------------

    def _init_game(self) -> vzd.DoomGame:
        e = self.env_cfg
        game = vzd.DoomGame()
        game.load_config(str(scenario_cfg_path(e["scenario"])))
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_resolution(getattr(vzd.ScreenResolution, e["render_resolution"]))
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_render_hud(bool(e["hud"]))
        game.set_render_crosshair(False)
        game.set_render_screen_flashes(True)
        game.set_labels_buffer_enabled(bool(e.get("labels_buffer", False)))
        game.set_automap_buffer_enabled(bool(e.get("automap_buffer", False)))
        game.set_available_buttons(self.actions.buttons())
        game.set_available_game_variables([getattr(vzd.GameVariable, v) for v in self.var_names])
        if e.get("episode_timeout"):
            game.set_episode_timeout(int(e["episode_timeout"]))
        if e.get("doom_skill"):
            game.set_doom_skill(int(e["doom_skill"]))
        game.init()
        return game

    # ----- helpers ----------------------------------------------------------------------

    def _read_vars(self) -> dict[str, float]:
        assert self.game is not None
        state = self.game.get_state()
        if state is not None and state.game_variables is not None:
            return {n: float(v) for n, v in zip(self.var_names, state.game_variables, strict=True)}
        return {
            n: float(self.game.get_game_variable(getattr(vzd.GameVariable, n)))
            for n in self.var_names
        }

    def _observe(self, state: vzd.GameState | None) -> np.ndarray:
        if state is None or state.screen_buffer is None:
            return self._last_obs
        h, w = self.obs_hw
        frame = cv2.resize(state.screen_buffer, (w, h), interpolation=cv2.INTER_AREA)
        self._last_obs = np.ascontiguousarray(frame.transpose(2, 0, 1))
        return self._last_obs

    @staticmethod
    def _enemies_visible(state: vzd.GameState | None) -> int:
        if state is None or not state.labels:
            return 0
        return sum(1 for lbl in state.labels if lbl.object_name in MONSTER_CLASSES)

    def draw_seed(self) -> int:
        return int(self._rng.integers(*self.seed_range))

    # ----- gym API ----------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode. `options` may hold `episode_seed` and `record_path` (a .lmp)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        if self.game is None:
            self.game = self._init_game()
        ep_seed = options.get("episode_seed")
        ep_seed = self.draw_seed() if ep_seed is None else int(ep_seed)
        lo, hi = self.seed_range
        if not lo <= ep_seed < hi:
            raise ValueError(f"episode seed {ep_seed} outside the {self.split} range [{lo}, {hi})")
        self.game.set_seed(ep_seed)
        record = options.get("record_path")
        if record:
            check_demo_path(record)
            Path(record).parent.mkdir(parents=True, exist_ok=True)
            self.game.new_episode(str(record))
        else:
            self.game.new_episode()
        state = self.game.get_state()
        obs = self._observe(state)
        self._prev_vars = self._read_vars()
        self._ep = {
            "seed": ep_seed,
            "steps": 0,
            "reward": 0.0,
            "start": dict(self._prev_vars),
            "shots": 0.0,
            "pickups": 0,
            "cells": set(),
            "enemy_steps": 0,
            "action_counts": [np.zeros(len(h.choices)) for h in self.actions.heads],
        }
        self._visit(self._prev_vars)
        info = {"game_vars": dict(self._prev_vars), "episode_seed": ep_seed, "enemies_visible": 0}
        info["enemies_visible"] = self._enemies_visible(state)
        return obs, info

    def _visit(self, gv: Mapping[str, float]) -> None:
        if "POSITION_X" in gv and "POSITION_Y" in gv:
            c = self.coverage_cell
            self._ep["cells"].add((int(gv["POSITION_X"] // c), int(gv["POSITION_Y"] // c)))

    def step(self, action: Sequence[int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.game is not None, "call reset() first"
        action = [int(a) for a in np.asarray(action).reshape(-1)]
        self.game.make_action(self.actions.to_buttons(action), self.frame_skip)
        finished = self.game.is_episode_finished()
        died = bool(self.game.is_player_dead())
        state = None if finished else self.game.get_state()
        obs = self._observe(state)
        cur = self._read_vars()
        prev = self._prev_vars
        rb = step_reward(prev, cur, self.weights, died, self.ammo_vars, self.weapon_vars)
        reward = rb.total

        ep = self._ep
        ep["steps"] += 1
        ep["reward"] += reward
        if cur.get("SELECTED_WEAPON") == prev.get("SELECTED_WEAPON"):
            ep["shots"] += max(
                0.0, prev.get("SELECTED_WEAPON_AMMO", 0.0) - cur.get("SELECTED_WEAPON_AMMO", 0.0)
            )
        ep["pickups"] += int(rb.weapon_ammo_pickup != 0) + int(rb.health_armor_pickup != 0)
        enemies = self._enemies_visible(state)
        ep["enemy_steps"] += int(enemies > 0)
        for counts, a in zip(ep["action_counts"], action, strict=True):
            counts[a] += 1
        self._visit(cur)
        self._prev_vars = cur

        terminated = finished and died
        truncated = finished and not died
        info: dict[str, Any] = {
            "game_vars": cur,
            "reward_terms": rb,
            "died": died,
            "enemies_visible": enemies,
            "episode_seed": ep["seed"],
        }
        if finished:
            info["episode_extra_stats"] = self.episode_stats(died)
        return obs, float(reward), terminated, truncated, info

    def episode_stats(self, died: bool) -> dict[str, float]:
        """Per-episode summary; Sample Factory averages these into its training stats."""
        ep, start, end = self._ep, self._ep["start"], self._prev_vars

        def d(key: str) -> float:
            return float(end.get(key, 0.0) - start.get(key, 0.0))

        shots = ep["shots"]
        hits = d("HITCOUNT")
        stats = {
            "kills": d("KILLCOUNT"),
            "deaths": float(died),
            "damage_dealt": d("DAMAGECOUNT"),
            "damage_taken": d("DAMAGE_TAKEN"),
            "shots": shots,
            "hits": hits,
            "accuracy": hits / shots if shots > 0 else 0.0,
            "survival_time_s": ep["steps"] * self.frame_skip / 35.0,
            "episode_steps": float(ep["steps"]),
            "episode_tics": float(self.game.get_episode_time()) if self.game else 0.0,
            "pickups": float(ep["pickups"]),
            "coverage": float(len(ep["cells"])),
            "enemy_visible_steps": float(ep["enemy_steps"]),
            "shaped_reward": ep["reward"],
        }
        steps = max(1, ep["steps"])
        for head, counts in zip(self.actions.heads, ep["action_counts"], strict=True):
            for choice, c in zip(head.choices, counts, strict=True):
                stats[f"act_{head.name}_{choice}"] = float(c) / steps
        return stats

    def render(self) -> np.ndarray | None:
        return self._last_obs.transpose(1, 2, 0) if self.render_mode == "rgb_array" else None

    def close(self) -> None:
        if self.game is not None:
            self.game.close()
            self.game = None


def make_env(
    cfg: Mapping[str, Any] | DictConfig,
    split: str = "train",
    rng_seed: int | None = None,
    render_mode: str | None = None,
) -> DoomEnv:
    """Build the env described by an experiment config."""
    return DoomEnv(cfg, split=split, rng_seed=rng_seed, render_mode=render_mode)
