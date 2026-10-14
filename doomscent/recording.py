"""Episode recording: a ViZDoom demo (.lmp) plus a gzipped sidecar JSON per episode.

The sidecar holds per-step actions, game variables, reward, and the policy's value
estimate, so any clip can be re-rendered and any detector re-run later.
"""

from __future__ import annotations

import gzip
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doomscent.env import DoomEnv, short_demo_path
from doomscent.rollout import EpisodeResult, Policy, run_episode

SIDECAR_SUFFIX = ".json.gz"


@dataclass(frozen=True)
class ClipName:
    """Filename stem: run, checkpoint step, scenario, seed, episode, and result."""

    run: str
    step: int
    scenario_short: str
    seed: int
    episode: int
    kills: int

    @property
    def stem(self) -> str:
        return (
            f"{self.run}__step_{self.step:07d}__{self.scenario_short}__seed_{self.seed}"
            f"__ep{self.episode:02d}__kills{self.kills}"
        )

    @staticmethod
    def parse(stem: str) -> ClipName:
        parts = stem.split("__")
        return ClipName(
            run=parts[0],
            step=int(parts[1].removeprefix("step_")),
            scenario_short=parts[2],
            seed=int(parts[3].removeprefix("seed_")),
            episode=int(parts[4].removeprefix("ep")),
            kills=int(parts[5].removeprefix("kills")),
        )


def write_sidecar(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.replace(path)


def read_sidecar(path: str | Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def sidecar_path_for(demo: str | Path) -> Path:
    demo = Path(demo)
    return demo.with_name(demo.stem + SIDECAR_SUFFIX)


def record_episode(
    env: DoomEnv,
    policy: Policy,
    seed: int,
    out_dir: str | Path,
    run: str,
    step: int,
    episode: int,
    meta: dict[str, Any] | None = None,
) -> tuple[EpisodeResult, Path, Path]:
    """Play and record one episode; returns the result, demo path, and sidecar path.

    The demo is recorded under a temporary name, then renamed once the kill count is known.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_demo = short_demo_path()
    res = run_episode(env, policy, seed, record_path=tmp_demo)
    env.reset(options={"episode_seed": seed})  # closes the demo file
    name = ClipName(
        run=run,
        step=step,
        scenario_short=env.env_cfg["scenario_short"],
        seed=seed,
        episode=episode,
        kills=int(res.stats["kills"]),
    )
    demo = out_dir / f"{name.stem}.lmp"
    shutil.move(tmp_demo, demo)
    sidecar = sidecar_path_for(demo)
    data = {
        "schema": 1,
        "run": run,
        "step": step,
        "scenario": env.env_cfg["scenario"],
        "scenario_short": env.env_cfg["scenario_short"],
        "seed": seed,
        "episode": episode,
        "demo": demo.name,
        "frame_skip": env.frame_skip,
        "episode_tics": res.stats["episode_tics"],
        "n_steps": len(res.steps),
        "reward": res.reward,
        "died": res.died,
        "stats": res.stats,
        "start_vars": res.start_vars,
        "action_space": env.actions.describe(),
        **(meta or {}),
        "steps": res.steps,
    }
    write_sidecar(sidecar, data)
    res.demo_path = str(demo)
    return res, demo, sidecar
