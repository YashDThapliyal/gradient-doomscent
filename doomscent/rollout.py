"""Run one episode with any policy and keep a per-step trace (the sidecar JSON content)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from doomscent.env import DoomEnv


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        """Return one choice per head, plus extras such as `value` and `probs`."""
        ...


@dataclass
class EpisodeResult:
    seed: int
    stats: dict[str, float]
    steps: list[dict[str, Any]]
    start_vars: dict[str, float]
    died: bool
    reward: float
    demo_path: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class RandomPolicy:
    """Uniform random choice per head."""

    def __init__(self, env: DoomEnv, seed: int = 0) -> None:
        self.sizes = [len(h.choices) for h in env.actions.heads]
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        return [int(self.rng.integers(n)) for n in self.sizes], {}


class SpinAndShootPolicy:
    """Scripted baseline: turn the same way at a fixed speed and hold fire every step."""

    def __init__(self, env: DoomEnv, turn_choice: int) -> None:
        names = env.actions.head_names
        if "turn" not in names or "attack" not in names:
            raise ValueError("spin-and-shoot needs the turn and attack heads")
        self.action = [0] * len(names)
        self.action[names.index("turn")] = int(turn_choice)
        self.action[names.index("attack")] = 1

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        return list(self.action), {}


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def run_episode(
    env: DoomEnv,
    policy: Policy,
    episode_seed: int,
    record_path: str | Path | None = None,
    keep_trace: bool = True,
) -> EpisodeResult:
    """Play one episode. With `record_path`, ViZDoom writes a .lmp demo of it."""
    obs, info = env.reset(
        options={"episode_seed": episode_seed, "record_path": str(record_path or "")}
    )
    policy.reset()
    start_vars = dict(info["game_vars"])
    heads = env.actions.heads
    turn_i = env.actions.head_names.index("turn") if "turn" in env.actions.head_names else None
    steps: list[dict[str, Any]] = []
    total = 0.0
    done = False
    t = 0
    while not done:
        action, extras = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total += reward
        if keep_trace:
            rec = {
                "t": t,
                "tic": (t + 1) * env.frame_skip,
                "action_idx": list(action),
                "action": {h.name: h.choices[a] for h, a in zip(heads, action, strict=True)},
                "turn_delta": env.actions.turn_delta(action[turn_i]) if turn_i is not None else 0,
                "vars": info["game_vars"],
                "reward": reward,
                "enemies_visible": info["enemies_visible"],
            }
            rec.update(extras)
            steps.append(_to_jsonable(rec))
        t += 1
    return EpisodeResult(
        seed=episode_seed,
        stats=info["episode_extra_stats"],
        steps=steps,
        start_vars=start_vars,
        died=bool(info["died"]),
        reward=total,
        demo_path=str(record_path) if record_path else None,
    )
