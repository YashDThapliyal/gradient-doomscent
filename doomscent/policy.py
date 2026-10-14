"""Load a Sample Factory checkpoint as a policy for our own tools (CPU inference).

The policy is recurrent: call reset() at the start of every episode. act() returns one
choice per head plus the value estimate and per-head action probabilities, which go into
the sidecar JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from doomscent.actions import build_action_space
from doomscent.runs import RunPaths

torch.set_num_threads(1)


def load_sf_config(paths: RunPaths) -> AttrDict:
    return AttrDict(json.loads((paths.sf_dir / "config.json").read_text()))


def build_model(paths: RunPaths) -> tuple[ActorCritic, AttrDict, list[str]]:
    """An untrained actor-critic with the run's architecture and spaces."""
    cfg = paths.load_config()
    sf_cfg = load_sf_config(paths)
    actions = build_action_space(cfg.actions.heads, cfg.actions.turn_speeds_deg_per_tic)
    h, w = int(cfg.env.obs_height), int(cfg.env.obs_width)
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 255, (3, h, w), dtype=np.uint8)})
    model = create_actor_critic(sf_cfg, obs_space, actions.gym_space())
    model.eval()
    return model, sf_cfg, list(actions.head_names)


def load_weights(model: ActorCritic, ckpt_path: str | Path) -> dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt


class SFPolicy:
    def __init__(self, model: ActorCritic, sf_cfg: AttrDict, heads: list[str], deterministic: bool):
        self.model = model
        self.cfg = sf_cfg
        self.heads = heads
        self.deterministic = deterministic
        self.rnn_size = get_rnn_size(sf_cfg)
        self.rnn = torch.zeros(1, self.rnn_size)

    def reset(self) -> None:
        self.rnn = torch.zeros(1, self.rnn_size)

    @torch.no_grad()
    def forward(self, obs: np.ndarray) -> dict[str, Any]:
        """One step: actions, value, per-head probabilities, and the new RNN state."""
        o = {"obs": torch.from_numpy(np.ascontiguousarray(obs)).unsqueeze(0)}
        out = self.model(prepare_and_normalize_obs(self.model, o), self.rnn)
        dist = self.model.action_distribution()
        actions = argmax_actions(dist) if self.deterministic else out["actions"]
        self.rnn = out["new_rnn_states"]
        dists = zip(self.heads, dist.distributions, strict=True)
        probs = {name: d.probs[0].tolist() for name, d in dists}
        return {
            "actions": [int(a) for a in actions.reshape(-1).tolist()],
            "value": self.denormalized_value(out["values"]),
            "probs": probs,
        }

    def denormalized_value(self, values: torch.Tensor) -> float:
        """Expected discounted return in reward units (the critic outputs normalized values)."""
        v = values.reshape(-1)[:1].clone()
        if self.cfg.get("normalize_returns", False):
            self.model.returns_normalizer(v, denormalize=True)
        return float(v[0])

    def act(self, obs: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        out = self.forward(obs)
        return out["actions"], {
            "value": round(out["value"], 4),
            "probs": {k: [round(p, 4) for p in v] for k, v in out["probs"].items()},
        }


def load_policy(paths: RunPaths, ckpt_path: str | Path, deterministic: bool = True) -> SFPolicy:
    model, sf_cfg, heads = build_model(paths)
    load_weights(model, ckpt_path)
    return SFPolicy(model, sf_cfg, heads, deterministic)
