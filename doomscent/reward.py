"""Reward from game-variable deltas. Pure functions, so they are unit-testable without a game.

The scenario's built-in reward is ignored; every term comes from config (`reward:`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

GameVars = Mapping[str, float]


@dataclass(frozen=True)
class RewardWeights:
    kill: float
    death: float
    damage_dealt_per_hp: float
    damage_taken_per_hp: float
    health_armor_pickup_per_point: float
    weapon_ammo_pickup: float
    scale: float = 1.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> RewardWeights:
        return cls(**{k: float(cfg[k]) for k in cls.__dataclass_fields__ if k in cfg})


@dataclass(frozen=True)
class RewardBreakdown:
    kills: float
    death: float
    damage_dealt: float
    damage_taken: float
    health_armor_pickup: float
    weapon_ammo_pickup: float

    @property
    def total(self) -> float:
        return (
            self.kills
            + self.death
            + self.damage_dealt
            + self.damage_taken
            + self.health_armor_pickup
            + self.weapon_ammo_pickup
        )


def _delta(prev: GameVars, cur: GameVars, key: str) -> float:
    if key not in cur or key not in prev:
        return 0.0
    return float(cur[key]) - float(prev[key])


def pickup_events(
    prev: GameVars, cur: GameVars, ammo_vars: Sequence[str], weapon_vars: Sequence[str]
) -> int:
    """Count weapon and ammo pickups: one event per variable that increased this step."""
    return sum(1 for v in (*ammo_vars, *weapon_vars) if _delta(prev, cur, v) > 0)


def step_reward(
    prev: GameVars,
    cur: GameVars,
    weights: RewardWeights,
    died: bool,
    ammo_vars: Sequence[str] = (),
    weapon_vars: Sequence[str] = (),
) -> RewardBreakdown:
    """Reward for one agent step, given game variables before and after it.

    Health gained while dying is impossible, and health or armor can only rise from a
    pickup in single-player, so positive HEALTH and ARMOR deltas count as pickups.
    """
    w = weights
    kills = max(0.0, _delta(prev, cur, "KILLCOUNT"))
    dealt = max(0.0, _delta(prev, cur, "DAMAGECOUNT"))
    taken = max(0.0, _delta(prev, cur, "DAMAGE_TAKEN"))
    gained = 0.0
    if not died:
        gained = max(0.0, _delta(prev, cur, "HEALTH")) + max(0.0, _delta(prev, cur, "ARMOR"))
    events = pickup_events(prev, cur, ammo_vars, weapon_vars)
    return RewardBreakdown(
        kills=w.scale * w.kill * kills,
        death=w.scale * (w.death if died else 0.0),
        damage_dealt=w.scale * w.damage_dealt_per_hp * dealt,
        damage_taken=w.scale * w.damage_taken_per_hp * taken,
        health_armor_pickup=w.scale * w.health_armor_pickup_per_point * gained,
        weapon_ammo_pickup=w.scale * w.weapon_ammo_pickup * events,
    )
