"""The factored discrete action space.

Every head is an independent categorical choice. A scenario enables a subset of heads
(v0 uses only `turn` and `attack`); later versions enable more without changing this file.
Turning uses ViZDoom's TURN_LEFT_RIGHT_DELTA button, whose value is applied every tic
(positive turns right), so a speed in degrees per tic is exact under frame skip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import vizdoom as vzd

ALL_HEADS = ("move", "strafe", "turn", "attack", "weapon")


@dataclass(frozen=True)
class Head:
    name: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class FactoredActionSpace:
    heads: tuple[Head, ...]
    turn_speeds: tuple[float, ...]

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(h.name for h in self.heads)

    def gym_space(self) -> gym.spaces.Tuple:
        return gym.spaces.Tuple([gym.spaces.Discrete(len(h.choices)) for h in self.heads])

    def buttons(self) -> list[vzd.Button]:
        """ViZDoom buttons to enable, in the order `to_buttons` fills them."""
        out: list[vzd.Button] = []
        for h in self.heads:
            out.extend(_HEAD_BUTTONS[h.name])
        return out

    def to_buttons(self, action: Sequence[int]) -> list[float]:
        """Convert one choice index per head into ViZDoom button values."""
        if len(action) != len(self.heads):
            raise ValueError(f"expected {len(self.heads)} action heads, got {len(action)}")
        values: list[float] = []
        for head, idx in zip(self.heads, action, strict=True):
            idx = int(idx)
            if not 0 <= idx < len(head.choices):
                raise ValueError(f"{head.name} choice {idx} out of range")
            values.extend(self._head_values(head.name, idx))
        return values

    def turn_delta(self, turn_idx: int) -> float:
        """Signed degrees per tic for a turn-head index (negative is left)."""
        if turn_idx == 0:
            return 0.0
        n = len(self.turn_speeds)
        if turn_idx <= n:
            return -self.turn_speeds[turn_idx - 1]
        return self.turn_speeds[turn_idx - 1 - n]

    def _head_values(self, name: str, idx: int) -> list[float]:
        if name == "move":
            return [float(idx == 1), float(idx == 2)]
        if name == "strafe":
            return [float(idx == 1), float(idx == 2)]
        if name == "turn":
            return [self.turn_delta(idx)]
        if name == "attack":
            return [float(idx)]
        if name == "weapon":
            return [float(idx)]
        raise KeyError(name)

    def describe(self) -> dict[str, Any]:
        """JSON-friendly description saved into every run folder."""
        return {
            "heads": [{"name": h.name, "choices": list(h.choices)} for h in self.heads],
            "turn_speeds_deg_per_tic": list(self.turn_speeds),
            "buttons": [b.name for b in self.buttons()],
        }


_HEAD_BUTTONS: dict[str, list[vzd.Button]] = {
    "move": [vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD],
    "strafe": [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT],
    "turn": [vzd.Button.TURN_LEFT_RIGHT_DELTA],
    "attack": [vzd.Button.ATTACK],
    "weapon": [vzd.Button.SELECT_NEXT_WEAPON],
}


def _make_head(name: str, turn_speeds: Sequence[float]) -> Head:
    if name == "move":
        return Head(name, ("none", "forward", "back"))
    if name == "strafe":
        return Head(name, ("none", "left", "right"))
    if name == "turn":
        labels = ["slow", "med", "fast", "x4", "x5", "x6"][: len(turn_speeds)]
        left = [f"left_{lbl}" for lbl in labels]
        right = [f"right_{lbl}" for lbl in labels]
        return Head(name, ("none", *left, *right))
    if name == "attack":
        return Head(name, ("off", "on"))
    if name == "weapon":
        return Head(name, ("none", "next"))
    raise ValueError(f"unknown action head {name!r}; choose from {ALL_HEADS}")


def build_action_space(heads: Sequence[str], turn_speeds: Sequence[float]) -> FactoredActionSpace:
    """Build the action space from config (`actions.heads`, `actions.turn_speeds_deg_per_tic`)."""
    if not heads:
        raise ValueError("at least one action head is required")
    if len(set(heads)) != len(heads):
        raise ValueError("duplicate action heads")
    speeds = tuple(float(s) for s in turn_speeds)
    if "turn" in heads and (not speeds or any(s <= 0 for s in speeds) or len(speeds) > 6):
        raise ValueError("turn speeds must be 1 to 6 positive values")
    return FactoredActionSpace(tuple(_make_head(h, speeds) for h in heads), speeds)
