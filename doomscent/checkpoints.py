"""Find a run's checkpoints and resolve selectors: a step, `latest`, `best`, or `final`.

Every kept checkpoint is a Sample Factory milestone saved when training crossed a boundary
(every dense interval, or every regular interval when dense checkpoints are off). A
checkpoint is named by its boundary, so the one saved at 2,519,424 env steps is step
2,500,000. Regular checkpoints (multiples of the regular interval) get full evaluation;
dense ones exist for the timelapse.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from doomscent.runs import RunPaths

_NAME = re.compile(r"checkpoint_(\d+)_(\d+)\.pth$")


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    train_step: int
    env_steps: int
    step: int  # the boundary this checkpoint represents
    regular: bool

    @property
    def label(self) -> str:
        return f"step_{self.step:07d}"


def parse_steps(text: str) -> int:
    """'2.5M', '500K', '500000' -> int env steps."""
    t = text.strip().upper()
    mult = {"K": 1_000, "M": 1_000_000}.get(t[-1:], 1)
    num = t[:-1] if mult > 1 else t
    return round(float(num) * mult)


def intervals(paths: RunPaths) -> tuple[int, int]:
    cfg = paths.load_config()
    regular = int(cfg.train.checkpoint_every_env_steps)
    dense = int(cfg.train.get("dense_checkpoint_every_env_steps", 0) or 0)
    return regular, (dense or regular)


def list_checkpoints(paths: RunPaths) -> list[Checkpoint]:
    """All milestone checkpoints, one per boundary (the first saved past it), by step."""
    regular, every = intervals(paths)
    found: dict[int, Checkpoint] = {}
    for f in sorted(paths.milestone_dir.glob("checkpoint_*.pth")):
        m = _NAME.search(f.name)
        if not m:
            continue
        train_step, env_steps = int(m.group(1)), int(m.group(2))
        boundary = (env_steps // every) * every
        ck = Checkpoint(f, train_step, env_steps, boundary, boundary % regular == 0)
        if boundary not in found or env_steps < found[boundary].env_steps:
            found[boundary] = ck
    return [found[k] for k in sorted(found)]


def best_pointer(paths: RunPaths) -> Path:
    return paths.root / "checkpoints_best.json"


def resolve_checkpoint(paths: RunPaths, spec: str) -> Checkpoint:
    cks = list_checkpoints(paths)
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {paths.milestone_dir}")
    s = spec.strip().lower()
    if s in ("latest", "final"):
        return cks[-1]
    if s == "best":
        p = best_pointer(paths)
        if not p.exists():
            raise FileNotFoundError(f"no best checkpoint recorded yet ({p})")
        step = int(json.loads(p.read_text())["step"])
        return next(c for c in cks if c.step == step)
    step = parse_steps(s.removeprefix("step_"))
    for c in cks:
        if c.step == step:
            return c
    raise FileNotFoundError(f"no checkpoint at step {step}; have {[c.step for c in cks]}")
