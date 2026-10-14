"""Run folder layout and reproducibility metadata.

runs/<run>/
  config.yaml            resolved experiment config
  meta/                  git commit, action space, scenario hashes, seeds
  sf/                    Sample Factory's experiment folder (checkpoints, summaries)
  metrics/               train.csv, eval.csv, eval_episodes/<ckpt>.csv
  clips/                 train/, eval/<ckpt>/, best/, flagged/, final/
  flagged.csv
  figures/  timelapse/   logs/
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from doomscent.actions import build_action_space
from doomscent.config import REPO_ROOT, save_config
from doomscent.env import scenario_files

RUNS_ROOT = REPO_ROOT / "runs"


@dataclass(frozen=True)
class RunPaths:
    name: str
    root: Path

    @classmethod
    def for_run(cls, name: str, runs_root: Path = RUNS_ROOT) -> RunPaths:
        if not name or "/" in name or name.startswith(("-", ".")):
            raise ValueError(f"bad run name {name!r}")
        return cls(name, runs_root / name)

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def meta(self) -> Path:
        return self.root / "meta"

    @property
    def sf_train_dir(self) -> Path:
        return self.root

    @property
    def sf_experiment(self) -> str:
        return "sf"

    @property
    def sf_dir(self) -> Path:
        return self.root / "sf"

    @property
    def checkpoint_dir(self) -> Path:
        return self.sf_dir / "checkpoint_p0"

    @property
    def milestone_dir(self) -> Path:
        return self.checkpoint_dir / "milestones"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def clips(self) -> Path:
        return self.root / "clips"

    @property
    def flagged_csv(self) -> Path:
        return self.root / "flagged.csv"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def timelapse(self) -> Path:
        return self.root / "timelapse"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def load_config(self) -> DictConfig:
        if not self.config.exists():
            raise FileNotFoundError(f"run {self.name} has no config at {self.config}")
        cfg = OmegaConf.load(self.config)
        assert isinstance(cfg, DictConfig)
        return cfg


def git_commit() -> dict[str, str | bool]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run_metadata(cfg: DictConfig, paths: RunPaths) -> None:
    """Everything needed to re-create the run from its folder alone."""
    paths.meta.mkdir(parents=True, exist_ok=True)
    save_config(cfg, paths.config)
    (paths.meta / "git.json").write_text(json.dumps(git_commit(), indent=2))
    actions = build_action_space(cfg.actions.heads, cfg.actions.turn_speeds_deg_per_tic)
    (paths.meta / "action_space.json").write_text(json.dumps(actions.describe(), indent=2))
    hashes = {p.name: sha256(p) for p in scenario_files(cfg.env.scenario)}
    (paths.meta / "scenario_hashes.json").write_text(json.dumps(hashes, indent=2))
    seeds = OmegaConf.to_container(cfg.seeds, resolve=True)
    (paths.meta / "seeds.json").write_text(
        json.dumps({"training_seed": int(cfg.seed), **seeds}, indent=2)  # type: ignore[dict-item]
    )
    (paths.meta / "bots.json").write_text(json.dumps({"bots": None, "note": "single-player"}))
