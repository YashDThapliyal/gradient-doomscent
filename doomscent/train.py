"""train: start or resume a run with Sample Factory APPO, configured from our YAML.

    uv run doomscent train --config configs/v0_defend_center_s1.yaml

The run folder (runs/<run_name>/) gets the resolved config and reproducibility metadata
first. Resuming refuses to continue if the config changed: any change needs a new run name.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from omegaconf import DictConfig, OmegaConf

from doomscent.config import load_config
from doomscent.runs import RunPaths, git_commit, write_run_metadata

# Keys that may differ when resuming: bookkeeping and evaluation settings that cannot change
# what the agent learns. Reward, actions, env, seeds, and training settings never may.
RESUME_SAFE_KEYS = ("train.wandb.mode",)
RESUME_SAFE_PREFIXES = ("eval.", "record.", "detectors.", "sync.", "timelapse.", "baseline.")


def sf_argv(cfg: DictConfig, paths: RunPaths) -> list[str]:
    """Sample Factory command-line arguments from an experiment config."""
    t = cfg.train
    args = [
        "--algo=APPO",
        f"--env={t.sf_env_name}",
        f"--experiment={paths.sf_experiment}",
        f"--train_dir={paths.sf_train_dir}",
        f"--seed={int(cfg.seed)}",
        f"--num_workers={int(t.num_workers)}",
        f"--num_envs_per_worker={int(t.num_envs_per_worker)}",
        f"--batch_size={int(t.batch_size)}",
        f"--rollout={int(t.rollout)}",
        f"--recurrence={int(t.recurrence)}",
        f"--use_rnn={bool(t.use_rnn)}",
        f"--rnn_type={t.rnn_type}",
        f"--gamma={float(t.gamma)}",
        f"--learning_rate={float(t.learning_rate)}",
        f"--device={t.device}",
        f"--env_frameskip={int(cfg.env.frame_skip)}",  # our env skips; SF uses it for counting
        f"--train_for_env_steps={int(t.train_for_env_steps)}",
        "--restart_behavior=resume",
        "--stats_avg=100",
        "--keep_checkpoints=100000",  # keep every periodic checkpoint too
        "--save_every_sec=300",  # rolling `latest` for resuming
        "--save_milestones_sec=-1",  # milestones come from StepCheckpointObserver
        "--save_best_after=1000000000000",  # `best` is decided by held-out eval, not here
    ]
    wb = t.wandb
    if wb.enabled:
        args += [
            "--with_wandb=True",
            f"--wandb_project={wb.project}",
            f"--wandb_group={cfg.version}",
            f"--wandb_tags {' '.join(wb.tags)}",
        ]
    args += list(t.get("sf_extra_args", []))
    return [a for arg in args for a in arg.split(" ")]


def checkpoint_interval(cfg: DictConfig) -> int:
    regular = int(cfg.train.checkpoint_every_env_steps)
    dense = int(cfg.train.get("dense_checkpoint_every_env_steps", 0) or 0)
    if dense and regular % dense:
        raise ValueError("the regular checkpoint interval must be a multiple of the dense one")
    return dense or regular


def is_resume_safe(key: str) -> bool:
    return key in RESUME_SAFE_KEYS or key.startswith(RESUME_SAFE_PREFIXES)


def config_diff(old: DictConfig, new: DictConfig) -> list[str]:
    a = OmegaConf.to_container(old, resolve=True)
    b = OmegaConf.to_container(new, resolve=True)
    diffs: list[str] = []

    def walk(x: Any, y: Any, prefix: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{prefix}.{k}" if prefix else str(k))
        elif (
            x != y
            and prefix not in RESUME_SAFE_KEYS
            and not prefix.startswith(RESUME_SAFE_PREFIXES)
        ):
            diffs.append(f"{prefix}: {x!r} -> {y!r}")

    walk(a, b, "")
    return diffs


def prepare_run(cfg: DictConfig) -> RunPaths:
    paths = RunPaths.for_run(str(cfg.run_name))
    if paths.config.exists():
        diffs = config_diff(paths.load_config(), cfg)
        if diffs:
            raise SystemExit(
                f"run {paths.name} exists with a different config; use a new run name.\n  "
                + "\n  ".join(diffs)
            )
        with open(paths.meta / "resumes.log", "a") as f:
            f.write(f"{git_commit()}\n")
    else:
        write_run_metadata(cfg, paths)
    paths.logs.mkdir(parents=True, exist_ok=True)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="doomscent train")
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. train.num_workers=4")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    paths = prepare_run(cfg)
    os.environ["DOOMSCENT_LEARNER_THREADS"] = str(int(cfg.train.get("learner_threads", 1)))
    if cfg.train.wandb.enabled:
        os.environ.setdefault("WANDB_MODE", str(cfg.train.wandb.mode))
        os.environ.setdefault("WANDB_DIR", str(paths.root))

    from sample_factory.algo.utils.misc import ExperimentStatus
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.train import make_runner
    from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults

    from doomscent.sf_integration import (
        StepCheckpointObserver,
        WallClockLimitObserver,
        patch_wandb_init,
        register_doomscent_env,
    )

    patch_wandb_init()
    os.environ["DOOMSCENT_RUN_NAME"] = str(cfg.run_name)

    register_doomscent_env(str(paths.config), cfg.train.sf_env_name)
    sfargs = sf_argv(cfg, paths)
    parser, _ = parse_sf_args(argv=sfargs, evaluation=False)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    sf_cfg = parse_full_cfg(parser, sfargs)
    sf_cfg, runner = make_runner(sf_cfg)
    runner.register_observer(StepCheckpointObserver(checkpoint_interval(cfg)))
    if cfg.train.get("max_wall_seconds"):
        runner.register_observer(WallClockLimitObserver(float(cfg.train.max_wall_seconds)))
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()
    steps = int(runner.env_steps.get(0, 0))
    if status == ExperimentStatus.SUCCESS and steps >= int(cfg.train.train_for_env_steps):
        done = {"status": int(status), "env_steps": steps, "time": time.strftime("%F %T")}
        (paths.root / "TRAINING_DONE.json").write_text(json.dumps(done))
    return int(status)


if __name__ == "__main__":
    sys.exit(main())
