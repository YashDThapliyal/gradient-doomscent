"""Sample Factory glue: env registration, per-head entropy stats, step-based checkpoints.

Sample Factory starts its workers with multiprocessing "spawn", so everything here lives at
module level: unpickling the registered env factory in a worker imports this module, which
re-applies the per-head entropy patch in that process too.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any

import sample_factory.algo.learning.learner_worker as _learner_worker
import torch
from omegaconf import OmegaConf
from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.algo.utils.action_distributions import TupleActionDistribution
from sample_factory.envs.env_utils import register_env

from doomscent.env import DoomEnv

SF_ENV_NAME = "doomscent_v0"


# ----- per-head entropy ---------------------------------------------------------------------


def _tuple_summaries(self: TupleActionDistribution) -> dict[str, torch.Tensor]:
    """Per-head entropy; the learner logs any `summaries()` of the action distribution."""
    return {f"entropy_head{i}": d.entropy().mean() for i, d in enumerate(self.distributions)}


TupleActionDistribution.summaries = _tuple_summaries  # type: ignore[attr-defined]


# ----- learner threads ---------------------------------------------------------------------

LEARNER_THREADS_ENV = "DOOMSCENT_LEARNER_THREADS"


def _learner_torch_runtime(cfg: Any, max_num_threads: int | None = 1) -> None:
    """Sample Factory pins the learner to one torch thread (fine on a GPU, a bottleneck on a
    CPU-only Mac). The learner process imports this module when it unpickles our env
    factory, before it initializes torch, so the thread count from config applies there."""
    threads = int(os.environ.get(LEARNER_THREADS_ENV, "1"))
    _original_init_torch_runtime(cfg, max_num_threads=threads)
    print(f"[doomscent] learner torch threads: {torch.get_num_threads()}", flush=True)


_original_init_torch_runtime = _learner_worker.init_torch_runtime
if _original_init_torch_runtime is not _learner_torch_runtime:
    _learner_worker.init_torch_runtime = _learner_torch_runtime


# ----- env --------------------------------------------------------------------------------


def make_sf_env(
    full_env_name: str,
    cfg: Any,
    env_config: Any = None,
    render_mode: str | None = None,
    doomscent_config: str = "",
) -> DoomEnv:
    """Sample Factory env factory: our DoomEnv on the training seed split.

    Each env instance gets its own RNG seed from the run seed and its worker and vector
    index, and draws a fresh training-range scenario seed for every episode.
    """
    exp_cfg = OmegaConf.load(doomscent_config)
    worker = int(getattr(env_config, "worker_index", 0) or 0) if env_config else 0
    vector = int(getattr(env_config, "vector_index", 0) or 0) if env_config else 0
    rng_seed = int(exp_cfg.seed) * 1_000_003 + worker * 1009 + vector
    return DoomEnv(exp_cfg, split="train", rng_seed=rng_seed, render_mode=render_mode)


def register_doomscent_env(config_path: str, env_name: str = SF_ENV_NAME) -> None:
    register_env(env_name, functools.partial(make_sf_env, doomscent_config=config_path))


# ----- checkpoints every N env steps --------------------------------------------------------


class StepCheckpointObserver(AlgoObserver):
    """Save a kept-forever milestone checkpoint at step 0 and every `every` env steps.

    Milestones are never pruned by Sample Factory, so together they are the full checkpoint
    history (regular 500K checkpoints plus optional dense ones for the timelapse).
    """

    def __init__(self, every: int) -> None:
        if every <= 0:
            raise ValueError("checkpoint interval must be positive")
        self.every = int(every)
        self.next_boundary: int | None = None

    def on_start(self, runner: Runner) -> None:
        steps = int(runner.env_steps.get(0, 0))
        if steps == 0:
            runner.save_milestone.emit()  # the untrained policy, for step-0 clips
        self.next_boundary = (steps // self.every + 1) * self.every

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        steps = int(runner.env_steps.get(0, 0))
        if self.next_boundary is None:
            self.next_boundary = (steps // self.every + 1) * self.every
        if steps >= self.next_boundary:
            runner.save_milestone.emit()
            self.next_boundary = (steps // self.every + 1) * self.every


class WallClockLimitObserver(AlgoObserver):
    """Stop training after a wall-clock budget.

    Sample Factory 2.1.1's `--train_for_seconds` never fires in the parallel runner (its
    `total_train_seconds` counter is never updated), so the throughput test uses this.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self.start: float | None = None

    def on_start(self, runner: Runner) -> None:
        self.start = time.time()

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        if self.start is not None and time.time() - self.start > self.seconds:
            runner._stop_training()


# ----- W&B ----------------------------------------------------------------------------------


def init_wandb_compat(cfg: Any) -> None:
    """Sample Factory's init_wandb without `Settings(start_method="fork")`, which wandb 0.30
    rejects. Same project, group, tags, TensorBoard sync, and resumable run id."""
    if not cfg.with_wandb:
        return
    import wandb

    if "wandb_unique_id" not in cfg:
        cfg.wandb_unique_id = f"{cfg.experiment}_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_user,
        sync_tensorboard=True,
        id=cfg.wandb_unique_id,
        name=os.environ.get("DOOMSCENT_RUN_NAME", cfg.wandb_unique_id),
        group=cfg.env if cfg.wandb_group is None else cfg.wandb_group,
        job_type=cfg.wandb_job_type,
        tags=cfg.wandb_tags,
        resume="allow",
    )
    wandb.config.update(dict(cfg), allow_val_change=True)


def patch_wandb_init() -> None:
    import sample_factory.algo.runners.runner as sf_runner

    sf_runner.init_wandb = init_wandb_compat
