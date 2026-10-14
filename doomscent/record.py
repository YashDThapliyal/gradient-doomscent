"""record: play episodes with a checkpoint (or a scripted or random policy) and render clips.

    uv run doomscent record --run v0_defend_center_s1 --ckpt best --episodes 3
    uv run doomscent record --run v0_random_agent --config configs/v0_defend_center.yaml \
        --ckpt random --episodes 3 --kind train

Clips land in runs/<run>/clips/<kind>/ (eval clips in clips/eval/<ckpt>/). Every episode
keeps its demo (.lmp) and sidecar (.json.gz); each rendered clip gets plain and minimap
versions, plus an overlay version when asked.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from doomscent.config import load_config
from doomscent.env import DoomEnv, make_env
from doomscent.recording import read_sidecar, record_episode
from doomscent.render import render_demo
from doomscent.rollout import Policy, RandomPolicy, SpinAndShootPolicy
from doomscent.runs import RunPaths, write_run_metadata

KINDS = ("train", "eval", "best", "flagged", "final")


def make_policy(
    spec: str, env: DoomEnv, paths: RunPaths, cfg: DictConfig
) -> tuple[Policy, int, str]:
    """Policy from `random`, `scripted`, or a checkpoint selector (step, latest, best).

    Returns the policy, its env step, and a label for folder names.
    """
    if spec == "random":
        return RandomPolicy(env, seed=int(cfg.seed)), 0, "random"
    if spec == "scripted":
        from doomscent.baseline import load_baseline

        label = load_baseline(cfg.env.scenario)["policy"].split("turn=")[1].split(",")[0]
        turn = env.actions.heads[env.actions.head_names.index("turn")].choices.index(label)
        return SpinAndShootPolicy(env, turn), 0, "scripted"
    from doomscent.checkpoints import resolve_checkpoint
    from doomscent.policy import load_policy

    ckpt = resolve_checkpoint(paths, spec)
    policy = load_policy(paths, ckpt.path, deterministic=bool(cfg.eval.deterministic))
    return policy, ckpt.env_steps, ckpt.label


def record_and_render(
    env: DoomEnv,
    policy: Policy,
    seeds: Sequence[int],
    out_dir: Path,
    run: str,
    step: int,
    record_cfg: dict[str, Any],
    variants: Iterable[str] = ("plain", "minimap"),
    meta: dict[str, Any] | None = None,
    first_episode: int = 0,
) -> list[dict[str, Any]]:
    """Record one episode per seed and render each; returns a summary row per episode."""
    rows = []
    for i, seed in enumerate(seeds):
        res, demo, sidecar = record_episode(
            env, policy, seed, out_dir, run, step, first_episode + i, meta
        )
        rr = render_demo(demo, read_sidecar(sidecar), variants, record_cfg=record_cfg)
        rows.append(
            {
                "seed": seed,
                "episode": first_episode + i,
                "kills": res.stats["kills"],
                "steps": len(res.steps),
                "demo": str(demo),
                "sidecar": str(sidecar),
                "clips": {k: str(v) for k, v in rr.outputs.items()},
                "replay_tics": rr.replay_tics,
                "replay_matches_sidecar": rr.matches_sidecar,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent record")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best", help="step, latest, best, random, or scripted")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--kind", choices=KINDS, default=None, help="clip folder (default: eval)")
    ap.add_argument("--config", help="only for a new run folder (random or scripted policies)")
    ap.add_argument("--first-seed-offset", type=int, default=0)
    ap.add_argument("--overlay", action="store_true", help="also render the overlay version")
    args = ap.parse_args(argv)

    paths = RunPaths.for_run(args.run)
    if args.config:
        cfg = load_config(args.config, [f"run_name={args.run}"])
        if not paths.config.exists():
            write_run_metadata(cfg, paths)
    else:
        cfg = paths.load_config()
    env = make_env(cfg, split="heldout")
    policy, step, label = make_policy(args.ckpt, env, paths, cfg)
    kind = args.kind or "eval"
    out_dir = paths.clips / kind / (label if kind == "eval" else "")
    start = int(cfg.seeds.eval_start) + args.first_seed_offset
    seeds = [start + i for i in range(args.episodes)]
    variants = ["plain", "minimap"] + (["overlay"] if args.overlay else [])
    rc = OmegaConf.to_container(cfg.record, resolve=True)
    rows = record_and_render(env, policy, seeds, out_dir, args.run, step, rc, variants)  # type: ignore[arg-type]
    env.close()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
