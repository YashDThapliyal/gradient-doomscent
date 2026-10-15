"""eval-watcher: a separate process that evaluates and records each new checkpoint.

    uv run doomscent eval-watcher --run v0_defend_center_s1

Training never pauses for this. For every regular checkpoint, in step order:
- 100 held-out episodes through `evaluate` (demos, sidecars, clips, flags, eval.csv);
- `best` moves when held-out mean kills improve, and 3 episodes from it are rendered
  into clips/best/ (plain, minimap, overlay);
- training-progress clips into clips/train/ (3 at step 0, 2 afterwards).
It also refreshes metrics/train.csv on each poll. Once training has finished and every
checkpoint is evaluated, it renders 5 episodes of the final checkpoint into clips/final/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from doomscent.checkpoints import (
    SETTLE_SECONDS,
    Checkpoint,
    best_pointer,
    is_complete,
    list_checkpoints,
)
from doomscent.evaluate import evaluate
from doomscent.metrics import write_train_csv
from doomscent.runs import RunPaths


def log(paths: RunPaths, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    with open(paths.logs / "eval_watcher.log", "a") as f:
        f.write(line + "\n")


def evaluated_steps(paths: RunPaths) -> set[int]:
    f = paths.metrics / "eval.csv"
    return set(pd.read_csv(f)["step"].astype(int)) if f.exists() else set()


def done_marker(paths: RunPaths) -> Any:
    return paths.root / "TRAINING_DONE.json"


def record_clips(
    paths: RunPaths, ck: Checkpoint, kind: str, n: int, variants: list[str], routine: bool = False
) -> list[dict]:
    from doomscent.env import make_env
    from doomscent.policy import load_policy
    from doomscent.record import record_and_render

    cfg = paths.load_config()
    env = make_env(cfg, split="heldout")
    policy = load_policy(paths, ck.path, deterministic=bool(cfg.eval.deterministic))
    seeds = [int(cfg.seeds.eval_start) + i for i in range(n)]
    rc = OmegaConf.to_container(cfg.record, resolve=True)
    if routine:  # per-checkpoint training clips: slimmer, re-renderable from the demos
        rc["crf"] = int(cfg.record.get("routine_crf", rc["crf"]))  # type: ignore[index]
        variants = list(cfg.eval.get("routine_clip_variants", variants))
    rows = record_and_render(
        env,
        policy,
        seeds,
        paths.clips / kind,
        paths.name,
        ck.step,
        rc,
        variants,  # type: ignore[arg-type]
    )
    env.close()
    return rows


def routine_clips(cfg: Any) -> bool:
    """Whether every checkpoint gets rendered clips, or only step 0 and the final best.

    Demos are always kept, so any skipped clip can be rendered later with `record`.
    """
    return bool(cfg.eval.get("render_routine_clips", True))


def update_best(paths: RunPaths, ck: Checkpoint, summary: dict) -> bool:
    """Point `best` at the argmax of held-out kills over every evaluated checkpoint.

    Recomputed from eval.csv each time, so re-evaluations can never leave a stale pointer.
    Returns True when the best checkpoint changed to `ck`.
    """
    ev = pd.read_csv(paths.metrics / "eval.csv")
    top = ev.sort_values(["kills", "reward"], ascending=False).iloc[0]
    p = best_pointer(paths)
    prev = json.loads(p.read_text())["step"] if p.exists() else None
    by_step = {c.step: c for c in list_checkpoints(paths)}
    best_ck = by_step.get(int(top.step), ck)
    info = {
        "step": int(top.step),
        "env_steps": int(top.env_steps),
        "path": str(best_ck.path),
        "kills": float(top.kills),
        "kills_ci": [float(top.kills_ci_lo), float(top.kills_ci_hi)],
        "reward": float(top.reward),
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=2))
    tmp.replace(p)
    return int(top.step) != prev and int(top.step) == ck.step


def sync(paths: RunPaths) -> None:
    """Copy the run folder to cloud storage after an eval, when a remote is configured."""
    remote = paths.load_config().get("sync", {}).get("rclone_remote")
    if not remote:
        return
    if shutil.which("rclone") is None:
        log(paths, "sync skipped: rclone not installed")
        return
    dest = f"{remote.rstrip('/')}/runs/{paths.name}"
    r = subprocess.run(["rclone", "copy", str(paths.root), dest], capture_output=True, text=True)
    log(paths, f"sync to {dest}: exit {r.returncode} {r.stderr.strip()[-200:]}")


def adopt_final_checkpoint(paths: RunPaths) -> None:
    """If training stopped past the last milestone, keep its final checkpoint as one too."""
    cks = list_checkpoints(paths)
    last = max((c.env_steps for c in cks), default=-1)
    periodic = sorted(paths.checkpoint_dir.glob("checkpoint_*.pth"))
    if not periodic:
        return
    newest = periodic[-1]
    env_steps = int(newest.stem.split("_")[-1])
    if env_steps > last:
        shutil.copy2(newest, paths.milestone_dir / newest.name)
        log(paths, f"adopted final checkpoint {newest.name} as a milestone")


def process_checkpoint(paths: RunPaths, ck: Checkpoint) -> None:
    cfg = paths.load_config()
    e = cfg.eval
    t = time.time()
    summary = evaluate(paths, ck, int(e.episodes), int(e.num_workers))
    log(
        paths,
        f"evaluated {ck.label} (env_steps {ck.env_steps}): kills {summary['kills']:.2f} "
        f"[{summary['kills_ci_lo']:.2f}, {summary['kills_ci_hi']:.2f}], "
        f"flagged {summary['flagged_episodes']}, {time.time() - t:.0f}s",
    )
    every = routine_clips(cfg)
    if ck.step == 0 or every:
        n_train = int(e.step0_clip_episodes) if ck.step == 0 else int(e.train_clip_episodes)
        record_clips(paths, ck, "train", n_train, ["plain", "minimap"], routine=True)
    if update_best(paths, ck, summary):
        if every:  # otherwise best clips are rendered once, when training ends
            record_clips(
                paths, ck, "best", int(e.best_clip_episodes), ["plain", "minimap", "overlay"]
            )
        log(paths, f"new best: {ck.label} with {summary['kills']:.2f} kills")
    sync(paths)


def record_end_clips(paths: RunPaths) -> None:
    """Final-checkpoint clips, or with routine clips off, clips of the overall best."""
    cfg = paths.load_config()
    variants = ["plain", "minimap", "overlay"]
    if routine_clips(cfg):
        kind, ck, n = "final", list_checkpoints(paths)[-1], int(cfg.eval.final_clip_episodes)
    else:
        step = json.loads(best_pointer(paths).read_text())["step"]
        by_step = {c.step: c for c in list_checkpoints(paths)}
        kind, ck, n = "best", by_step[step], int(cfg.eval.best_clip_episodes)
    if any((paths.clips / kind).glob(f"*__step_{ck.step:07d}__*.mp4")):
        return
    record_clips(paths, ck, kind, n, variants)
    log(paths, f"{kind} clips recorded from {ck.label}")
    sync(paths)


def run_watcher(paths: RunPaths, poll: float, once: bool = False) -> None:
    log(paths, f"watching {paths.milestone_dir}")
    while True:
        if (paths.sf_dir / ".summary").exists():
            write_train_csv(paths)
        training_done = done_marker(paths).exists()
        if training_done:
            adopt_final_checkpoint(paths)
        done = evaluated_steps(paths)
        cks = list_checkpoints(paths) if paths.milestone_dir.exists() else []
        todo = [c for c in cks if c.regular and c.step not in done]
        if training_done and cks and cks[-1].step not in done and cks[-1] not in todo:
            todo.append(cks[-1])  # the final checkpoint is always evaluated
        for ck in todo:
            try:
                process_checkpoint(paths, ck)
            except Exception as e:  # keep watching; this checkpoint is retried next poll
                log(paths, f"evaluation of {ck.label} failed, will retry: {type(e).__name__}: {e}")
        settling = [f for f in paths.milestone_dir.glob("checkpoint_*.pth") if not is_complete(f)]
        if training_done and not todo and settling:
            time.sleep(SETTLE_SECONDS)  # a checkpoint is still being written or just adopted
            continue
        if training_done and not todo:
            record_end_clips(paths)
            write_train_csv(paths)
            log(paths, "training finished and every checkpoint is evaluated; exiting")
            return
        if once:
            return
        time.sleep(poll)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent eval-watcher")
    ap.add_argument("--run", required=True)
    ap.add_argument("--poll", type=float, default=None)
    ap.add_argument("--once", action="store_true", help="process what exists, then exit")
    args = ap.parse_args(argv)
    paths = RunPaths.for_run(args.run)
    poll = args.poll or float(paths.load_config().eval.poll_interval_s)
    run_watcher(paths, poll, once=args.once)


if __name__ == "__main__":
    main()
