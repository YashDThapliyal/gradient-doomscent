"""montage: 2x2 progress video of the same scenario and seed at 4 checkpoints.

    uv run doomscent montage --run v0_defend_center_s1 --ckpts 0,1M,5M,final
    uv run doomscent montage --run v0_defend_center_s1 --ckpts auto

`auto` picks untrained (step 0), early learning (first checkpoint whose held-out kills
beat the scripted baseline), mid (halfway between that and best), and best. Each tile is
a 640x360 replay of that checkpoint's episode on the first held-out seed, with a corner
tag giving its step and kills; episodes that end early hold their last frame, dimmed.
Output: runs/<run>/montage/montage__<labels>.mp4 (1280x720, 35 fps).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from doomscent.checkpoints import Checkpoint, list_checkpoints, resolve_checkpoint
from doomscent.plots.common import fmt_steps
from doomscent.recording import read_sidecar, sidecar_path_for
from doomscent.render import render_demo
from doomscent.runs import RunPaths
from doomscent.video import VideoWriter, draw_tag, iter_frames, probe_video

TILE_W, TILE_H = 640, 360


def auto_checkpoints(paths: RunPaths) -> list[Checkpoint]:
    from doomscent.baseline import load_baseline

    cfg = paths.load_config()
    ev = pd.read_csv(paths.metrics / "eval.csv").sort_values("step")
    base = load_baseline(cfg.env.scenario)["kills_mean"]
    best = resolve_checkpoint(paths, "best")
    above = ev[(ev.kills > base) & (ev.step > 0) & (ev.step <= best.step)]
    early_step = int(above.step.iloc[0]) if len(above) else int(ev[ev.step > 0].step.iloc[0])
    regular = [c for c in list_checkpoints(paths) if c.regular]
    mid_target = (early_step + best.step) / 2
    mid = min(
        (c for c in regular if early_step < c.step < best.step),
        key=lambda c: abs(c.step - mid_target),
        default=None,
    )
    picks = [resolve_checkpoint(paths, "0"), resolve_checkpoint(paths, str(early_step))]
    picks += [mid] if mid else []
    picks.append(best)
    out: list[Checkpoint] = []
    for p in picks:
        if p.step not in [o.step for o in out]:
            out.append(p)
    return out


def find_or_record_demo(paths: RunPaths, ck: Checkpoint, seed: int) -> Path:
    """The checkpoint's training clip demo on `seed`, recording it if it does not exist."""
    pattern = f"{paths.name}__step_{ck.step:07d}__*__seed_{seed}__ep*.lmp"
    hits = sorted((paths.clips / "train").glob(pattern))
    if hits:
        return hits[0]
    from doomscent.env import make_env
    from doomscent.policy import load_policy
    from doomscent.recording import record_episode

    cfg = paths.load_config()
    env = make_env(cfg, split="heldout")
    policy = load_policy(paths, ck.path, deterministic=bool(cfg.eval.deterministic))
    _, demo, _ = record_episode(env, policy, seed, paths.clips / "train", paths.name, ck.step, 0)
    env.close()
    return demo


def render_tile(demo: Path, tmp: Path, record_cfg: dict) -> Path:
    sc = read_sidecar(sidecar_path_for(demo))
    rc = {**record_cfg, "width": TILE_W, "height": TILE_H}
    return render_demo(demo, sc, ["plain"], out_dir=tmp, record_cfg=rc).outputs["plain"]


def compose(tile_paths: list[Path], tags: list[list[str]]) -> Iterator[np.ndarray]:
    """Stream grid frames; a finished tile holds its last frame, dimmed."""
    n = max(int(probe_video(p)["frames"]) for p in tile_paths)
    blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    streams = [iter_frames(p) for p in tile_paths]
    last: list[np.ndarray | None] = [None] * len(streams)
    done = [False] * len(streams)
    for _ in range(n):
        cells = []
        for k, (stream, tag) in enumerate(zip(streams, tags, strict=True)):
            f = None if done[k] else next(stream, None)
            if f is None:
                done[k] = True
                prev = last[k]
                f = (prev * 0.45).astype(np.uint8) if prev is not None else blank
            else:
                last[k] = f
            cells.append(draw_tag(f, tag, "tl", size=18, margin=10))
        while len(cells) < 4:
            cells.append(blank)
        top = np.concatenate(cells[:2], axis=1)
        bottom = np.concatenate(cells[2:4], axis=1)
        yield np.concatenate([top, bottom], axis=0)


def make_montage(paths: RunPaths, cks: list[Checkpoint], seed: int | None = None) -> Path:
    from omegaconf import OmegaConf

    cfg = paths.load_config()
    seed = int(cfg.seeds.eval_start) if seed is None else seed
    rc = OmegaConf.to_container(cfg.record, resolve=True)
    labels = "_".join(fmt_steps(c.step) for c in cks)
    out = paths.root / "montage" / f"montage__seed_{seed}__{labels}.mp4"
    tags = []
    with tempfile.TemporaryDirectory(prefix="montage_") as td:
        tiles = []
        for ck in cks:
            demo = find_or_record_demo(paths, ck, seed)
            kills = int(read_sidecar(sidecar_path_for(demo))["stats"]["kills"])
            tiles.append(render_tile(demo, Path(td), rc))  # type: ignore[arg-type]
            tags.append([f"step {fmt_steps(ck.step)}", f"{kills} kill{'s' if kills != 1 else ''}"])
        fps, crf = int(cfg.record.fps), int(cfg.record.crf)
        with VideoWriter(out, 2 * TILE_W, 2 * TILE_H, fps, crf) as w:
            for f in compose(tiles, tags):
                w.write(f)
        n_frames = w.frames
    meta = {"checkpoints": [c.step for c in cks], "seed": seed, "frames": n_frames, "tags": tags}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent montage")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpts", default="auto", help="'auto' or e.g. 0,1M,5M,final (up to 4)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    paths = RunPaths.for_run(args.run)
    if args.ckpts == "auto":
        cks = auto_checkpoints(paths)
    else:
        cks = [resolve_checkpoint(paths, s) for s in args.ckpts.split(",")][:4]
    out = make_montage(paths, cks, args.seed)
    print(out)


if __name__ == "__main__":
    main()
