"""sidebyside: two checkpoints playing the same held-out game, next to each other.

    uv run doomscent sidebyside --run v0_defend_center_s1 --ckpts 0,best

Both tiles replay the same scenario seed at 960x540 with the HUD on, under a header that
names each checkpoint and its kills. The shorter game holds its last frame, dimmed.
Output: runs/<run>/sidebyside/sidebyside__seed_<seed>__<a>_vs_<b>.mp4 (1920x620, 35 fps).
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from doomscent.checkpoints import resolve_checkpoint
from doomscent.montage import find_or_record_demo
from doomscent.plots.common import fmt_steps
from doomscent.recording import read_sidecar, sidecar_path_for
from doomscent.render import render_demo
from doomscent.runs import RunPaths
from doomscent.style import DARK, hex_to_rgb
from doomscent.timelapse.panels import text
from doomscent.video import VideoWriter, iter_frames, probe_video

TILE_W, TILE_H, HEADER = 960, 540, 80


def frames_with_hold(path: Path, n: int) -> Iterator[np.ndarray]:
    last = None
    for f in iter_frames(path):
        last = f
        yield f
    for _ in range(n - int(probe_video(path)["frames"])):
        yield (last * 0.45).astype(np.uint8)


def make_sidebyside(paths: RunPaths, specs: list[str], seed: int | None) -> Path:
    cfg = paths.load_config()
    seed = int(cfg.seeds.eval_start) if seed is None else seed
    rc = OmegaConf.to_container(cfg.record, resolve=True)
    rc.update({"width": 1024, "height": 576})  # type: ignore[union-attr]
    cks = [resolve_checkpoint(paths, s) for s in specs]
    names = ["before" if i == 0 else "after" for i in range(len(cks))]
    a, b = (fmt_steps(c.step) for c in cks)
    out = paths.root / "sidebyside" / f"sidebyside__seed_{seed}__{a}_vs_{b}.mp4"
    with tempfile.TemporaryDirectory(prefix="sbs_") as td:
        clips, heads = [], []
        for ck, name in zip(cks, names, strict=True):
            demo = find_or_record_demo(paths, ck, seed)
            sc = read_sidecar(sidecar_path_for(demo))
            clips.append(render_demo(demo, sc, ["plain"], Path(td), rc).outputs["plain"])  # type: ignore[arg-type]
            kills = int(sc["stats"]["kills"])
            heads.append(
                (f"{name}: step {fmt_steps(ck.step)}", f"{kills} kill{'s' if kills != 1 else ''}")
            )
        n = max(int(probe_video(c)["frames"]) for c in clips)
        header = np.full((HEADER, 2 * TILE_W, 3), hex_to_rgb(DARK.background), np.uint8)
        for k, (title, sub) in enumerate(heads):
            header = text(header, (k * TILE_W + 24, 16), title, 24, DARK.text, "semibold")
            header = text(
                header, (k * TILE_W + 24, 48), sub, 17, DARK.accent if k else DARK.subtext
            )
        streams = [frames_with_hold(c, n) for c in clips]
        with VideoWriter(
            out, 2 * TILE_W, TILE_H + HEADER, int(cfg.record.fps), int(cfg.record.crf)
        ) as w:
            for frames in zip(*streams, strict=True):
                tiles = [
                    cv2.resize(f, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA) for f in frames
                ]
                w.write(np.concatenate([header, np.concatenate(tiles, axis=1)], axis=0))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent sidebyside")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpts", default="0,best", help="two checkpoints, e.g. 0,best")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    specs = args.ckpts.split(",")
    if len(specs) != 2:
        raise SystemExit("--ckpts takes exactly two checkpoints")
    print(make_sidebyside(RunPaths.for_run(args.run), specs, args.seed))


if __name__ == "__main__":
    main()
