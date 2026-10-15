"""highlights: auto-cut multi-kill streaks and close escapes from a checkpoint's evaluation.

    uv run doomscent highlights --run v0_defend_center_s1 --ckpt best

Moments are found from the sidecar JSON of every held-out episode (never by hand):
- streak: at least `streak_min` kills within `streak_window` agent steps;
- close escape: health falls to `escape_health` or below and the agent keeps fighting
  (alive and at least one more kill) for `escape_steps` agent steps.
The best moments are cut from 1280x720 HUD-on replays and joined, each with a corner tag.
Output: runs/<run>/highlights/highlights__<ckpt>.mp4 and a JSON list of the moments.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from doomscent.checkpoints import resolve_checkpoint
from doomscent.plots.common import fmt_steps
from doomscent.recording import read_sidecar
from doomscent.render import render_demo
from doomscent.runs import RunPaths
from doomscent.video import FrameReader, VideoWriter, draw_tag

GAP_FRAMES = 18  # black frames between moments


@dataclass(frozen=True)
class Moment:
    kind: str
    seed: int
    episode: int
    demo: str
    sidecar: str
    start_step: int
    end_step: int
    kills: int
    score: float  # for ranking: kills per second for streaks, steps survived for escapes
    label: str


def kill_steps(steps: list[dict]) -> list[int]:
    out, prev = [], None
    for s in steps:
        k = s["vars"]["KILLCOUNT"]
        if prev is not None and k > prev:
            out += [s["t"]] * int(k - prev)
        prev = k
    return out


def find_streaks(steps: list[dict], window: int, min_kills: int) -> list[tuple[int, int, int]]:
    """Maximal runs of kills with at least min_kills inside any `window` steps."""
    ks = kill_steps(steps)
    found, i = [], 0
    while i < len(ks):
        j = i
        while j + 1 < len(ks) and ks[j + 1] - ks[i] < window:
            j += 1
        if j - i + 1 >= min_kills:
            found.append((ks[i], ks[j], j - i + 1))
            i = j + 1
        else:
            i += 1
    return found


def find_escapes(steps: list[dict], health: float, hold: int) -> list[tuple[int, int, int]]:
    """First step at low health after which the agent survives `hold` steps and kills again."""
    for i, s in enumerate(steps):
        if s["vars"]["HEALTH"] <= health and s["vars"]["HEALTH"] > 0:
            end = i + hold
            if end < len(steps):
                kills = steps[end]["vars"]["KILLCOUNT"] - s["vars"]["KILLCOUNT"]
                if kills >= 1:
                    return [(s["t"], steps[end]["t"], int(kills))]
            return []
    return []


def collect_moments(paths: RunPaths, label: str, cfg: dict) -> list[Moment]:
    df = pd.read_csv(paths.metrics / "eval_episodes" / f"{label}.csv")
    fs = 4
    moments = []
    for r in df.itertuples():
        sc = read_sidecar(r.sidecar)
        steps = sc["steps"]
        for a, b, k in find_streaks(steps, cfg["streak_window"], cfg["streak_min"]):
            secs = max(1, b - a) * fs / 35
            moments.append(
                Moment(
                    "streak",
                    r.seed,
                    r.episode,
                    r.demo,
                    r.sidecar,
                    a - 12,
                    b + 8,
                    k,
                    k / secs,
                    f"{k} kills in {secs:.1f} s",
                )
            )
        for a, b, k in find_escapes(steps, cfg["escape_health"], cfg["escape_steps"]):
            hp = int(steps[a]["vars"]["HEALTH"])
            moments.append(
                Moment(
                    "escape",
                    r.seed,
                    r.episode,
                    r.demo,
                    r.sidecar,
                    a - 15,
                    b + 5,
                    k,
                    float(b - a),
                    f"down to {hp} health, {k} more kill{'s' if k > 1 else ''}",
                )
            )
    return moments


def pick(moments: list[Moment], n_streaks: int, n_escapes: int) -> list[Moment]:
    streaks = sorted((m for m in moments if m.kind == "streak"), key=lambda m: (-m.kills, -m.score))
    escapes = sorted((m for m in moments if m.kind == "escape"), key=lambda m: -m.score)
    chosen, used = [], set()
    for group, n in ((streaks, n_streaks), (escapes, n_escapes)):
        for m in group:
            if len([c for c in chosen if c.kind == m.kind]) >= n:
                break
            if m.demo not in used:  # one moment per episode keeps the reel varied
                chosen.append(m)
                used.add(m.demo)
    return chosen


def make_reel(paths: RunPaths, ckpt_spec: str, n_streaks: int, n_escapes: int) -> Path:
    run_cfg = paths.load_config()
    ck = resolve_checkpoint(paths, ckpt_spec)
    hcfg = {"streak_window": 35, "streak_min": 3, "escape_health": 30, "escape_steps": 35}
    if "highlights" in run_cfg:  # optional overrides in the run config
        hcfg.update(OmegaConf.to_container(run_cfg.highlights, resolve=True))  # type: ignore[arg-type]
    moments = pick(collect_moments(paths, ck.label, hcfg), n_streaks, n_escapes)
    if not moments:
        raise SystemExit(f"no streaks or escapes found for {ck.label}")
    rc = OmegaConf.to_container(run_cfg.record, resolve=True)
    out = paths.root / "highlights" / f"highlights__{ck.label}.mp4"
    fs = int(run_cfg.env.frame_skip)
    with (
        tempfile.TemporaryDirectory(prefix="highlights_") as td,
        VideoWriter(
            out,
            int(rc["width"]),
            int(rc["height"]),
            int(rc["fps"]),
            int(rc["crf"]),  # type: ignore[index]
        ) as w,
    ):
        black = np.zeros((int(rc["height"]), int(rc["width"]), 3), np.uint8)  # type: ignore[index]
        for n, m in enumerate(moments):
            clip = render_demo(m.demo, read_sidecar(m.sidecar), ["plain"], Path(td), rc).outputs[
                "plain"
            ]  # type: ignore[arg-type]
            reader = FrameReader(clip)
            a, b = max(0, m.start_step * fs), min(reader.n - 1, m.end_step * fs)
            tag = [m.label, f"step {fmt_steps(ck.step)} · seed {m.seed}"]
            for i in range(a, b + 1):
                w.write(draw_tag(reader.get(i), tag, "tl"))
            reader.close()
            if n < len(moments) - 1:
                for _ in range(GAP_FRAMES):
                    w.write(black)
    meta = {"checkpoint": ck.step, "moments": [asdict(m) for m in moments]}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent highlights")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--streaks", type=int, default=5)
    ap.add_argument("--escapes", type=int, default=3)
    args = ap.parse_args(argv)
    print(make_reel(RunPaths.for_run(args.run), args.ckpt, args.streaks, args.escapes))


if __name__ == "__main__":
    main()
