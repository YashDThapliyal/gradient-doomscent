"""timelapse: render the training timelapse from the brain-data cache.

    uv run doomscent timelapse --run v0_defend_center_s1

Builds (or reuses) the cache, then writes to runs/<run>/timelapse/:
- timelapse_16x9.mp4      1920x1080, 60 fps: brain (left), learning curve (right), fight
                          with attention heatmap and decision bars (bottom), step/score tag
- timelapse_mobile.mp4    1080x1350 version for mobile feeds
- brain_only.mp4          views 1 and 2 full frame
- attention_before_after.mp4   the same fight moment, first vs best checkpoint, heatmap on
- mind_map.mp4            probe frames placed by UMAP of the network's features
Edge colors and the curve cursor are interpolated between checkpoints so nothing jumps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from doomscent.checkpoints import Checkpoint, best_pointer, list_checkpoints
from doomscent.plots.common import fmt_steps
from doomscent.runs import RunPaths
from doomscent.style import DARK, Theme, hex_to_rgb
from doomscent.timelapse.brain import LAYERS
from doomscent.timelapse.cache import build_cache, cache_path, tl_dir
from doomscent.timelapse.panels import (
    PAIRS,
    BrainLayout,
    CurveData,
    CurvePanel,
    draw_brain,
    draw_decision,
    draw_tag_block,
    saliency_overlay,
    text,
)
from doomscent.video import FrameReader, VideoWriter

GAME_FPS = 35


@dataclass
class Cached:
    ck: Checkpoint
    change: dict[str, np.ndarray]
    acts: np.ndarray
    probs: np.ndarray
    value: np.ndarray
    sal: np.ndarray
    kills: np.ndarray
    n_frames: int
    probe_logits: np.ndarray


def load_cached(paths: RunPaths, cks: list[Checkpoint]) -> list[Cached]:
    out = []
    for ck in cks:
        d = np.load(cache_path(paths, ck))
        out.append(
            Cached(
                ck,
                {p: d[f"change:{p}"] for p in PAIRS},
                d["fight_acts"],
                d["fight_probs"],
                d["fight_value"],
                d["fight_sal"],
                d["fight_kills"],
                int(d["footage_frames"]),
                d["probe_logits"],
            )
        )
    return out


def scales(cached: list[Cached], units: dict[str, list[int]]) -> tuple[dict, dict, tuple]:
    change_scale = {
        p: float(np.quantile(np.concatenate([c.change[p].ravel() for c in cached[1:]]), 0.99))
        if len(cached) > 1
        else 1.0
        for p in PAIRS
    }
    all_acts = np.abs(np.concatenate([c.acts for c in cached]))
    act_scale, start = {}, 0
    for layer in LAYERS:
        n = len(units[layer])
        act_scale[layer] = float(np.quantile(all_acts[:, start : start + n], 0.99))
        start += n
    vals = np.concatenate([c.value for c in cached])
    return change_scale, act_scale, (float(np.quantile(vals, 0.01)), float(np.quantile(vals, 0.99)))


def curve_data(paths: RunPaths, x_max: float) -> CurveData:
    from doomscent.plots.data import load_baseline_for

    ev = pd.read_csv(paths.metrics / "eval.csv").sort_values("step")
    base = load_baseline_for(paths)
    return CurveData(
        ev["step"].to_numpy(float),
        ev["kills"].to_numpy(float),
        base["kills_mean"] if base else None,
        base["gate_kills"] if base else None,
        x_max,
    )


class Footage:
    """Open readers for the checkpoints currently on screen."""

    def __init__(self, paths: RunPaths):
        self.dir = tl_dir(paths) / "cache" / "footage"
        self.readers: dict[str, FrameReader] = {}

    def frame(self, label: str, i: int) -> np.ndarray:
        if label not in self.readers:
            if len(self.readers) > 3:
                old = next(iter(self.readers))
                self.readers.pop(old).close()
            self.readers[label] = FrameReader(self.dir / f"{label}.mp4")
        return self.readers[label].get(i)


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def render_main(
    paths: RunPaths, cached: list[Cached], units_info: dict, theme: Theme
) -> list[Path]:
    cfg = paths.load_config()
    tl = cfg.timelapse
    fps, n_out = int(tl.fps), int(tl.fps) * int(tl.duration_s)
    units, sizes = units_info["units"], units_info["sizes"]
    change_scale, act_scale, vrange = scales(cached, units)
    heads = json.loads((paths.meta / "action_space.json").read_text())["heads"]
    x_max = float(cached[-1].ck.step)
    curve_wide = CurvePanel(curve_data(paths, x_max), 960, 560, theme)
    curve_tall = CurvePanel(curve_data(paths, x_max), 1080, 320, theme)
    lay_wide = BrainLayout(960, 560, units, sizes)
    lay_full = BrainLayout(1920, 1080, units, sizes)
    lay_tall = BrainLayout(1080, 560, units, sizes)
    out_dir = tl_dir(paths)
    outs = {
        "wide": out_dir / "timelapse_16x9.mp4",
        "tall": out_dir / "timelapse_mobile.mp4",
        "brain": out_dir / "brain_only.mp4",
    }
    writers = {
        "wide": VideoWriter(outs["wide"], 1920, 1080, fps, 20),
        "tall": VideoWriter(outs["tall"], 1080, 1350, fps, 20),
        "brain": VideoWriter(outs["brain"], 1920, 1080, fps, 20),
    }
    footage = Footage(paths)
    bg = hex_to_rgb(theme.background)
    n_ck = len(cached)
    blur = int(tl.saliency_blur_px)
    try:
        for t in range(n_out):
            u = t / max(1, n_out - 1) * (n_ck - 1)
            c = min(int(u), n_ck - 1)
            f = u - c
            cur, nxt = cached[c], cached[min(c + 1, n_ck - 1)]
            change = {p: (1 - f) * cur.change[p] + f * nxt.change[p] for p in PAIRS}
            step = (1 - f) * cur.ck.step + f * nxt.ck.step
            tic = int(t / fps * GAME_FPS)
            fi = tic % max(1, cur.n_frames)
            si = min(fi // int(cfg.env.frame_skip), len(cur.acts) - 1)
            acts = cur.acts[si]
            score = curve_wide.value_at(step)
            game = footage.frame(cur.ck.label, fi)
            game = saliency_overlay(game, cur.sal[si], theme, blur)
            kills_now = float(cur.kills[si])

            brain_w = draw_brain(lay_wide, change, change_scale, acts, act_scale, theme)
            frame = np.empty((1080, 1920, 3), np.uint8)
            frame[:] = bg
            frame[0:560, 0:960] = brain_w
            frame[0:560, 960:1920] = curve_wide.draw(step)
            frame[560:1080, 480:1404] = fit(game, 924, 520)
            frame[560:1080, 0:480] = draw_decision(
                480, 520, cur.probs[si], heads, float(cur.value[si]), vrange, theme
            )
            frame[560:1080, 1404:1920] = draw_tag_block(516, 520, step, score, kills_now, theme)
            frame = text(
                frame,
                (500, 1050),
                "attention: gradient saliency of the chosen action",
                13,
                theme.subtext,
            )
            writers["wide"].write(frame)

            tall = np.empty((1350, 1080, 3), np.uint8)
            tall[:] = bg
            tall[0:560] = draw_brain(
                lay_tall,
                change,
                change_scale,
                acts,
                act_scale,
                theme,
                title="weight change · activity",
            )
            tall[560:880] = curve_tall.draw(step)
            tall[880:1350, 0:1080] = _letterbox(game, 1080, 470, bg)
            tall = text(tall, (24, 900), f"step {fmt_steps(step)}", 30, theme.text, "semibold")
            tall = text(
                tall, (24, 940), f"held-out kills {score:.1f}", 20, theme.accent, "semibold"
            )
            writers["tall"].write(tall)

            brain_full = draw_brain(lay_full, change, change_scale, acts, act_scale, theme)
            brain_full = text(
                brain_full,
                (1896, 1040),
                f"step {fmt_steps(step)}",
                26,
                theme.text,
                "semibold",
                "ra",
            )
            writers["brain"].write(brain_full)
            if t % (fps * 5) == 0:
                print(f"[timelapse] frame {t}/{n_out} step {fmt_steps(step)}", flush=True)
    finally:
        for w in writers.values():
            w.close()
        for r in footage.readers.values():
            r.close()
    return list(outs.values())


def _letterbox(img: np.ndarray, w: int, h: int, bg: tuple[int, int, int]) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s) // 2 * 2, int(ih * s) // 2 * 2
    out = np.empty((h, w, 3), np.uint8)
    out[:] = bg
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y : y + nh, x : x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def render_attention(paths: RunPaths, cached: list[Cached], theme: Theme) -> Path:
    """The same fight moment (same seed, same tics) from the first and the best checkpoint."""
    cfg = paths.load_config()
    bp = best_pointer(paths)
    best_step = json.loads(bp.read_text())["step"] if bp.exists() else cached[-1].ck.step
    first = cached[0] if cached[0].ck.step > 0 or len(cached) == 1 else cached[1]
    best = next(c for c in cached if c.ck.step == best_step)
    start = GAME_FPS
    end = min(first.n_frames, best.n_frames, start + 7 * GAME_FPS)
    start = min(start, max(0, end - GAME_FPS))
    out = tl_dir(paths) / "attention_before_after.mp4"
    footage = Footage(paths)
    bg = hex_to_rgb(theme.background)
    fs = int(cfg.env.frame_skip)
    with VideoWriter(out, 1920, 620, GAME_FPS, 20) as w:
        for fi in range(start, end):
            frame = np.empty((620, 1920, 3), np.uint8)
            frame[:] = bg
            for k, c in enumerate((first, best)):
                si = min(fi // fs, len(c.sal) - 1)
                g = saliency_overlay(
                    footage.frame(c.ck.label, fi),
                    c.sal[si],
                    theme,
                    int(cfg.timelapse.saliency_blur_px),
                )
                frame[80:620, k * 960 : (k + 1) * 960] = fit(g, 960, 540)
                name = "early" if k == 0 else "best"
                frame = text(
                    frame,
                    (k * 960 + 24, 22),
                    f"{name}: step {fmt_steps(c.ck.step)}",
                    22,
                    theme.text,
                    "semibold",
                )
                frame = text(
                    frame,
                    (k * 960 + 24, 52),
                    "accent: pixels that most change the chosen action",
                    14,
                    theme.subtext,
                )
            w.write(frame)
    return out


def render_mind_map(paths: RunPaths, cached: list[Cached], theme: Theme) -> Path:
    """Probe frames placed by UMAP of the GRU features, colored by turn direction."""
    cfg = paths.load_config()
    tl = cfg.timelapse
    fps, n_out = int(tl.fps), int(tl.fps) * max(10, int(tl.duration_s) // 2)
    coords = np.load(tl_dir(paths) / "umap.npz")
    final = coords[cached[-1].ck.label]
    lo, hi = np.quantile(final, 0.01, axis=0), np.quantile(final, 0.99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo, hi = lo - 0.15 * span, hi + 0.15 * span
    size, top = 1080, 140
    out = tl_dir(paths) / "mind_map.mp4"
    colors = {"left": theme.second, "none": theme.muted, "right": theme.accent}
    bg = hex_to_rgb(theme.background)

    def turn_dir(logits: np.ndarray) -> list[str]:
        a = logits[:, :7].argmax(1)
        return ["none" if x == 0 else ("left" if x <= 3 else "right") for x in a]

    with VideoWriter(out, size, size, fps, 20) as w:
        n_ck = len(cached)
        for t in range(n_out):
            u = t / max(1, n_out - 1) * (n_ck - 1)
            c = min(int(u), n_ck - 1)
            f = u - c
            nxt = min(c + 1, n_ck - 1)
            p = (1 - f) * coords[cached[c].ck.label] + f * coords[cached[nxt].ck.label]
            dirs = turn_dir(cached[c].probe_logits)
            fire = cached[c].probe_logits[:, 7:9].argmax(1) == 1
            img = np.empty((size, size, 3), np.uint8)
            img[:] = bg
            xy = (p - lo) / (hi - lo)
            xy = np.clip(xy, 0, 1)
            for (x, y), d, on in zip(xy, dirs, fire, strict=True):
                px = (int(60 + x * (size - 120)), int(top + (1 - y) * (size - top - 60)))
                col = hex_to_rgb(colors[d])
                cv2.circle(img, px, 6, col, -1 if on else 2, cv2.LINE_AA)
            step = (1 - f) * cached[c].ck.step + f * cached[nxt].ck.step
            img = text(
                img, (40, 34), f"mind map at step {fmt_steps(step)}", 28, theme.text, "semibold"
            )
            img = text(
                img,
                (40, 76),
                f"{len(xy)} fixed probe frames placed by UMAP of the memory "
                "state; filled: fires, ring: holds fire",
                15,
                theme.subtext,
            )
            lx = 40
            for name in ("left", "none", "right"):
                label = {"left": "turns left", "none": "no turn", "right": "turns right"}[name]
                cv2.circle(img, (lx + 8, 112), 6, hex_to_rgb(colors[name]), -1, cv2.LINE_AA)
                img = text(img, (lx + 22, 103), label, 15, colors[name])
                lx += 150
            w.write(img)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent timelapse")
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--skip-cache", action="store_true", help="render from the existing cache")
    args = ap.parse_args(argv)
    paths = RunPaths.for_run(args.run)
    if args.skip_cache:
        cks = list_checkpoints(paths)
    else:
        cks = build_cache(paths, workers=args.workers, force=args.rebuild_cache)
    units_info = json.loads((tl_dir(paths) / "units.json").read_text())
    cached = load_cached(paths, cks)
    theme = DARK
    outs = render_main(paths, cached, units_info, theme)
    outs.append(render_attention(paths, cached, theme))
    outs.append(render_mind_map(paths, cached, theme))
    meta = {"checkpoints": [c.step for c in cks], "outputs": [str(o) for o in outs]}
    (tl_dir(paths) / "timelapse.json").write_text(json.dumps(meta, indent=2))
    for o in outs:
        print(o)


if __name__ == "__main__":
    main()
