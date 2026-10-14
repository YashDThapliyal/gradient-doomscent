"""F6: frames per second vs rollout workers, one line per envs-per-worker setting."""

from __future__ import annotations

import pandas as pd

from doomscent.plots.common import FigureSpec, direct_label, main_axes, ref_line
from doomscent.runs import RUNS_ROOT, RunPaths

MACHINE_NAMES = {"mac_m3pro": "M3 Pro Mac (12 cores)"}


def load_throughput() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted((RUNS_ROOT / "throughput").glob("*/results.csv"))]
    if not frames:
        raise FileNotFoundError("no throughput results; run `doomscent throughput`")
    df = pd.concat(frames, ignore_index=True)
    if "learner_threads" in df:
        df = df[df["learner_threads"].fillna(1) == 1]
    return df[df["exit_code"] == 0]


def load_concurrent() -> pd.DataFrame | None:
    files = sorted((RUNS_ROOT / "throughput").glob("*/concurrent.csv"))
    return pd.concat([pd.read_csv(f) for f in files]) if files else None


def build(paths: RunPaths) -> FigureSpec:
    df = load_throughput()
    cfg = paths.load_config()
    chosen = (int(cfg.train.num_workers), int(cfg.train.num_envs_per_worker))
    best = df.loc[df["fps"].idxmax()]
    machines = ", ".join(MACHINE_NAMES.get(m, m) for m in df["machine"].unique())
    conc = load_concurrent()
    if conc is not None:
        total = conc.groupby("concurrent_runs").fps.sum()
        n_runs, tot = int(total.idxmax()), float(total.max())
        title = (
            f"One run tops out near {best.fps / 1000:.1f}K frames per second whatever the worker "
            f"count; {n_runs} runs at once reach {tot / 1000:.1f}K"
        )
    else:
        title = (
            f"Throughput peaks at {best.fps / 1000:.1f}K frames per second with "
            f"{int(best.workers)} workers x {int(best.envs_per_worker)} envs"
        )
    sub = (
        f"{machines}  ·  Defend the Center, 128x72, frame skip 4, APPO batch 2048  ·  "
        "100 s per setting, steady state after 40 s warm-up; the CPU learner is the bottleneck"
    )

    def draw(fig, theme, size):
        ax = main_axes(fig)
        envs = sorted(df["envs_per_worker"].unique())
        shades = [theme.muted2, theme.muted, theme.subtext, theme.text][-len(envs) :]
        for e, c in zip(envs, shades, strict=False):
            d = df[df.envs_per_worker == e].sort_values("workers")
            ax.plot(d.workers, d.fps / 1000, color=c, lw=1.8, marker="o", ms=4)
            direct_label(ax, d.workers.iloc[-1], d.fps.iloc[-1] / 1000, f"{e} envs/worker", c)
        sel = df[(df.workers == chosen[0]) & (df.envs_per_worker == chosen[1])]
        if len(sel):
            ax.plot(sel.workers, sel.fps / 1000, "o", ms=10, mfc="none", mec=theme.accent, mew=2)
            ax.annotate(
                "chosen for training",
                (sel.workers.iloc[0], sel.fps.iloc[0] / 1000),
                xytext=(-10, 16),
                textcoords="offset points",
                ha="right",
                color=theme.accent,
                fontsize=10,
            )
        if conc is not None:
            ref_line(ax, tot / 1000, f"{n_runs} concurrent runs, total", theme)
        ax.set_ylim(0, max(df.fps.max(), tot if conc is not None else 0) / 1000 * 1.18)
        ax.set_xticks(sorted(df.workers.unique()))
        ax.set_xlabel("rollout workers")
        ax.set_ylabel("thousand frames per second")

    return FigureSpec("F6", "throughput", title, sub, draw, right_margin_in=1.6)
