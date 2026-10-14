"""evaluate: metrics over N held-out episodes for one checkpoint, with demos and clips.

    uv run doomscent evaluate --run v0_defend_center_s1 --ckpt latest --episodes 100

Writes:
- runs/<run>/clips/eval/<step_N>/: a demo and sidecar for every episode; rendered clips
  (plain and minimap) for the first 3, the best-scoring, and any multi-kill streak.
- runs/<run>/clips/flagged/ plus a line in runs/<run>/flagged.csv per detector hit.
- runs/<run>/metrics/eval_episodes/<step_N>.csv (per episode) and one row of
  runs/<run>/metrics/eval.csv (replaced if this checkpoint is evaluated again).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from doomscent.checkpoints import Checkpoint, resolve_checkpoint
from doomscent.detectors import run_detectors
from doomscent.recording import read_sidecar
from doomscent.runs import RunPaths
from doomscent.stats import bootstrap_ci

# ----- per-episode analysis from sidecar steps ----------------------------------------------


def max_kills_in_window(steps: Sequence[dict], window: int) -> int:
    """Most kills inside any window of `window` agent steps."""
    times: list[int] = []
    prev = None
    for s in steps:
        k = s["vars"].get("KILLCOUNT", 0)
        if prev is not None and k > prev:
            times += [s["t"]] * int(k - prev)
        prev = k
    best, j = 0, 0
    for i in range(len(times)):
        while times[i] - times[j] >= window:
            j += 1
        best = max(best, i - j + 1)
    return best


def policy_summary(steps: Sequence[dict], heads: Sequence[dict]) -> dict[str, float]:
    """Mean per-head entropy (from recorded probabilities) and per-choice usage shares."""
    out: dict[str, float] = {}
    n = max(1, len(steps))
    for i, h in enumerate(heads):
        counts = np.zeros(len(h["choices"]))
        ent = []
        for s in steps:
            counts[s["action_idx"][i]] += 1
            p = (s.get("probs") or {}).get(h["name"])
            if p is not None:
                pa = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
                ent.append(float(-(pa * np.log(pa)).sum()))
        for c, cnt in zip(h["choices"], counts, strict=True):
            out[f"share_{h['name']}_{c}"] = float(cnt) / n
        if ent:
            out[f"entropy_{h['name']}"] = float(np.mean(ent))
    return out


# ----- workers --------------------------------------------------------------------------------


def _eval_worker(job: dict[str, Any]) -> list[dict[str, Any]]:
    from doomscent.env import make_env
    from doomscent.policy import load_policy
    from doomscent.recording import record_episode

    paths = RunPaths.for_run(job["run"])
    cfg = paths.load_config()
    env = make_env(cfg, split="heldout")
    policy = load_policy(paths, job["ckpt_path"], deterministic=job["deterministic"])
    rows = []
    for episode, seed in job["episodes"]:
        meta = {"ckpt": job["label"], "deterministic": job["deterministic"]}
        res, demo, sidecar = record_episode(
            env, policy, seed, job["out_dir"], job["run"], job["step"], episode, meta
        )
        heads = env.actions.describe()["heads"]
        rows.append(
            {
                "episode": episode,
                "seed": seed,
                **res.stats,
                "reward": res.reward,
                "died": res.died,
                "multi_kill_max": max_kills_in_window(res.steps, job["window"]),
                **policy_summary(res.steps, heads),
                "demo": str(demo),
                "sidecar": str(sidecar),
            }
        )
    env.close()
    return rows


def _render_worker(job: dict[str, Any]) -> dict[str, Any]:
    from doomscent.render import render_demo

    rr = render_demo(
        job["demo"], read_sidecar(job["sidecar"]), job["variants"], job["out_dir"], job["rc"]
    )
    return {
        "demo": job["demo"],
        "clips": {k: str(v) for k, v in rr.outputs.items()},
        "replay_matches_sidecar": rr.matches_sidecar,
    }


def _pool(n: int):
    return mp.get_context("spawn").Pool(processes=max(1, n))


def render_many(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if not jobs:
        return []
    with _pool(min(workers, len(jobs))) as pool:
        return pool.map(_render_worker, jobs)


# ----- evaluation -------------------------------------------------------------------------------


def reward_history(paths: RunPaths, label: str, window: int) -> list[float]:
    """Episode rewards from the most recent earlier evaluation, for the reward-spike check."""
    d = paths.metrics / "eval_episodes"
    earlier = sorted(p for p in d.glob("step_*.csv") if p.stem < label) if d.exists() else []
    if not earlier:
        return []
    return pd.read_csv(earlier[-1])["reward"].tolist()[-window:]


def flag_episodes(paths: RunPaths, df: pd.DataFrame, label: str, cfg: Any) -> list[dict]:
    det_cfg = OmegaConf.to_container(cfg.detectors, resolve=True)
    history = reward_history(paths, label, int(cfg.detectors.reward_spike.window))
    flags = []
    for row in df.itertuples():
        sc = read_sidecar(row.sidecar)
        hist = history + df["reward"].iloc[: row.Index].tolist()
        found = run_detectors(sc["steps"], sc["start_vars"], sc["died"], row.reward, hist, det_cfg)
        for f in found:
            flags.append(
                {
                    "episode": row.episode,
                    "seed": row.seed,
                    "detector": f.detector,
                    "detail": f.detail,
                    "demo": row.demo,
                    "sidecar": row.sidecar,
                }
            )
    return flags


def summarize(df: pd.DataFrame, ck: Checkpoint, label: str, n_flagged: int, base: float) -> dict:
    lo, hi = bootstrap_ci(df["kills"])
    deaths = float(df["deaths"].mean())
    row: dict[str, Any] = {
        "ckpt": label,
        "step": ck.step,
        "env_steps": ck.env_steps,
        "episodes": len(df),
        "kills": float(df["kills"].mean()),
        "kills_ci_lo": lo,
        "kills_ci_hi": hi,
        "kills_std": float(df["kills"].std()),
        "deaths": deaths,
        "kd_ratio": float(df["kills"].mean()) / deaths if deaths > 0 else float("nan"),
        "survival_time": float(df["survival_time_s"].mean()),
        "accuracy": float(df["hits"].sum() / max(1.0, df["shots"].sum())),
        "damage_dealt": float(df["damage_dealt"].mean()),
        "damage_taken": float(df["damage_taken"].mean()),
        "pickups": float(df["pickups"].mean()),
        "coverage": float(df["coverage"].mean()),
        "reward": float(df["reward"].mean()),
        "normalized_score": float(df["kills"].mean()) / base if base else float("nan"),
        "flagged_episodes": n_flagged,
    }
    for c in df.columns:
        if c.startswith(("share_", "entropy_")):
            row[c] = float(df[c].mean())
    return row


def upsert_eval_row(paths: RunPaths, row: dict) -> pd.DataFrame:
    out = paths.metrics / "eval.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if not df.empty:
        df = df[df["step"] != row["step"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).sort_values("step")
    tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    return df


def append_flagged(paths: RunPaths, rows: list[dict]) -> None:
    if not rows:
        return
    out = paths.flagged_csv
    df = pd.DataFrame(rows)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)


def evaluate(
    paths: RunPaths, ck: Checkpoint, episodes: int, workers: int, render: bool = True
) -> dict[str, Any]:
    """Run the evaluation for one checkpoint and write every output; returns the summary row."""
    cfg = paths.load_config()
    e = cfg.eval
    label = ck.label
    out_dir = paths.clips / "eval" / label
    seeds = [int(cfg.seeds.eval_start) + i for i in range(episodes)]
    chunks = [list(enumerate(seeds))[i::workers] for i in range(workers)]
    jobs = [
        {
            "run": paths.name,
            "ckpt_path": str(ck.path),
            "label": label,
            "step": ck.step,
            "episodes": chunk,
            "out_dir": str(out_dir),
            "deterministic": bool(e.deterministic),
            "window": int(e.multi_kill_window_steps),
        }
        for chunk in chunks
        if chunk
    ]
    with _pool(len(jobs)) as pool:
        rows = [r for part in pool.map(_eval_worker, jobs) for r in part]
    df = pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)

    flags = flag_episodes(paths, df, label, cfg)
    df["flags"] = [
        ";".join(f["detector"] for f in flags if f["episode"] == ep) for ep in df["episode"]
    ]
    try:
        from doomscent.baseline import load_baseline

        base = float(load_baseline(cfg.env.scenario)["kills_mean"])
    except FileNotFoundError:
        base = float("nan")

    rc = OmegaConf.to_container(cfg.record, resolve=True)
    clip_eps = set(df["episode"].iloc[: int(e.clips_first_n)])
    best_i = df.sort_values(["kills", "reward"], ascending=False).index[0]
    clip_eps.add(int(df.loc[best_i, "episode"]))
    clip_eps |= set(df.loc[df["multi_kill_max"] >= int(e.multi_kill_min), "episode"])
    render_jobs = []
    if render:
        for ep in sorted(clip_eps):
            r = df[df.episode == ep].iloc[0]
            render_jobs.append(
                {
                    "demo": r.demo,
                    "sidecar": r.sidecar,
                    "variants": ["plain", "minimap"],
                    "out_dir": str(out_dir),
                    "rc": rc,
                }
            )
        cap = int(cfg.detectors.get("max_clips_per_detector", 3))
        per_detector: dict[str, int] = {}
        flagged_demos: dict[str, dict] = {}
        for f in flags:
            if per_detector.get(f["detector"], 0) < cap and f["demo"] not in flagged_demos:
                per_detector[f["detector"]] = per_detector.get(f["detector"], 0) + 1
                flagged_demos[f["demo"]] = f
        for demo, f in flagged_demos.items():
            render_jobs.append(
                {
                    "demo": demo,
                    "sidecar": f["sidecar"],
                    "variants": ["plain", "minimap"],
                    "out_dir": str(paths.clips / "flagged"),
                    "rc": rc,
                }
            )
    rendered = render_many(render_jobs, workers)
    flagged_clip = {
        r["demo"]: r["clips"]["plain"]
        for r in rendered
        if str(paths.clips / "flagged") in r["clips"]["plain"]
    }
    append_flagged(
        paths,
        [
            {"run": paths.name, "step": ck.step, "ckpt": label, **f}
            | {"clip": flagged_clip.get(f["demo"], "")}
            for f in flags
        ],
    )

    (paths.metrics / "eval_episodes").mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.metrics / "eval_episodes" / f"{label}.csv", index=False)
    summary = summarize(df, ck, label, len({f["episode"] for f in flags}), base)
    summary["clips_rendered"] = len(rendered)
    summary["replays_match"] = all(r["replay_matches_sidecar"] for r in rendered)
    upsert_eval_row(paths, summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent evaluate")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="latest", help="step (e.g. 2.5M), latest, best, final")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    paths = RunPaths.for_run(args.run)
    cfg = paths.load_config()
    ck = resolve_checkpoint(paths, args.ckpt)
    episodes = args.episodes or int(cfg.eval.episodes)
    workers = args.workers or int(cfg.eval.num_workers)
    summary = evaluate(paths, ck, episodes, workers, render=not args.no_render)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
