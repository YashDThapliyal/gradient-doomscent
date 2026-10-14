"""Training metrics: Sample Factory's TensorBoard events to runs/<run>/metrics/train.csv.

Figures read only CSVs, never TensorBoard or W&B. Columns are keyed by env_steps (frames,
counting frame skip, as Sample Factory counts them). Different stats are logged at
different steps, so cells are left empty where a stat was not logged; nothing is filled in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from doomscent.runs import RunPaths

RENAMES = {
    "perf/_fps": "fps",
    "reward/reward": "ep_reward_mean",
    "policy_stats/avg_kills": "kills_mean",
    "policy_stats/avg_deaths": "deaths_mean",
    "policy_stats/avg_damage_dealt": "damage_dealt_mean",
    "policy_stats/avg_damage_taken": "damage_taken_mean",
    "policy_stats/avg_accuracy": "accuracy_mean",
    "policy_stats/avg_survival_time_s": "survival_time_s_mean",
    "policy_stats/avg_shots": "shots_mean",
    "len/len": "ep_len_mean",
    "train/value_loss": "value_loss",
    "train/policy_loss": "policy_loss",
    "train/kl_divergence": "kl",
    "train/entropy": "policy_entropy",
    "train/grad_norm": "grad_norm",
    "train/lr": "lr",
}


def head_names(paths: RunPaths) -> list[str]:
    spec = json.loads((paths.meta / "action_space.json").read_text())
    return [h["name"] for h in spec["heads"]]


def read_tb_scalars(paths: RunPaths) -> pd.DataFrame:
    """Long table: tag, env_steps, value, wall_time from every event file of the run."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows = []
    for f in sorted((paths.sf_dir / ".summary" / "0").glob("events*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            rows += [(tag, e.step, e.value, e.wall_time) for e in ea.Scalars(tag)]
    return pd.DataFrame(rows, columns=["tag", "env_steps", "value", "wall_time"])


def column_name(tag: str, heads: list[str]) -> str | None:
    if tag in RENAMES:
        return RENAMES[tag]
    if tag.startswith("train/entropy_head"):
        i = int(tag.removeprefix("train/entropy_head"))
        return f"policy_entropy_{heads[i]}" if i < len(heads) else None
    if tag.startswith("policy_stats/avg_act_"):
        return "action_hist_" + tag.removeprefix("policy_stats/avg_act_")
    if tag.startswith("policy_stats/avg_") and not tag.endswith(("_min", "_max")):
        return tag.removeprefix("policy_stats/avg_") + "_mean"
    return None


def write_train_csv(paths: RunPaths) -> Path | None:
    """Rebuild metrics/train.csv from the event files. Returns its path, or None if no data."""
    long = read_tb_scalars(paths)
    if long.empty:
        return None
    heads = head_names(paths)
    long["column"] = [column_name(t, heads) for t in long["tag"]]
    long = long.dropna(subset=["column"])
    t0 = long["wall_time"].min()
    wide = long.pivot_table(index="env_steps", columns="column", values="value", aggfunc="last")
    wall = long.groupby("env_steps")["wall_time"].max() - t0
    wide.insert(0, "wall_time_s", wall)
    wide = wide.reset_index().sort_values("env_steps")
    out = paths.metrics / "train.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    wide.to_csv(tmp, index=False)
    tmp.replace(out)
    return out
