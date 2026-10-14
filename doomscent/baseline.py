"""Scripted spin-and-shoot baseline: sets the v0 gate.

The turn speed is chosen on training seeds (best mean kills), then the chosen policy is
measured on held-out seeds. Choosing the strongest speed keeps the gate honest.

    uv run doomscent baseline --config configs/v0_defend_center.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from doomscent.config import REPO_ROOT, load_config
from doomscent.env import make_env
from doomscent.rollout import SpinAndShootPolicy, run_episode
from doomscent.stats import bootstrap_ci


def baseline_dir(scenario: str) -> Path:
    return REPO_ROOT / "runs" / "baseline" / scenario


def load_baseline(scenario: str) -> dict:
    """Read the measured baseline (mean kills, gate) for a scenario."""
    path = baseline_dir(scenario) / "baseline.json"
    if not path.exists():
        raise FileNotFoundError(f"no baseline at {path}; run `doomscent baseline` first")
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent baseline")
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    b = cfg.baseline

    train_env = make_env(cfg, split="train")
    tr0 = cfg.seeds.train.start
    selection = {}
    for choice in b.turn_choices_to_try:
        policy = SpinAndShootPolicy(train_env, int(choice))
        kills = [
            run_episode(train_env, policy, tr0 + i, keep_trace=False).stats["kills"]
            for i in range(b.selection_episodes)
        ]
        label = train_env.actions.heads[train_env.actions.head_names.index("turn")].choices[choice]
        selection[label] = float(np.mean(kills))
        print(f"selection (train seeds): turn={label}: {selection[label]:.2f} kills/episode")
    train_env.close()
    best_label = max(selection, key=selection.get)

    env = make_env(cfg, split="heldout")
    turn_choices = env.actions.heads[env.actions.head_names.index("turn")].choices
    policy = SpinAndShootPolicy(env, turn_choices.index(best_label))
    rows = []
    for i in range(b.episodes):
        seed = cfg.seeds.eval_start + i
        res = run_episode(env, policy, seed, keep_trace=False)
        rows.append({"episode": i, "seed": seed, **res.stats, "reward": res.reward})
    env.close()

    df = pd.DataFrame(rows)
    lo, hi = bootstrap_ci(df["kills"])
    mean = float(df["kills"].mean())
    out = {
        "scenario": cfg.env.scenario,
        "policy": f"spin_and_shoot(turn={best_label}, attack=on)",
        "selection_train_seeds": selection,
        "episodes": int(b.episodes),
        "heldout_seeds": [int(df.seed.min()), int(df.seed.max())],
        "kills_mean": mean,
        "kills_ci95": [lo, hi],
        "kills_std": float(df["kills"].std()),
        "deaths_mean": float(df["deaths"].mean()),
        "accuracy_mean": float(df["accuracy"].mean()),
        "survival_time_s_mean": float(df["survival_time_s"].mean()),
        "gate_multiplier": float(b.gate_multiplier),
        "gate_kills": mean * float(b.gate_multiplier),
    }
    d = baseline_dir(cfg.env.scenario)
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "episodes.csv", index=False)
    (d / "baseline.json").write_text(json.dumps(out, indent=2))
    OmegaConf.save(cfg, d / "config.yaml", resolve=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
