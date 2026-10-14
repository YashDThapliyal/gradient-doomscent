"""Data access for figures: run CSVs, sidecars, sibling seeds, and subtitles."""

from __future__ import annotations

import json
import re

import pandas as pd

from doomscent.runs import RUNS_ROOT, RunPaths

_SEED_SUFFIX = re.compile(r"_s\d+$")


def run_group(name: str) -> str:
    return _SEED_SUFFIX.sub("", name)


def sibling_runs(paths: RunPaths) -> list[RunPaths]:
    """All seeds of the same experiment (runs named <group>_s<k>) that have eval data."""
    group = run_group(paths.name)
    if group == paths.name:
        return [paths]
    found = []
    for d in sorted(RUNS_ROOT.glob(f"{group}_s*")):
        if _SEED_SUFFIX.sub("", d.name) == group and (d / "metrics" / "eval.csv").exists():
            found.append(RunPaths.for_run(d.name))
    return found or [paths]


def load_eval(paths: RunPaths) -> pd.DataFrame:
    f = paths.metrics / "eval.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run the eval watcher or `doomscent evaluate`")
    return pd.read_csv(f).sort_values("step").reset_index(drop=True)


def load_train(paths: RunPaths) -> pd.DataFrame:
    f = paths.metrics / "train.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing")
    return pd.read_csv(f).sort_values("env_steps").reset_index(drop=True)


def load_baseline_for(paths: RunPaths) -> dict | None:
    scenario = paths.load_config().env.scenario
    f = RUNS_ROOT / "baseline" / scenario / "baseline.json"
    return json.loads(f.read_text()) if f.exists() else None


def load_flagged(paths: RunPaths) -> pd.DataFrame:
    f = paths.flagged_csv
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["step", "detector"])


SCENARIO_NAMES = {"defend_the_center": "Defend the Center"}


def subtitle(paths: RunPaths, n_seeds: int | None = None, extra: str = "") -> str:
    cfg = paths.load_config()
    scen = SCENARIO_NAMES.get(cfg.env.scenario, cfg.env.scenario)
    parts = [paths.name, scen]
    if n_seeds is not None:
        parts.append(f"{n_seeds} seed{'s' if n_seeds != 1 else ''}")
    if extra:
        parts.append(extra)
    return "  ·  ".join(parts)
