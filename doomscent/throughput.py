"""throughput: frames per second across rollout-worker and envs-per-worker settings (F6).

    uv run doomscent throughput --config configs/v0_defend_center.yaml --machine mac_m3pro \
        --workers 2,4,6,8,10 --envs 4,8,16 --seconds 100

Each setting trains for a fixed wall time through the real training path (our env, APPO,
the run's batch size). FPS is the median of Sample Factory's 10-second FPS readings after a
warm-up. Results: runs/throughput/<machine>/results.csv; each probe run folder is kept.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from doomscent.config import REPO_ROOT
from doomscent.metrics import read_tb_scalars
from doomscent.runs import RUNS_ROOT, RunPaths


def measure(paths: RunPaths, warmup_s: float) -> dict[str, float]:
    tb = read_tb_scalars(paths)
    fps = tb[tb.tag == "perf/_fps"].sort_values("wall_time")
    if fps.empty:
        return {"fps_median": float("nan"), "fps_p90": float("nan"), "n_readings": 0}
    t0 = tb.wall_time.min()
    after = fps[fps.wall_time - t0 >= warmup_s]
    steady = after["value"].to_numpy()
    slope = float("nan")
    if len(after) >= 2:
        dt = after.wall_time.iloc[-1] - after.wall_time.iloc[0]
        slope = float((after.env_steps.iloc[-1] - after.env_steps.iloc[0]) / max(dt, 1e-9))
    return {
        "fps": slope,
        "fps_median": float(np.median(steady)) if len(steady) else float("nan"),
        "fps_p90": float(np.quantile(steady, 0.9)) if len(steady) else float("nan"),
        "n_readings": len(steady),
        "env_steps": int(tb.env_steps.max()),
    }


def remeasure(out_dir: Path, warmup: float) -> None:
    """Recompute every row from the kept probe-run folders (no training)."""
    csv = out_dir / "results.csv"
    df = pd.read_csv(csv)
    rows = []
    for r in df.to_dict("records"):
        lt = (
            int(r.get("learner_threads", 1))
            if r.get("learner_threads") == r.get("learner_threads")
            else 1
        )
        name = f"tp_{r['machine']}_w{int(r['workers']):02d}_e{int(r['envs_per_worker']):02d}"
        name += f"_lt{lt:02d}" if lt != 1 else ""
        rows.append({**r, **measure(RunPaths.for_run(name, runs_root=out_dir), warmup)})
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(pd.DataFrame(rows).sort_values("fps", ascending=False).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent throughput")
    ap.add_argument("--config", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--workers", default="2,4,6,8,10")
    ap.add_argument("--envs", default="4,8,16")
    ap.add_argument("--learner-threads", default="1")
    ap.add_argument("--seconds", type=int, default=100)
    ap.add_argument("--warmup", type=float, default=40.0)
    ap.add_argument("--remeasure", action="store_true", help="recompute from saved runs")
    args = ap.parse_args(argv)
    out_dir = RUNS_ROOT / "throughput" / args.machine
    if args.remeasure:
        remeasure(out_dir, args.warmup)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    rows = []
    if results_csv.exists():
        prev = pd.read_csv(results_csv)
        if "learner_threads" not in prev:
            prev["learner_threads"] = 1
        rows = prev.fillna({"learner_threads": 1}).to_dict("records")
    done = {(r["workers"], r["envs_per_worker"], int(r.get("learner_threads", 1))) for r in rows}
    grid = [
        (w, e, lt)
        for lt in [int(x) for x in args.learner_threads.split(",")]
        for w in [int(x) for x in args.workers.split(",")]
        for e in [int(x) for x in args.envs.split(",")]
    ]
    for w, e, lt in grid:
        if (w, e, lt) in done:
            continue
        name = f"tp_{args.machine}_w{w:02d}_e{e:02d}" + (f"_lt{lt:02d}" if lt != 1 else "")
        paths = RunPaths.for_run(name)
        cmd = [
            "uv",
            "run",
            "doomscent",
            "train",
            "--config",
            args.config,
            f"run_name={name}",
            f"train.num_workers={w}",
            f"train.num_envs_per_worker={e}",
            "train.train_for_env_steps=10000000000",
            "train.wandb.enabled=false",
            f"train.max_wall_seconds={args.seconds}",
            f"train.learner_threads={lt}",
        ]
        print(f"[throughput] workers={w} envs={e} learner_threads={lt}", flush=True)
        t = time.time()
        log = RUNS_ROOT / f"{name}.log"
        with open(log, "w") as f:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
        rc = proc.returncode
        m = measure(paths, args.warmup)
        row = {
            "machine": args.machine,
            "workers": w,
            "envs_per_worker": e,
            "learner_threads": lt,
            "total_envs": w * e,
            **m,
            "wall_s": time.time() - t,
            "exit_code": rc,
        }
        rows.append(row)
        print(f"[throughput] {row}", flush=True)
        pd.DataFrame(rows).to_csv(results_csv, index=False)
        dest = out_dir / name
        if not dest.exists():
            shutil.move(str(paths.root), dest)
            shutil.move(str(log), dest / "train.log")
    df = pd.DataFrame(rows).sort_values("fps_median", ascending=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
