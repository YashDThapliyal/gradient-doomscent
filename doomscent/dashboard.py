"""dashboard: a live, read-only view of training runs at http://127.0.0.1:8765.

    uv run doomscent dashboard --runs "v0_defend_center_s*" --port 8765

Reads only run folders (logs, eval.csv, train.csv, best pointers); never touches training.
The page polls /api/state every few seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from doomscent.runs import RUNS_ROOT

FRAMES_RE = re.compile(
    r"Fps is \(10 sec: ([\d.na]+), 60 sec: ([\d.na]+), 300 sec: ([\d.na]+)\)\. "
    r"Total num frames: (\d+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PAGE = Path(__file__).with_name("dashboard.html")


def tail(path: Path, n_bytes: int = 40_000) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - n_bytes))
        return ANSI_RE.sub("", f.read().decode("utf-8", errors="replace"))


def num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def run_state(root: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(root / "config.yaml")
    target = int(cfg.train.train_for_env_steps)
    log = root / "logs" / "train.log"
    matches = FRAMES_RE.findall(tail(log))
    frames, fps = 0, None
    if matches:
        _f10, f60, f300, total = matches[-1]
        frames, fps = int(total), num(f60) or num(f300)
    done_file = root / "TRAINING_DONE.json"
    age = time.time() - log.stat().st_mtime if log.exists() else 1e9
    if done_file.exists():
        status = "finished"
        frames = max(frames, json.loads(done_file.read_text())["env_steps"])
    elif age < 90:
        status = "training"
    else:
        status = "queued"  # waiting its turn (seeds train one at a time)
    out: dict[str, Any] = {
        "name": root.name,
        "status": status,
        "frames": frames,
        "target": target,
        "fps": fps if status == "training" else None,
        "eta_s": (target - frames) / fps if status == "training" and fps else None,
        "eval": [],
        "train": [],
        "best": None,
        "events": [],
    }
    ev = root / "metrics" / "eval.csv"
    if ev.exists():
        df = pd.read_csv(ev).sort_values("step")
        out["eval"] = [
            {
                "step": int(r.step),
                "kills": num(r.kills),
                "lo": num(r.kills_ci_lo),
                "hi": num(r.kills_ci_hi),
                "acc": num(r.accuracy),
            }
            for r in df.itertuples()
        ]
    tr = root / "metrics" / "train.csv"
    if tr.exists():
        df = pd.read_csv(tr, usecols=lambda c: c in {"env_steps", "kills_mean", "accuracy_mean"})
        df = df.dropna(subset=["kills_mean"])
        step = max(1, len(df) // 300)
        out["train"] = [
            {"step": int(r.env_steps), "kills": num(r.kills_mean), "acc": num(r.accuracy_mean)}
            for r in df.iloc[::step].itertuples()
        ]
    bp = root / "checkpoints_best.json"
    if bp.exists():
        b = json.loads(bp.read_text())
        out["best"] = {"step": b["step"], "kills": b["kills"], "ci": b.get("kills_ci")}
    lines = tail(root / "logs" / "eval_watcher.log", 20000).strip().splitlines()
    starts = [i for i, ln in enumerate(lines) if "watching" in ln]
    lines = lines[starts[-1] :] if starts else lines  # only the current watcher session
    out["events"] = [ln for ln in lines if "evaluated" in ln or "best" in ln][-6:]
    return out


def state(pattern: str) -> dict[str, Any]:
    runs = [
        run_state(d)
        for d in sorted(RUNS_ROOT.glob(pattern))
        if (d / "config.yaml").exists() and "_invalid" not in d.name
    ]
    base = RUNS_ROOT / "baseline" / "defend_the_center" / "baseline.json"
    b = json.loads(base.read_text()) if base.exists() else {}
    return {
        "time": time.strftime("%H:%M:%S"),
        "baseline": b.get("kills_mean"),
        "gate": b.get("gate_kills"),
        "runs": runs,
    }


def make_handler(pattern: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/api/state"):
                body = json.dumps(state(pattern)).encode()
                ctype = "application/json"
            elif self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                ctype = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent dashboard")
    ap.add_argument("--runs", default="v0_defend_center_s*", help="glob of run folders")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.runs))
    print(f"dashboard at http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
