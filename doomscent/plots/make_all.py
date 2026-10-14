"""figures: regenerate every v0 figure for a run from its CSVs and sidecars.

uv run doomscent figures --run v0_defend_center_s1
"""

from __future__ import annotations

import argparse
import importlib
import json
import traceback

from doomscent.plots.common import export
from doomscent.runs import RunPaths

FIGURES = {
    "F1": "doomscent.plots.f1_learning_curve",
    "F2": "doomscent.plots.f2_exchange",
    "F3": "doomscent.plots.f3_actions",
    "F4": "doomscent.plots.f4_entropy",
    "F5": "doomscent.plots.f5_scorecard",
    "F6": "doomscent.plots.f6_throughput",
    "F8": "doomscent.plots.f8_path_maps",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="doomscent figures")
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default="", help="comma-separated figure IDs")
    args = ap.parse_args(argv)
    paths = RunPaths.for_run(args.run)
    wanted = [f.strip() for f in args.only.split(",") if f.strip()] or list(FIGURES)
    index, failed = {}, []
    for fid in wanted:
        try:
            spec = importlib.import_module(FIGURES[fid]).build(paths)
            files = export(spec, paths.figures)
            index[fid] = {
                "title": spec.title,
                "subtitle": spec.subtitle,
                "files": [f.name for f in files],
            }
            print(f"{fid}: {spec.title}  ({len(files)} files)")
        except Exception as e:  # report every figure, then fail at the end
            failed.append(fid)
            print(f"{fid}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()
    paths.figures.mkdir(parents=True, exist_ok=True)
    (paths.figures / "INDEX.json").write_text(json.dumps(index, indent=2))
    if failed:
        raise SystemExit(f"figures failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
