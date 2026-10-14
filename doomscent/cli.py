"""Single entry point: `doomscent <command> ...` dispatches to each command module."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "train": "doomscent.train",
    "evaluate": "doomscent.evaluate",
    "record": "doomscent.record",
    "montage": "doomscent.montage",
    "figures": "doomscent.plots.make_all",
    "timelapse": "doomscent.timelapse.render",
    "eval-watcher": "doomscent.eval_watcher",
    "baseline": "doomscent.baseline",
    "throughput": "doomscent.throughput",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        names = ", ".join(sorted(COMMANDS))
        print(f"usage: doomscent <command> [args]\ncommands: {names}", file=sys.stderr)
        raise SystemExit(2)
    module = importlib.import_module(COMMANDS[argv[0]])
    module.main(argv[1:])


if __name__ == "__main__":
    main()
