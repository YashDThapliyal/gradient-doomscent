#!/usr/bin/env bash
# v0 "Done when": a fresh clone reproduces a run from its config with nothing beyond the
# repo and its dependencies. Clones the committed repo, runs `uv sync`, the test suite, a
# short training run from configs/v0_defend_center_s1.yaml, and an evaluation of it.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DIR="/tmp/doomscent_clone_check"
rm -rf "$DIR"
git clone -q "$REPO" "$DIR"
cd "$DIR"
echo "commit: $(git rev-parse HEAD)"
uv sync -q
uv run pytest -q -m "not game" | tail -1
uv run doomscent train --config configs/v0_defend_center_s1.yaml run_name=repro_check \
  train.train_for_env_steps=600000 train.wandb.enabled=false > train.log 2>&1
cat runs/repro_check/TRAINING_DONE.json; echo
uv run doomscent evaluate --run repro_check --ckpt latest --episodes 20 --workers 4 --no-render \
  | grep -E '"(ckpt|kills|kills_ci_lo|kills_ci_hi|episodes)"'
echo "fresh clone check passed"
