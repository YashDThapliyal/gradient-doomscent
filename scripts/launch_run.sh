#!/usr/bin/env bash
# Start (or resume) one run in tmux: training plus its eval watcher, each logging to the run folder.
#   scripts/launch_run.sh configs/v0_defend_center_s1.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="$1"
RUN=$(uv run python -c "from doomscent.config import load_config; print(load_config('$CONFIG').run_name)" 2>/dev/null | tail -1)
mkdir -p "runs/$RUN/logs"
tmux new-session -d -s "train_$RUN" "uv run doomscent train --config $CONFIG >> runs/$RUN/logs/train.log 2>&1"
sleep 20  # let training write its config before the watcher starts
tmux new-session -d -s "watch_$RUN" "uv run doomscent eval-watcher --run $RUN >> runs/$RUN/logs/eval_watcher.out 2>&1"
echo "started tmux sessions train_$RUN and watch_$RUN"
