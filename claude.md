# CLAUDE.md

Working rules for this repo. `SPEC.md` is the source of truth for what to build; this file is the source of truth for how to work.

## Current scope

- **Build v0 only** ("v0: It learns to fight" in `SPEC.md`).
- Do not start v1 or later. Where v0 infrastructure must leave room for later versions (multiplayer, curriculum, timelapse), design for it, but do not build it.
- Do not edit `SPEC.md`. If the spec is wrong or ambiguous, make the most reasonable call, keep going, and record it under **Decisions** in `PROGRESS.md`.

## Working autonomously

- Work without stopping to ask. Gates replace human review.
- **Never mark a gate or task done without evidence**: the exact command run, its key output or numbers, and file paths of anything produced.
- If a gate fails, debug and fix it yourself. If something is truly impossible on this machine, take the fallback the prompt or spec names, and record why.
- Commit after each passed gate.

## PROGRESS.md

Keep `PROGRESS.md` at the repo root, updated as you work, with these sections:

1. **Final report**: written at the top when v0 is finished. What works, the gate results, where the assets are, and known issues.
2. **Status**: the current step and what is running (including tmux session names and log paths).
3. **Gates**: one entry per gate with pass or fail, the evidence, and the date.
4. **Decisions**: every judgment call where the spec was silent or ambiguous, and why.
5. **Deviations from spec**: anything done differently from `SPEC.md`, and why.
6. **Open issues**: known problems not yet fixed.

## Code

- Python 3.12, managed with **uv** only: `uv add` for dependencies, `uv run` to execute. Never use bare `pip`.
- Package code lives in `doomrl/`, following the repo structure in `SPEC.md`.
- Type hints on public functions. Small, testable modules.
- `uv run ruff check . && uv run ruff format --check . && uv run pytest` must pass before any commit.
- Every experiment setting comes from an OmegaConf YAML file in `configs/`. No hardcoded hyperparameters, reward weights, seeds, or thresholds in code.
- Seed everything. Training and held-out seed ranges come from config and must never overlap.

## Experiments

- **Never change the reward, action space, or hyperparameters of a run in progress.** Any change means a new config file and a new run name. Record the reason in `PROGRESS.md`.
- Each run folder must contain: the resolved config, git commit hash, action space, scenario file hashes, and seeds.
- Long jobs (training, large evals, rendering) run in **tmux**, log to a file in the run folder, and must not block the session. Check on them periodically and keep building in the meantime.
- Report numbers from held-out seeds unless a metric is explicitly about training.

## Data and assets

- **Never delete** checkpoints, demo files, clips, metrics, or figures. Keep all checkpoints, overriding Sample Factory's default retention.
- `runs/` and `data/` are gitignored. Never commit checkpoints, videos, or large binaries.
- Follow the asset capture rules and filenames in `SPEC.md` exactly. Recording is automatic, never a manual afterthought.
- When a clip or figure is good enough to share, copy it to `assets/` and add a line to `assets/INDEX.md`.
- Log every degenerate or surprising behavior in `WEIRD.md` with its clip, using the format in `SPEC.md`.

## Game content

- Use only Freedoom (bundled with ViZDoom), ViZDoom's bundled scenarios, Sample Factory's maps, and maps created in this repo.
- Never download or add commercial Doom game files (`doom.wad`, `doom2.wad`, or similar), or any other copyrighted game files.

## Machine

- Development machine: Apple Silicon Mac (M3 Pro, 18 GB memory), macOS 14 or newer required for ViZDoom.
- Train on CPU. Do not depend on Apple's GPU (MPS) for correctness.
- Use rollout worker and env counts from the throughput test, not guesses.

## Writing style

- No em dashes in any written output: README, `PROGRESS.md`, `WEIRD.md`, docstrings, figure titles, or commit messages. Use commas, colons, or separate sentences.
- Figure titles state the finding, not the topic, per the style guide in `SPEC.md`.
