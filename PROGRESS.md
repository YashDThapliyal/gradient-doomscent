# PROGRESS

## Final report

Not written yet. v0 is in progress.

## Status

- Current step: 4 (scripted baseline over 100 held-out episodes).

## Gates

### Step 1: repo scaffold: PASS (2026-10-14)

- Command: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
- Output: `All checks passed!`, `5 files already formatted`, `4 passed in 0.62s`.
- Files: `pyproject.toml`, `uv.lock`, `doomscent/config.py`, `tests/test_config.py`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`.

### Step 2: ViZDoom and Sample Factory run on this Mac: PASS (2026-10-14)

- Installed: vizdoom 1.3.2, sample-factory 2.1.1, torch 2.14.1, numpy 1.26.4, gymnasium 0.29.1 (all in `uv.lock`).
- ViZDoom example: `basic.cfg` with random actions, 3 episodes, 665 steps in 0.19 s, rewards -114, -139, -385. No errors.
- Sample Factory benchmark env: `uv run python -m sf_examples.vizdoom.train_vizdoom --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=6 --num_envs_per_worker=4 --batch_size=2048 --device=cpu --train_for_env_steps=200000 --train_dir=runs/_bench --experiment=sf_benchmark_smoke --with_wandb=False`
  - Exit 0. `Collected {0: 221184}, FPS: 2151.9`; steady state about 2.4K FPS (log lines `Fps is (10 sec: 2457.1 ...)`).
  - Log: `runs/_bench/sf_benchmark.log`.
- No fallback to Stable-Baselines3 needed.

### Step 3: make_env, wrappers, sanity checks: PASS (2026-10-14)

- Command: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q` gives `43 passed`.
- Reward unit tests: `tests/test_reward.py` (10 tests: kill, damage dealt vs taken, death, pickups, scale, config).
- Detector unit tests: `tests/test_detectors.py` (15 tests: spinning, stuck, camping, wasted fire, pacifist, suicide, reward spike, enable flags).
- Kill count rises on a kill: `test_kill_count_rises_on_a_kill` (KILLCOUNT never decreases, every rise carries reward of at least +1, total matches the episode stat).
- Health drops when hit: `test_health_drops_when_hit` (every step where DAMAGE_TAKEN rises has HEALTH falling).
- Random agent completes 10 episodes: `test_random_agent_completes_10_episodes`.
- Spin-and-shoot beats random: `test_spin_and_shoot_beats_random`, seeds 200 to 219: `random 2.60 vs spin-and-shoot 4.70 kills/episode`.
- Files: `doomscent/env.py`, `doomscent/actions.py`, `doomscent/reward.py`, `doomscent/detectors.py`, `doomscent/rollout.py`, `configs/v0_defend_center*.yaml`.

## Decisions

- Package name is `doomscent` (not `doomrl`), per the project brief. W&B project name is `gradient-doomscent`.
- The package lives at the repo root (`doomscent/`) as SPEC.md's repo structure shows, using uv_build with `module-root = ""`.
- **Own Gymnasium env instead of ViZDoom's Gymnasium wrapper.** The wrapper has no factored action heads and no per-episode seed or demo-recording control. `doomscent/env.py` wraps `vzd.DoomGame` directly, and the same class is used by Sample Factory and by every tool, so the policy sees identical inputs everywhere.
- **Observation:** rendered at 256x144 (16:9, a native ViZDoom resolution), area-downsampled by 2 to 128x72, RGB, channels first, HUD off.
- **Turn head:** 7 choices (none, plus slow, medium, fast each way) at 1.5, 4.0, 9.0 degrees per tic through `TURN_LEFT_RIGHT_DELTA`. Doom's keyboard turn is about 5.6 to 7 degrees per tic, so this spans fine aim to fast sweeps. Verified the delta applies every tic (5 per tic over 4 tics turned exactly 20 degrees).
- **Reward is computed only from game-variable deltas;** the scenario's built-in reward is ignored. Positive HEALTH and ARMOR deltas count as pickups (they cannot rise any other way in single-player). A weapon or ammo pickup is one event per configured ammo or weapon variable that rises.
- **Shots fired** is counted as ammo spent with the weapon unchanged (the game has no shots variable). Accuracy is HITCOUNT divided by that.
- **Detectors live in `doomscent/detectors.py`,** not `wrappers.py`. They are pure functions over the sidecar step records, so they can be re-run on old recordings. Same for the reward, in `doomscent/reward.py`. Small modules, per CLAUDE.md.
- **Seed ranges:** training episodes draw from [0, 1,000,000); held-out evaluation uses [900,000,000, 900,100,000), with episode i of an evaluation using seed 900,000,000 + i. `load_config` rejects overlapping ranges. ViZDoom replays a seed exactly (tested).
- **Enemy visibility** for the pacifist detector counts labels-buffer objects whose class is a Doom monster. Defend the Center's monsters report as `Demon` and `MarineChainsawVzd`.
- `spec.md` and `claude.md` were provided in lowercase. macOS is case-insensitive, so they resolve as `SPEC.md` and `CLAUDE.md`; left untouched.

## Deviations from spec

- **numpy 1.26 and gymnasium 0.29 instead of the newest.** Sample Factory 2.1.1 (the latest release) requires `numpy<2` and `gymnasium<1.0`. ViZDoom 1.3.2 accepts both. Pinned in `pyproject.toml`.

## Open issues

None yet.
