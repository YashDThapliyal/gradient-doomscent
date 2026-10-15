# PROGRESS

## Final report

**v0 ("It learns to fight") is done. All three gates in v0's "Done when" list pass.**

An APPO agent (Sample Factory 2.1.1, CNN plus GRU, CPU only on an M3 Pro Mac) learned Defend the Center from 128x72 pixels in 20M env steps per seed. It beats the scripted spin-and-shoot baseline (4.42 kills) by about 2.4x on held-out seeds, in all 3 of 3 seeds:

| Seed | Best checkpoint | Held-out kills, selection seeds (95% CI) | Held-out kills, fresh seeds (95% CI) | Accuracy | First checkpoint past the gate |
| --- | --- | --- | --- | --- | --- |
| 1 | 20M | 10.68 [10.43, 10.93] | **10.48 [10.21, 10.74]** | 41% | 5M |
| 2 | 19M | 10.36 [10.10, 10.63] | **10.49 [10.21, 10.77]** | 40% | 6.5M |
| 3 | 18M | 10.46 [10.21, 10.71] | **10.56 [10.26, 10.84]** | 40% | 4.5M |

Gate: 1.5 x 4.42 = 6.63 kills per episode over 100 held-out episodes. "Selection seeds" are held-out seeds 900,000,000 to 900,000,099, on which `best` was chosen; "fresh seeds" are 100 different held-out seeds (900,001,000 onward) used only to re-score the chosen checkpoint, so they are free of selection bias. Deterministic policy throughout.

**What the agent learned.** It starts by never firing (seeds 1 and 3) or, by chance, already spinning and firing (seed 2). In all three seeds it then rediscovers the scripted spin-and-shoot strategy (0.5M to 1.5M steps: turn at full speed every step, fire every step, about 2.6 kills), then breaks out of it by learning to stop turning when a target is centered and hold fire otherwise: accuracy rises from 10% to 41%, kills from 2.6 to 10.7, from the same 26 bullets. Every game still ends in death once the ammo runs out (no pickups in this scenario), and the value estimate visibly drops as ammo runs low. The brain timelapse shows the vision layers settling early and the memory-to-action connections changing last; the attention overlay shows the trained network looking at the monsters it is about to shoot, while the 100K network barely attends to anything (its gradients are about 70x weaker).

**Done when:**
- [x] Best checkpoint averages at least 1.5x the scripted baseline's kills over 100 held-out-seed episodes in at least 2 of 3 seeds: 3 of 3 (10.48, 10.49, 10.56 on fresh seeds; gate 6.63).
- [x] One command each regenerates the montage, the timelapse, and all v0 figures from the run folder: `doomscent montage --run <run> --ckpts auto`, `doomscent timelapse --run v0_defend_center_s1`, `doomscent figures --run <run>`; all three were run by `scripts/finalize_v0.sh` (log: `runs/queue_v0.log`).
- [x] A fresh clone reproduces a run from its config with nothing beyond the repo and its dependencies: `scripts/check_fresh_clone.sh` cloned commit 07a89d3 into `/tmp`, ran `uv sync`, 60 tests passed, trained 516K steps from `configs/v0_defend_center_s1.yaml`, and evaluated the trained 500K checkpoint (5.05 kills on 20 held-out episodes). Log: `runs/fresh_clone_check.log`.

**Assets** (curated copies in `assets/`, listed in `assets/INDEX.md`; videos are not committed):
- Figures F1 to F6 and F8, 3 sizes x 2 themes x PNG and PDF: `assets/figures/` (also `runs/<run>/figures/` for every seed).
- Step-0 clips, before and after, best-checkpoint clips (plain, minimap, overlay), 2x2 progress montage, highlight reel, side-by-side: `assets/clips/`.
- Training timelapse (1920x1080 60 fps, 1080x1350 mobile, brain-only, attention before/after, mind map) from 201 checkpoints: `assets/timelapse/`.
- Weird behaviors with clips: `WEIRD.md` (spin-and-shoot rediscovery in all seeds, seed 2 born a spinner, step-0 pacifists); clip in `assets/weird/`.
- Raw archive: `runs/v0_defend_center_s{1,2,3}/` (every checkpoint, every eval demo and sidecar, eval.csv, train.csv, flagged.csv, clips, figures), `runs/baseline/`, `runs/throughput/`, `runs/v0_smoke/`.

**Known issues** (details under Open issues):
- No cloud sync (no bucket on this machine); W&B runs are offline in `runs/<run>/wandb/` and can be uploaded with `wandb sync`.
- `runs/` is about 12 GB after its 21 GB of videos were deleted on 2026-10-15 (see Deviations). Every demo, sidecar, and milestone checkpoint is kept, so any clip can be re-rendered with `doomscent record`; the curated videos live in `assets/`.
- Defend the Center cannot show close escapes (low health only happens after the ammo is gone), so the highlight reel has streaks only; `kd_ratio` is always kills/1.
- Several engine and framework quirks were found and worked around (ViZDoom demo paths over 112 characters hang; ViZDoom draws its console over the second and later recorded episodes of a game; Sample Factory's `--train_for_seconds` never fires; Sample Factory pins its learner to one thread and its W&B init breaks with wandb 0.30; Mac sleep stops Sample Factory). Each is recorded with its fix below.

## Status

- **v0 complete, 2026-10-15 04:45.** Nothing is training. The live dashboard (`tmux` session `dashboard`, http://127.0.0.1:8765) is left running; it is read-only.

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

### Step 4: scripted baseline and v0 gate value: DONE (2026-10-14)

- Command: `uv run doomscent baseline --config configs/v0_defend_center.yaml`
- Turn speed chosen on training seeds 0 to 29 (mean kills): left slow 3.90, left med 3.80, left fast 2.67, right slow 4.10, **right med 4.43**, right fast 2.90.
- Held-out measurement, spin-and-shoot (turn right 4 deg/tic, attack every step), 100 episodes, seeds 900000000 to 900000099:
  - **Mean kills 4.42, 95% bootstrap CI [4.22, 4.63]**, sd 0.99, accuracy 0.17, survival 13.8 s, deaths 1.0.
  - **v0 gate: 1.5 x 4.42 = 6.63 mean kills per episode** on 100 held-out episodes.
- Files: `runs/baseline/defend_the_center/baseline.json`, `episodes.csv`, `config.yaml`.

### Step 5: demo recording, sidecars, renderer: PASS (2026-10-14)

- Command: `uv run doomscent record --run v0_random_agent --config configs/v0_defend_center.yaml --ckpt random --episodes 3 --kind train --overlay` (52 s wall).
- Output in `runs/v0_random_agent/clips/train/`: per episode a `.lmp` demo, a `.json.gz` sidecar, and three clips (plain, `__minimap`, `__overlay`), for example `v0_random_agent__step_0000000__dtc__seed_900000002__ep02__kills4__minimap.mp4`.
- Clips play back: every clip decoded end to end with PyAV (`doomscent.video.probe_video`): 1280x720 (overlay 1680x720), 35.0 fps, H.264.
  - seed 900000000: 278 frames (7.94 s), seed 900000001: 304 frames, seed 900000002: 442 frames.
- Sidecars match episode lengths: sidecar step counts 70, 76, 111 equal the env's episode step counts; sidecar `episode_tics` 288, 314, 452 equal the replay's episode time exactly; replayed kills equal recorded kills (3, 0, 4). Frame count is episode tics minus the scenario's 10 start-up tics.
- Test: `tests/test_recording.py::test_record_and_replay_match` (records, replays, decodes, checks lengths), plus filename and path-guard tests. Suite: `51 passed`.

### Step 6: throughput test and config choice: DONE (2026-10-14)

- Command: `uv run doomscent throughput --config configs/v0_defend_center.yaml --machine mac_m3pro --workers 2,4,6,8,10,12 --envs 4,8,16 --seconds 100`, then `--remeasure` (FPS = slope of env steps over wall time after a 40 s warm-up). Plus `--workers 3 --envs 8`, and 3 concurrent runs at 3 workers x 8 envs.
- Single run, frames per second (env steps counting frame skip): every setting falls between 1.8K and 2.9K. Best 2,949 (4 workers x 4 envs); 3 x 8 gives 2,662; 12 x 16 drops to 1,843.
- **Bottleneck: the CPU learner, not env sampling.** Sample Factory's end-of-run profile (6 x 8): learner `train` 91 s of 107 s, of which `forward_head` (the conv encoder) 68 s; inference worker `forward` 55 s. A batch of 2048 frames takes about 1.4 s through the conv encoder on this Mac whatever the torch thread count (1 to 10 threads: 1.6 s to 1.4 s). The torch macOS arm64 wheels have no oneDNN (`torch.backends.mkldnn.is_available() == False`); torch 2.5.1, 2.7.1, 2.9.1, 2.11.0, 2.13.0 were all benchmarked and none is materially faster, so torch stays at 2.14.1.
- **3 concurrent runs (3 workers x 8 envs each): 1,639 + 1,522 + 1,522 = 4,683 FPS total**, 1.6x the best single run, because each run's single-core learner is the limit.
- **Chosen config: 3 seeds trained at the same time, each with 3 rollout workers x 8 envs per worker** (`configs/v0_defend_center.yaml`).
- Files: `runs/throughput/mac_m3pro/results.csv`, `concurrent.csv`, and every probe run folder with its `train.log`.

### Step 7: training, checkpoint retention, eval watcher: PASS (re-confirmed 2026-10-15)

- **Corrected held-out kills, smoke run** (re-evaluated after the console-overlay fix, 100 held-out episodes each, deterministic policy, `runs/v0_smoke/metrics/eval.csv`): step 0: 0.00; 500K: 2.83 [2.62, 3.04]; 1M: 3.11 [2.94, 3.28]; 1.5M: 3.50 [3.32, 3.69]; 2M: 4.52 [4.32, 4.72]; 2.5M: 4.76 [4.58, 4.94]; 3M: 5.17 [4.98, 5.36]; 3.5M: 5.34 [5.15, 5.53]; 4M: 6.02 [5.82, 6.22]. Rising at every checkpoint.
- The numbers listed below this line were the original evaluations, which were corrupted by the console-overlay bug and are kept here only as the record of what was first reported.

- Command: `scripts/launch_run.sh configs/v0_smoke.yaml` (4M env steps, checkpoints every 500K, dense every 100K, W&B offline), tmux sessions `train_v0_smoke` and `watch_v0_smoke`.
- Training finished: `runs/v0_smoke/TRAINING_DONE.json` (`env_steps 4022272`). Every checkpoint kept: 41 milestones in `runs/v0_smoke/sf/checkpoint_p0/milestones/` (step 0 plus every 100K).
- **Held-out kills rise** (eval watcher, 100 held-out episodes each, deterministic policy, `runs/v0_smoke/metrics/eval.csv`): step 0: 0.00; 500K: 2.96 [2.80, 3.12]; 1M: 3.00 [2.85, 3.15]; 1.5M: 3.31 [3.04, 3.57]; 2M: 4.04 [3.86, 4.23].
- Training kills (stochastic policy, `train.csv`, 500K bins): 3.13, 3.91, 3.86, 3.96, 4.19, 4.44, 5.19, 5.71 by 3.5M, with accuracy rising from 0.15 to 0.22.
- **Diagnosis of the early plateau.** From 0.7M to 2M training kills sat near 3.9. Shots per episode were 25.8 of the 26 rounds the pistol starts with, and each hit kills, so kills were capped by accuracy (about 15%). Two probes ran beside the smoke run to test whether the reward or the discount was the problem: `configs/v0_probe_gamma995.yaml` (gamma 0.995) and `configs/v0_probe_shotcost.yaml` (an extra -0.03 per round spent). Through 1M steps both matched the base run exactly (kills 3.80 and 3.91 vs 3.86, shots about 26). The base run then broke through on its own after 2M by aiming better. The probes were stopped at about 1M (folders kept) and **the spec's reward and Sample Factory's default discount are kept**. The ammo-spent term stays in the code, set to 0 in every config.
- Also exercised on the smoke run: `doomscent figures` (F1 to F6 and F8, 84 files in `runs/v0_smoke/figures/`), `doomscent montage --ckpts auto` (`runs/v0_smoke/montage/montage__seed_900000000__0_500K_1M_1.5M.mp4`, 562 frames).

### Step 8: evaluate, record, montage, figures, timelapse commands: PASS (2026-10-14)

- `evaluate` and `record`: exercised by every eval watcher (100 held-out episodes per checkpoint with demos, sidecars, clips; train, best, and final clips through `record_and_render`). Seed 1, corrected evaluations: `runs/v0_defend_center_s1/metrics/eval.csv`.
- `figures --run v0_smoke`: all 7 figures (F1 to F6, F8) at 3 sizes x 2 themes x PNG and PDF, 84 files. Layout issues found by viewing them and fixed: F3 tick labels overlapped the next panel, F8 subtitle overflowed the square size, F5 rows were cramped, F6 direct labels overlapped. (Those smoke figures were later quarantined with the invalid evaluations; they regenerate from corrected CSVs.)
- `montage --run v0_smoke --ckpts auto`: 1280x720 2x2 grid, 562 frames, step tags (also quarantined; regenerated at the end).
- `timelapse --run v0_defend_center_s1 --subdir timelapse_test --max-checkpoints 6 --duration 8`: brain-data cache for 6 checkpoints (0 to 7.3M), then `timelapse_16x9.mp4` (1920x1080, 480 frames at 60 fps), `timelapse_mobile.mp4` (1080x1350), `brain_only.mp4`, `attention_before_after.mp4` (245 frames), `mind_map.mp4` (300 frames), in `runs/v0_defend_center_s1/timelapse_test/`. Frames viewed: brain with weight-change edges and node activity, progressive curve with gate and baseline, decision bars, saliency on the footage, step and score tag; the mind map separates the probe frames by turn direction. The smoke run had no valid evaluations at this point (quarantined), so the timelapse was checked on a subset of seed 1 instead; the test folder is separate from the real `timelapse/`.
- The brain's functional forward pass matches Sample Factory's model outputs to 1e-4 (`tests/test_brain.py`).

### Independent code review (2026-10-15 01:00): findings fixed

A read-only review agent went through `doomscent/`. Fixed (commit a7441e2, tests added where testable; 70 tests pass):
- Sample Factory writes milestone checkpoints in place, so the eval watcher could read a half-written file, crash, and stop evaluating with no restart. `list_checkpoints` now skips files younger than 20 s or not a complete zip; the watcher logs a failed evaluation and retries it on the next poll.
- `best` was updated incrementally; it is now the argmax over `eval.csv` (written atomically), so a re-evaluation can never leave it stale. Re-evaluating a checkpoint replaced its `eval.csv` row but appended duplicate `flagged.csv` rows; flags are now replaced per checkpoint too.
- Winner's curse: `best` is chosen on the same 100 held-out seeds it is reported on. `doomscent evaluate --confirm-offset 1000` re-scores the best checkpoint on 100 disjoint held-out seeds (900,001,000 onward); F5 and the final report use that unbiased number, with the selection-seed number beside it.
- Timelapse attention overlays renormalized every frame to full intensity, so a near-random network looked as attentive as a trained one. Saliency now uses one scale across all checkpoints (step 0's gradients turned out about 70x weaker than the trained network's). The timelapse tag showed an interpolated score no evaluated network produced; it now shows the on-screen checkpoint and its latest real held-out score.
- Smaller: F4 counted detector hits as flagged episodes; pickups depended on reward weights; `record` named clips by raw env steps instead of the checkpoint boundary; the timelapse assumed `normalize_returns`; the mind map hard-coded the turn-head size; the resume-safe config keys were narrowed so `eval.episodes`, `eval.deterministic`, and detector thresholds can never change mid-run.
- Not changed, noted: training accuracy (mean of per-episode ratios) and eval accuracy (pooled hits / shots) use different aggregations; figures compare like with like. `kd_ratio` is always kills/1 in this scenario (one death per episode).

### Step 9: 3 seeds trained, dense checkpoints on seed 1: DONE (2026-10-15)

- `configs/v0_defend_center_s{1,2,3}.yaml`, 20M env steps each: seed 1 finished 2026-10-15 01:14 (`runs/v0_defend_center_s1/TRAINING_DONE.json`, env_steps 20,004,864), seeds 2 and 3 at 03:28 (20,013,056 each). Seed 1 kept 212 milestone checkpoints (every 100K), seeds 2 and 3 kept 41 each (every 500K); none deleted.
- Seed 1 trained alone, then seeds 2 and 3 concurrently overnight (`scripts/queue_v0_parallel.sh`) at about 2.1K FPS each. Interruptions (Mac sleep, disk full, a renice mistake) are recorded under Open issues; each run resumed from its checkpoint.

### Step 10: final evaluations, assets, figures, timelapse, flagged clips, WEIRD.md: DONE (2026-10-15)

- Every regular checkpoint of every seed evaluated on 100 held-out episodes after the console-overlay fix (41 rows in each `eval.csv`, including 20M); final clips from the 20M checkpoint (`clips/final/`); best clips at every new best (`clips/best/`).
- Unbiased confirmation of each best checkpoint on 100 fresh held-out seeds: `runs/v0_defend_center_s*/metrics/confirm/` (numbers in the final report).
- `scripts/finalize_v0.sh` (log `runs/queue_v0.log`, 03:31 to 04:28): smoke re-evaluation, confirmations, `figures` for all seeds and the smoke run (84 files each), `montage --ckpts auto` per seed, `highlights` and `sidebyside` per seed, `timelapse` for seed 1 (201 checkpoints, cache in `runs/v0_defend_center_s1/timelapse/cache/`, outputs in `runs/v0_defend_center_s1/timelapse/`; `timelapse_16x9.mp4` is 1920x1080, 60.0 fps, 60.0 s).
- Flagged clips reviewed: seed 1 flagged 255 episodes, all at steps 0 to 1.5M (pacifist at step 0, spinning at 0.5M to 1.5M), none after 2M; seed 2: 74 (spinning, mostly the untrained policy); seed 3: 124 (pacifist at step 0, spinning at 0.5M to 1M). Real oddities written to `WEIRD.md`; the feature clip copied to `assets/weird/`.

### Step 11: v0 "Done when": PASS (2026-10-15)

- Gate in at least 2 of 3 seeds: 3 of 3 (see the final report). One command each for montage, timelapse, figures: yes. Fresh clone reproduces a run: yes (`runs/fresh_clone_check.log`).

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
- **Baseline turn speed is picked on training seeds** from all six turn choices (best mean kills over 30 episodes each), then measured on held-out seeds. Picking the strongest spin speed makes the gate harder, not easier.
- **Sidecars are gzipped JSON (`.json.gz`).** About 10x smaller; 100 episodes per evaluation over dozens of checkpoints would otherwise reach gigabytes.
- **Clip filenames** follow the spec's format, with the step zero-padded to at least 7 digits (`step_0000000`, `step_2500000`), which also matches the spec's `clips/train/step_0000000_*` pattern up to the run-name prefix. Plain clips have no suffix; the other versions end in `__minimap.mp4` and `__overlay.mp4`.
- **Overlay version** is 1680x720: the 1280x720 game view plus a 400 px side panel (step, seed, kills, health, value meter, per-head action probabilities), so nothing covers the game.
- **Clip seeds:** step-0 and per-checkpoint training clips use the first held-out seeds (900000000, 900000001, ...), so the same seed is available at every checkpoint for the montage.
- **Env steps are frames:** Sample Factory counts env steps as agent steps times frame skip (4), so checkpoint intervals, the x-axes of every figure, and `train_for_env_steps` are in game frames. This matches how Sample Factory reports Doom FPS.
- **`learner_threads` config option.** Sample Factory pins its learner to one torch thread; `doomscent/sf_integration.py` lets config raise it (applied inside the learner process). It did not help on this Mac (see step 6), so v0 uses 1.
- **Wall-clock limit observer.** Sample Factory 2.1.1's `--train_for_seconds` never fires in the parallel runner (its `total_train_seconds` is never updated); `train.max_wall_seconds` uses our own observer instead. Only the throughput test uses it.
- **Detector tuning (before the main runs).** On the smoke run's 500K evaluation, spinning fired on 45 of 100 episodes, wasted fire on 30, reward spike on 6. Tightened: spinning needs 95% of steps turning and 95% one way; wasted fire needs at least 22 shots at no more than 5% accuracy; the reward spike compares an episode with the other episodes of the same checkpoint (comparing with the previous checkpoint flagged ordinary learning as an exploit). Re-running the detectors on those same 100 sidecars: wasted fire 1, reward spike 0, spinning still 45. In Defend the Center the agent cannot move, so a checkpoint that spins does so in every episode; that is a policy-wide strategy, and no threshold makes it rare. Rendered flagged clips are capped at 3 per detector per evaluation; every hit is still logged in `flagged.csv`.
- **Clip budget per evaluation:** first 3 episodes, the best-scoring one, and the 2 strongest multi-kill streaks (the streak rule matched 25 of 100 episodes and made each evaluation take about 5 minutes). Eval watchers use 2 processes each so three of them fit beside three trainings.
- **Highlight reels contain only streaks in Defend the Center.** A close escape (health at or below 30, then still fighting) never happens: in 99 of 100 held-out games the agent drops below 40 health only after its ammo is gone, and never kills again after that. The detector exists and would find escapes in scenarios with pickups or movement.
- **Eval-only settings may change when a run resumes.** `train.py` refuses to resume a run whose config changed, except under `eval.`, `record.`, `detectors.`, `sync.`, `timelapse.`, `baseline.`, which cannot affect learning. Used once: slimmer routine clips after the disk filled.
- **Seeds train one at a time, not concurrently.** The throughput test favored 3 concurrent runs (4.7K vs 2.9K FPS total), but with eval watchers on top the Mac was overloaded (load 299) while the user was working on it. One run at a time gets about 2.5K FPS, so total wall time is nearly the same. Each run's config is unchanged (still 3 workers x 8 envs).
- **Training length: 20M env steps per seed.** The smoke run was still improving at 4M (held-out 4.04 at 2M, training kills 5.7 at 3.5M); 20M is affordable on this machine (about 4 hours with 3 concurrent runs).
- `spec.md` and `claude.md` were provided in lowercase. They were renamed (contents untouched) to `SPEC.md` and `CLAUDE.md`, the names CLAUDE.md and the README use, so links work on case-sensitive systems such as Linux and GitHub.

## Deviations from spec

- **Step order:** the 3 seed runs (step 9) were launched before the timelapse command was checked on the smoke run (the last item of step 8), because training is the long pole and the timelapse code cannot affect training. The timelapse was checked on the smoke run while the seeds trained.

- **Minimap drawn from ViZDoom's map geometry, not the automap buffer pixels.** The replay exposes sector lines and object positions (`sectors_info`, `objects_info`), the same data the automap draws. Drawing it ourselves gives the agent's exact recent path and the figure style (thin gray walls, accent path), and the same code draws the F8 path maps. Monsters are not drawn because dead monsters stay in the object list as corpses and would mislead.

- **numpy 1.26 and gymnasium 0.29 instead of the newest.** Sample Factory 2.1.1 (the latest release) requires `numpy<2` and `gymnasium<1.0`. ViZDoom 1.3.2 accepts both. Pinned in `pyproject.toml`.

- **Run-folder videos deleted; routine clips off for future runs (2026-10-15, user's call).** The spec renders clips at every checkpoint and evaluation and says to keep them all. The user decided only a start clip and a best clip are worth keeping. All 2,270 MP4s in `runs/` (about 21 GB) were deleted with their approval; free disk went from 2.6 GB to 25 GB. Kept: every demo (`.lmp`) and sidecar (14,213 each, so any clip re-renders with `doomscent record`), every checkpoint, all metrics, and the curated videos in `assets/`. New config key `eval.render_routine_clips` (false in `configs/v0_defend_center.yaml`): the watcher renders clips only for step 0 and, when training ends, the overall best checkpoint; per-checkpoint eval, train, flagged, and final clips are skipped. Missing key means the old behavior, so v0 run configs still describe what they did.

- **Checkpoints thinned and smoke run deleted (2026-10-15, user's call).** First, timelapses were built for seeds 2 and 3 (41 checkpoints each; `assets/timelapse/s2/`, `s3/`). Then every checkpoint except step 0, best, final, and the progress-montage steps was deleted, along with off-grid and Sample Factory resume checkpoints: kept 4 for seed 1 (0, 2M, 11M, 20M) and 5 each for seeds 2 (0, 2.5M, 10.5M, 19M, 20M) and 3 (0, 2M, 10M, 18M, 20M). `runs/v0_smoke` was deleted, and `scripts/finalize_v0.sh` no longer touches it and now builds all three timelapses. `runs/` went from 12 GB to 2.1 GB; free disk 34 GB. What still works: v1 can start from each best checkpoint; montage, highlights, side-by-side, figures, and confirmation evals; and the timelapse, which now lists checkpoints from its cache when their files are gone (`timelapse_checkpoints`) and re-renders deleted footage from the cached fight demos (`ensure_footage`). What is lost: new evaluations or analyses of intermediate checkpoints need retraining.

## Open issues

- **Disk full (2026-10-14 about 21:30; resolved 22:50).** Training and eval stopped with `No space left on device` (the disk had about 375 GB of other data; this project used 28 GB in `runs/` plus caches). Rendered MP4s were the bulk (seed 1: 7.8 GB of clips vs 40 MB of demos and sidecars). With the user's approval: deleted the quarantined corrupted evaluation outputs (`_invalid_eval_console_bug/`, 8.8 GB), Sample Factory's rolling resume checkpoints except the newest per run (milestone checkpoints all kept), and the throughput probes' checkpoints (results and logs kept). Free space went from 0.66 GB to 12 GB. From 2026-10-15 02:25 a tmux loop (`prune_rolling`) keeps only the newest two rolling resume checkpoints per run (every milestone checkpoint is kept), since they otherwise grow about 0.5 GB per run per hour. For the rest of training, routine per-checkpoint eval and train clips are plain only at CRF 30; best, final, and flagged clips stay CRF 20 with all versions; every demo is kept, so any clip can be re-rendered. Seed 1 resumed at 15,130,624 env steps. Not touched: uv's 12 GB cache (locked by another project's running uv processes) and the user's 10 GB Hugging Face cache (other projects).

- **Mac sleep stops Sample Factory (handled 2026-10-14 20:20).** The Mac slept from about 19:15 to 20:05; on waking, Sample Factory found no heartbeats for 3,000 s and stopped seed 1 cleanly at 14.4M (checkpoint saved, nothing lost). The queue was waiting for `TRAINING_DONE.json`, so nothing ran until it was noticed. Fixes: `caffeinate -dims` in tmux while runs are going, and `scripts/queue_v0.sh` now resumes any run whose training session ends before it finishes. Seed 1 resumed at 14,417,920 env steps.
- **The 20:20 resume did not actually train (fixed 21:01).** Earlier, to relieve the overloaded Mac, every process whose command line matched "doomscent" was reniced to 19; that included the tmux server, so every later session inherited nice 19. Sample Factory's rollout workers set their own niceness to 10 at startup, which is a priority increase from 19 and is refused without root (`psutil.AccessDenied` in `init_rollout_worker_process`). The workers died, frames stayed at 14,417,920, and Sample Factory stopped on missing heartbeats at 20:47 (the Mac also slept from 20:35, on battery with the lid closed). Fix: restarted the tmux server from a nice-0 shell; workers now run at their own nice 10. Lesson: never renice by command-line match; it caught the tmux server.
- **Sleep on battery.** `caffeinate -dims` cannot stop lid-close sleep on battery. Runs survive sleep (the queue resumes them from checkpoints), but training only progresses while the Mac is awake: lid open and plugged in.

- **ViZDoom console overlay on recorded episodes (fixed 2026-10-14 15:45).** From the second recorded episode on a game instance, ViZDoom drew its console log over the top of the screen, so the policy saw a corrupted view. Every evaluation episode after each eval worker's first was affected; the smoke run's 3.5M checkpoint scored 1.56 held-out kills under the bug and 5.35 when played without recording. Found because held-out kills fell while training kills rose; per-episode results showed each worker's first episode scoring 5 to 7 kills and the rest 1 to 2. `render_messages`, `console_enabled`, and `+con_notifytime 0` do not help. Fix: a fresh game for every recorded episode (`doomscent/env.py`), with a regression test (`test_recorded_episodes_show_the_policy_the_same_screen`, which fails without the fix). Training never records and is unaffected; the baseline does not record and is unaffected; random-agent clips are unaffected (the random policy ignores the screen). All earlier evaluation outputs were moved, not deleted, to `runs/<run>/_invalid_eval_console_bug/`, and every checkpoint is being re-evaluated.

- **Cloud sync is not configured.** The spec asks for rclone sync of `runs/` after every eval. The hook exists (`sync.rclone_remote` in config, called by the eval watcher) but no bucket or credentials are available on this machine, so it is set to null and skipped.
- **W&B runs are offline** (no W&B credentials on this machine). Each run's `runs/<run>/wandb/offline-run-*` can be uploaded later with `wandb sync`.

- **ViZDoom demo path bug (worked around).** `new_episode(path)` and `replay_episode(path)` hang forever, with no error, when the path is longer than 112 characters or has a component starting with `-`. Every demo is recorded and replayed through a short path in `/tmp/doomscent_<uid>/` and then moved; `check_demo_path` raises on unusable paths. Found by bisecting path lengths (112 works, 113 hangs).
