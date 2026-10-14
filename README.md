# Gradient Doomscent

Reinforcement learning agents that learn to fight in Doom from raw pixels, using [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) with the free Freedoom game content and [Sample Factory](https://www.samplefactory.dev/) APPO. Every clip, model, and file here can be shared: no commercial Doom files are used.

This repo is built in layered versions (see `SPEC.md`). **v0, "It learns to fight"**, trains an agent on ViZDoom's Defend the Center scenario (stand in the middle of a round room and kill monsters closing in from every side) and produces the whole asset pipeline: held-out evaluation, demo recordings rendered to 1280x720 clips, anomaly flags, figures, a progress montage, and a training timelapse of the network's "brain".

Results, gate numbers, and decisions are in [`PROGRESS.md`](PROGRESS.md). Odd behaviors the agents found are in [`WEIRD.md`](WEIRD.md). Curated clips and figures are listed in [`assets/INDEX.md`](assets/INDEX.md).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). On macOS, ViZDoom needs macOS 14 or newer.

```bash
uv sync
uv run pytest            # unit tests plus a few real short Doom episodes
```

`uv sync` is the only setup step; every dependency is pinned in `uv.lock`.

## Commands

Every command reads an OmegaConf YAML from `configs/` (saved into the run folder) and writes into `runs/<run>/`.

```bash
# scripted spin-and-shoot baseline over 100 held-out episodes (sets the v0 gate)
uv run doomscent baseline --config configs/v0_defend_center.yaml

# train (starts or resumes); in practice use the tmux launcher, which also starts the eval watcher
uv run doomscent train --config configs/v0_defend_center_s1.yaml
scripts/launch_run.sh configs/v0_defend_center_s1.yaml

# the eval watcher: evaluates and records every new checkpoint while training runs
uv run doomscent eval-watcher --run v0_defend_center_s1

uv run doomscent evaluate  --run v0_defend_center_s1 --ckpt best --episodes 100
uv run doomscent record    --run v0_defend_center_s1 --ckpt best --episodes 3 --overlay
uv run doomscent montage   --run v0_defend_center_s1 --ckpts auto     # or 0,1M,5M,final
uv run doomscent figures   --run v0_defend_center_s1
uv run doomscent timelapse --run v0_defend_center_s1
uv run doomscent throughput --config configs/v0_defend_center.yaml --machine mac_m3pro
```

`--ckpt` takes a step (`2.5M`, `500K`, `2500000`), `latest`, `final`, or `best` (best held-out mean kills). Thin shell wrappers live in `scripts/`.

## What a run folder holds

```
runs/<run>/
  config.yaml          resolved config
  meta/                git commit, action space, scenario file hashes, seeds, map geometry
  sf/                  Sample Factory experiment: every checkpoint is kept (checkpoint_p0/milestones/)
  metrics/             train.csv (from training stats), eval.csv (one row per checkpoint),
                       eval_episodes/<ckpt>.csv (one row per held-out episode)
  clips/               train/ eval/<ckpt>/ best/ flagged/ final/: .lmp demos, .json.gz sidecars, .mp4 clips
  flagged.csv          every anomaly-detector hit
  figures/             F1-F6 and F8, 16:9 / square / paper, light and dark, PNG and PDF
  montage/  timelapse/ derived videos
```

A run can be re-created from its folder alone: `uv run doomscent train --config runs/<run>/config.yaml`.

## How it works

- **Env** (`doomscent/env.py`): Defend the Center rendered at 256x144 and downsampled to 128x72 RGB, HUD off, frame skip 4. Factored discrete action heads (`doomscent/actions.py`): turn (none, or 1.5, 4, or 9 degrees per tic each way) and attack. Reward comes only from game-variable deltas (`doomscent/reward.py`): +1 kill, -1 death, +0.01 per hp of damage dealt, -0.005 per hp taken, pickups. Game variables are never shown to the policy.
- **Seeds**: training episodes draw scenario seeds from [0, 1,000,000); evaluation uses the reserved held-out range starting at 900,000,000. ViZDoom replays a seed exactly, so held-out seeds prevent memorization.
- **Training** (`doomscent/train.py`, `doomscent/sf_integration.py`): Sample Factory APPO with its tested Doom defaults, a CNN plus GRU policy, batch 2048, on CPU. Checkpoints are saved every 500K env steps (every 100K on the timelapse seed) and never deleted.
- **Evaluation** (`doomscent/evaluate.py`, `doomscent/eval_watcher.py`): 100 held-out episodes per checkpoint with the deterministic policy, a demo and per-step sidecar for each, anomaly detectors (`doomscent/detectors.py`), and rendered clips.
- **Clips** (`doomscent/render.py`): demos replay at 1280x720 with the HUD on at 35 fps; each clip also gets a minimap version and, for best and final checkpoints, an overlay version with action probabilities and the value estimate beside the game view.
- **Timelapse** (`doomscent/timelapse/`): real sampled units and weights from every checkpoint; edges lit by weight change, nodes by activity on the frame shown, gradient-saliency attention on the fight footage, and a UMAP mind map of a fixed probe set.

## Content and licences

Game content: Freedoom (bundled with ViZDoom) and ViZDoom's bundled scenarios. Font: Inter (SIL Open Font License, `assets/fonts/`).
