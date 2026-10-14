# Doom RL: Project Spec

Last updated: 2026-10-14. This file is the source of truth for the project; see CLAUDE.md for working rules.

## Overview

We train reinforcement learning agents to fight in Doom from raw pixels, using ViZDoom and the free Freedoom game content, then build toward deathmatch personalities, self-play, a mode where people can play against the agent, and world-model "dream" videos. The project ships in layered versions: each one runs end to end on its own, and each one builds the infrastructure the next needs.

Guiding rules for every version:

1. **End to end or it is not done.** A version is complete only when it trains, evaluates, records clips, and regenerates its figures from one command each.
2. **Assets are outputs, not afterthoughts.** Every evaluation writes clips, metrics, and figures into a fixed folder layout (see Asset capture). Nothing is recorded by hand after the fact.
3. **One config per experiment.** Every run is defined by a YAML file that is saved with its outputs. No hardcoded hyperparameters.
4. **Log the weird stuff.** Any exploit, stall, or degenerate strategy goes in the weird behaviors log with a clip, from v0 onward.
5. **Free content only.** Only Freedoom, ViZDoom's bundled scenarios, Sample Factory's maps, and maps we make ourselves. No commercial Doom game files, so every clip, model, and repo file can be shared publicly.

Scope by version:

| Version | Name | Adds | Gate to move on |
| --- | --- | --- | --- |
| v0 | It learns to fight | Full pipeline on one combat scenario (Defend the Center) | Mean kills per episode at least 1.5x a scripted spin-and-shoot baseline over 100 held-out-seed episodes |
| v1 | It clears the arenas | Curriculum through harder combat scenarios up to Battle | Battle score at least 75% of Sample Factory's published pretrained agent, measured with our eval script |
| v2 | Fighting personalities | Multiplayer deathmatch vs bots, reward variants, duels | Duel round-robin results for 4 personalities |
| v3 | Self-play league | Population self-play in duels, Elo | Elo rising across snapshots; beats every v2 personality |
| Stretch A | Play against it | Human joins a live match vs any checkpoint | A person can play a full match against any checkpoint |
| Stretch B | World model dreams | Learned world model of Doom, dream rollouts | Side-by-side real vs imagined fight video |

## Core design decisions

These hold across all versions unless a version section says otherwise.

| Decision | Choice | Why |
| --- | --- | --- |
| Game | Doom via ViZDoom 1.3.x, Freedoom 2 content (bundled with ViZDoom) | Free and shareable; very fast engine |
| Scenarios | ViZDoom built-ins (Defend the Center, Defend the Line, Deadly Corridor, Deathmatch), Sample Factory's Battle maps, and ViZDoom's multiplayer maps (`cig.wad` map01, `multi_duel.wad`) | Ready-made, with known baselines |
| Policy input | RGB pixels, 128x72, HUD off | Sample Factory's tested resolution for Doom |
| Memory | Recurrent policy: CNN encoder plus GRU | Doom is partially observable; enemies leave the screen |
| Reward input | Game variables only (kills, frags, deaths, health, armor, ammo, damage dealt and taken, hits, position) | Ground truth for reward and detectors; never fed to the policy |
| Action space | Factored discrete heads: move (forward, back, none), strafe (left, right, none), turn (3 speeds each way, or none), attack (on, off), weapon (next, none; multiplayer only) | Smooth, human-like control |
| Frame skip | 4 in single-player, 2 in multiplayer | Sample Factory's settings |
| Start variety | A random seed per episode; a reserved held-out seed range for evaluation; random spawns in multiplayer | ViZDoom replays identically for a given seed, so held-out seeds prevent memorization |
| Algorithm | Sample Factory 2, APPO (asynchronous PPO), for every version | Fastest Doom trainer available, with built-in multiplayer, self-play, and population-based training (PBT) |
| Parallelism | Sample Factory rollout workers, several envs per worker | Emulation throughput sets training speed |
| Logging | Weights & Biases plus local CSV per run | W&B for live dashboards; CSV feeds the figure scripts |
| Compute | Local Mac for v0; rented many-core machine when a version needs it | Throughput scales with cores |

**Base reward (single-player).** All constants live in config; these are starting values:

| Event | Reward |
| --- | --- |
| Kill | +1.0 |
| Death | -1.0 |
| Damage dealt | +0.01 per health point |
| Damage taken | -0.005 per health point |
| Health or armor picked up | +0.005 per point |
| Weapon or ammo picked up | +0.1 |

Damage dealt is weighted above damage taken on purpose, to discourage "cowardice" (agents learning to hide to avoid losing reward), a failure prior fighting-game work ran into.

**Base reward (multiplayer).** Frag +1.0, death -0.5, plus the same damage and pickup terms. Suicide counts as a death and a -1.0 frag.

**Episode boundary.** Single-player: the scenario's own end (death or time limit). Multiplayer: a fixed-length match with automatic respawns; one match is one episode.

**Game variables and buffers.** Each scenario config enables the game variables the reward and detectors need. The labels buffer (which objects are on screen, and where) is turned on during training and evaluation for detectors and personality rewards, but never fed to the policy, and it is off in matches against humans.

**Training settings.** Start from Sample Factory's tested ViZDoom configs and change only what this table lists:

| Setting | Start value | Source |
| --- | --- | --- |
| Base config | `sf_examples.vizdoom` defaults for the nearest scenario | Sample Factory |
| Batch size | 2048 | Sample Factory |
| Rollout workers x envs per worker | Chosen by the v0 throughput test on the Mac; 72 x 8 on a 36-core machine | Sample Factory guidance |
| Resolution | 128x72 | Sample Factory |
| Discount (gamma) | Sample Factory default single-player; 0.995 multiplayer | Sample Factory |
| Recurrent core | GRU, on | Sample Factory |
| Policies | 1 in v0 and v1; 8 with PBT in v3 | Sample Factory |

## Lessons from prior work

Doom is one of the most studied RL games, so we start from tested settings and published baselines instead of tuning from scratch. This is a learning source, not a template: our repo, evaluation, asset pipeline, and versions are our own.

| Source | What they did | What we take |
| --- | --- | --- |
| [Sample Factory](https://www.samplefactory.dev/09-environment-integrations/vizdoom/) | APPO trainer reaching 50K+ frames per second on a 10-core machine; Battle and Battle2 maps; duel and deathmatch against bots; self-play with PBT; pretrained agents on Hugging Face | The training framework, its Doom configs, and its pretrained Battle agent as our v1 benchmark |
| [ViZDoom competitions paper](https://ar5iv.labs.arxiv.org/html/1809.03470) | Deathmatch-from-pixels competitions; strong entries predicted game features (such as "enemy visible") alongside the policy | An optional auxiliary "enemy visible" prediction head in v1, trained from the labels buffer |
| [World Models (Ha and Schmidhuber)](https://worldmodels.github.io/) | Learned a world model of a ViZDoom scenario and trained a policy inside the model's "dream" | The precedent for Stretch B, including acting inside the dream |
| [GameNGen](https://arxiv.org/abs/2408.14837) | A diffusion model that simulates Doom in real time | Reference point for Stretch B visuals, at a far larger scale than ours |
| [ViZDoom examples](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python) | Multiplayer hosting, bots added with `addbot`, spectator mode for humans, top-down automap buffer, episode recording and replay | v2 multiplayer, Stretch A, minimap assets, and our recording pipeline |

**Lessons that change the plan:**

- **Benchmark against published agents.** Sample Factory's pretrained models can be downloaded and run through our own eval script. That gives v1 an honest target instead of a number we invent.
- **Bots are ours to define.** Sample Factory notes that current ViZDoom no longer ships the bots used in their original curriculum. Bot strength comes from a bots config file, so we fix one per version and save it with every run.
- **Determinism cuts both ways.** A seeded ViZDoom episode replays exactly, which makes demos and debugging easy but also makes memorization easy. Evaluation always uses held-out seeds.
- **Known Doom failure modes:** spinning in place, getting stuck on walls, camping, firing nonstop at nothing, and killing yourself with your own rocket. The anomaly detectors target each of these.
- **Multiplayer means several processes.** Each player is its own ViZDoom instance connected over a local port. Sample Factory handles this during training; our own code needs it only for tournaments and human play.

**What we will not copy:** Sample Factory's experiment folder layout as our repo structure (we wrap it), or anyone's reward shaping without testing it.

## Tech stack

The project runs on Python 3.12 with ViZDoom, Sample Factory, and PyTorch, managed by uv. Every dependency is pinned in `uv.lock`, so the Mac and any rented machine run identical versions.

**Core**

| Job | Tool | Notes |
| --- | --- | --- |
| Language | Python 3.12 | ViZDoom ships Apple Silicon wheels (macOS 14 or newer) and Linux wheels for 3.10 to 3.14 |
| Dependencies and virtualenv | uv, `pyproject.toml` + `uv.lock` | `uv sync` is the only setup step on a new machine |
| Lint and format | ruff | Runs in CI and as a pre-commit hook |
| Tests | pytest | Reward, action-space, and detector logic; because the game content is free, CI also runs a few real short episodes |
| Config | OmegaConf YAML | One file per experiment, command-line overrides, resolved config saved into each run folder |

**Game and RL**

| Job | Tool | Notes |
| --- | --- | --- |
| Game engine | ViZDoom 1.3.x | Single-player and multiplayer, bots, game variables, depth, labels, and automap buffers |
| Game content | Freedoom 2 (bundled with ViZDoom) | Free to use and share |
| Scenarios and maps | ViZDoom's bundled scenarios, Sample Factory's Battle maps, our own maps in `scenarios/` | Each scenario is a `.wad` map plus a `.cfg` settings file |
| Bots | ViZDoom `addbot` with our own bots config file | Bot config saved with every multiplayer run |
| Training | Sample Factory 2, APPO | Every version; built-in multiplayer, self-play, PBT |
| Env API for our own tools | Gymnasium, via ViZDoom's Gymnasium wrapper | Used by evaluate, record, and the tournament runner |
| Neural networks | PyTorch | Learner on CPU on the Mac; a CUDA GPU speeds up the learner on rented machines |
| Elo | Own module, `doomrl/elo.py` | Standard Elo with fixed K; anchors held constant |

**Logging, data, and storage**

| Job | Tool | Notes |
| --- | --- | --- |
| Live dashboards | Weights & Biases | Project `doom-rl`, one W&B run per training run, tagged by version |
| Metrics for figures | CSV files, read with pandas | Written by our eval and training hooks; figure scripts never call W&B |
| Episode recordings | ViZDoom demo files (`.lmp`) | Tiny files that replay an episode exactly; rendered to video at any resolution later |
| Trajectory dataset | Compressed NumPy `.npz` chunks | Lossless frames, actions, and game variables, for Stretch B |
| Backup and sync | rclone to an object storage bucket (S3, GCS, or Cloudflare R2) | Syncs `runs/` and `data/` after every eval; checkpoints included |

**Video, figures, and timelapse**

| Job | Tool | Notes |
| --- | --- | --- |
| Rendering clips | ViZDoom demo replay at 1280x720 with HUD on | High-quality footage without slowing training |
| Minimap | ViZDoom automap buffer | Top-down view and movement trails |
| Video encode and stitching | FFmpeg via PyAV | H.264 MP4; grids and side-by-sides composed as NumPy frames, then encoded |
| Overlays and labels | NumPy + Pillow | Corner tags, decision bars, heatmap blending; Inter font bundled in `assets/fonts/` |
| Figures | matplotlib with `doomrl/plots/style.mplstyle` | Light and dark themes from the same script |
| Neuron activity | PyTorch forward hooks | Activations on the probe set, cached per checkpoint |
| Attention heatmaps | Captum (gradient saliency) | On the probe set and on rendered clip frames |
| Mind map | umap-learn | Fit once on the final checkpoint's features, then applied to every checkpoint |

**Stretch goals**

| Job | Tool | Notes |
| --- | --- | --- |
| Play against it | ViZDoom multiplayer with a human-controlled ViZDoom window | Keyboard and mouse, no extra game library needed |
| World model | TorchRL DreamerV3 components | Same building blocks as the pokeRL repo, which is a code reference |

**Compute and repo**

| Job | Tool | Notes |
| --- | --- | --- |
| Development and v0 training | Mac, M3 Pro | Rollout workers on CPU cores |
| v1 to v3 training | Rented many-core Linux machine, optionally with one GPU | Same `uv sync` setup |
| Stretch B training | Rented single-GPU Linux machine | Needs a GPU |
| Long runs on remote machines | tmux | Runs survive SSH disconnects |
| Code hosting and CI | GitHub + GitHub Actions | ruff, pytest, and a short real-episode smoke test on every push |

**Rules**

- Never add commercial Doom game files (the original `doom.wad` or `doom2.wad`) to the repo or any run.
- Upgrade a pinned dependency only on its own commit, with a short smoke-training run to confirm nothing changed.

## Repo structure and shared infrastructure

The repo wraps Sample Factory rather than living inside it, and is organized so every later version adds files rather than rewriting v0.

```
doom-rl/
  configs/              one YAML per experiment (v0_defend_center.yaml, ...)
  scenarios/            scenario .cfg files, our own maps, bots configs
  doomrl/
    env.py              make_env(): ViZDoom game setup, action space, seeding
    wrappers.py         reward, game-variable tracking, demo recording, detectors
    actions.py          the factored action space definition
    train.py            launches Sample Factory with our config and hooks
    eval_watcher.py     picks up each new checkpoint, runs eval and clip capture
    evaluate.py         metrics over N held-out episodes for any checkpoint
    record.py           renders demo files into clips (see Asset capture)
    tournament.py       multiplayer matches between any set of checkpoints and bots
    elo.py              ratings
    timelapse/          probe set, brain-data cache, composite renderer
    plots/              one script per figure, reads only from runs/
  scripts/              thin shell wrappers: train.sh, eval.sh, montage.sh, figures.sh
  runs/                 all outputs (gitignored, synced to cloud storage)
  data/                 trajectory dataset for Stretch B (gitignored, synced)
  assets/               curated clips and figures promoted for posts and the README
  WEIRD.md              weird behaviors log
  README.md
```

**Commands every version must support:**

- `train --config <yaml>` starts or resumes a run.
- `evaluate --run <name> --ckpt <step|latest|best> --episodes 100` writes metrics to the run folder, using held-out seeds.
- `record --run <name> --ckpt <...> --episodes 3` writes clips to the run folder.
- `montage --run <name> --ckpts 0,1M,5M,final` builds the side-by-side progress video.
- `figures --run <name>` regenerates every figure for that run from its CSVs.
- `timelapse --run <name>` renders the training timelapse.

**Evaluation runs beside training, not inside it.** The eval watcher is a separate process that notices each new checkpoint, evaluates it, and records clips. Training never pauses for evaluation.

**Checkpointing.** Save every 500K env steps, plus `best` whenever held-out score improves, plus `latest`. Override Sample Factory's default of keeping only recent checkpoints: keep all of them, since they are the raw material for montages and the timelapse.

**Reproducibility.** Each run folder stores its config, the git commit hash, the action space, the bots config, the scenario files' hashes, and the seed. A run must be re-creatable from its folder alone.

## Asset capture: recording and clipping

Recording is automatic and continuous. Every evaluated episode is saved as a ViZDoom demo file, which replays the episode exactly, and clips are rendered from those demos at full quality. Nothing about capture depends on remembering to do it. These recordings are the project's main assets; treat losing them like losing a checkpoint.

**How recording works.** Demo files are tiny, so every eval episode keeps one, along with a sidecar JSON of per-step actions, game variables, reward, and the policy's value estimate. Rendering replays the demo at 1280x720 with the HUD on, so footage looks like real Doom, not the 128x72 frames the policy sees. Anything can be re-rendered later at a different size or with different overlays.

**What gets recorded, and when:**

| Trigger | What is captured | Saved to |
| --- | --- | --- |
| Step 0 (before training) | 3 episodes of the untrained policy | `runs/<run>/clips/train/step_0000000_*` |
| Every checkpoint (500K steps) | 2 episodes from the eval watcher | `runs/<run>/clips/train/step_<N>_*` |
| Every evaluation | Demos for all 100 episodes; rendered clips for the first 3, the best-scoring, and any multi-kill streak | `runs/<run>/clips/eval/<ckpt>/` |
| New best checkpoint | 3 rendered episodes from `best` | `runs/<run>/clips/best/` |
| Flagged behavior | Any episode where an anomaly detector fires (below) | `runs/<run>/clips/flagged/` |
| End of run | 5 rendered episodes from the final checkpoint | `runs/<run>/clips/final/` |

**Clip format.**

- 1280x720, 35 fps (Doom's native tic rate), H.264 MP4, HUD on.
- Every rendered clip also gets a version with a top-down minimap inset (ViZDoom's automap) in a corner, showing the agent's position and recent path.
- Filename encodes run, checkpoint step, scenario, seed, episode, and result: `v0_defend_center_s1__step_2500000__dtc__seed_900007__ep02__kills14.mp4`.

**Anomaly detectors that auto-flag episodes.** Each writes the clip plus a line to `runs/<run>/flagged.csv`:

- **Spinning:** turning on most steps while barely changing position.
- **Stuck:** moving forward for many steps with no change in position (wall-hugging).
- **Camping:** staying inside a small area for most of an episode (multiplayer).
- **Wasted fire:** many shots with almost no hits.
- **Pacifist:** never firing in an episode with enemies on screen.
- **Suicide:** death from self-inflicted damage.
- **Reward spike:** episode reward far above the recent average (possible exploit).

Thresholds live in config; tune them so they fire on a handful of episodes per eval, not every episode.

**Derived assets** (built from recordings, never hand-edited):

- **Progress montage:** 2x2 grid of the same scenario and seed at 4 checkpoints, each labeled with its step count in a small corner tag.
- **Highlight reel:** auto-cut multi-kill streaks and close escapes (low health, survived), found from the sidecar JSON.
- **Overlay versions:** the same clip with action bars and a value meter beside the game view, never covering the action.
- **Path maps:** the agent's movement over many episodes drawn on the top-down map, early checkpoint vs late (F8).

**Storage and promotion.**

- `runs/` is the raw archive. Sync it to cloud storage after every eval so a dead laptop or a terminated rental machine loses nothing.
- When a clip or figure is good enough to post or put in the README, copy it into `assets/` with a descriptive name and add a line to `assets/INDEX.md`: what it shows, which run and checkpoint, and the command that made it.

### Training timelapse

A composite video shows the agent's "brain" changing as it learns, beside its score curve and the matching fight footage, with a whole training run compressed into about a minute. It is rendered from saved checkpoints after or alongside training, so it never slows training down. Command: `timelapse --run <name>`.

**Layout** (all panels synced to the same checkpoint):

| Panel | Position | Shows |
| --- | --- | --- |
| Brain | Left | Stylized network: edges lit by weight change, nodes lit by activity on the current frame |
| Learning curve | Right | Eval score vs env steps (kills per episode in v0 and v1, win rate in v2 and v3), drawn progressively, with a cursor marking the current checkpoint |
| Fight | Bottom | Footage from the current checkpoint, with optional attention heatmap, decision bars, and minimap inset |
| Tag | Corner | Step count and current score |

**Brain views.** Views 1 to 3 are in the main timelapse; 4 and 5 are rendered as separate clips.

1. **Weight-change map:** the network drawn as layers of nodes, sampling 16 to 32 real units per layer. Edge brightness is the size of that connection's weight change since the previous checkpoint, so active learning glows and settled parts fade.
2. **Live activity:** node brightness is that unit's activation on the frame currently shown in the fight panel, so the brain reacts in sync with the fight.
3. **Attention heatmap:** a soft overlay on the fight footage showing which pixels most influence the chosen action (gradient saliency).
4. **Decision panel:** top action probabilities plus a value meter (the agent's estimate that it is winning).
5. **Mind map:** each probe frame placed as a dot in 2D from the network's internal features (UMAP), colored by chosen action; clusters form as training goes on.

**Honesty rules.** Every visual is computed from the real network's values; nothing is animated for effect. Subsampled units are chosen once and kept fixed across checkpoints. The brain panel carries a small "simplified view, N of M units shown" note.

**What it requires**

- **Dense checkpoints** for any run meant for a timelapse: at least 100 across the run (every 100K steps is a good default), set in config separately from the regular checkpoint interval.
- **A fixed probe set:** a few hundred frames from held-out states covering distinct situations (no enemy visible, enemy far, enemy close, several enemies, low health, low ammo). Every checkpoint is shown the same probe set, so changes in the brain come from learning, not from different inputs.
- **A brain-data cache** per checkpoint (weight deltas, activations, saliency maps, UMAP coordinates) in `runs/<run>/timelapse/cache/`, so re-rendering with a new layout or style never recomputes the data.
- **Smooth motion:** interpolate edge colors and the curve cursor between checkpoints so the video does not stutter at each checkpoint boundary.

**Style.** Follows the figure style guide: same fonts and palette, dark theme by default for video, accent color reserved for the score line and the strongest edges. Export 1920x1080 at 60 fps, plus a 1080x1350 version for mobile feeds.

**Assets this produces** (saved to `runs/<run>/timelapse/`, promoted to `assets/timelapse/`):

- The full composite timelapse.
- A brain-only clip (views 1 and 2).
- An attention before and after: the same fight moment from the first and best checkpoints, heatmap on.
- A mind map evolution clip.

The timelapse is first built for the v0 run, then re-rendered for v1 (with the current curriculum scenario shown under the curve) and v3 (with Elo in place of win rate).

## Metrics

Every metric is written both to W&B and to a CSV in the run folder, keyed by environment step. Figures are built only from the CSVs.

**Training metrics** (`runs/<run>/metrics/train.csv`, from Sample Factory's training stats):

| Metric | What it tells you |
| --- | --- |
| `env_steps`, `wall_time_s`, `fps` | Throughput; the x-axis for everything |
| `ep_reward_mean` | Whether learning is happening at all |
| `kills_mean`, `deaths_mean` | Per episode, rolling over the last 100 training episodes |
| `damage_dealt_mean`, `damage_taken_mean` | Per episode, in health points |
| `policy_entropy` (per action head) | Collapse toward one action shows here first |
| `value_loss`, `policy_loss`, `kl` | Whether the update is healthy |
| `action_hist` (per action head) | Frequency of each choice in each head |

**Evaluation metrics** (`runs/<run>/metrics/eval.csv`, every checkpoint, 100 held-out-seed episodes, deterministic policy):

| Metric | Definition |
| --- | --- |
| `kills` | Mean kills per episode, with a 95% bootstrap interval |
| `deaths`, `kd_ratio` | Mean deaths per episode; kills divided by deaths (multi-life scenarios) |
| `survival_time` | Seconds alive per episode |
| `accuracy` | Hits divided by shots fired |
| `damage_dealt`, `damage_taken` | Health points per episode |
| `pickups` | Health, armor, weapon, and ammo pickups per episode |
| `coverage` | Distinct map cells visited per episode (64x64 map-unit grid) |
| `normalized_score` | Score divided by the scenario's reference (scripted baseline in v0; published agent in v1) |
| `flagged_episodes` | Count caught by the anomaly detectors |

**Multiplayer metrics** (v2 onward): frags per minute, deaths per minute, frag share in deathmatch, duel win rate with a 95% Wilson interval, and average engagement distance (from the labels buffer).

**Version-specific metrics** are listed in each version section.

## Graphs and figures

Figures should look like they came from a careful researcher's paper or a good engineering blog, not a default plotting library or a slide template. Every figure is generated by a script in `doomrl/plots/`, reads only from run CSVs and demo sidecars, and is regenerated with `figures --run <name>`. Never edit a figure by hand.

### Style guide

Implement this once as a shared matplotlib style file (`doomrl/plots/style.mplstyle`) plus a small helper module, and use it for every figure.

- **Typography:** one sans-serif family throughout (Inter or IBM Plex Sans, bundled in the repo so it renders the same everywhere). Title 15pt semibold, axis labels 11pt regular, tick labels 10pt in a mid gray.
- **Title states the finding, not the topic:** "Kills per episode pass the baseline by 3M steps", not "Kills vs steps". A smaller gray subtitle line gives the run name, scenario, and seed count.
- **Color:** one accent color for the thing that matters, grays for everything else. At most 4 hues when series must be told apart. No rainbow colormaps, no default matplotlib color cycle. Heatmaps use a single-hue sequential map, or a diverging map centered on 50% for win rates.
- **Chart furniture:** no top or right spines; light horizontal gridlines only; no boxes around legends; no drop shadows, gradients, 3D, or background images.
- **Direct labels over legends:** put the series name at the right end of its line, in the line's color. Use a legend only when lines overlap too much to label.
- **Noise handling:** draw the raw series as a thin, faint line and an exponential moving average on top in full color. Across seeds, draw the mean with a shaded band (min to max, or 95% interval; say which in the subtitle).
- **Annotations:** at most two per figure, small gray text with a thin leader line, marking real events (curriculum stage change, exploit discovered).
- **Axes:** x-axis in environment steps with short units (0, 10M, 20M); rates from 0% to 100%, with a faint reference line where a gate or baseline applies.
- **Map figures:** top-down maps drawn as thin light-gray wall lines on the background color, with paths and heatmaps in the accent hue; no in-game textures.
- **Sizes and exports:** a 16:9 version (1600x900) for posts, a square version (1200x1200) for mobile-first posts, and a paper-width version for reports. Export PNG at 2x scale plus a vector PDF. Produce a light and a dark theme from the same script.
- **Small multiples over cramming:** if a figure needs more than 4 lines, split it into a grid of small panels sharing axes.

### Figure set

| ID | Figure | Version | What it shows |
| --- | --- | --- | --- |
| F1 | Learning curve | v0 | Held-out kills per episode vs env steps, mean plus seed band, with the scripted baseline and the 1.5x gate as dashed reference lines |
| F2 | Kill and death exchange | v0 | Kills and deaths (or damage dealt and taken) per episode vs steps, two lines with direct labels |
| F3 | Action evolution | v0 | One small heatmap per action head: choices (rows) by checkpoint (columns), shade is usage share |
| F4 | Entropy and exploits | v0 | Policy entropy per head vs steps, with flagged-behavior events as small ticks on the x-axis |
| F5 | Eval scorecard | v0 | A clean table image for the best checkpoint: kills with interval, deaths, accuracy, survival time, coverage |
| F6 | Throughput | v0 | Frames per second vs rollout workers and envs per worker, for the Mac and rented machines |
| F7 | Scenario ladder | v1 | Heatmap: scenarios in curriculum order (rows) by checkpoint (columns), shade is normalized score; promotions as vertical rules |
| F8 | Path maps | v0, v1 | Top-down map with the agent's paths over many episodes, early checkpoint vs best, side by side |
| F9 | Duel matrix | v2 | Square heatmap of personality vs personality duel win rate, diverging map centered on 50%, values printed in cells |
| F10 | Personality fingerprints | v2 | Small multiples per personality: a movement heatmap on the same map plus action-usage bars, shared scales |
| F11 | Elo curve | v3 | Elo per snapshot vs steps, with bots and v2 personalities as labeled horizontal reference lines |
| F12 | Matchup map | v3 | Win-rate heatmap of later snapshots vs earlier ones |
| F13 | World model quality | Stretch B | Reconstruction loss and game-variable prediction error vs steps, two small panels |
| F14 | Dream strip | Stretch B | Two rows of frames over the same timesteps: real game on top, imagined on the bottom |

Every figure is saved to `runs/<run>/figures/<ID>_<name>_{16x9,square,paper}_{light,dark}.{png,pdf}`; promoted figures are copied to `assets/figures/`.

## v0: It learns to fight

v0 proves the whole pipeline: an agent learns to fight in Defend the Center (stand in the middle of a circular room and kill monsters closing in from every side) from pixels, and every asset in this spec gets produced for that run.

**Scope**

- Defend the Center, with a training seed range and a reserved held-out seed range.
- Base reward, factored action space (turn and attack heads only, since the scenario allows nothing else), pixel input, recurrent policy, 3 training seeds.
- All shared infrastructure: config system, Sample Factory wrapper, eval watcher, checkpoint retention, demo recording and rendering, anomaly flags, montage, figure scripts, timelapse.

**Build list**

1. Install ViZDoom and Sample Factory on Python 3.12; run one ViZDoom example and Sample Factory's Doom benchmark env to confirm both work on the Mac.
2. `make_env`: scenario config, 128x72 RGB, frame skip 4, action space, game variables, reward wrapper, and seeding with the train and held-out split.
3. Sanity checks before any training: a random agent runs 10 episodes without errors; a scripted spin-and-shoot agent gets more kills than random; game variables change as expected (kill count rises on a kill, health drops when hit).
4. Measure the scripted baseline over 100 held-out episodes; this sets the v0 gate.
5. Demo recording plus sidecar JSON, and the renderer that turns demos into 1280x720 clips with HUD and minimap. Verify by recording and rendering the random agent at step 0.
6. Throughput test: frames per second across rollout worker and envs-per-worker settings (feeds F6); choose the config.
7. Training through Sample Factory with checkpoint retention, and the eval watcher evaluating and recording each new checkpoint.
8. `evaluate`, `record`, `montage`, `figures`, and `timelapse` commands.
9. Train 3 seeds to the gate, with dense checkpoints on at least one seed for the timelapse.

**Metrics and figures:** all training and eval metrics; F1 through F6, plus F8.

**Assets this version must produce**

- Step-0 clip of the untrained agent.
- Progress montage at 4 checkpoints (untrained, early learning, mid, best).
- Best-checkpoint clips, with and without overlays and minimap.
- Every flagged clip, reviewed, with real oddities written into `WEIRD.md`.
- F1 to F6 and F8 in all sizes and both themes.
- The training timelapse for the dense-checkpoint seed.

**Done when**

- [ ] Best checkpoint averages at least 1.5x the scripted baseline's kills over 100 held-out-seed episodes, in at least 2 of 3 seeds.
- [ ] One command each regenerates the montage, the timelapse, and all v0 figures from the run folder.
- [ ] A fresh clone can reproduce a run from its config with no files beyond the repo and its dependencies.

**Not in v0:** other scenarios, movement actions, multiplayer.

## v1: It clears the arenas

v1 turns the single-scenario agent into one policy that moves, aims, and fights through a ladder of harder scenarios, ending in Sample Factory's Battle maps, and measures itself against a published agent.

**Scope**

- One policy with the full single-player action space (move, strafe, turn, attack).
- A curriculum ladder, in order: Defend the Center, Defend the Line, Deadly Corridor, Deathmatch (monsters), Battle, Battle2.
- A benchmark: Sample Factory's pretrained Battle agent, run through our own eval script on our held-out seeds.
- Optional bonus rung: reach the exit of Freedoom's first map (Episode 1, Map 1), which ViZDoom provides as a Gymnasium environment.

**Build list**

1. Shared action space across scenarios: every scenario uses the full factored heads, with choices a scenario does not allow masked out.
2. Reference scores per scenario: the scripted baseline for the built-in scenarios, and the pretrained Sample Factory agents for Battle and Battle2, all measured with our eval script.
3. Curriculum manager: tracks a frontier scenario; each env samples the frontier 70% of the time and earlier scenarios 30%, so old skills are kept.
4. Promotion rule: when held-out normalized score at the frontier reaches the scenario's threshold over 50 episodes, the frontier moves up. Log every promotion with its step count.
5. Optional auxiliary head that predicts "enemy visible" from the labels buffer, the trick strong competition entries used. Run it as an ablation, with and without.
6. Initialize from the best v0 checkpoint and switch on the trajectory logger (see Stretch B) so world model data accumulates from here on.

**Metrics added:** frontier scenario vs steps; normalized score per scenario (training and held-out); Battle and Battle2 score as a fraction of the published agent's; coverage per scenario.

**Figures:** F7 (scenario ladder) and F8 (path maps per scenario).

**Assets this version must produce**

- The first successful episode on each new scenario, captured automatically at each promotion.
- A long, uncut Battle episode from the best checkpoint, plus a cut-down highlight version.
- Side-by-side: our agent and the published agent on the same Battle seed.
- Path maps for every scenario, early vs best.
- If attempted, the first run that reaches the exit of Freedoom's first map.

**Done when**

- [ ] Battle score is at least 75% of the published pretrained agent's score on the same held-out seeds.
- [ ] Every earlier scenario still meets its promotion threshold (no forgetting).
- [ ] F7, F8, and the Battle videos regenerate from the run folder with one command each.

**Not in v1:** multiplayer, bots.

## v2: Fighting personalities

v2 moves to multiplayer deathmatch, trains four agents with different reward personalities, and runs a duel tournament between them. The multiplayer and tournament infrastructure built here is reused by v3 and Stretch A.

**Scope**

- Multiplayer deathmatch on ViZDoom's `cig.wad` map01 against bots, then 1v1 duels on the same map and on `multi_duel.wad`.
- Four personalities, each fine-tuned from the best v1 checkpoint.
- Round-robin duel tournament with a win-rate matrix and style metrics.

**Build list**

1. Bots config with fixed skill settings, saved with every run; a default lineup used for every "vs bots" number.
2. Multiplayer training through Sample Factory's Doom multiplayer environments, with frame skip 2 and our multiplayer reward.
3. Personality reward terms, each added on top of the base reward with a weight in config. Enemy distance comes from the labels buffer plus positions, during training only.

| Personality | Extra reward term |
| --- | --- |
| Rusher | Bonus for closing distance to visible enemies and for close-range frags |
| Sniper | Bonus for frags from long range; penalty for time spent in close range |
| Survivor | Bonus for time alive and for health kept; extra penalty on deaths |
| Scavenger | Bonus for pickups and for map coverage |

4. Train each personality against the bot lineup first, so every one stays competent, then against a pool of the other three.
5. Tournament runner: hosts a match as several ViZDoom processes on local ports, loads any checkpoint or bot into each slot, randomizes spawns, and records a demo per match. Every pair plays 50 duels from held-out seeds.
6. Clip overlay that labels each player with its personality name in a small corner tag; renders from either player's view or from the top-down map.

**Metrics added:** duel win rate per pair; per personality: frags and deaths per minute, average engagement distance, distance moved per minute, coverage, pickups, accuracy. These confirm the personalities really play differently.

**Figures:** F9 (duel matrix) and F10 (personality fingerprints).

**Assets this version must produce**

- The best duel for every pairing, from both players' views, with personality labels.
- A tournament bracket video built from those duels.
- All four personalities started from the same spawn and seed, shown side by side, so the difference in style is obvious.
- Top-down movement heatmaps per personality, animated over a match.
- Every flagged exploit clip, since reward variants are where reward hacking shows up most.

**Done when**

- [ ] Every personality finishes first in frags against the default bot lineup in at least 60% of held-out matches.
- [ ] Style metrics differ clearly between personalities in F10.
- [ ] Tournament results and the bracket video regenerate from the run folders.

**Not in v2:** training against its own past selves (v3), human players (Stretch A).

## v3: Self-play league

v3 trains duelists against each other and against their own past snapshots instead of bots, using a population of policies, and measures progress with Elo.

**Scope**

- Duel self-play on the v2 maps, using Sample Factory's self-play setup with population-based training (PBT): 8 policies that train against each other and periodically copy the strongest member's weights and settings.
- A snapshot pool of past policies, so the population keeps beating old strategies.
- Elo ratings for every snapshot, plus fixed anchors: the default bot lineup and the four v2 personalities.

**Build list**

1. Self-play config built from Sample Factory's duel self-play setup, initialized from the best v2 checkpoints so the league starts competent.
2. Snapshot pool: save each population member at a fixed interval and include pool members as opponents alongside the live population.
3. Elo service: after each snapshot is added, the tournament runner plays it against a fixed sample of pool members and anchors on held-out seeds, then updates ratings.
4. Strategy drift tracking: distance between consecutive snapshots' action usage and movement heatmaps, to spot when a new style appears.

**Metrics added:** Elo per snapshot; win rate of the latest snapshot vs each anchor; PBT events (which member was copied and when); strategy drift.

**Figures:** F11 (Elo curve with anchors) and F12 (matchup map of later vs earlier snapshots).

**Assets this version must produce**

- Duels of the latest snapshot vs early snapshots at regular intervals, to show the gap growing.
- Clips whenever strategy drift spikes, since that is when a new style has appeared.
- A final showdown: the league champion vs each v2 personality.

**Done when**

- [ ] Elo rises across snapshots, and the latest champion beats every v2 personality in at least 60% of held-out duels.
- [ ] The champion still finishes first in frags against the default bot lineup (no collapse into self-play-only habits).
- [ ] F11, F12, and the showdown clips regenerate from the run folder.

**Not in v3:** team modes, maps beyond the v2 set.

## Stretch A: Play against it

Stretch A lets a person join a live Doom match against any trained agent. It reuses the v2 tournament runner, with one slot filled by a human playing in a ViZDoom window with keyboard and mouse. Because the game content is free, anyone can clone the repo and play.

**Scope**

- `play` command: hosts a match, puts the human in one slot and any checkpoint (or several, plus bots) in the others, and opens the human's game window.
- Modes: 1v1 duel, or free-for-all deathmatch with agents and bots.
- Opponent menu: pick a version, a personality, or "matched", which picks the snapshot whose Elo is closest to the player's running rating.
- Fairness settings: an optional reaction delay for the agent (in tics) and a cap on its turn speed, so it cannot snap-aim faster than a person could.

**Build list**

1. Human slot: a ViZDoom instance in a human-input mode with the window visible at full resolution, joining the hosted match.
2. Agent slots run at Doom's real-time rate (35 tics per second); check that the policy never falls behind on the Mac's CPU.
3. Opponent menu and the Elo-matched picker, reusing the v3 Elo table.
4. Every human match is recorded as a demo file with the result and stats in `runs/human/`, after a startup prompt asking the player whether recording is fine.
5. Play over a local network as well as on one machine; document the ports.

**Metrics added:** human win rate per opponent; human and agent frags per minute; the player's running Elo.

**Assets this version must produce**

- Human vs agent clips from both players' views, especially matches the agent wins with a learned trick.
- A short screen recording of the play menu and a match, for the README and posts.

**Done when**

- [ ] A person can play a full duel or deathmatch against any checkpoint with no slowdown.
- [ ] Matches are recorded and land in `runs/human/` with their stats.
- [ ] Setup works from the README alone on a fresh machine.

## Stretch B: World model dreams

Stretch B trains a world model of Doom on recorded play and shows real fights next to the model's imagined versions. World Models and GameNGen showed this works for Doom; ours is a much smaller model aimed at a striking visual, trained offline and separately from the policies.

**Scope**

- A trajectory dataset of frames, actions, and game variables from v1 to v3 agents, bots, and human matches.
- A DreamerV3-style world model (recurrent latent state, image decoder, heads predicting health, ammo, and kills), trained offline on that dataset.
- Dream rollouts: given a few real frames and the real action sequence, the model imagines what happens next.

**Build list**

1. Trajectory logger, switched on from v1 onward: saves frames losslessly with actions and game variables, in chunks, to `data/trajectories/`. Mix scenarios, agents, and opponents so the model sees varied play.
2. World model training script at 64x64 first; move to 128x72 only if dreams are too blurry to read.
3. Dream renderer: condition on 8 real frames, then roll out 70 imagined frames (2 seconds of game time) using the real actions; output real and imagined side by side, same size and frame rate.
4. Open-loop error measurement: frame error vs rollout length, to show how far ahead dreams stay faithful.
5. Optional, following World Models: let a policy act inside the world model with no game engine at all, and render the fully dreamed fight.

**Metrics added:** reconstruction loss, game-variable prediction error, frame error vs rollout horizon.

**Figures:** F13 (world model quality) and F14 (dream strip).

**Assets this version must produce**

- Real vs dream side-by-side videos, rendered from several world model checkpoints so a "dreams getting sharper" montage can be cut.
- The most interesting failure dreams (monsters melting into walls, rooms rearranging, ammo counts drifting), saved and labeled.
- A fully dreamed fight if step 5 works.

**Done when**

- [ ] Real vs dream videos render from one command for any world model checkpoint.
- [ ] Dreams stay recognizable (room layout, enemies, HUD numbers) for at least the first 35 imagined frames.
- [ ] F13 and F14 regenerate from the run folder.

## Weird behaviors log

`WEIRD.md` records every strange, exploitative, or funny behavior with a clip, from the first v0 run onward. It is the source for the reward hacking post and for deciding which reward changes were real fixes.

Each entry is one row, newest first:

| Date | Run | Checkpoint | Scenario and seed | Category | What happened | Suspected cause | Clip | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-20 (example) | v0\_defend\_center\_s1 | 3.5M | Defend the Center, 900007 | Spinning | Turns in one direction nonstop and fires on every step; still scores well | No penalty on wasted shots; monsters approach from all sides | `runs/.../flagged/ep07.mp4` | Kept as a feature; noted for post |

**Categories:** spinning, stuck on wall, camping, wasted fire, pacifist, suicide, spawn exploit, glitch, cowardice.

**Outcome** is one of: fixed (with the config change that fixed it), kept (legitimate strategy), or feature (worth posting about). Anything marked feature gets its clip copied into `assets/weird/`.
