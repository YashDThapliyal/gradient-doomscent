# Assets index

Curated clips and figures promoted from `runs/`. One line each: what it shows, run and checkpoint, and the command that made it. Videos are not committed to git (CLAUDE.md); they live here locally and in cloud sync.

| File | What it shows | Run and checkpoint | Command |
| --- | --- | --- | --- |
| `clips/v0_before_untrained_0kills.mp4` | Untrained agent (step 0) on held-out seed 900000000: never fires, 0 kills | v0_defend_center_s1, step 0 | `doomscent eval-watcher` (clips/train), copied |
| `clips/v0_after_15M_steps_10kills.mp4` | Same game after 15M training steps: waits for targets, 10 kills from 26 bullets | v0_defend_center_s1, step 15M (best at the time) | `doomscent eval-watcher` (clips/best), copied |
| `weird/v0_s1_1M_rediscovers_spin_and_shoot.mp4` | At 1M steps the agent spins at full speed and fires nonstop, reinventing the scripted baseline before learning to aim | v0_defend_center_s1, step 1M, seed 900000000 | eval watcher flagged clip (spinning detector), copied |
| `figures/F1_learning_curve_{16x9,square,paper}_{light,dark}.{png,pdf}` | F1: Mean held-out kills per episode pass the 1.5x baseline gate by 6M steps | v0_defend_center_s1 (F1, F2 use all 3 seeds) | `uv run doomscent figures --run v0_defend_center_s1` |
| `figures/F2_kill_death_exchange_{16x9,square,paper}_{light,dark}.{png,pdf}` | F2: Damage taken still exceeds damage dealt at 20M steps | v0_defend_center_s1 (F1, F2 use all 3 seeds) | `uv run doomscent figures --run v0_defend_center_s1` |
| `figures/F3_action_evolution_{16x9,square,paper}_{light,dark}.{png,pdf}` | F3: By 20M steps the top choices are turn 'none' 36%, attack 'on' 85% | v0_defend_center_s1 (F1, F2 use all 3 seeds) | `uv run doomscent figures --run v0_defend_center_s1` |
| `figures/F4_entropy_and_exploits_{16x9,square,paper}_{light,dark}.{png,pdf}` | F4: Turn entropy falls 71%, from 1.95 to 0.57 nats | v0_defend_center_s1 (F1, F2 use all 3 seeds) | `uv run doomscent figures --run v0_defend_center_s1` |
| `figures/F5_eval_scorecard_{16x9,square,paper}_{light,dark}.{png,pdf}` | F5: The best checkpoint averages 10.5 kills per episode on fresh held-out games, 2.4x the scripted baseline | v0_defend_center_s1 (F1, F2 use all 3 seeds) | `uv run doomscent figures --run v0_defend_center_s1` |
| `figures/F6_throughput_{16x9,square,paper}_{light,dark}.{png,pdf}` | F6: One run tops out near 2.9K frames per second whatever the worker count; 3 runs at once reach 4.7K | v0_defend_center_s1 (F1, F2 use all 3 seeds) | `uv run doomscent figures --run v0_defend_center_s1` |
| `figures/F8_path_maps_{16x9,square,paper}_{light,dark}.{png,pdf}` | F8: The agent stays put (knocked back at most 126 units); at its best it spends 19% of steps facing its 3 favourite directions, vs 14% early | v0_defend_center_s1 (F1, F2 use all 3 seeds) | `uv run doomscent figures --run v0_defend_center_s1` |
