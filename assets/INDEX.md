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
| `clips/v0_s1_progress_montage.mp4` | 2x2 montage, same held-out game at untrained (0), early learning (2M), mid (11M), best (20M) | v0_defend_center_s1, steps 0/2M/11M/20M, seed 900000000 | `uv run doomscent montage --run v0_defend_center_s1 --ckpts auto` |
| `clips/v0_s1_best_{plain,minimap,overlay}.mp4` | Best checkpoint, 11 kills: plain, with minimap inset, and with action-probability and value panel | v0_defend_center_s1, step 20M, seed 900000000 | eval watcher (clips/best) |
| `clips/v0_s1_highlights_best.mp4` | Auto-cut multi-kill streaks from the best checkpoint's 100 held-out games | v0_defend_center_s1, step 20M | `uv run doomscent highlights --run v0_defend_center_s1 --ckpt best` |
| `clips/v0_s1_sidebyside_untrained_vs_best.mp4` | Untrained vs best on the same held-out game, side by side | v0_defend_center_s1, steps 0 and 20M, seed 900000000 | `uv run doomscent sidebyside --run v0_defend_center_s1 --ckpts 0,best` |
| `timelapse/timelapse_16x9.mp4` | Training timelapse, 1920x1080 60 fps: brain (weight change, activity), learning curve, fight with attention, decision bars | v0_defend_center_s1, 201 checkpoints 0 to 20M | `uv run doomscent timelapse --run v0_defend_center_s1` |
| `timelapse/timelapse_mobile.mp4` | Same timelapse at 1080x1350 for mobile feeds | v0_defend_center_s1 | same command |
| `timelapse/brain_only.mp4` | Brain views 1 and 2 full frame | v0_defend_center_s1 | same command |
| `timelapse/attention_before_after.mp4` | Same fight moment, 100K vs best (20M), gradient saliency on one shared scale: the early network barely attends, the trained one locks onto the monsters | v0_defend_center_s1 | same command |
| `timelapse/mind_map.mp4` | UMAP of the memory state on 300 fixed probe frames, colored by turn direction, across checkpoints | v0_defend_center_s1 | same command |
| `timelapse/s2/` | All five timelapse videos (16x9, mobile, brain only, attention before/after, mind map) for seed 2 | v0_defend_center_s2, 41 checkpoints 0 to 20M (every 500K) | `uv run doomscent timelapse --run v0_defend_center_s2` |
| `timelapse/s3/` | All five timelapse videos (16x9, mobile, brain only, attention before/after, mind map) for seed 3 | v0_defend_center_s3, 41 checkpoints 0 to 20M (every 500K) | `uv run doomscent timelapse --run v0_defend_center_s3` |
