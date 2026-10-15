# Assets index

Curated clips and figures promoted from `runs/`. One line each: what it shows, run and checkpoint, and the command that made it. Videos are not committed to git (CLAUDE.md); they live here locally and in cloud sync.

| File | What it shows | Run and checkpoint | Command |
| --- | --- | --- | --- |
| `clips/v0_before_untrained_0kills.mp4` | Untrained agent (step 0) on held-out seed 900000000: never fires, 0 kills | v0_defend_center_s1, step 0 | `doomscent eval-watcher` (clips/train), copied |
| `clips/v0_after_15M_steps_10kills.mp4` | Same game after 15M training steps: waits for targets, 10 kills from 26 bullets | v0_defend_center_s1, step 15M (best at the time) | `doomscent eval-watcher` (clips/best), copied |
| `weird/v0_s1_1M_rediscovers_spin_and_shoot.mp4` | At 1M steps the agent spins at full speed and fires nonstop, reinventing the scripted baseline before learning to aim | v0_defend_center_s1, step 1M, seed 900000000 | eval watcher flagged clip (spinning detector), copied |
