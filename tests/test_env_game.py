"""Real ViZDoom episodes: game variables behave, random agent runs, scripted beats random."""

import numpy as np
import pytest

from doomscent.env import make_env, scenario_files
from doomscent.rollout import RandomPolicy, SpinAndShootPolicy, run_episode

pytestmark = pytest.mark.game


@pytest.fixture(scope="module")
def env(v0_cfg):
    e = make_env(v0_cfg, split="train", rng_seed=0)
    yield e
    e.close()


def test_observation_and_spaces(env):
    obs, info = env.reset(options={"episode_seed": 5})
    assert obs.shape == (3, 72, 128) and obs.dtype == np.uint8
    assert obs.std() > 0, "screen should not be blank"
    assert [s.n for s in env.action_space.spaces] == [7, 2]
    assert "KILLCOUNT" in info["game_vars"]


def test_scenario_files_exist(v0_cfg):
    files = scenario_files(v0_cfg.env.scenario)
    assert [f.suffix for f in files] == [".cfg", ".wad"]
    assert all(f.exists() for f in files)


def test_heldout_split_rejects_train_seed(v0_cfg):
    e = make_env(v0_cfg, split="heldout")
    with pytest.raises(ValueError):
        e.reset(options={"episode_seed": 5})
    e.close()


def test_same_seed_replays_identically(env):
    policy = SpinAndShootPolicy(env, turn_choice=5)
    a = run_episode(env, policy, 11)
    b = run_episode(env, policy, 11)
    assert a.stats == b.stats


def test_kill_count_rises_on_a_kill(env):
    policy = SpinAndShootPolicy(env, turn_choice=5)
    res = run_episode(env, policy, 3)
    prev = res.start_vars["KILLCOUNT"]
    kills_seen = 0.0
    for s in res.steps:
        k = s["vars"]["KILLCOUNT"]
        assert k >= prev, "kill count never decreases"
        if k > prev:
            kills_seen += k - prev
            assert s["reward"] >= 1.0, "a kill step carries the kill reward"
        prev = k
    assert kills_seen > 0, "spin-and-shoot should kill at least once on seed 3"
    assert res.stats["kills"] == kills_seen


def test_health_drops_when_hit(env):
    res = run_episode(env, RandomPolicy(env, seed=0), 7)
    prev = res.start_vars
    hit_steps = 0
    for s in res.steps:
        cur = s["vars"]
        if cur["DAMAGE_TAKEN"] > prev["DAMAGE_TAKEN"]:
            hit_steps += 1
            assert cur["HEALTH"] < prev["HEALTH"], "health drops on a damage step"
            assert s["reward"] < 0 or cur["KILLCOUNT"] > prev["KILLCOUNT"]
        prev = cur
    assert hit_steps > 0
    assert res.died and res.stats["deaths"] == 1.0


def test_random_agent_completes_10_episodes(env):
    policy = RandomPolicy(env, seed=1)
    results = [run_episode(env, policy, 100 + i, keep_trace=False) for i in range(10)]
    assert len(results) == 10
    assert all(r.stats["episode_steps"] > 0 for r in results)


def test_spin_and_shoot_beats_random(env):
    seeds = range(200, 220)
    rnd = RandomPolicy(env, seed=2)
    spin = SpinAndShootPolicy(env, turn_choice=5)
    rand_kills = np.mean([run_episode(env, rnd, s, keep_trace=False).stats["kills"] for s in seeds])
    spin_kills = np.mean(
        [run_episode(env, spin, s, keep_trace=False).stats["kills"] for s in seeds]
    )
    print(f"random {rand_kills:.2f} vs spin-and-shoot {spin_kills:.2f} kills/episode")
    assert spin_kills > rand_kills
