import pytest

from doomscent.env import check_demo_path, make_env
from doomscent.recording import ClipName, read_sidecar, record_episode
from doomscent.render import render_demo
from doomscent.rollout import RandomPolicy
from doomscent.video import probe_video


def test_clip_name_matches_spec_format_and_roundtrips():
    n = ClipName("v0_defend_center_s1", 2500000, "dtc", 900007, 2, 14)
    assert n.stem == "v0_defend_center_s1__step_2500000__dtc__seed_900007__ep02__kills14"
    assert ClipName.parse(n.stem) == n
    assert ClipName("r", 0, "dtc", 1, 0, 0).stem.startswith("r__step_0000000__")


@pytest.mark.parametrize("bad", ["/tmp/-x/a.lmp", "/tmp/a b.lmp", "/tmp/" + "a" * 120 + ".lmp"])
def test_demo_paths_that_hang_vizdoom_are_rejected(bad):
    with pytest.raises(ValueError):
        check_demo_path(bad)


@pytest.mark.game
def test_record_and_replay_match(v0_cfg, tmp_path):
    out = tmp_path / "clips"
    env = make_env(v0_cfg, split="heldout")
    res, demo, sidecar = record_episode(env, RandomPolicy(env, 0), 900000003, out, "t", 0, 0)
    env.close()
    data = read_sidecar(sidecar)
    assert data["n_steps"] == len(data["steps"]) == res.stats["episode_steps"]
    rr = render_demo(demo, data, ["plain"], record_cfg={"width": 640, "height": 360})
    assert rr.matches_sidecar
    info = probe_video(rr.outputs["plain"])
    assert info["frames"] == rr.replay_tics > 0
    assert (info["width"], info["height"], info["fps"]) == (640, 360, 35.0)
