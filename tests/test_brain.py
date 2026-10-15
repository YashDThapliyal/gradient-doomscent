import gymnasium as gym
import numpy as np
import torch
from sample_factory.model.actor_critic import create_actor_critic
from sf_examples.vizdoom.doom.doom_params import default_doom_cfg

from doomscent.timelapse.brain import (
    edge_change,
    edge_strength,
    forward_step,
    layer_sizes,
    raw_blocks,
    saliency,
    sample_units,
)


def _model():
    cfg = default_doom_cfg()
    cfg.use_rnn, cfg.rnn_type = True, "gru"
    obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 255, (3, 72, 128), dtype=np.uint8)})
    act_space = gym.spaces.Tuple([gym.spaces.Discrete(7), gym.spaces.Discrete(2)])
    torch.manual_seed(0)
    model = create_actor_critic(cfg, obs_space, act_space)
    model.eval()
    return model


def test_functional_forward_matches_model():
    model = _model()
    obs = np.random.default_rng(0).integers(0, 255, (2, 3, 72, 128), dtype=np.uint8)
    h = torch.randn(2, 512) * 0.1
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

    with torch.no_grad():
        ref = model(prepare_and_normalize_obs(model, {"obs": torch.from_numpy(obs)}), h)
    ours = forward_step(model, obs, h)
    assert torch.allclose(ours.logits, ref["action_logits"], atol=1e-4)
    assert torch.allclose(ours.value, ref["values"].reshape(-1), atol=1e-4)
    assert torch.allclose(ours.h, ref["new_rnn_states"], atol=1e-4)
    assert ours.acts["conv3"].shape == (2, 128)


def test_units_fixed_and_blocks_shapes():
    model = _model()
    sizes = layer_sizes(model)
    assert sizes == {
        "input": 3,
        "conv1": 32,
        "conv2": 64,
        "conv3": 128,
        "fc": 512,
        "gru": 512,
        "out": 10,
    }
    u1, u2 = sample_units(sizes, 24, 0), sample_units(sizes, 24, 0)
    assert u1 == u2 and len(u1["fc"]) == 24 and u1["input"] == [0, 1, 2]
    blocks = raw_blocks(model, u1)
    assert blocks["conv3->fc"].shape == (24, 24, 18)
    assert blocks["fc->gru"].shape == (24, 24, 3)
    assert blocks["gru->out"].shape == (10, 24, 1)
    s = edge_strength(blocks)
    assert s["conv1->conv2"].shape == (24, 24)
    assert all(np.all(v == 0) for v in edge_change(blocks, blocks).values())


def test_saliency_shape_and_nonnegative():
    model = _model()
    obs = np.random.default_rng(1).integers(0, 255, (3, 72, 128), dtype=np.uint8)
    sal = saliency(model, obs, torch.zeros(1, 512), [7, 2])
    assert sal.shape == (72, 128) and (sal >= 0).all() and sal.max() > 0


def test_timelapse_keeps_cached_checkpoints_whose_files_are_gone(tmp_path, monkeypatch) -> None:
    from doomscent.checkpoints import Checkpoint
    from doomscent.timelapse import cache

    (tmp_path / "cache").mkdir()
    for step in (0, 500_000, 1_000_000):
        (tmp_path / "cache" / f"step_{step:07d}.npz").touch()
    (tmp_path / "cache" / "step_1500000.tmp.npz").touch()  # half-written, ignored
    on_disk = Checkpoint(tmp_path / "c.pth", 1, 1_000_000, 1_000_000, True)
    monkeypatch.setattr(cache, "tl_dir", lambda paths: tmp_path)
    monkeypatch.setattr(cache, "list_checkpoints", lambda paths: [on_disk])
    cks = cache.timelapse_checkpoints(None)
    assert [c.step for c in cks] == [0, 500_000, 1_000_000]
    assert cks[-1] is on_disk  # a real checkpoint file wins over its cache entry
