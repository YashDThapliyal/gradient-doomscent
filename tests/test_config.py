import pytest
from omegaconf import OmegaConf

from doomscent.config import load_config, save_config, validate_seed_split


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_overlapping_seed_ranges_rejected():
    cfg = OmegaConf.create(
        {"seeds": {"train": {"start": 0, "stop": 100}, "heldout": {"start": 50, "stop": 200}}}
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_split(cfg)


def test_disjoint_seed_ranges_accepted():
    cfg = OmegaConf.create(
        {"seeds": {"train": {"start": 0, "stop": 100}, "heldout": {"start": 100, "stop": 200}}}
    )
    validate_seed_split(cfg)


def test_base_inheritance_and_overrides(tmp_path):
    _write(tmp_path, "base.yaml", "a: 1\nb: {c: 2}\n")
    child = _write(tmp_path, "child.yaml", "base: base.yaml\nb: {d: 3}\n")
    cfg = load_config(child, ["a=5"])
    assert cfg.a == 5 and cfg.b.c == 2 and cfg.b.d == 3


def test_save_roundtrip(tmp_path):
    cfg = OmegaConf.create({"x": 1, "y": "${x}"})
    out = tmp_path / "run" / "config.yaml"
    save_config(cfg, out)
    assert OmegaConf.load(out).y == 1


def test_routine_clips_switch() -> None:
    from doomscent.eval_watcher import routine_clips

    assert routine_clips(OmegaConf.create({"eval": {}}))  # older run configs: every clip
    assert not routine_clips(OmegaConf.create({"eval": {"render_routine_clips": False}}))
    assert not routine_clips(load_config("configs/v0_defend_center.yaml"))
