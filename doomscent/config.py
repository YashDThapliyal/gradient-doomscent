"""Experiment config loading: one OmegaConf YAML per experiment, with CLI overrides."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config, merge dotlist overrides, and validate the seed split.

    A config may name a parent with a top-level `base:` key (path relative to its own
    directory); the parent is loaded first and the child merged on top.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    cfg = OmegaConf.load(path)
    if "base" in cfg and cfg.base:
        parent = load_config(path.parent / str(cfg.base))
        cfg = OmegaConf.merge(parent, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(cfg, DictConfig)
    validate_seed_split(cfg)
    return cfg


def validate_seed_split(cfg: DictConfig) -> None:
    """Fail fast if the training and held-out seed ranges overlap."""
    if "seeds" not in cfg:
        return
    tr = cfg.seeds.train
    ho = cfg.seeds.heldout
    if tr.start >= tr.stop or ho.start >= ho.stop:
        raise ValueError("seed ranges must satisfy start < stop")
    if tr.start < ho.stop and ho.start < tr.stop:
        raise ValueError(
            f"train seeds [{tr.start}, {tr.stop}) overlap held-out seeds [{ho.start}, {ho.stop})"
        )


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Write the fully resolved config next to a run's outputs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
