"""Brain-data cache: everything the timelapse draws, computed once per checkpoint.

runs/<run>/timelapse/
  units.json            sampled unit indices per layer (fixed for the whole run)
  probe_set.npz         the fixed probe frames, their context, and categories
  cache/step_N.npz      raw sampled weights, edge strength and change, probe activations
                        and features, and the fight episode's per-step activations,
                        probabilities, value, saliency, and kills
  cache/fight/          the fight episode's demo and sidecar per checkpoint
  cache/footage/        the fight episode replayed at 1024x576, HUD off (matches saliency)
  umap.npz              2D mind-map coordinates of probe features per checkpoint

Re-rendering with a new layout or style reads only this folder.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch

from doomscent.checkpoints import Checkpoint, list_checkpoints
from doomscent.runs import RunPaths
from doomscent.timelapse.brain import (
    LAYERS,
    edge_change,
    edge_strength,
    forward_step,
    layer_sizes,
    raw_blocks,
    saliency,
    sample_units,
)

FOOTAGE_RES = (1024, 576)
TL_SUBDIR = {"name": "timelapse"}  # `timelapse --subdir` renders a test into another folder


def tl_dir(paths: RunPaths) -> Path:
    return paths.root / TL_SUBDIR["name"]


def cache_path(paths: RunPaths, ck: Checkpoint) -> Path:
    return tl_dir(paths) / "cache" / f"{ck.label}.npz"


def ensure_units(paths: RunPaths) -> dict[str, list[int]]:
    f = tl_dir(paths) / "units.json"
    if f.exists():
        return json.loads(f.read_text())["units"]
    from doomscent.policy import build_model

    cfg = paths.load_config()
    model, _, _ = build_model(paths)
    sizes = layer_sizes(model)
    units = sample_units(sizes, int(cfg.timelapse.units_per_layer), int(cfg.timelapse.units_seed))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"sizes": sizes, "units": units}, indent=2))
    return units


def ensure_probe(paths: RunPaths) -> dict[str, np.ndarray]:
    """Build the probe set once, from held-out eval demos of early, middle, and late checkpoints."""
    f = tl_dir(paths) / "probe_set.npz"
    if f.exists():
        return dict(np.load(f))
    from omegaconf import OmegaConf

    from doomscent.timelapse.probe import build_probe_set

    cfg = paths.load_config()
    labels = sorted(p.stem for p in (paths.metrics / "eval_episodes").glob("step_*.csv"))
    if not labels:
        raise FileNotFoundError("the probe set needs at least one evaluated checkpoint")
    picks = sorted({labels[min(1, len(labels) - 1)], labels[len(labels) // 2], labels[-1]})
    demos = []
    for lbl in picks:
        demos += sorted((paths.clips / "eval" / lbl).glob("*.lmp"))[:15]
    spec = json.loads((paths.meta / "action_space.json").read_text())
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    tl_cfg = OmegaConf.to_container(cfg.timelapse, resolve=True)
    probe = build_probe_set(
        demos,
        env_cfg,
        tl_cfg,
        spec["buttons"],
        int(cfg.env.frame_skip),
        int(cfg.seed),  # type: ignore[arg-type]
    )
    np.savez_compressed(f, **probe)
    return probe


class BrainPolicy:
    """Deterministic policy driven by the functional forward, logging brain data per step."""

    def __init__(self, model, units: dict[str, list[int]], head_sizes: list[int], heads: list[str]):
        self.model, self.units, self.head_sizes, self.heads = model, units, head_sizes, heads
        self.reset()

    def reset(self) -> None:
        self.h = torch.zeros(1, self.model.core.core.hidden_size)
        self.log: dict[str, list] = {"acts": [], "probs": [], "value": [], "sal": []}

    def act(self, obs: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        sal = saliency(self.model, obs, self.h, self.head_sizes)
        out = forward_step(self.model, obs, self.h)
        actions, probs, start = [], {}, 0
        for name, n in zip(self.heads, self.head_sizes, strict=True):
            p = torch.softmax(out.logits[0, start : start + n], dim=0)
            actions.append(int(p.argmax()))
            probs[name] = [round(float(x), 4) for x in p]
            start += n
        v = out.value.reshape(1).clone()
        if getattr(self.model, "returns_normalizer", None) is not None:
            self.model.returns_normalizer(v, denormalize=True)
        self.log["acts"].append(
            np.concatenate([out.acts[lyr][0, self.units[lyr]].numpy() for lyr in LAYERS])
        )
        self.log["probs"].append(np.concatenate([probs[h] for h in self.heads]))
        self.log["value"].append(float(v[0]))
        self.log["sal"].append(sal)
        self.h = out.h
        return actions, {"value": round(float(v[0]), 4), "probs": probs}


def probe_forward(model, probe: dict[str, np.ndarray], units: dict[str, list[int]]) -> dict:
    """Run every probe frame after its context frames; return sampled acts and features."""
    obs, ctx = probe["obs"], probe["context"]
    h = torch.zeros(len(obs), model.core.core.hidden_size)
    for t in range(ctx.shape[1]):
        h = forward_step(model, ctx[:, t], h).h
    out = forward_step(model, obs, h)
    acts = np.concatenate([out.acts[lyr][:, units[lyr]].numpy() for lyr in LAYERS], axis=1)
    return {
        "probe_acts": acts.astype(np.float32),
        "probe_feat": out.h.numpy().astype(np.float32),
        "probe_logits": out.logits.numpy().astype(np.float32),
    }


def _cache_one(job: dict[str, Any]) -> str:
    from omegaconf import OmegaConf

    from doomscent.env import make_env
    from doomscent.policy import build_model, load_weights
    from doomscent.recording import read_sidecar, record_episode
    from doomscent.render import render_demo

    torch.set_num_threads(1)
    TL_SUBDIR["name"] = job["subdir"]  # worker processes do not inherit the parent's setting
    paths = RunPaths.for_run(job["run"])
    cfg = paths.load_config()
    ck = Checkpoint(Path(job["path"]), 0, job["env_steps"], job["step"], job["regular"])
    units = json.loads((tl_dir(paths) / "units.json").read_text())["units"]
    probe = dict(np.load(tl_dir(paths) / "probe_set.npz"))
    model, _, heads = build_model(paths)
    load_weights(model, ck.path)
    blocks = raw_blocks(model, units)
    data: dict[str, Any] = {f"block:{k}": v for k, v in blocks.items()}
    data.update({f"strength:{k}": v for k, v in edge_strength(blocks).items()})
    with torch.no_grad():
        data.update(probe_forward(model, probe, units))

    env = make_env(cfg, split="heldout")
    head_sizes = [len(h.choices) for h in env.actions.heads]
    policy = BrainPolicy(model, units, head_sizes, heads)
    seed = int(cfg.seeds.eval_start) + int(cfg.timelapse.fight_seed_offset)
    fight_dir = tl_dir(paths) / "cache" / "fight"
    res, demo, sidecar = record_episode(env, policy, seed, fight_dir, paths.name, ck.step, 0)
    env.close()
    sal = np.stack(policy.log["sal"])
    scale = float(np.quantile(sal, 0.995)) or 1.0
    data.update(
        {
            "fight_acts": np.stack(policy.log["acts"]).astype(np.float32),
            "fight_probs": np.stack(policy.log["probs"]).astype(np.float32),
            "fight_value": np.array(policy.log["value"], dtype=np.float32),
            "fight_sal": np.clip(sal / scale * 255, 0, 255).astype(np.uint8),
            "fight_sal_scale": np.array(scale, dtype=np.float32),  # raw = u8 / 255 * scale
            "fight_kills": np.array([s["vars"]["KILLCOUNT"] for s in res.steps], np.float32),
            "fight_meta": np.array(
                json.dumps({"seed": seed, "demo": str(demo), "kills": res.stats["kills"]})
            ),
        }
    )
    rc = OmegaConf.to_container(cfg.record, resolve=True)
    rc.update({"width": FOOTAGE_RES[0], "height": FOOTAGE_RES[1], "hud": False, "gop": 35})  # type: ignore[union-attr]
    footage_dir = tl_dir(paths) / "cache" / "footage"
    rr = render_demo(demo, read_sidecar(sidecar), ["plain"], footage_dir, rc)  # type: ignore[arg-type]
    final = footage_dir / f"{ck.label}.mp4"
    rr.outputs["plain"].replace(final)
    data["footage_frames"] = np.array(rr.replay_tics)
    out = tl_dir(paths) / "cache" / f"{ck.label}.npz"
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(out)
    return ck.label


def add_changes(paths: RunPaths, cks: list[Checkpoint]) -> None:
    """Edge change since the previous checkpoint (zero for the first)."""
    prev = None
    for ck in cks:
        f = cache_path(paths, ck)
        d = dict(np.load(f))
        blocks = {k.split(":", 1)[1]: v for k, v in d.items() if k.startswith("block:")}
        change = edge_change(prev or blocks, blocks)
        d.update({f"change:{k}": v for k, v in change.items()})
        np.savez_compressed(f.with_name(f.stem + ".tmp.npz"), **d)
        f.with_name(f.stem + ".tmp.npz").replace(f)
        prev = blocks


def fit_umap(paths: RunPaths, cks: list[Checkpoint]) -> None:
    """Fit UMAP once on the final checkpoint's probe features; apply to every checkpoint."""
    import umap

    cfg = paths.load_config().timelapse.umap
    feats = {ck.label: np.load(cache_path(paths, ck))["probe_feat"] for ck in cks}
    reducer = umap.UMAP(
        n_neighbors=int(cfg.n_neighbors), min_dist=float(cfg.min_dist), random_state=int(cfg.seed)
    )
    reducer.fit(feats[cks[-1].label])
    coords = {lbl: reducer.transform(f).astype(np.float32) for lbl, f in feats.items()}
    np.savez_compressed(tl_dir(paths) / "umap.npz", **coords)


def subsample(cks: list[Checkpoint], n: int | None) -> list[Checkpoint]:
    """Up to n checkpoints spread evenly from first to last (always both ends)."""
    if not n or len(cks) <= n:
        return cks
    idx = sorted({round(i * (len(cks) - 1) / (n - 1)) for i in range(n)})
    return [cks[i] for i in idx]


def build_cache(
    paths: RunPaths, workers: int = 4, force: bool = False, max_checkpoints: int | None = None
) -> list[Checkpoint]:
    ensure_units(paths)
    ensure_probe(paths)
    cks = subsample(list_checkpoints(paths), max_checkpoints)
    jobs = [
        {
            "run": paths.name,
            "path": str(c.path),
            "env_steps": c.env_steps,
            "step": c.step,
            "regular": c.regular,
            "subdir": TL_SUBDIR["name"],
        }
        for c in cks
        if force or not cache_path(paths, c).exists()
    ]
    (tl_dir(paths) / "cache").mkdir(parents=True, exist_ok=True)
    if jobs:
        with mp.get_context("spawn").Pool(max(1, min(workers, len(jobs)))) as pool:
            for i, label in enumerate(pool.imap_unordered(_cache_one, jobs)):
                print(f"[timelapse cache] {label} ({i + 1}/{len(jobs)})", flush=True)
    add_changes(paths, cks)
    fit_umap(paths, cks)
    return cks
