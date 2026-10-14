"""The network as a stylized brain: real weights, real activations, real gradients.

The actor-critic's conv encoder is TorchScript, so instead of forward hooks this module
recomputes the forward pass from the same weights (tested to match the model's own outputs)
and records every layer on the way. Units shown per layer are sampled once and kept fixed.

Layers: input (RGB) -> conv1 (32) -> conv2 (64) -> conv3 (128) -> fc (512) -> gru (512)
-> out (9 action logits + value).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import Saliency
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import ActorCritic

ENC = "encoder.encoders.obs.enc."
LAYERS = ("input", "conv1", "conv2", "conv3", "fc", "gru", "out")
CONV_STRIDES = (4, 2, 2)


def layer_sizes(model: ActorCritic) -> dict[str, int]:
    sd = model.state_dict()
    return {
        "input": 3,
        "conv1": sd[ENC + "conv_head.0.weight"].shape[0],
        "conv2": sd[ENC + "conv_head.2.weight"].shape[0],
        "conv3": sd[ENC + "conv_head.4.weight"].shape[0],
        "fc": sd[ENC + "mlp_layers.0.weight"].shape[0],
        "gru": sd["core.core.weight_hh_l0"].shape[1],
        "out": sd["action_parameterization.distribution_linear.weight"].shape[0] + 1,
    }


def sample_units(sizes: dict[str, int], per_layer: int, seed: int) -> dict[str, list[int]]:
    """Choose which units to draw, once per run; small layers are shown whole."""
    rng = np.random.default_rng(seed)
    return {
        name: sorted(rng.choice(n, size=per_layer, replace=False).tolist())
        if n > per_layer
        else list(range(n))
        for name, n in sizes.items()
    }


def normalize(model: ActorCritic, obs_u8: torch.Tensor) -> torch.Tensor:
    return prepare_and_normalize_obs(model, {"obs": obs_u8.clone()})["obs"]


@dataclass
class StepOut:
    acts: dict[str, torch.Tensor]  # per layer: (B, units) activations
    h: torch.Tensor  # new GRU state (B, 512)
    logits: torch.Tensor  # (B, 9)
    value: torch.Tensor  # (B,)


def forward_from_normalized(model: ActorCritic, x: torch.Tensor, h: torch.Tensor) -> StepOut:
    """One recurrent step from normalized obs, keeping every layer's activations.

    Conv activations are reported per channel as the spatial mean of the ELU output.
    """
    sd = model.state_dict()
    acts: dict[str, torch.Tensor] = {"input": x.mean(dim=(2, 3))}
    y = x
    for i, (idx, stride) in enumerate(zip((0, 2, 4), CONV_STRIDES, strict=True)):
        w, b = sd[ENC + f"conv_head.{idx}.weight"], sd[ENC + f"conv_head.{idx}.bias"]
        y = F.elu(F.conv2d(y, w, b, stride=stride))
        acts[f"conv{i + 1}"] = y.mean(dim=(2, 3))
    fc = F.elu(
        F.linear(y.flatten(1), sd[ENC + "mlp_layers.0.weight"], sd[ENC + "mlp_layers.0.bias"])
    )
    acts["fc"] = fc
    h_new = torch._VF.gru_cell(
        fc,
        h,
        sd["core.core.weight_ih_l0"],
        sd["core.core.weight_hh_l0"],
        sd["core.core.bias_ih_l0"],
        sd["core.core.bias_hh_l0"],
    )
    acts["gru"] = h_new
    logits = F.linear(
        h_new,
        sd["action_parameterization.distribution_linear.weight"],
        sd["action_parameterization.distribution_linear.bias"],
    )
    value = F.linear(h_new, sd["critic_linear.weight"], sd["critic_linear.bias"]).squeeze(-1)
    acts["out"] = torch.cat([logits, value.unsqueeze(-1)], dim=1)
    return StepOut(acts, h_new, logits, value)


@torch.no_grad()
def forward_step(model: ActorCritic, obs_u8: np.ndarray, h: torch.Tensor) -> StepOut:
    x = normalize(model, torch.from_numpy(np.ascontiguousarray(obs_u8)))
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return forward_from_normalized(model, x, h)


def head_slices(head_sizes: list[int]) -> list[slice]:
    out, start = [], 0
    for n in head_sizes:
        out.append(slice(start, start + n))
        start += n
    return out


def saliency(
    model: ActorCritic, obs_u8: np.ndarray, h: torch.Tensor, head_sizes: list[int]
) -> np.ndarray:
    """|d log p(chosen action) / d pixel|, max over color channels: (H, W) float32.

    The chosen action is the argmax of each head; its log-probability is the sum over heads.
    """
    x = normalize(model, torch.from_numpy(np.ascontiguousarray(obs_u8)).unsqueeze(0)).detach()
    slices = head_slices(head_sizes)

    def score(inp: torch.Tensor) -> torch.Tensor:
        out = forward_from_normalized(model, inp, h.detach())
        total = 0
        for s in slices:
            lp = F.log_softmax(out.logits[:, s], dim=1)
            total = total + lp.max(dim=1).values
        return total

    grads = Saliency(score).attribute(x, abs=True)
    return grads[0].max(dim=0).values.numpy().astype(np.float32)


def raw_blocks(model: ActorCritic, units: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """The raw weights behind each sampled edge, shaped (dst, src, k).

    An edge's strength is the L2 norm over k: the kernel joining two conv channels, the
    spatial positions of a conv3 channel feeding an fc unit, the three GRU gates, or the
    single output weight. Keeping the raw block makes weight changes between checkpoints
    exact (the norm of the difference, not the difference of norms).
    """
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    blocks: dict[str, torch.Tensor] = {}
    for i, idx in enumerate((0, 2, 4)):
        blocks[f"{LAYERS[i]}->{LAYERS[i + 1]}"] = sd[ENC + f"conv_head.{idx}.weight"].flatten(2)
    wfc = sd[ENC + "mlp_layers.0.weight"]
    c3 = sd[ENC + "conv_head.4.weight"].shape[0]
    blocks["conv3->fc"] = wfc.reshape(wfc.shape[0], c3, -1)
    wih = sd["core.core.weight_ih_l0"]
    blocks["fc->gru"] = wih.reshape(3, -1, wih.shape[1]).permute(1, 2, 0)
    wout = torch.cat(
        [sd["action_parameterization.distribution_linear.weight"], sd["critic_linear.weight"]]
    )
    blocks["gru->out"] = wout.unsqueeze(-1)
    out = {}
    for key, b in blocks.items():
        src, dst = key.split("->")
        out[key] = b[np.ix_(units[dst], units[src])].numpy().astype(np.float32)
    return out


def edge_strength(blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.linalg.norm(v, axis=-1) for k, v in blocks.items()}


def edge_change(prev: dict[str, np.ndarray], cur: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.linalg.norm(cur[k] - prev[k], axis=-1) for k in cur}
