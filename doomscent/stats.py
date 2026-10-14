"""Small statistics helpers shared by evaluate, baseline, and figures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def ema(values: Sequence[float], alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average, for the smoothed line drawn over raw series."""
    x = np.asarray(values, dtype=float)
    out = np.empty_like(x)
    acc = None
    for i, v in enumerate(x):
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out
