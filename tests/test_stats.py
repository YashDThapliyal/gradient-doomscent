import numpy as np

from doomscent.stats import bootstrap_ci, ema


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).normal(10, 2, 200)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert hi - lo < 1.0


def test_bootstrap_constant():
    assert bootstrap_ci([3, 3, 3]) == (3.0, 3.0)


def test_ema():
    assert list(ema([0, 10], alpha=0.5)) == [0.0, 5.0]
