import pytest

from doomscent.evaluate import max_kills_in_window, policy_summary


def _steps(kill_times, n=50):
    k, out = 0, []
    for t in range(n):
        k += kill_times.count(t)
        out.append({"t": t, "vars": {"KILLCOUNT": k}})
    return out


def test_multi_kill_window():
    assert max_kills_in_window(_steps([1, 3, 5, 40]), window=10) == 3
    assert max_kills_in_window(_steps([1, 20, 40]), window=10) == 1
    assert max_kills_in_window(_steps([]), window=10) == 0
    assert max_kills_in_window(_steps([5, 5]), window=1) == 2


def test_policy_summary_shares_and_entropy():
    heads = [{"name": "attack", "choices": ["off", "on"]}]
    steps = [
        {"action_idx": [1], "probs": {"attack": [0.5, 0.5]}},
        {"action_idx": [1], "probs": {"attack": [0.5, 0.5]}},
        {"action_idx": [0], "probs": {"attack": [0.5, 0.5]}},
        {"action_idx": [1], "probs": {"attack": [0.5, 0.5]}},
    ]
    s = policy_summary(steps, heads)
    assert s["share_attack_on"] == 0.75 and s["share_attack_off"] == 0.25
    assert s["entropy_attack"] == pytest.approx(0.6931, abs=1e-3)
