import pytest

from doomscent.reward import RewardWeights, pickup_events, step_reward

W = RewardWeights(
    kill=1.0,
    death=-1.0,
    damage_dealt_per_hp=0.01,
    damage_taken_per_hp=-0.005,
    health_armor_pickup_per_point=0.005,
    weapon_ammo_pickup=0.1,
)
BASE = {"KILLCOUNT": 0, "DAMAGECOUNT": 0, "DAMAGE_TAKEN": 0, "HEALTH": 100, "ARMOR": 0, "AMMO2": 26}


def test_no_change_is_zero():
    assert step_reward(BASE, dict(BASE), W, died=False).total == 0.0


def test_kill_and_damage_dealt():
    cur = {**BASE, "KILLCOUNT": 1, "DAMAGECOUNT": 30}
    rb = step_reward(BASE, cur, W, died=False)
    assert rb.kills == 1.0
    assert rb.damage_dealt == pytest.approx(0.3)
    assert rb.total == pytest.approx(1.3)


def test_damage_taken_is_penalized_less_than_dealt_is_rewarded():
    taken = step_reward(BASE, {**BASE, "DAMAGE_TAKEN": 20, "HEALTH": 80}, W, died=False)
    dealt = step_reward(BASE, {**BASE, "DAMAGECOUNT": 20}, W, died=False)
    assert taken.total == pytest.approx(-0.1)
    assert abs(taken.total) < dealt.total


def test_health_drop_is_not_a_pickup():
    rb = step_reward(BASE, {**BASE, "HEALTH": 60, "DAMAGE_TAKEN": 40}, W, died=False)
    assert rb.health_armor_pickup == 0.0


def test_death_penalty_and_no_pickup_while_dying():
    cur = {**BASE, "HEALTH": 0, "DAMAGE_TAKEN": 100}
    rb = step_reward(BASE, cur, W, died=True)
    assert rb.death == -1.0
    assert rb.total == pytest.approx(-1.5)


def test_health_and_armor_pickups():
    cur = {**BASE, "HEALTH": 125, "ARMOR": 100}
    rb = step_reward({**BASE, "HEALTH": 100}, cur, W, died=False)
    assert rb.health_armor_pickup == pytest.approx(0.005 * 125)


def test_ammo_pickup_event():
    cur = {**BASE, "AMMO2": 36}
    assert pickup_events(BASE, cur, ["AMMO2"], []) == 1
    assert step_reward(BASE, cur, W, False, ammo_vars=["AMMO2"]).weapon_ammo_pickup == 0.1


def test_spending_ammo_is_not_a_pickup():
    assert pickup_events(BASE, {**BASE, "AMMO2": 25}, ["AMMO2"], []) == 0


def test_scale_multiplies_everything():
    w2 = RewardWeights(**{**W.__dict__, "scale": 2.0})
    cur = {**BASE, "KILLCOUNT": 1, "DAMAGE_TAKEN": 10}
    assert step_reward(BASE, cur, w2, False).total == pytest.approx(
        2 * step_reward(BASE, cur, W, False).total
    )


def test_weights_from_config(v0_cfg):
    w = RewardWeights.from_config(v0_cfg.reward)
    assert w.kill == 1.0 and w.death == -1.0 and w.damage_taken_per_hp == -0.005


def test_ammo_spent_term_off_by_default_and_charged_when_set():
    prev = {**BASE, "SELECTED_WEAPON": 2, "SELECTED_WEAPON_AMMO": 26}
    cur = {**prev, "SELECTED_WEAPON_AMMO": 25}
    assert step_reward(prev, cur, W, False).total == 0.0
    w2 = RewardWeights(**{**W.__dict__, "ammo_spent_per_unit": -0.03})
    assert step_reward(prev, cur, w2, False).ammo_spent == pytest.approx(-0.03)
    switched = {**cur, "SELECTED_WEAPON": 3}
    assert step_reward(prev, switched, w2, False).ammo_spent == 0.0
