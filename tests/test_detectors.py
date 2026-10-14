from doomscent.detectors import (
    camping,
    pacifist,
    reward_spike,
    run_detectors,
    spinning,
    stuck,
    suicide,
    wasted_fire,
)

START = {
    "POSITION_X": 0.0,
    "POSITION_Y": 0.0,
    "SELECTED_WEAPON": 2,
    "SELECTED_WEAPON_AMMO": 50,
    "HITCOUNT": 0,
}


def step(turn=0.0, attack="off", move="none", x=0.0, y=0.0, ammo=50, hits=0, enemies=0, weapon=2):
    return {
        "action": {"turn": "x", "attack": attack, "move": move},
        "turn_delta": turn,
        "vars": {
            "POSITION_X": x,
            "POSITION_Y": y,
            "SELECTED_WEAPON": weapon,
            "SELECTED_WEAPON_AMMO": ammo,
            "HITCOUNT": hits,
        },
        "enemies_visible": enemies,
        "reward": 0.0,
    }


SPIN = {"min_turn_fraction": 0.9, "same_direction_fraction": 0.9, "max_displacement": 64.0}


def test_spinning_fires_on_constant_one_way_turn():
    assert spinning([step(turn=4.0)] * 100, SPIN) is not None


def test_spinning_ignores_aiming_back_and_forth():
    steps = [step(turn=4.0 if i % 2 else -4.0) for i in range(100)]
    assert spinning(steps, SPIN) is None


def test_spinning_ignores_mostly_still_agent():
    steps = [step(turn=4.0)] * 50 + [step()] * 50
    assert spinning(steps, SPIN) is None


STUCK = {"min_forward_steps": 10, "max_displacement": 16.0}


def test_stuck_fires_when_walking_into_wall():
    assert stuck([step(move="forward", x=1.0)] * 20, STUCK) is not None


def test_stuck_ignores_real_movement():
    assert stuck([step(move="forward", x=10.0 * i) for i in range(20)], STUCK) is None


def test_camping():
    cfg = {"radius": 100.0, "min_fraction": 0.8}
    assert camping([step(x=5.0)] * 90 + [step(x=900.0)] * 10, cfg) is not None
    assert camping([step(x=50.0 * i) for i in range(100)], cfg) is None


WASTE = {"min_shots": 20, "max_accuracy": 0.1}


def test_wasted_fire_many_shots_no_hits():
    steps = [step(attack="on", ammo=50 - i - 1) for i in range(30)]
    assert wasted_fire(steps, START, WASTE) is not None


def test_wasted_fire_ignores_accurate_shooting():
    steps = [step(attack="on", ammo=50 - i - 1, hits=i + 1) for i in range(30)]
    assert wasted_fire(steps, START, WASTE) is None


def test_wasted_fire_ignores_few_shots():
    steps = [step(attack="on", ammo=50 - i - 1) for i in range(5)]
    assert wasted_fire(steps, START, WASTE) is None


def test_pacifist():
    cfg = {"min_enemy_visible_steps": 10}
    assert pacifist([step(enemies=2)] * 20, START, cfg) is not None
    assert pacifist([step(enemies=2)] * 19 + [step(attack="on", ammo=49)], START, cfg) is None
    assert pacifist([step(enemies=0)] * 20, START, cfg) is None


def test_suicide_after_rocket():
    cfg = {"rocket_weapon_slot": 5, "window_steps": 4}
    steps = [step(weapon=5, ammo=50)] * 10 + [step(weapon=5, ammo=49), step(weapon=5, ammo=49)]
    start = {**START, "SELECTED_WEAPON": 5}
    assert suicide(steps, start, died=True, cfg=cfg) is not None
    assert suicide(steps, start, died=False, cfg=cfg) is None
    assert suicide([step(ammo=49)] * 3, START, died=True, cfg=cfg) is None


def test_reward_spike():
    cfg = {"window": 100, "min_history": 5, "z": 3.0}
    hist = [1.0, 1.2, 0.9, 1.1, 1.0, 0.8]
    assert reward_spike(10.0, hist, cfg) is not None
    assert reward_spike(1.1, hist, cfg) is None
    assert reward_spike(10.0, hist[:3], cfg) is None


def test_run_detectors_respects_enabled(v0_cfg):
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(v0_cfg.detectors)
    steps = [step(turn=4.0)] * 100
    names = [f.detector for f in run_detectors(steps, START, False, 0.0, [], cfg)]
    assert names == ["spinning"]
    cfg["spinning"]["enabled"] = False
    assert run_detectors(steps, START, False, 0.0, [], cfg) == []
