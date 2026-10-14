import pytest
import vizdoom as vzd

from doomscent.actions import build_action_space


def test_v0_heads_and_space():
    a = build_action_space(["turn", "attack"], [1.5, 4.0, 9.0])
    assert a.head_names == ("turn", "attack")
    assert [s.n for s in a.gym_space().spaces] == [7, 2]
    assert a.buttons() == [vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.ATTACK]


def test_turn_deltas_signed_left_negative():
    a = build_action_space(["turn", "attack"], [1.5, 4.0, 9.0])
    assert [a.turn_delta(i) for i in range(7)] == [0.0, -1.5, -4.0, -9.0, 1.5, 4.0, 9.0]
    assert a.to_buttons([6, 1]) == [9.0, 1.0]
    assert a.to_buttons([0, 0]) == [0.0, 0.0]


def test_full_space_button_layout():
    a = build_action_space(["move", "strafe", "turn", "attack", "weapon"], [2.0])
    assert len(a.buttons()) == len(a.to_buttons([1, 2, 1, 1, 1]))
    assert a.to_buttons([1, 2, 1, 1, 1]) == [1.0, 0.0, 0.0, 1.0, -2.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "heads,speeds", [([], [1.0]), (["turn", "turn"], [1.0]), (["turn"], []), (["jump"], [1.0])]
)
def test_invalid_spaces_rejected(heads, speeds):
    with pytest.raises(ValueError):
        build_action_space(heads, speeds)


def test_out_of_range_choice_rejected():
    a = build_action_space(["turn", "attack"], [1.0])
    with pytest.raises(ValueError):
        a.to_buttons([3, 0])
    with pytest.raises(ValueError):
        a.to_buttons([0])
