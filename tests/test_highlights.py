from doomscent.highlights import Moment, find_escapes, find_streaks, pick


def _steps(kills_at=(), health=None, n=100):
    k, out = 0, []
    for t in range(n):
        k += list(kills_at).count(t)
        hp = health[t] if health else 100
        out.append({"t": t, "vars": {"KILLCOUNT": k, "HEALTH": hp}})
    return out


def test_streaks_found_and_separated():
    steps = _steps(kills_at=(10, 14, 20, 60, 90))
    assert find_streaks(steps, window=35, min_kills=3) == [(10, 20, 3)]
    assert find_streaks(steps, window=35, min_kills=4) == []


def test_escape_needs_survival_and_a_kill():
    health = [100] * 30 + [25] * 70
    assert find_escapes(_steps(kills_at=(50,), health=health), 30, 35) == [(30, 65, 1)]
    assert find_escapes(_steps(health=health), 30, 35) == []  # no kill after the hit
    assert find_escapes(_steps(kills_at=(50,), health=health, n=40), 30, 35) == []  # died


def test_pick_one_moment_per_episode():
    def m(kind, demo, kills, score):
        return Moment(kind, 1, 0, demo, "", 0, 1, kills, score, "")

    ms = [
        m("streak", "a", 5, 1.0),
        m("streak", "a", 4, 2.0),
        m("streak", "b", 3, 1.0),
        m("escape", "c", 1, 40.0),
        m("escape", "a", 1, 50.0),
    ]
    chosen = pick(ms, n_streaks=5, n_escapes=5)
    assert [(c.kind, c.demo, c.kills) for c in chosen] == [
        ("streak", "a", 5),
        ("streak", "b", 3),
        ("escape", "c", 1),
    ]
