"""CPU-only test for the team layout JSON.

No Isaac, no GPU: this is pure parsing and validation. It is worth its own suite
because every field here fails *late* if it is wrong -- a bad model name surfaces
as a missing YAML halfway through scene load, and a bad room name surfaces as a
robot standing in the wrong place with no error at all -- so the loader is
supposed to reject them up front, and that promise needs a test.

Run:
    python feasibility_verify/test_team_config_stubbed.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.behavior_env.team_config import (
    BUILTIN_ROBOT_MODELS,
    homogeneous_layout,
    known_robot_models,
    load_team_layout,
    parse_team_layout,
    register_robot_model,
)


def ok(message: str) -> None:
    print(f"  ok: {message}")


def rejects(payload, needle: str) -> str:
    """Assert parse fails, and that the message says why. Returns the message."""
    try:
        parse_team_layout(payload)
    except ValueError as error:
        assert needle in str(error), f"expected {needle!r} in error, got: {error}"
        return str(error)
    raise AssertionError(f"expected a ValueError mentioning {needle!r}, got none")


def main() -> int:
    print("test 1: a heterogeneous layout parses, and keeps both ways of placing")
    layout = parse_team_layout({
        "robots": [
            {"name": "agent_0", "model": "R1", "position": [-9.1, -1.8], "team": "alpha"},
            {"name": "agent_1", "model": "Tiago", "room": "living_room_0", "team": "alpha"},
            {"name": "agent_2", "model": "r1pro", "room": "kitchen_0", "team": "bravo"},
        ]
    })
    assert layout.n_agents == 3
    assert layout.models == ("R1", "Tiago", "R1Pro"), layout.models
    assert layout.is_heterogeneous
    a0 = layout.spec_for("agent_0")
    assert a0.placed_explicitly and a0.position == (-9.1, -1.8, 0.05), a0.position
    assert not layout.spec_for("agent_1").placed_explicitly
    assert layout.spec_for("agent_1").room == "living_room_0"
    # Case is normalised, because it becomes a filename: r1pro_primitives.yaml.
    assert layout.spec_for("agent_2").model == "R1Pro"
    assert layout.spec_for("agent_2").config_name() == "r1pro"
    ok("3 robots, 2 models by position and 1 by room, model names normalised")

    print("test 2: teams group agents, and a teammate list includes the agent itself")
    assert layout.teams == {"alpha": ("agent_0", "agent_1"), "bravo": ("agent_2",)}
    assert layout.team_of("agent_1") == "alpha"
    # Including self is deliberate: a brain that planned for everyone *but* the
    # agent that woke it would be a silent hole.
    assert layout.teammates_of("agent_1") == ("agent_0", "agent_1")
    assert layout.teammates_of("agent_2") == ("agent_2",)
    ok("teams resolved from per-robot fields; teammates_of includes self")

    print("test 3: an explicit teams map is accepted, and must agree with the fields")
    both = parse_team_layout({
        "robots": [
            {"name": "a", "model": "R1", "room": "r0", "team": "x"},
            {"name": "b", "model": "R1", "room": "r0"},
        ],
        "teams": {"x": ["a", "b"]},
    })
    assert both.teams == {"x": ("a", "b")}, both.teams
    rejects(
        {
            "robots": [{"name": "a", "model": "R1", "room": "r0", "team": "x"}],
            "teams": {"y": ["a"]},
        },
        "make them agree",
    )
    rejects(
        {
            "robots": [{"name": "a", "model": "R1", "room": "r0"}],
            "teams": {"x": ["a"], "y": ["a"]},
        },
        "appears in two teams",
    )
    rejects(
        {
            "robots": [{"name": "a", "model": "R1", "room": "r0"}],
            "teams": {"x": ["ghost"]},
        },
        "not in 'robots'",
    )
    ok("map and per-robot fields reconciled; disagreement, double membership and "
       "unknown names all rejected")

    print("test 4: a robot with no team gets a team of its own")
    solo = parse_team_layout({
        "robots": [
            {"name": "a", "model": "R1", "room": "r0"},
            {"name": "b", "model": "R1", "room": "r0"},
        ]
    })
    # This is what makes one-LLM-per-robot a special case of the team topology
    # rather than a second code path.
    assert solo.teams == {"a": ("a",), "b": ("b",)}, solo.teams
    ok("unteamed robots become teams of one")

    print("test 5: placement is exactly one of position or room")
    rejects({"robots": [{"name": "a", "model": "R1"}]}, "needs 'position'")
    rejects(
        {"robots": [{"name": "a", "model": "R1", "position": [1, 2], "room": "r0"}]},
        "not both",
    )
    rejects({"robots": [{"name": "a", "model": "R1", "position": [1]}]}, "[x, y]")
    rejects({"robots": [{"name": "a", "model": "R1", "position": ["x", 2]}]}, "must be numbers")
    # [x, y] gains the spawn height place_robots uses, rather than the floor.
    z = parse_team_layout(
        {"robots": [{"name": "a", "model": "R1", "position": [1, 2]}]}
    ).spec_for("a").position[2]
    assert z == 0.05, z
    ok("exactly one of position/room enforced; [x,y] gains the spawn height")

    print("test 6: an unknown model is refused by name, and told how to add one")
    message = rejects(
        {"robots": [{"name": "a", "model": "Fetch", "room": "r0"}]},
        "unknown robot model",
    )
    for model in BUILTIN_ROBOT_MODELS:
        assert model in message, f"{model} missing from the error: {message}"
    # The error has to carry the escape hatch, or the next person to import a
    # non-BEHAVIOR robot reads it as "this is impossible" rather than "declare it".
    assert "config" in message and "register_robot_model" in message, message
    ok("Fetch refused up front, with both ways to register a robot in the message")

    print("test 6b: a non-BEHAVIOR robot is declared by naming its primitives config")
    with tempfile.TemporaryDirectory() as directory:
        custom = os.path.join(directory, "myrobot_primitives.yaml")
        with open(custom, "w", encoding="utf-8") as handle:
            handle.write("robots: [{type: MyRobot}]\n")
        layout = parse_team_layout({
            "robots": [
                {"name": "a", "model": "MyRobot", "room": "r0", "config": custom},
                {"name": "b", "model": "R1", "room": "r0"},
            ]
        })
        assert layout.spec_for("a").model == "MyRobot"
        assert layout.spec_for("a").config_path == os.path.abspath(custom)
        # A built-in still resolves by name, so the two kinds coexist.
        assert layout.spec_for("b").config_path is None
        assert "MyRobot" in known_robot_models()
        # Registered once, it needs no 'config' the second time.
        again = parse_team_layout({"robots": [{"name": "c", "model": "myrobot", "room": "r0"}]})
        assert again.spec_for("c").model == "MyRobot"
        # A config path that is not there is refused now, not at scene load.
        rejects(
            {"robots": [{"name": "d", "model": "Ghost", "room": "r0",
                         "config": os.path.join(directory, "absent.yaml")}]},
            "does not exist",
        )
        # Re-registering the same name against a different file is a conflict.
        other = os.path.join(directory, "other_primitives.yaml")
        with open(other, "w", encoding="utf-8") as handle:
            handle.write("robots: [{type: MyRobot}]\n")
        try:
            register_robot_model("MyRobot", other)
            raise AssertionError("expected a ValueError for a conflicting re-registration")
        except ValueError as error:
            assert "already registered" in str(error), error
    ok("a custom robot registers from the layout, coexists with built-ins, and "
       "a missing or conflicting config is refused up front")

    print("test 7: duplicate names are refused")
    rejects(
        {"robots": [{"name": "a", "model": "R1", "room": "r0"},
                    {"name": "a", "model": "R1", "room": "r0"}]},
        "duplicate robot names",
    )
    ok("names are the key of the action and observation dicts, so they must be unique")

    print("test 8: the file loader reports which file was wrong")
    with tempfile.TemporaryDirectory() as directory:
        good = os.path.join(directory, "team.json")
        with open(good, "w", encoding="utf-8") as handle:
            json.dump({"robots": [{"name": "a", "model": "R1", "room": "r0"}]}, handle)
        assert load_team_layout(good).n_agents == 1

        bad = os.path.join(directory, "bad.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        try:
            load_team_layout(bad)
            raise AssertionError("expected a ValueError for malformed JSON")
        except ValueError as error:
            assert "bad.json" in str(error) and "not valid JSON" in str(error), error

        try:
            load_team_layout(os.path.join(directory, "absent.json"))
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass
    ok("valid file loads; malformed and missing files name the path")

    print("test 9: --agents N is expressible as a layout, with optional teams")
    flat = homogeneous_layout(9, room="empty_room_0")
    assert flat.n_agents == 9 and not flat.is_heterogeneous
    assert len(flat.teams) == 9, "no team_size means one team each"
    grouped = homogeneous_layout(9, room="empty_room_0", team_size=4)
    assert [len(m) for m in grouped.teams.values()] == [4, 4, 1], grouped.teams
    assert grouped.teammates_of("agent_0") == ("agent_0", "agent_1", "agent_2", "agent_3")
    ok("9 agents in teams of 4 -> 4/4/1; the remainder is its own team")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
