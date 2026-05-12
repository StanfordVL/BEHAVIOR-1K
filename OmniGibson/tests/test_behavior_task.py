import sys

import omnigibson as og
from omnigibson.macros import gm


def test_behavior_task():
    gm.ENABLE_OBJECT_STATES = True
    gm.HEADLESS = True
    config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": "putting_away_Halloween_decorations",
            "activity_definition_id": 0,
            "online_object_sampling": True,
            "use_presampled_robot_pose": False,
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb"],
            }
        ],
    }
    try:
        env = og.Environment(configs=config)
        print(
            "BehaviorTask instantiated successfully! Ground goal state options:",
            len(env.task.ground_goal_state_options),
        )
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


def test_behavior_task_wildcard():
    gm.ENABLE_OBJECT_STATES = True
    gm.HEADLESS = True
    config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "house_double_floor_upper",
            "load_room_types": ["bathroom"],
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": "setting_mousetraps",
            "activity_definition_id": 0,
            "online_object_sampling": True,
            "use_presampled_robot_pose": False,
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb"],
            }
        ],
    }
    try:
        env = og.Environment(configs=config)
        scope = env.task.compiled_task.object_scope
        assert "sink.n.01_*" not in scope, f"wildcard left unexpanded in {sorted(scope)}"
        assert "floor.n.01_*" not in scope, f"wildcard left unexpanded in {sorted(scope)}"
        assert "sink.n.01_1" in scope, f"sink.n.01_1 missing from {sorted(scope)}"
        assert "floor.n.01_1" in scope, f"floor.n.01_1 missing from {sorted(scope)}"
        print(
            "Wildcard BehaviorTask instantiated successfully! Ground goal state options:",
            len(env.task.ground_goal_state_options),
        )
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


def test_wildcard_room_anchor_preserved():
    # installing_a_fence anchors garden ONLY via wildcard-synset bases
    # (rail_fence.n.01_1, floor.n.01_1). Without the fix in
    # initialize_activity, unbinding before room determination drops garden
    # from the scene layout and expand_wildcards raises "found 0 in rooms of
    # type garden". sample_states is just one (ontop ...) so it samples reliably.
    gm.ENABLE_OBJECT_STATES = True
    gm.HEADLESS = True
    config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "house_double_floor_lower",
            "load_room_types": ["garden"],
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": "installing_a_fence",
            "activity_definition_id": 0,
            "online_object_sampling": True,
            "use_presampled_robot_pose": False,
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb"],
            }
        ],
    }
    try:
        env = og.Environment(configs=config)
        scope = env.task.compiled_task.object_scope
        assert "rail_fence.n.01_*" not in scope, f"wildcard left unexpanded in {sorted(scope)}"
        assert "rail_fence.n.01_1" in scope, f"rail_fence.n.01_1 missing from {sorted(scope)}"
        print(
            "Wildcard BehaviorTask instantiated successfully! Ground goal state options:",
            len(env.task.ground_goal_state_options),
        )
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_behavior_task()
    test_behavior_task_wildcard()
    test_wildcard_room_anchor_preserved()
