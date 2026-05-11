import sys
from unittest.mock import MagicMock, patch

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


def test_inroom_object_scope_tentative_assignment():
    """Fix 1: _build_inroom_object_scope must populate _object_scope with a
    candidate for every inroom object, even before MBM runs.

    Without this, _object_scope has None for all inroom objects when
    _determine_room_instances is called, so wildcard compilation gets an empty
    scene layout and fails with "found 0 instances".
    """
    from omnigibson.utils.bddl_utils import BDDLSampler

    mock_obj = MagicMock()
    sampler = object.__new__(BDDLSampler)
    sampler._object_scope = {"bookcase.n.01_1": None}
    sampler._inroom_object_scope = {"living_room": {"bookcase.n.01_1": {"living_room_0": [mock_obj]}}}

    # Apply only the tentative-assignment block added by Fix 1
    for obj_to_rooms in sampler._inroom_object_scope.values():
        for obj_inst, room_inst_to_objs in obj_to_rooms.items():
            for objs in room_inst_to_objs.values():
                if objs:
                    sampler._object_scope[obj_inst] = objs[0]
                    break

    assert sampler._object_scope["bookcase.n.01_1"] is mock_obj


def test_parse_inroom_updates_synset_map_after_conditions_switch():
    """Fix 2: _parse_inroom_object_room_assignment must refresh
    _object_instance_to_synset from the *current* _activity_conditions.

    Without this, switching to compiled conditions (which adds wildcard-expanded
    instances) leaves those new names absent from the map, causing a KeyError
    when processing their inroom conditions.
    """
    from omnigibson.utils.bddl_utils import BDDLSampler

    mock_synset = MagicMock()
    mock_synset.abilities = {"sceneObject": True}

    with patch("omnigibson.utils.bddl_utils.get_knowledge_base") as mock_kb:
        mock_kb.return_value.get_synset.return_value = mock_synset

        sampler = object.__new__(BDDLSampler)
        sampler._scene_model = None
        sampler._object_instance_to_synset = {"bookcase.n.01_1": "bookcase.n.01"}
        sampler._room_type_to_object_instance = {}
        sampler._inroom_object_instances = set()

        # Simulate compiled conditions with two extra wildcard-expanded instances
        compiled_conds = MagicMock()
        compiled_conds.parsed_objects = {"bookcase.n.01": ["bookcase.n.01_1", "bookcase.n.01_2", "bookcase.n.01_3"]}
        compiled_conds.parsed_initial_conditions = [
            ("inroom", "bookcase.n.01_1", "living_room"),
            ("inroom", "bookcase.n.01_2", "living_room"),
            ("inroom", "bookcase.n.01_3", "living_room"),
        ]
        sampler._activity_conditions = compiled_conds

        error = sampler._parse_inroom_object_room_assignment()

        assert error is None
        assert "bookcase.n.01_2" in sampler._object_instance_to_synset
        assert "bookcase.n.01_3" in sampler._object_instance_to_synset
        assert "bookcase.n.01_2" in sampler._inroom_object_instances


def test_sample_states_switches_to_compiled_conditions():
    """Fix 3: sample_states must set _activity_conditions to compiled_task.conditions
    and re-run inroom setup before building the sampling order.

    Without this, wildcard-expanded instances are never added to
    _inroom_object_instances, so _build_sampling_order flags them as having no
    kinematic condition and raises a ValueError.
    """
    from omnigibson.utils.bddl_utils import BDDLSampler

    mock_synset = MagicMock()
    mock_synset.abilities = {"sceneObject": True}

    with patch("omnigibson.utils.bddl_utils.get_knowledge_base") as mock_kb:
        mock_kb.return_value.get_synset.return_value = mock_synset

        sampler = object.__new__(BDDLSampler)
        sampler._scene_model = None
        sampler._sampling_whitelist = None
        sampler._sampling_blacklist = None
        sampler._object_instance_to_synset = {"bookcase.n.01_1": "bookcase.n.01"}
        sampler._inroom_object_instances = {"bookcase.n.01_1"}
        sampler._room_type_to_object_instance = {"living_room": ["bookcase.n.01_1"]}

        base_conds = MagicMock()
        base_conds.parsed_objects = {"bookcase.n.01": ["bookcase.n.01_1"]}
        base_conds.parsed_initial_conditions = [("inroom", "bookcase.n.01_1", "living_room")]
        sampler._activity_conditions = base_conds

        compiled_conds = MagicMock()
        compiled_conds.parsed_objects = {"bookcase.n.01": ["bookcase.n.01_1", "bookcase.n.01_2", "bookcase.n.01_3"]}
        compiled_conds.parsed_initial_conditions = [
            ("inroom", "bookcase.n.01_1", "living_room"),
            ("inroom", "bookcase.n.01_2", "living_room"),
            ("inroom", "bookcase.n.01_3", "living_room"),
        ]
        compiled_task = MagicMock()
        compiled_task.conditions = compiled_conds

        sampler._build_inroom_object_scope = MagicMock(return_value=None)
        sampler._build_sampling_order = MagicMock(return_value=None)
        sampler._sample_all_conditions = MagicMock(return_value=(True, None))

        sampler.sample_states(compiled_task)

        assert sampler._activity_conditions is compiled_conds
        assert "bookcase.n.01_2" in sampler._inroom_object_instances
        assert "bookcase.n.01_3" in sampler._inroom_object_instances


if __name__ == "__main__":
    test_behavior_task()
