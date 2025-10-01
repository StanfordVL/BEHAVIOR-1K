import torch as th
from omnigibson.object_states import ToggledOn, OnTop, Inside, Open, NextTo

ROBOT_OBJECT_DISTANCE_THRESHOLD = 0.5  # meters


def check_progress(env, check_specs):
    """Generic progress checker using declarative specs."""
    objs = env.task.object_scope
    results = {}

    for name, spec in check_specs.items():
        check_type = spec[0]

        if check_type == "near":
            # ("near", obj1_key, obj2_key)
            obj1, obj2 = objs[spec[1]].unwrapped, objs[spec[2]].unwrapped
            pos1, pos2 = obj1.get_position_orientation()[0], obj2.get_position_orientation()[0]
            # Only consider x and y coordinates (horizontal distance)
            dist = th.linalg.norm(pos1[:2] - pos2[:2])
            results[name] = bool(dist < ROBOT_OBJECT_DISTANCE_THRESHOLD)

        elif check_type == "state":
            # Check if it's a relational or non-relational state based on argument pattern
            if len(spec) == 4 and isinstance(spec[3], bool):
                # ("state", obj_key, state_name, expected_bool) - non-relational state, e.g. ToggledOn, Open
                obj = objs[spec[1]].unwrapped
                results[name] = obj.states[spec[2]].get_value() == spec[3]
            elif len(spec) == 5:
                # ("state", obj1_key, state_name, obj2_key, expected_bool) - relational state with explicit bool, e.g. Inside, OnTop
                obj1, obj2 = objs[spec[1]].unwrapped, objs[spec[3]].unwrapped
                results[name] = obj1.states[spec[2]].get_value(obj2) == spec[4]
            else:
                raise ValueError(f"Invalid state spec: {spec}")

    return results


# Task specifications
CHALLENGE_TASKS_PROGRESS_APPROXIMATION = {
    "turning_on_radio": lambda env: check_progress(
        env,
        {
            "robot_near_radio": ("near", "agent.n.01_1", "radio_receiver.n.01_1"),
            "radio_picked_up": ("state", "radio_receiver.n.01_1", OnTop, "table.n.02_1", False),
            "radio_on": ("state", "radio_receiver.n.01_1", ToggledOn, True),
        },
    ),
    "picking_up_trash": lambda env: check_progress(
        env,
        {
            "robot_near_trash_can": ("near", "agent.n.01_1", "ashcan.n.01_1"),
            "robot_near_can_of_soda_1": ("near", "agent.n.01_1", "can__of__soda.n.01_1"),
            "robot_near_can_of_soda_2": ("near", "agent.n.01_1", "can__of__soda.n.01_2"),
            "robot_near_can_of_soda_3": ("near", "agent.n.01_1", "can__of__soda.n.01_3"),
            "can_of_soda_1_picked_up": ("state", "can__of__soda.n.01_1", OnTop, "floor.n.01_1", False),
            "can_of_soda_2_picked_up": ("state", "can__of__soda.n.01_2", OnTop, "floor.n.01_1", False),
            "can_of_soda_3_picked_up": ("state", "can__of__soda.n.01_3", OnTop, "floor.n.01_1", False),
            "can_of_soda_1_in_trash": ("state", "can__of__soda.n.01_1", Inside, "ashcan.n.01_1", True),
            "can_of_soda_2_in_trash": ("state", "can__of__soda.n.01_2", Inside, "ashcan.n.01_1", True),
            "can_of_soda_3_in_trash": ("state", "can__of__soda.n.01_3", Inside, "ashcan.n.01_1", True),
        },
    ),
    "putting_away_Halloween_decorations": lambda env: check_progress(
        env,
        {
            "cabinet_open": ("state", "cabinet.n.01_1", Open, True),
            "robot_near_candle_1": ("near", "agent.n.01_1", "candle.n.01_1"),
            "robot_near_candle_2": ("near", "agent.n.01_1", "candle.n.01_2"),
            "robot_near_candle_3": ("near", "agent.n.01_1", "candle.n.01_3"),
            "candle_1_picked_up": ("state", "candle.n.01_1", OnTop, "floor.n.01_1", False),
            "candle_2_picked_up": ("state", "candle.n.01_2", OnTop, "floor.n.01_1", False),
            "candle_3_picked_up": ("state", "candle.n.01_3", OnTop, "floor.n.01_1", False),
            "candle_1_in_cabinet": ("state", "candle.n.01_1", Inside, "cabinet.n.01_1", True),
            "candle_2_in_cabinet": ("state", "candle.n.01_2", Inside, "cabinet.n.01_1", True),
            "candle_3_in_cabinet": ("state", "candle.n.01_3", Inside, "cabinet.n.01_1", True),
            "robot_near_pumpkin_1": ("near", "agent.n.01_1", "pumpkin.n.02_1"),
            "robot_near_pumpkin_2": ("near", "agent.n.01_1", "pumpkin.n.02_2"),
            "pumpkin_1_picked_up": ("state", "pumpkin.n.02_1", OnTop, "floor.n.01_1", False),
            "pumpkin_2_picked_up": ("state", "pumpkin.n.02_2", OnTop, "floor.n.01_1", False),
            "pumpkin_1_in_cabinet": ("state", "pumpkin.n.02_1", Inside, "cabinet.n.01_1", True),
            "pumpkin_2_in_cabinet": ("state", "pumpkin.n.02_2", Inside, "cabinet.n.01_1", True),
            "robot_near_caldron": ("near", "agent.n.01_1", "cauldron.n.01_1"),
            "caldron_picked_up": ("state", "cauldron.n.01_1", OnTop, "floor.n.01_1", False),
            "cauldron_next_to_table": ("state", "cauldron.n.01_1", NextTo, "table.n.02_1", True),
        },
    ),
    # TODO: add more tasks here
}
