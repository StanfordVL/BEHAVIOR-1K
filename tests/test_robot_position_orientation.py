import pytest

import omnigibson as og
from omnigibson.macros import gm


def test_scene_graph():

    if og.sim is None:
        # Set global flags
        gm.ENABLE_OBJECT_STATES = True
    else:
        # Make sure sim is stopped
        og.sim.stop()

    # Define the environment configuration
    config = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "type": "BehaviorRobot",
                "obs_modalities": "all",
                "position": [1, 1, 1],
                "orientation": [0, 0, 0, 1],
                "controller_config": {
                    "arm_0": {
                        "name": "NullJointController",
                        "motor_type": "position",
                    },
                },
            }
        ],
    }

    env = og.Environment(configs=config)

    robot = og.sim.scenes[0].robots[0]
    breakpoint()
    robot.reset()


if __name__ == "__main__":
    test_scene_graph()
