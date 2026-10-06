import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.object_states import Draped
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import multi_dim_linspace
from omnigibson.utils.ui_utils import KeyboardRobotController

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can be draped on a hanger.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene + cloth object + hanger + table + rack + robot
    cfg = {
        "env": {
            "external_sensors": [
                {
                    "sensor_type": "VisionSensor",
                    "name": "helper_camera",
                    "relative_prim_path": "/external_sensor0",
                    "modalities": ["rgb"],
                    "sensor_kwargs": {
                        "image_height": 1024,
                        "image_width": 512,
                    },
                    "position": [1.7426, -0.0601, 1.4616],
                    "orientation": [0.4808, 0.3535, 0.4753, 0.6465],
                    "pose_frame": "parent",
                }
            ],
        },
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "pants",
                "category": "pants",
                "model": "tnirgd",
                "prim_type": PrimType.CLOTH,
                "abilities": {"cloth": {}},
                "position": [100.0, 100.0, 100.0],
                "orientation": [-0.5000, -0.5000, 0.5000, 0.5000],
                "scale": [0.6, 0.6, 0.6],
            },
            {
                "type": "DatasetObject",
                "name": "hanger",
                "category": "hanger",
                "model": "agrpio",
                "prim_type": PrimType.RIGID,
                "position": [110.0, 100.0, 100.0],
                "orientation": [0.0000, 0.0000, 0.7071, 0.7071],
                "scale": [2.0, 2.0, 2.0],
            },
            {
                "type": "DatasetObject",
                "name": "breakfast_table",
                "category": "breakfast_table",
                "model": "skczfi",
                "prim_type": PrimType.RIGID,
                "position": [0.0, 0.0, 0.9],
                "fixed_base": True,
                "scale": [1.0, 1.5, 2.5],
            },
            {
                "type": "DatasetObject",
                "name": "rack",
                "category": "drying_rack",
                "model": "rygebd",
                "prim_type": PrimType.RIGID,
                "position": [1.0, -0.3, 0.75],
                "orientation": [0.0, 0.0, 0.7071, 0.7071],
                "fixed_base": True,
            },
        ],
        "robots": [
            {
                "type": "R1",
                "obs_modalities": [],
                "position": [-0.15, 0.8, 0.0],
                "orientation": [0.0, 0.0, -0.7071, 0.7071],
                "grasping_mode": "sticky",
                "controller_config": {
                    "arm_left": {
                        "name": "OperationalSpaceController",
                    },
                    "arm_right": {
                        "name": "OperationalSpaceController",
                    },
                },
                "action_type": "continuous",
                "action_normalize": True,
            }
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Grab object references
    breakfast_table = env.scene.object_registry("name", "breakfast_table")
    hanger = env.scene.object_registry("name", "hanger")
    pants = env.scene.object_registry("name", "pants")
    rack = env.scene.object_registry("name", "rack")
    R1 = env.scene.robots[0]

    # TODO: fix this when we have a better way to deal with gravity compensation
    hanger.root_link.density = 1.0
    pants.root_link.mass = 0.01

    # Set camera pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.8550, 1.5565, 1.8658]),
        orientation=th.tensor([0.1224, 0.5098, 0.8280, 0.1989]),
    )

    action_generator = KeyboardRobotController(robot=R1)
    action_generator.print_keyboard_teleop_info()

    # open the gripper
    open_action = action_generator.get_teleop_action()
    open_action[R1.controller_action_idx["gripper_left"]] = 1.0

    for _ in range(10):
        pants.keep_still()
        hanger.keep_still()
        env.step(open_action)

    # teleport the objects
    pants.set_position_orientation([0.1861, 0.1940, 1.2226], [-0.5000, -0.5000, 0.5000, 0.5000])
    hanger.set_position_orientation([0.2579, 0.2352, 1.2843], [0.0000, 0.0000, 0.7071, 0.7071])

    # close the gripper
    close_action = action_generator.get_teleop_action()
    close_action[R1.controller_action_idx["gripper_left"]] = -1.0

    for _ in range(10):
        env.step(close_action)

    print("Running demo, try to put the hanger onto the rack. ")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    while step != max_steps:
        action = action_generator.get_teleop_action()
        # Make sure R1 doesn't release the hanger
        action[R1.controller_action_idx["gripper_left"]] = -1.0
        env.step(action=action)
        step += 1

    # Shut down env at the end
    og.clear()


if __name__ == "__main__":
    main()
