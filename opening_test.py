
import os

import yaml
import numpy as np
import random

import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_og_avg_category_specs,
    get_object_models_of_category,
)
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
import omnigibson.utils.transform_utils as T
from omnigibson.objects import PrimitiveObject
from omnigibson.object_states import ContactBodies
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import detect_robot_collision
from omnigibson.utils.grasping_planning_utils import get_grasp_position_for_open

def pause(time):
    for _ in range(int(time*100)):
        # og.sim.render()
        og.sim.step()

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

def place_marker(pos):
    marker = PrimitiveObject(
        prim_path=f"/World/marker",
        name="marker",
        primitive_type="Cube",
        size=0.07,
        visual_only=True,
        rgba=[1.0, 0, 0, 1.0],
    )
    og.sim.import_object(marker)
    marker.set_position(pos)
    og.sim.step()

def main(random_selection=False, headless=False, short_exec=False):
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table", "bottom_cabinet"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    for o in scene.objects:
        if o.prim_path == "/World/bottom_cabinet_bamfsz_0":
            cabinet = o

    # marker = PrimitiveObject(
    #     prim_path=f"/World/marker",
    #     name="marker",
    #     primitive_type="Cube",
    #     size=0.07,
    #     visual_only=True,
    #     rgba=[1.0, 0, 0, 1.0],
    # )
    # og.sim.import_object(marker)
    # marker.set_position([-0.3, -0.8, 0.5])
    # og.sim.step()

    # from IPython import embed; embed()

    controller = StarterSemanticActionPrimitives(None, scene, robot)
    robot.set_position([0.0, -0.5, 0.05])
    robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
    start_joint_pos = np.array(
        [
            0.0,
            0.0,  # wheels
            0.2,  # trunk
            0.0,
            1.1707963267948966,
            0.0,  # head
            1.4707963267948965,
            -0.4,
            1.6707963267948966,
            0.0,
            1.5707963267948966,
            0.0,  # arm
            0.05,
            0.05,  # gripper
        ]
    )
    robot.set_joint_positions(start_joint_pos)
    og.sim.step()

    grasp_data = get_grasp_position_for_open(robot, cabinet, True)
    grasp_pose, target_poses, object_direction, joint_info, grasp_required = grasp_data
    place_marker(grasp_pose[0])
    # execute_controller(controller._move_hand_direct_cartesian(grasp_pose), env)
    # execute_controller(controller._execute_grasp(), env)
    pause(10)


if __name__ == "__main__":
    main()