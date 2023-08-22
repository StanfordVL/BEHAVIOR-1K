import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
from omnigibson.objects.primitive_object import PrimitiveObject
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

import argparse
    

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)

def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action.tolist())
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def main():
    # Load the config
    config_filename = "test_tiago.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    grasp_obj = DatasetObject(
        name="cologne",
        category="cologne",
        model="lyipur",
        scale=[0.01, 0.01, 0.01]
    )
    og.sim.import_object(grasp_obj)
    grasp_obj.set_position([-0.3, -0.8, 0.5])
    og.sim.step()

    pause(2)
    controller = StarterSemanticActionPrimitives(None, scene, robot)

    def set_start_pose():
        control_idx = np.concatenate([robot.trunk_control_idx])
        robot.set_joint_positions([0.1], control_idx)
        og.sim.step()

    def test_grasp_no_navigation():
        # Need to set start pose to reset_hand because default tuck pose for Tiago collides with itself
        set_start_pose()
        pose = controller._get_robot_pose_from_2d_pose([-0.433881, -0.230183, -1.87])
        robot.set_position_orientation(*pose)
        og.sim.step()
        execute_controller(controller._grasp(grasp_obj), env)
    
    test_grasp_no_navigation()

if __name__ == "__main__":
    main()



