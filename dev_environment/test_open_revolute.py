import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext, StarterSemanticActionPrimitiveSet
from omnigibson.objects.primitive_object import PrimitiveObject
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.grasping_planning_utils import get_grasp_position_for_open

import cProfile, pstats, io
import time
import os
import argparse

gm.DEBUG = True

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
        env.step(action[0])
    #     actions.append(action.tolist())
    # if filename is not None:
    #     with open(filename, "w") as f:
    #         yaml.dump(actions, f)

def main():
    # Load the config
    config_filename = "test_IK.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]
    og.sim.stop()
    og.sim.play()

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = StarterSemanticActionPrimitives(None, scene, robot)

    open_obj = DatasetObject(
        name="fridge",
        category="fridge",
        model="dszchb",
        scale=[0.7, 0.7, 0.7]
    )

    # open_obj = DatasetObject(
    #     name="bottom_cabinet",
    #     category="bottom_cabinet",
    #     model="bamfsz"
    # )

    og.sim.import_object(open_obj)
    # open_obj.set_position([-1.2, -0.4, 0.5])
    open_obj.set_position_orientation([-1.2, -0.4, 0.5], T.euler2quat([0, 0, np.pi/2]))
    og.sim.step()
    

    def set_start_pose():
        reset_pose_tiago = np.array([
            -1.78029833e-04,  3.20231302e-05, -1.85759447e-07, -1.16488536e-07,
            4.55182843e-08,  2.36128806e-04,  1.50000000e-01,  9.40000000e-01,
            -1.10000000e+00,  0.00000000e+00, -0.90000000e+00,  1.47000000e+00,
            0.00000000e+00,  2.10000000e+00,  2.71000000e+00,  1.50000000e+00,
            1.71000000e+00,  1.30000000e+00, -1.57000000e+00, -1.40000000e+00,
            1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
            4.50000000e-02,  4.50000000e-02,  4.50000000e-02,
        ])
        robot.set_joint_positions(reset_pose_tiago)
        og.sim.step()

    def test_open_no_navigation():
        # Need to set start pose to reset_hand because default tuck pose for Tiago collides with itself
        set_start_pose()
        pose = controller._get_robot_pose_from_2d_pose([-1.0, -0.5, np.pi/2])
        robot.set_position_orientation(*pose)
        og.sim.step()
        get_grasp_position_for_open(robot, open_obj, True)
        # execute_controller(controller._open_or_close(cabinet), env)

    def test_open():
        set_start_pose()
        # pose_2d = [-0.231071, -0.272773, 2.55196]

        pose_2d = [-0.282843, 0.0, -3.07804]
        pose = controller._get_robot_pose_from_2d_pose(pose_2d)
        robot.set_position_orientation(*pose)
        og.sim.step()

        # joint_pos = [0.0133727 ,0.216775 ,0.683931 ,2.04371 ,1.88204 ,0.720747 ,1.23276 ,1.72251]
        # control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["left"]])
        # robot.set_joint_positions(joint_pos, control_idx)
        # og.sim.step()
        # pause(100)
        # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.OPEN, *(open_obj,)), env)
        execute_controller(controller._open_or_close(open_obj, True), env)

    markers = []
    for i in range(20):
        marker = PrimitiveObject(
            prim_path=f"/World/test_{i}",
            name=f"test_{i}",
            primitive_type="Cube",
            size=0.07,
            visual_only=True,
            rgba=[1.0, 0, 0, 1.0])
        markers.append(marker)
        og.sim.import_object(marker)
    
    # from omnigibson.object_states.open import _get_relevant_joints
    controller.markers = markers
    # j = _get_relevant_joints(open_obj)[1][0]
    # j.set_pos(0.5)
    # pause(2)
    # open_obj.joints["j_link_2"].set_pos(0.4)
    og.sim.step()
    test_open()
    # pause(5)

if __name__ == "__main__":
    main()



