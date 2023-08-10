import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omni.physx.scripts import utils
from omnigibson.utils.motion_planning_utils import (
    plan_base_motion,
    plan_arm_motion,
    detect_robot_collision,
    detect_robot_collision_in_sim,
    arm_planning_validity_fn
)

import cProfile, pstats, io
import time
import os
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
        # print(env.robots[0].get_joint_positions())
        # break
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def main():
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # config["scene"]["not_load_object_categories"] = ["sofa", "carpet"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    table = DatasetObject(
        name="table",
        category="breakfast_table",
        model="rjgmmy",
        scale = 0.3
    )
    og.sim.import_object(table)
    table.set_position([1.0, 1.0, 0.58])

    grasp_obj = DatasetObject(
        name="potato",
        category="cologne",
        model="lyipur",
        scale=0.01
    )

    og.sim.import_object(grasp_obj)
    grasp_obj.set_position([-0.3, -0.8, 0.5])
    og.sim.step()

    controller = StarterSemanticActionPrimitives(None, scene, robot)

    # Need to set start pose because default tuck pose for Fetch collides with itself
    def set_start_pose():
        default_pose = np.array(
            [
                0.0,
                0.0,  # wheels
                0.0,  # trunk
                0.0,
                -1.0,
                0.0,  # head
                -1.0,
                1.53448,
                2.2,
                0.0,
                1.36904,
                1.90996,  # arm
                0.05,
                0.05,  # gripper
            ]
        )
        robot.set_joint_positions(default_pose)
        og.sim.step()
    
    robot.tuck()
    og.sim.step()

    def test_navigate_to_obj():
        set_start_pose()
        execute_controller(controller._navigate_to_obj(table), env)

    def test_grasp_no_navigation():
        # set_start_pose()
        robot.set_position([0.0, -0.5, 0.05])
        robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
        og.sim.step()
        execute_controller(controller._grasp(grasp_obj), env)

    def test_grasp():
        set_start_pose()
        execute_controller(controller._grasp(grasp_obj), env)

    def test_place():
        test_grasp()
        pause(1)
        execute_controller(controller.place_on_top(table), env)

    def test_grasp_replay_and_place():
        set_start_pose()
        robot.set_position([0.0, -0.5, 0.05])
        robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
        og.sim.step()
        controller._fix_robot_base()
        replay_controller(env, "./replays/grasp.yaml")
        controller._unfix_robot_base()
        execute_controller(controller.place_on_top(table), env)

    # Work more reliably
    # test_navigate_to_obj()
    # test_grasp_no_navigation()
    # test_grasp_replay_and_place()

    # Don't work as reliably
    # test_grasp()
    # test_place()

    # test_grasp_no_navigation()
    # test_grasp()
    # test_grasp_replay_and_place()

#     joint_pos = np.array([-2.7455899e-03,  2.6479384e-02,  9.1548145e-02,  1.2876080e-01,
#   1.1599542e-01, -6.4017418e-05 , 1.2593279e+00 ,-1.6949688e+00,
#   1.3330945e+00 , 2.5068374e+00 , 2.9042342e-01 , 2.3277307e+00,
#   4.9997143e-02 , 5.0000001e-02]
# )
#     control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["0"]])

#     robot.set_joint_positions(joint_pos)
#     robot.set_position([0.0, -0.5, 0.05])
#     robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
#     og.sim.step()
#     with UndoableContext(robot, "arm") as context:        
#         print(not arm_planning_validity_fn(context, joint_pos[control_idx]))
#         pause(100)
    
#     pause(10)

    # execute_controller(controller._navigate_to_pose([0.0, 1.0, 0]), env)
    pause(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test script")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If set, profile code and generate .prof file",
    )
    args = parser.parse_args()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()
        s = io.StringIO()
        results = pstats.Stats(pr)
        filename = f'profile-{os.path.basename(__file__)}-{time.strftime("%Y%m%d-%H%M%S")}'
        results.dump_stats(f"./profiles/{filename}.prof")
        os.system(f"flameprof ./profiles/{filename}.prof > ./profiles/{filename}.svg")
        # Run `snakeviz ./profiles/<filename>.prof` to visualize stack trace or open <filename>.svg in a browser
    else:
        main()