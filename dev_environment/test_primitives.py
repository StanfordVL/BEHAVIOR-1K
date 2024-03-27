import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import set_arm_and_detect_collision, set_base_and_detect_collision

import cProfile, pstats, io
import time
import os
import argparse

def pause(time):
    for _ in range(int(time*100)):
        og.sim.render()

def pause_step(time):
    for _ in range(int(time*100)):
        og.sim.step()

def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action)
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)

def main():
    # Load the config
    config_filename = "fetch_grasp.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    # robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # grasp_obj = DatasetObject(
    #     name="cologne",
    #     category="bottle_of_cologne",
    #     model="lyipur"
    # )
    # og.sim.import_object(grasp_obj)
    # grasp_obj.set_position([-0.3, -0.8, 0.5])
    # og.sim.step()
    

    # controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    def set_start_pose():
        # reset_pose_tiago = np.array([
        #     -1.78029833e-04,  3.20231302e-05, -1.85759447e-07, -1.16488536e-07,
        #     4.55182843e-08,  2.36128806e-04,  1.50000000e-01,  9.40000000e-01,
        #     -1.10000000e+00,  0.00000000e+00, -0.90000000e+00,  1.47000000e+00,
        #     0.00000000e+00,  2.10000000e+00,  2.71000000e+00,  1.50000000e+00,
        #     1.71000000e+00,  1.30000000e+00, -1.57000000e+00, -1.40000000e+00,
        #     1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
        #     4.50000000e-02,  4.50000000e-02,  4.50000000e-02,
        # ])
        # robot.set_joint_positions(reset_pose_tiago)
        pose = controller._get_robot_pose_from_2d_pose([0.3, -0.6, np.pi])
        robot.set_position_orientation(*pose)
    
        
        # joint_pos = [0.18195947259189682,
        #     -0.16227011792488005,
        #     -0.01629889539052818,
        #     1.4345131159531648,
        #     1.4604627465341495,
        #     -1.5426523883381118,
        #     1.4373635452237794,
        #     4.381678886700396]

        # joint_pos = [0.18118853795707054, -0.19147231822670935, 0.028304982732447265, 1.4356467035592764, 1.3324823185895067, -1.5425956205806692, 1.4446004667004895, 4.277606296262872]

        # joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        # robot.set_joint_positions(joint_pos, joint_control_idx)
        # print(joint_pos, joint_control_idx)
        og.sim.step()


    # set_start_pose()
    # from IPython import embed; embed()
    # execute_controller(controller._navigate_to_pose([-0.5, 0.5, 0.0]), env)
    # for i in range(100):
    #     og.sim.step()

    # from IPython import embed; embed()
    pause_step(100)
    # execute_controller(controller._grasp(grasp_obj), env)
    # from IPython import embed; embed()
    # pause_step(10)
    # while True:
    #     action = np.array([ 0.        ,  0.        ,  0.24733247,  0.45000017,  0.18195947259189682,
    #         -0.16227011792488005,
    #         -0.01629889539052818,
    #         1.4345131159531648,
    #         1.4604627465341495,
    #         -1.5426523883381118,
    #         1.4373635452237794,
    #         4.381678886700396,  0.        ])
    #     env.step(action)
    # pause(10)

if __name__ == "__main__":
    main()