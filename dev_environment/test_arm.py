import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import detect_robot_collision, arm_planning_validity_fn
from omnigibson.utils.control_utils import FKSolver

import cProfile, pstats, io
import time
import os
import argparse

from omni.usd.commands import CopyPrimsCommand, DeletePrimsCommand, CopyPrimCommand, TransformPrimsCommand, TransformPrimCommand
from omnigibson.prims import CollisionGeomPrim
from pxr import Gf, Usd
from omni.isaac.core.utils.prims import get_prim_at_path

def pause(time):
    for _ in range(int(time*100)):
        og.sim.render()

def main():
    # Load the config
    config_filename = "test_tiago.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]
    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()
    
    # robot.tuck()
    # pause(2)

    # import random
    def get_random_joint_position():
        joint_positions = []
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
        joints = np.array([joint for joint in robot.joints.values()])
        arm_joints = joints[joint_control_idx]
        for i, joint in enumerate(arm_joints):
            val = np.random.uniform(joint.lower_limit, joint.upper_limit)
            joint_positions.append(val)
        return joint_positions
    
    def set_joint_position(joint_pos):
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
        robot.set_joint_positions(joint_pos, joint_control_idx)

    robot.tuck()
    og.sim.step()
    # while True:
    #     joint_pos = get_random_joint_position()
    #     set_joint_position(joint_pos)
    #     pause(2)
    #     print(joint_pos)
    #     print("------------------------")

    # sample_table_collision = [0.05725323620041453, 0.5163557640853469, 1.510323032160434, -4.410407307232964, -1.1433260958390707, -5.606768602222553, 1.0313821643138894, -4.181284701460742]
    # sample_self_collision = [0.03053552120088664, 1.0269865478752571, 1.1344740372495958, 6.158997020615134, 1.133466907494042, -4.544473644642829, 0.6930819484783561, 4.676661155308317]
    # collision_free = [0.17814310139520295, -0.8082173382782226, 1.3469484097869393, 1.6222072455290446, 2.0591874971218145, -2.9063608379063557, -0.04634827601286595, 4.505122702016582]
    
    #Tiago
    collision = [0.14065403286781475, 0.7298650222286143, -0.4780975016232605, 1.0888713731557247, -0.03729107004351029, 3.274825625013916, 1.2937221767307419, 1.8178545818287346, 0.269868125704424, -1.1858447020249343, 1.079587475865726, 0.11286700467163624, -1.3232706255151934, 0.3340342084010399, 1.4264938203455721]
    no_collision = [0.3184793294422698, 1.5142631693122297, 1.2405191873837995, 0.21394545305074741, -0.6831575211130013, -0.7389958913494964, 2.725925427761072, 1.1755425218590514, 0.6547571278019166, 0.7133282478444771, 0.5163046628994854, -1.0098026849625532, 0.09229435376315243, 1.9379299096653453, 2.0534229844998677]

    # set_joint_position(no_collision)
    # pause(100)
    # pause(100)
    # pos = np.array([
    #             0.0,  
    #             0.0, 
    #             0.0, 
    #             0.0,
    #             0.0, 
    #             0.0,
    #             0.1, # trunk
    #             -1.1,
    #             -1.1,  
    #             0.0,  
    #             1.47,  
    #             1.47,
    #             0.0,  
    #             2.71,  
    #             2.71,  
    #             1.71,
    #             1.71, 
    #             -1.57, 
    #             -1.57,  
    #             1.39,
    #             1.39,  
    #             0.0,  
    #             0.0,  
    #             0.045,
    #             0.045,  
    #             0.045,  
    #             0.045,
    #         ])
    joint_pos = np.array(
            [
                0.0,  
                0.0, 
                0.0, 
                0.0,
                0.0, 
                0.0,
                0.1, # trunk
                -1.1,
                -1.1,  
                0.0,  
                1.47,  
                1.47,
                0.0,  
                2.71,  
                2.71,  
                1.71,
                1.71, 
                -1.57, 
                -1.57,  
                1.39,
                1.39,  
                0.0,  
                0.0,  
                0.045,
                0.045,  
                0.045,  
                0.045,
            ]
        )
    
    
    
    pose_2d = [0.299906, -0.918024, 2.94397]

    pos = np.array([pose_2d[0], pose_2d[1], 0.05])
    orn = T.euler2quat([0, 0, pose_2d[2]])

    # robot.set_position_orientation([-1.08215380e+00, -3.35281938e-01, -2.77837131e-07], [ 1.78991655e-07, -4.65450078e-08, -2.67762393e-01,  9.63485003e-01])
    # robot.set_position_orientation(pos, orn)
    # og.sim.step()
    positions = [[0.09640930593013763, -1.0999783277511597, 1.470136046409607, 2.7100629806518555, 1.710019826889038, -1.5699725151062012, 1.3899997472763062, -2.2541275939147454e-06], [0.11789778377430027, 1.57079632679, -0.00844635003731165, 1.6066719362098862, 0.4473218694991109, 0.019161401102889112, -1.2949643256956296, -1.9651135606361847]]
    
    joint_combined_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
    joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    control_idx_in_joint_pos = np.where(np.in1d(joint_combined_idx, joint_control_idx))[0]
    joint_pos = joint_pos[joint_combined_idx]
    joint_pos[control_idx_in_joint_pos] = [0.263221, 1.51202 ,0.277794 ,1.5376, 0.852972, -0.23253 ,-1.41372 ,1.55155]
    # print(pos)
    # robot.set_joint_positions([0.263221, 1.51202 ,0.277794 ,1.5376, 0.852972, -0.23253 ,-1.41372 ,1.55155], joint_control_idx)
    # pause(100)
    # from omnigibson.controllers.controller_base import ControlType
    # action = np.zeros(robot.action_dim)
    # for name, controller in robot._controllers.items():
    #     joint_idx = controller.dof_idx
    #     action_idx = robot.controller_action_idx[name]
    #     if controller.control_type == ControlType.POSITION and len(joint_idx) == len(action_idx):
    #         action[action_idx] = robot.get_joint_positions()[joint_idx]
    # action[robot.controller_action_idx["arm_left"]] = positions[1]
    # for p in positions:
    #     robot.set_joint_positions(p, joint_control_idx)
    #     og.sim.step()
    #     pause(2)
    # with UndoableContext(robot, "arm") as context: 
    #     while True:
    #         jp = np.array(robot.get_joint_positions()[joint_combined_idx])
    #         print(not arm_planning_validity_fn(context, jp))
    #         env.step(action)

#     joint_pos = [-1.0821538e+00, -3.3528194e-01, -2.7783713e-07,  3.8149398e-07,
#  -2.0263911e-07, -5.4213971e-01 , 1.1684235e-01 , 1.5707957e+00,
#  -1.0999898e+00 ,-2.8282230e-07 , 5.6211746e-01 , 1.4701120e+00,
#   1.0374244e-08 , 1.6099000e+00 , 2.6821127e+00 , 4.4674629e-01,
#   1.8163613e+00 ,-2.2369886e-02, -1.5652136e+00, -1.2442690e+00,
#   1.3900158e+00, -2.0943952e+00, -5.9008621e-06,  4.4999883e-02,
#   4.5000002e-02,  4.4999868e-02,  4.5000002e-02]
#     joint_pos = np.array(joint_pos)[joint_combined_idx]
    # with UndoableContext(robot, "arm") as context:        
    #     print(not arm_planning_validity_fn(context, joint_pos))
    #     for i in range(10000):
    #         og.sim.render()

    with UndoableContext(robot, "base") as context:        
        print(detect_robot_collision(context, [pos, orn]))
        for link in context.robot_meshes_copy:
            for mesh in context.robot_meshes_copy[link]:
                mesh.collision_enabled = True
        for i in range(10000):
            og.sim.render()
    
    # breakpoint()

if __name__ == "__main__":
        main()