#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:    Ji Yingwei
Created:   2025-11-20
Description:
    Example script for interacting with OmniGibson scenes with VR, pink ik and BehaviorRobot.
"""


import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.teleop_utils2 import OVXRSystem
from omnigibson.utils.constants import PrimType
import torch as th

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
# gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = True

CONTROLLER_VISIBLE = False
ENABLE_CAMERA_LIMITS = False


def main():
    """
    Spawn a BehaviorRobot in Rs_int and users can navigate around and interact with the scene using VR.
    """
    # Create the config for generating the environment we want
    cfg={
        "scene": {
            "type": "Scene",
        },
        # "objects": [
        #         {
        #             "type": "DatasetObject",
        #             "name": "shirt",
        #             "category": "t_shirt",
        #             "model": "kvidcx",
        #             # "bounding_box": [0.472, 1.243, 1.158],
        #             "prim_type": PrimType.CLOTH,
        #             "abilities": {"cloth": {}},
        #             "position": [1, 1, 2.0],
        #             "scale":[0.5,0.5,0.5]
        #         },
        #         {
        #             "type": "DatasetObject",
        #             "name": "breakfast_table",
        #             "category": "breakfast_table",
        #             "model": "rjgmmy",
        #             "bounding_box": [1.36, 1.081, 0.84],
        #             "prim_type": PrimType.RIGID,
        #             "position": [1, 1, 0.58],
        #         },
        #     ],
    
        # "scene": {
        #     "type": "InteractiveTraversableScene",
        #     "scene_model": "house_single_floor",
        #     "load_room_types": [ "kitchen"]
        #     },

        "robots":[
            {
                "type": "R1",
                "obs_modalities": ["rgb"],
                "controller_config": {
                    "base":{
                        "name": "HolonomicBaseJointController",
                        # "mode": "absolute_pose",
                        # "use_delta_commands":False,
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    "trunk":{
                        "name": "JointController",
                        # "mode": "absolute_pose",
                        "use_delta_commands":False,
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    "arm_left": {
                        "name": "JointController",
                        # "mode": "absolute_pose",
                        "use_delta_commands":False,
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    "arm_right": {
                        "name": "JointController",
                        # "mode": "absolute_pose",
                        "use_delta_commands":False,
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    "gripper_left": {"name": "MultiFingerGripperController", "command_input_limits": "default"},
                    "gripper_right": {"name": "MultiFingerGripperController", "command_input_limits": "default"},
                },
                "action_normalize": False,
                "reset_joint_pos": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.8000,
                    -0.8000,
                    0.0000,
                    -0.0068,
                    0.0059,
                    2.6054,
                    2.5988,
                    -1.4515,
                    -1.4478,
                    -0.0065,
                    0.0052,
                    1.5670,
                    -1.5635,
                    -1.1428,
                    1.1610,
                    0.0087,
                    0.0087,
                    0.0087,
                    0.0087,
                ]
            }
            ],
        # "task":{
        #     "type": "BehaviorTask",
        #     "activity_name": "putting_dishes_away_after_cleaning",
        #     "activity_definition_id": 0,
        #     "activity_instance_id": 0,
        #     "predefined_problem": None,
        #     "online_object_sampling": False,
        #     "debug_object_sampling": None,
        #     "highlight_task_relevant_objects": False,
        #     "use_presampled_robot_pose": True
        # }
    }


    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()

    robot = env.robots[0]
    # robot._teleop_rotation_offset = {
    #     "left": th.tensor([-0.7071, 0.7071, 0, 0], dtype=th.float32),
    #     "right": th.tensor([-0.7071, 0.7071, 0, 0], dtype=th.float32),
    # }
    # # Then override the property
    # def teleop_rotation_offset(self):
    #     return self._teleop_rotation_offset
    # type(robot).teleop_rotation_offset = property(teleop_rotation_offset)

    # start vrsys
    vrsys = OVXRSystem(
        robot=robot,
        show_control_marker=CONTROLLER_VISIBLE,
        # disable_display_output=True,
        system="SteamVR",
        eef_tracking_mode="controller",
        align_anchor_to="camera",
        # roll, pitch, yaw
        view_angle_limits=[180, 30, 30] if ENABLE_CAMERA_LIMITS else None,
    )
    vrsys.start()

    for _ in range(300000000):
        # update the VR system
        vrsys.update()
        # get the action from the VR system and step the environment
        action=vrsys.get_robot_teleop_action()
        # print(action)
        act_new=action.clone()#
        act1=vrsys.act1[0]

        act_new[3:7]  = th.as_tensor(act1[:4],  dtype=act_new.dtype, device=act_new.device)
        act_new[7:13] = th.as_tensor(act1[4:10], dtype=act_new.dtype, device=act_new.device)
        act_new[14:20]= th.as_tensor(act1[10:], dtype=act_new.dtype, device=act_new.device)
   
        env.step(act_new)




    print("Cleaning up...")
    vrsys.stop()
    og.shutdown()


if __name__ == "__main__":
    main()
