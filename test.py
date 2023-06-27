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

def pause(time):
    for _ in range(int(time*100)):
        # og.sim.render()
        og.sim.step()

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)
        

def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select a type of scene and loads a turtlebot into it, generating a Point-Goal navigation
    task within the environment.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table", "bottom_cabinet"]
    # config["scene"]["load_object_categories"] = None

    # config["objects"] = [obj_cfg]
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
        category="potato",
        model="lqjear"
    )
    og.sim.import_object(grasp_obj)
    grasp_obj.set_position([-0.3, -0.8, 0.5])

    for o in scene.objects:
        if o.prim_path == "/World/bottom_cabinet_bamfsz_0":
            cabinet = o

    # cabinet = DatasetObject(
    #     name="cabinet",
    #     category="bottom_cabinet",
    #     model="jhymlr"
    # )
    # og.sim.import_object(cabinest)
    # cabinet.set_position([1.0, 0.0, 0.5])
    # scene._add_object(cabinet)

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

    # marker2 = PrimitiveObject(
    #     prim_path=f"/World/marker",
    #     name="marker2",
    #     primitive_type="Sphere",
    #     radius=0.1,
    #     visual_only=True,
    #     rgba=[1.0, 0, 0, 1.0],
    # )

    # og.sim.import_object(marker2)
    # marker2.set_position([1.0, 1.0, 0.5])
    # og.sim.step()

    # from IPython import embed; embed()

    # robot.set_position([0.0, -0.5, 0.05])
    # robot.set_orientation(T.euler2quat([0, 0, -np.pi/1.5]))
    og.sim.step()

    controller = StarterSemanticActionPrimitives(None, scene, robot)
    # navigate_controller = controller._navigate_to_pose_direct([0.0, -1.0, 0.0], low_precision=True)
    

    # navigate_controller = controller._navigate_to_pose([0.5, 2.5, 0.0])
    # navigate_controller = controller._navigate_to_pose([0.0, 0.5, 0.0])
    # navigate_controller_marker = controller._navigate_to_obj(marker)
    navigate_controller_table = controller._navigate_to_obj(table)
    navigate_controller = controller._navigate_to_obj(grasp_obj)
    hand_controller = controller.grasp(grasp_obj)
    place_controller = controller.place_on_top(table)
    open_controller = controller.open(cabinet)

    # robot.untuck()
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
    # robot.tuck()
    og.sim.step()

    robot.set_position([-1.0, -0.5, 0.05])
    robot.set_orientation(T.euler2quat([0, 0, np.pi]))
    og.sim.step()

    execute_controller(open_controller, env)
    
    # while True:
    #     with UndoableContext():
    #         pose_2d = [1.21189, 0.625961, 1.87304]
    #         pose = controller._get_robot_pose_from_2d_pose(pose_2d)
    #         robot.set_position_orientation(*pose)
    #         og.sim.step()
    #         print(detect_robot_collision(robot))
    #         print("-------")
    #         pause(2)


    # pose_2d = [1.43015, 1.0724, -2.80638] 
    # pose_2d = [1.21189, 0.625961, 1.87304]
    # # print(controller._test_pose(pose_2d))
    # # pause(10)
    # pose = controller._get_robot_pose_from_2d_pose(pose_2d)
    # # hand_pose = ([1.2112058 , 0.91612303, 0.43050346], [0.7996582 , 0.29997651, 0.01271829, 0.51999913])
    # # start_joint_pos = [0.0127592, 1.17074, 1.46882, -0.399443, 1.66954, 0.0000686818, 1.57074, -0.00000181259]
    # # control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["0"]])
    # robot.set_position_orientation(*pose)
    # # robot.set_joint_positions(start_joint_pos, control_idx)
    # og.sim.step()
    # print(detect_robot_collision(robot))
    # pause(100)

    # execute_controller(navigate_controller_table, env)
    # controller._navigate_to_obj(table)
    # execute_controller(hand_controller, env)
    # pause(1)
    # execute_controller(place_controller, env)
    # pause(10)

    # test_pose = ([-0.3, 10.0, 0.05], T.euler2quat([0, np.pi/2, 0]))
    # print(controller._target_in_reach_of_robot(test_pose))
    # execute_controller(navigate_controller_table, env)

    # def test_collision(joint_pos):
    #     with UndoableContext():
    #         set_joint_position(joint_pos)
    #         og.sim.step()
    #         print(detect_robot_collision(robot))
    #         print("-------")
    #         pause(2)

    # def set_joint_position(joint_pos):
    #     joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    #     robot.set_joint_positions(joint_pos, joint_control_idx)

    # def set_random_joint_position():
    #     joint_positions = []
    #     joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    #     joints = np.array([joint for joint in robot.joints.values()])
    #     arm_joints = joints[joint_control_idx]
    #     for i, joint in enumerate(arm_joints):
    #         val = random.uniform(joint.lower_limit, joint.upper_limit)
    #         joint_positions.append(val)
    #     return joint_positions
    
    # sample_table_collision = [0.05725323620041453, 0.5163557640853469, 1.510323032160434, -4.410407307232964, -1.1433260958390707, -5.606768602222553, 1.0313821643138894, -4.181284701460742]
    # sample_self_collision = [0.03053552120088664, 1.0269865478752571, 1.1344740372495958, 6.158997020615134, 1.133466907494042, -4.544473644642829, 0.6930819484783561, 4.676661155308317]
    # while True:
    #     random_pos = set_random_joint_position()
    #     test_collision(sample_self_collision)

    # for action in navigate_controller:
    #     state, reward, done, info = env.step(action)
    #     if done:
    #         og.log.info("Episode finished after {} timesteps".format(i + 1))
    #         break

    # # Always close the environment at the end
    # env.close()


# def check_contact(obj1, obj2):
#     # One way, use the Object State API
#     from omnigibson.object_states import ContactBodies
#     # Other way, directly check for Touching
#     in_contact = obj1.states[Touching].get_value(obj2)

#     # This returns list of links obj1 is in contact with
#     in_contact = len(set(obj2.links.values()).intersection(obj1.states[ContactBodies].get_value())) > 0

#     # Last way (most fine grained)
#     contact_list = obj1.contact_list()


# def check_contact_multiple_times_and_then_revert(robot):
#     # Goal:
#     # 1. Check if robot is in contact with anything N times
#     # 2. Revert state to the initial state deterministically

#     # Pseudocode:
#     # 1. Record sim state
#     # 2. Modify some omni physics flags to "prepare" for checking
#     # 3. Run contact checks
#     # 4. Revert state
#     # 5. Revert physics flags

#     # 1.
#     state = og.sim.dump_state(serialized=False)

#     # 2.
#     # We need to (a) disable gravity, (b) disable "solving contacts", (c) make sure all objects are "kept still"
#     og.sim.set_gravity(0.0)
#     for obj in og.sim.scene.objects:
#         for link in obj.links.values():
#             PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(False)
#         obj.keep_still()

#     # 3.
#     for pos in robot_poses:
#         robot.set_position(pos)
#         check_contact(robot)

#     # 4.
#     og.sim.load_state(state, serialized=False)

#     # 5.
#     og.sim.set_gravity(9.81)
#     for obj in og.sim.scene.objects:
#         for link in obj.links.values():
#             link.set_attribute("SolveContacts", True)


if __name__ == "__main__":
    main()
