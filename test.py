import os

import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

from omnigibson.utils.motion_planning_utils import write_to_file, detect_self_collision
    

def pause(time):
    for _ in range(int(time*100)):
        # og.sim.render()
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
        if detect_self_collision(env.robots[0]):
            pause(2)
            print("collision detected")
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def main():
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

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
    og.sim.step()

    controller = StarterSemanticActionPrimitives(None, scene, robot)
    
    def test_navigate_to_obj():
        execute_controller(controller._navigate_to_obj(table), env)

    def test_grasp():
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
        robot.set_position([0.0, -0.5, 0.05])
        robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
        og.sim.step()
        execute_controller(controller.grasp(grasp_obj), env)

    def test_full_grasp():
        execute_controller(controller.grasp(grasp_obj), env)

    def test_place():
        test_grasp()
        pause(1)
        execute_controller(controller.place_on_top(table), env)

    # start_joint_pos = np.array(
    #     [
    #         0.0,
    #         0.0,  # wheels
    #         0.2,  # trunk
    #         0.0,
    #         1.1707963267948966,
    #         0.0,  # head
    #         1.4707963267948965,
    #         -0.4,
    #         1.6707963267948966,
    #         0.0,
    #         1.5707963267948966,
    #         0.0,  # arm
    #         0.05,
    #         0.05,  # gripper
    #     ]
    # )
    # robot.set_joint_positions(start_joint_pos)
    # robot.set_position([0.0, -0.5, 0.05])
    # robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
    # pause(1)
    # replay_controller(env, "grasp.yaml")
    # replay_controller(env, "place.yaml")
    # execute_controller(controller._execute_release(), env)
    # pause(10)
    # test_place()

    def test_collision(joint_pos):
        with UndoableContext():
            set_joint_position(joint_pos)
            og.sim.step()
            print(detect_self_collision(robot))
            print("-------")
            pause(2)

    def set_joint_position(joint_pos):
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        robot.set_joint_positions(joint_pos, joint_control_idx)

    sample_self_collision = [0.03053552120088664, 1.0269865478752571, 1.1344740372495958, 6.158997020615134, 1.133466907494042, -4.544473644642829, 0.6930819484783561, 4.676661155308317]
    while True:
        test_collision(sample_self_collision)
        # set_joint_position(sample_self_collision)
        # og.sim.step()



if __name__ == "__main__":
    main()
