import os

import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

from omnigibson.utils.motion_planning_utils import write_to_file

def pause(time):
    for _ in range(int(time*100)):
        # og.sim.render()
        og.sim.step()

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

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

    controller = StarterSemanticActionPrimitives(None, scene, robot)
    
    def test_navigate_to_obj():
        ctrl_navigate = controller._navigate_to_obj(table)
        execute_controller(ctrl_navigate, env)

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
        ctrl_grasp = controller.grasp(grasp_obj)
        robot.set_position([0.0, -0.5, 0.05])
        robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
        og.sim.step()
        execute_controller(ctrl_grasp, env)

    def test_full_grasp():
        ctrl_grasp = controller.grasp(grasp_obj)
        execute_controller(ctrl_grasp, env)

    def test_place():
        test_grasp()
        pause(1)
        ctrl_grasp = controller.place_on_top(table)
        execute_controller(ctrl_grasp, env)

    test_place()
    # test_full_grasp()
    # write_to_file({"-----------": "-----------"})

if __name__ == "__main__":
    main()
