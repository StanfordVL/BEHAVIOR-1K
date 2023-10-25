"""
Example script demo'ing robot control to solve a task

Options for keyboard control to solve task or programmtic action primitive
"""
import os
import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.symbolic_semantic_action_primitives import SymbolicSemanticActionPrimitives, SymbolicSemanticActionPrimitiveSet
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController


# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

def main(keyboard_control=False):
    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    if keyboard_control:
        config["robots"][0]["action_normalize"] = True

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    if keyboard_control:
        action_generator = KeyboardRobotController(robot=robot)
        action_generator.print_keyboard_teleop_info()
        action = action_generator.get_teleop_action()
        print("Running demo.")
        print("Press ESC to quit")
        for _ in range(10):
            env.step(action=action)
    else:
        controller = SymbolicSemanticActionPrimitives(env)
        # Grasp bottle of vodka
        grasp_obj, = scene.object_registry("category", "bottle_of_vodka")
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, grasp_obj), env)

        # place bottle of vodka in box
        box, = scene.object_registry("category", "storage_box")
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_INSIDE, box), env)

if __name__ == "__main__":
    main()