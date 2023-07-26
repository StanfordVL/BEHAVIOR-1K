"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import os
import numpy as np
import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController


# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False


def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    for component, controller_options in default_config.items():
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options, name="{} controller".format(component), random_selection=random_selection
        )

        # Add to user responses
        controller_choices[component] = choice

    return controller_choices


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Compile config
    config_filename = os.path.join(og.example_config_path, "homeboy.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config["robots"][0]["action_normalize"] = True

    # Create the environment
    env = og.Environment(configs=config, action_timestep=1/60., physics_timestep=1/60.)

    # Choose robot controller to use
    robot = env.robots[0]
    # controller_choices = choose_controllers(robot=robot, random_selection=random_selection)
    # controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    # robot.reload_controllers(controller_config=controller_config)

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment
    env.reset()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    while True:
        action = action_generator.get_teleop_action()
        for _ in range(10):
            env.step(action=action)
            step += 1

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
