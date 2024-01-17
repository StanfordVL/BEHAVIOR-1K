"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import numpy as np
import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController


CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


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

    # Create the config for generating the environment we want
    cfg = yaml.safe_load(open("omni_grpc.yaml", "r"))

    # # Add the robot we want to load
    # robot0_cfg = dict()
    # robot0_cfg["type"] = robot_name
    # robot0_cfg["obs_modalities"] = ["rgb", "depth", "seg_instance", "normal", "scan", "occupancy_grid"]
    # robot0_cfg["action_type"] = "continuous"
    # robot0_cfg["action_normalize"] = True

    # # Create the environment
    env = og.Environment(configs=cfg)

    # # Choose robot controller to use
    robot = env.robots[0]
    # controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # # Choose control mode
    control_mode = "teleop"
    # if random_selection:
    #     control_mode = "random"
    # else:
    #     control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # # Update the control mode of the robot
    # controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    # robot.reload_controllers(controller_config=controller_config)

    # # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # # is preserved
    # env.scene.update_initial_state()

    # Update the simulator's viewer camera's pose so it points towards the robot
    # og.sim.viewer_camera.set_position_orientation(
    #     position=np.array([1.46949, -3.97358, 2.21529]),
    #     orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    # )

    # Reset environment and robot
    env.reset()
    robot.reset()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0
    while step != max_steps:
        action = action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        _, reward, _, _, _ = env.step(action=action)
        print(reward)
        step += 1

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
