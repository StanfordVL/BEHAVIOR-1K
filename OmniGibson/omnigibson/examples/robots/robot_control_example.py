"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics for performance boost
gm.USE_GPU_DYNAMICS = False


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
    controller_names = robot.controller_order
    for controller_name in controller_names:
        controller_options = default_config[controller_name]
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options,
            name=f"{controller_name} controller",
            random_selection=random_selection,
        )

        # Add to user responses
        controller_choices[controller_name] = choice

    return controller_choices


def main(
    random_selection=False,
    headless=False,
    short_exec=False,
    quickstart=False,
    robot_name=None,
    scene_model=None,
    control_mode=None,
):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load
    if scene_model is None and quickstart:
        scene_model = "Rs_int"
    elif scene_model is None:
        scene_model = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)
    else:
        assert scene_model in SCENES, f"Unknown scene '{scene_model}'. Valid options are: {list(SCENES)}"

    # Choose robot to create
    if robot_name is None and quickstart:
        robot_name = "fetch"
    elif robot_name is None:
        robot_name = choose_from_options(
            options=list(sorted(REGISTERED_ROBOTS)), name="robot", random_selection=random_selection
        )
    else:
        assert robot_name in REGISTERED_ROBOTS, f"{robot_name} is not a registered robot."

    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["model"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)

    # Choose robot controller to use
    robot = env.robots[0]
    if quickstart or (robot_name is not None and scene_model is not None and control_mode is not None):
        controller_choices = {component: robot._default_controllers[component] for component in robot.controller_order}
    else:
        controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # Choose control mode
    if random_selection:
        control_mode = "random"
    elif quickstart:
        control_mode = control_mode or "teleop"
    elif control_mode is None:
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")
    else:
        assert control_mode in CONTROL_MODES, (
            f"Unknown control mode '{control_mode}'. Valid options are: {list(CONTROL_MODES)}"
        )

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_file()

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.46949, -3.97358, 2.21529]),
        orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment and robot
    env.reset()
    robot.reset()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    if len(action_generator.ik_arms) > 0:
        print(f"Detected IK arms: {action_generator.ik_arms}. Use 3/4 or Z/X to switch active arm.")

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    if not og.sim.is_playing():
        og.sim.play()

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    random_action = None
    while step != max_steps:
        if not og.sim.is_playing():
            og.sim.play()

        if control_mode == "random":
            # Sample new random action every 30 steps
            if step % 30 == 0:
                random_action = action_generator.get_random_action() * 0.05
            action = random_action
        else:
            action = action_generator.get_teleop_action()

        env.step(action=action)
        step += 1

    # Always shut down the environment cleanly at the end
    og.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teleoperate a robot in a BEHAVIOR scene.")

    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Whether the example should be loaded with default settings for a quick start.",
    )
    parser.add_argument(
        "--robot",
        default=None,
        help="Robot model to load. If omitted, the example prompts for a robot unless --quickstart is used.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        choices=sorted(SCENES),
        help="Scene to load. If omitted, the example prompts for a scene unless --quickstart is used.",
    )
    parser.add_argument(
        "--control-mode",
        default=None,
        choices=sorted(CONTROL_MODES),
        help="Control mode to use. If omitted, the example prompts for a mode unless --quickstart is used.",
    )
    parser.add_argument(
        "--short-exec",
        action="store_true",
        help="Run only a short fixed-length rollout, useful for smoke testing.",
    )
    args = parser.parse_args()
    main(
        quickstart=args.quickstart,
        short_exec=args.short_exec,
        robot_name=args.robot,
        scene_model=args.scene,
        control_mode=args.control_mode,
    )
