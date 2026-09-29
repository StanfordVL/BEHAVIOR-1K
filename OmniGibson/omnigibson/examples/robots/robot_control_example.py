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
    renderer="kit",
    output=None,
    num_steps=100,
):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)

    :param renderer: "kit" (default, interactive Omniverse viewport) or "rtx"/"gl" to run physics on
        Newton (Kit-free) instead. Newton physics and a live Kit render must not share a process
        (see docs/other/newton_migration.md's known-workarounds section), so "rtx"/"gl" force
        PHYSICS_BACKEND="newton" and RENDER_BACKEND="none".
    :param output: for renderer "rtx"/"gl": if given, runs headless and writes a recorded video to
        this path (works without a display). If omitted, opens a live on-screen window instead
        (requires a real DISPLAY -- a local X server or an X11-forwarded SSH session) and does not
        save anything.
    :param num_steps: number of steps to run when renderer is "rtx" or "gl".
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    if renderer != "kit":
        if renderer == "gl" and output:
            # pyglet.options["headless"] forces pyglet's EGL surfaceless backend globally, which is
            # what lets the GL viewer init without a display -- but it also pre-empts ever opening a
            # real window, so only set it for the headless-recording case. For a live window, leave
            # this unset so pyglet falls back to its normal X11 backend against the caller's DISPLAY.
            import pyglet

            pyglet.options["headless"] = True
        gm.PHYSICS_BACKEND = "newton"
        gm.RENDER_BACKEND = "none"

    # Choose scene to load
    scene_model = "Rs_int"
    if not quickstart:
        scene_model = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)

    # Choose robot to create
    robot_name = "fetch"
    if not quickstart:
        robot_name = choose_from_options(
            options=list(sorted(REGISTERED_ROBOTS)), name="robot", random_selection=random_selection
        )

    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["model"] = robot_name
    # rgb capture requires Kit's Replicator renderer -- unavailable under RENDER_BACKEND="none"
    robot0_cfg["obs_modalities"] = ["rgb"] if gm.RENDER_BACKEND != "none" else []
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
    if renderer != "kit":
        cfg["env"] = {"device": "cuda:0"}

    # Create the environment
    env = og.Environment(configs=cfg)

    # Choose robot controller to use
    robot = env.scene.robots[0]
    controller_choices = {
        "base": "DifferentialDriveController",
        "arm_0": "InverseKinematicsController",
        "gripper_0": "MultiFingerGripperController",
        "camera": "JointController",
    }
    if not quickstart:
        controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # Choose control mode
    if random_selection:
        control_mode = "random"
    elif quickstart:
        control_mode = "teleop"
    else:
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_file()

    # Update the simulator's viewer camera's pose so it points towards the robot. Under the
    # Newton-physics + Kit-render hybrid, viewer_camera creation is not yet implemented (deferred to a
    # later phase of the RenderBackend abstraction) -- skip rather than crash.
    if og.sim.viewer_camera is not None:
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([1.46949, -3.97358, 2.21529]),
            orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
        )

    # Reset environment and robot
    env.reset()
    robot.reset()

    if renderer != "kit":
        # The stock KeyboardRobotController needs Kit's carb.input, unavailable under
        # RENDER_BACKEND="none" -- but Newton's own viewers have their own (non-Kit) windows/input,
        # so control_mode="teleop" drives KeyboardRobotController through that instead (see
        # newton_viewer_recording.py's teleop helpers). control_mode="random" keeps recording through
        # a fixed number of steps, same as before.
        if control_mode == "teleop":
            from omnigibson.utils.newton_viewer_recording import run_teleop_with_newton_viewer

            run_teleop_with_newton_viewer(env, robot, renderer)
        else:
            from omnigibson.utils.newton_viewer_recording import record_with_newton_viewer

            record_with_newton_viewer(env, robot, renderer, output, num_steps=num_steps)
        og.shutdown()
        return

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

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

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    random_action = None
    while step != max_steps:
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
        "--renderer",
        choices=["kit", "rtx", "gl"],
        default="kit",
        help=(
            "'kit' (default) is the interactive Omniverse Kit viewport. 'rtx' and 'gl' run physics "
            "on Newton (Kit-free) through Newton's own ray-traced or OpenGL viewer instead, using "
            "random actions. Pass --output to record a headless video (works without a display); "
            "omit it to open a live on-screen window instead (requires a real DISPLAY -- a local X "
            "server or an X11-forwarded SSH session)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Video output path for --renderer rtx/gl (headless recording). Omit for a live window.",
    )
    parser.add_argument("--num-steps", type=int, default=100, help="Number of steps to run for --renderer rtx/gl.")
    args = parser.parse_args()
    main(quickstart=args.quickstart, renderer=args.renderer, output=args.output, num_steps=args.num_steps)
