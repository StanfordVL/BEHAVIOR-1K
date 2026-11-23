"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import os

import numpy as np
import torch as th

import omnigibson as og
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


class RobotCameraWindow:
    """
    On-screen pyglet window showing the robot's cameras side by side, for running without Kit (which has no
    viewport). Also the window that KeyboardRobotController reads keyboard input from.
    """

    def __init__(self, height=512):
        import pyglet

        self._pyglet = pyglet
        self._height = height
        self.window = pyglet.window.Window(width=height, height=height, caption="Robot cameras", resizable=True)
        # Closing the window quits the demo, like ESC
        self.window.push_handlers(on_close=og.shutdown)
        self._texture = None

    def show(self, obs):
        """
        Draws every rgb image found in @obs side by side, and processes the window's pending input events

        Args:
            obs (dict or list): Observation returned by env.step()
        """
        rgbs = [rgb.cpu().numpy() if isinstance(rgb, th.Tensor) else np.asarray(rgb) for rgb in self._find_rgbs(obs)]
        self.window.switch_to()
        if len(rgbs) > 0:
            # Pad every image to the tallest one's height, then tile them left to right
            h = max(rgb.shape[0] for rgb in rgbs)
            rgb = np.concatenate(
                [np.pad(rgb[..., :4], ((0, h - rgb.shape[0]), (0, 0), (0, 0))) for rgb in rgbs], axis=1
            )
            w = rgb.shape[1]
            if self._texture is None:
                # Keep the cameras' aspect ratio
                self.window.set_size(int(self._height * w / h), self._height)
            # Negative pitch flips rows top-to-bottom, since pyglet's origin is the bottom-left corner
            image = self._pyglet.image.ImageData(w, h, "RGBA", np.ascontiguousarray(rgb).tobytes(), pitch=-w * 4)
            self._texture = image.get_texture()
        self.window.clear()
        if self._texture is not None:
            self._texture.blit(0, 0, width=self.window.width, height=self.window.height)
        self.window.flip()
        self.window.dispatch_events()

    @classmethod
    def _find_rgbs(cls, obs):
        """
        Args:
            obs (dict or list): (Nested) observation

        Returns:
            list: Every rgb image in @obs, in traversal order
        """
        if isinstance(obs, dict):
            if "rgb" in obs:
                return [obs["rgb"]]
            obs = list(obs.values())
        if isinstance(obs, (list, tuple)):
            return [rgb for value in obs for rgb in cls._find_rgbs(value)]
        return []


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
    physics_backend="physx",
    renderer="kit",
):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)

    :param physics_backend: "physx" or "newton". See SUPPORTED_BACKEND_COMBINATIONS for which renderers each supports
    :param renderer: "kit" for Omniverse, or "gl"/"ovrtx" to render the robot's cameras through Newton's
        OpenGL/ray-traced viewer (requires Newton physics). With a display, the robot's camera is shown in its own
        window, which takes the keyboard input for teleop. Without one (or with @headless), it always uses random
        actions
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Without Kit, there's an on-screen window only if there's a display to put it on
    has_window = renderer == "kit" or (not headless and bool(os.environ.get("DISPLAY")))
    if renderer != "kit":
        if not has_window:
            import pyglet

            # Render the cameras through EGL, which needs no display
            pyglet.options["headless"] = True
    gm.PHYSICS_BACKEND = physics_backend
    gm.RENDER_BACKEND = renderer

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
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
    if physics_backend == "newton":
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
    if not has_window:
        control_mode = "random"
    elif random_selection:
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

    # Update the simulator's viewer camera's pose so it points towards the robot.
    if og.sim.viewer_camera is not None:
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([1.46949, -3.97358, 2.21529]),
            orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
        )

    # Reset environment and robot
    env.reset()
    robot.reset()

    # Create teleop controller. Without Kit, it reads the keyboard from the robot camera window
    camera_window = RobotCameraWindow() if renderer != "kit" and has_window else None
    action_generator = None
    if has_window:
        action_generator = KeyboardRobotController(
            robot=robot, window=None if camera_window is None else camera_window.window
        )

        # Register custom binding to reset the environment
        action_generator.register_custom_keymapping(
            key="R",
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
                random_action = (th.rand(robot.action_dim, device=og.sim.device) * 2 - 1) * 0.05
            action = random_action
        else:
            action = action_generator.get_teleop_action()

        obs, *_ = env.step(action=action)
        if camera_window is not None:
            camera_window.show(obs)
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
        "--physics-backend",
        choices=["physx", "newton"],
        default="physx",
        help="Physics backend. 'gl' and 'ovrtx' renderers require 'newton'.",
    )
    parser.add_argument(
        "--renderer",
        choices=["kit", "gl", "ovrtx"],
        default="kit",
        help=(
            "'kit' (default) is the interactive Omniverse Kit viewport. 'gl' and 'ovrtx' render the robot's "
            "cameras through Newton's OpenGL or ray-traced viewer (requires --physics-backend newton), shown in "
            "a window that takes the keyboard input (random actions only, if there's no display)."
        ),
    )
    args = parser.parse_args()
    main(quickstart=args.quickstart, physics_backend=args.physics_backend, renderer=args.renderer)
