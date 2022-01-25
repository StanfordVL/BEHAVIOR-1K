"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import logging
import platform
import random
import sys
import time
from collections import OrderedDict

import numpy as np


from igibson.robots import REGISTERED_ROBOTS, ManipulationRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

CONTROL_MODES = OrderedDict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = OrderedDict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

GUIS = OrderedDict(
    ig="iGibson GUI (default)",
    pb="PyBullet GUI",
)

ARROWS = {
    0: "up_arrow",
    1: "down_arrow",
    2: "left_arrow",
    3: "right_arrow",
    65295: "left_arrow",
    65296: "right_arrow",
    65297: "up_arrow",
    65298: "down_arrow",
}

gui = "ig"


def choose_from_options(options, name, random_selection=False):
    """
    Prints out options from a list, and returns the requested option.

    :param options: dict or Array, options to choose from. If dict, the value entries are assumed to be docstrings
        explaining the individual options
    :param name: str, name of the options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return str: Requested option
    """
    # Select robot
    print("\nHere is a list of available {}s:\n".format(name))

    for k, option in enumerate(options):
        docstring = ": {}".format(options[option]) if isinstance(options, dict) else ""
        print("[{}] {}{}".format(k + 1, option, docstring))
    print()

    if not random_selection:
        try:
            s = input("Choose a {} (enter a number from 1 to {}): ".format(name, len(options)))
            # parse input into a number within range
            k = min(max(int(s), 1), len(options)) - 1
        except:
            k = 0
            print("Input is not valid. Use {} by default.".format(list(options)[k]))
    else:
        k = random.choice(range(len(options)))

    # Return requested option
    return list(options)[k]


def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return OrderedDict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = OrderedDict()

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


class KeyboardController:
    """
    Simple class for controlling iGibson robots using keyboard commands
    """

    def __init__(self, robot, simulator):
        """
        :param robot: BaseRobot, robot to control
        """
        # Store relevant info from robot
        self.simulator = simulator
        self.action_dim = robot.action_dim
        self.controller_info = OrderedDict()
        idx = 0
        for name, controller in robot._controllers.items():
            self.controller_info[name] = {
                "name": type(controller).__name__,
                "start_idx": idx,
                "command_dim": controller.command_dim,
            }
            idx += controller.command_dim

        # Other persistent variables we need to keep track of
        self.joint_control_idx = None  # Indices of joints being directly controlled via joint control
        self.current_joint = -1  # Active joint being controlled for joint control
        self.gripper_direction = 1.0  # Flips between -1 and 1
        self.persistent_gripper_action = None  # Whether gripper actions should persist between commands,
        # i.e.: if using binary gripper control and when no keypress is active, the gripper action should still the last executed gripper action
        self.last_keypress = None  # Last detected keypress
        self.keypress_mapping = None
        self.populate_keypress_mapping()
        self.time_last_keyboard_input = time.time()

    def populate_keypress_mapping(self):
        """
        Populates the mapping @self.keypress_mapping, which maps keypresses to action info:

            keypress:
                idx: <int>
                val: <float>
        """
        self.keypress_mapping = {}
        self.joint_control_idx = set()

        # Add mapping for joint control directions (no index because these are inferred at runtime)
        self.keypress_mapping["]"] = {"idx": None, "val": 0.1}
        self.keypress_mapping["["] = {"idx": None, "val": -0.1}

        # Iterate over all controller info and populate mapping
        for component, info in self.controller_info.items():
            if info["name"] == "JointController":
                for i in range(info["command_dim"]):
                    ctrl_idx = info["start_idx"] + i
                    self.joint_control_idx.add(ctrl_idx)
            elif info["name"] == "DifferentialDriveController":
                self.keypress_mapping["i"] = {"idx": info["start_idx"] + 0, "val": 0.2}
                self.keypress_mapping["k"] = {"idx": info["start_idx"] + 0, "val": -0.2}
                self.keypress_mapping["l"] = {"idx": info["start_idx"] + 1, "val": 0.1}
                self.keypress_mapping["j"] = {"idx": info["start_idx"] + 1, "val": -0.1}
            elif info["name"] == "InverseKinematicsController":
                self.keypress_mapping["up_arrow"] = {"idx": info["start_idx"] + 0, "val": 0.5}
                self.keypress_mapping["down_arrow"] = {"idx": info["start_idx"] + 0, "val": -0.5}
                self.keypress_mapping["right_arrow"] = {"idx": info["start_idx"] + 1, "val": -0.5}
                self.keypress_mapping["left_arrow"] = {"idx": info["start_idx"] + 1, "val": 0.5}
                self.keypress_mapping["p"] = {"idx": info["start_idx"] + 2, "val": 0.5}
                self.keypress_mapping[";"] = {"idx": info["start_idx"] + 2, "val": -0.5}
                self.keypress_mapping["n"] = {"idx": info["start_idx"] + 3, "val": 0.5}
                self.keypress_mapping["b"] = {"idx": info["start_idx"] + 3, "val": -0.5}
                self.keypress_mapping["o"] = {"idx": info["start_idx"] + 4, "val": 0.5}
                self.keypress_mapping["u"] = {"idx": info["start_idx"] + 4, "val": -0.5}
                self.keypress_mapping["v"] = {"idx": info["start_idx"] + 5, "val": 0.5}
                self.keypress_mapping["c"] = {"idx": info["start_idx"] + 5, "val": -0.5}
            elif info["name"] == "MultiFingerGripperController":
                if info["command_dim"] > 1:
                    for i in range(info["command_dim"]):
                        ctrl_idx = info["start_idx"] + i
                        self.joint_control_idx.add(ctrl_idx)
                else:
                    self.keypress_mapping[" "] = {"idx": info["start_idx"], "val": 1.0}
                    self.persistent_gripper_action = 1.0
            elif info["name"] == "NullGripperController":
                # We won't send actions if using a null gripper controller
                self.keypress_mapping[" "] = {"idx": info["start_idx"], "val": None}
            else:
                raise ValueError("Unknown controller name received: {}".format(info["name"]))

    def get_random_action(self):
        """
        :return Array: Generated random action vector (normalized)
        """
        return np.random.uniform(-1, 1, self.action_dim)

    def get_teleop_action(self):
        """
        :return Array: Generated action vector based on received user inputs from the keyboard
        """
        action = np.zeros(self.action_dim)
        keypress = self.get_keyboard_input()

        if keypress is not None:
            # If the keypress is a number, the user is trying to select a specific joint to control
            if keypress.isnumeric():
                if int(keypress) in self.joint_control_idx:
                    self.current_joint = int(keypress)

            elif keypress in self.keypress_mapping:
                action_info = self.keypress_mapping[keypress]
                idx, val = action_info["idx"], action_info["val"]

                # Non-null gripper
                if val is not None:
                    # If the keypress is a spacebar, this is a gripper action
                    if keypress == " ":
                        # We toggle the gripper direction if the last keypress is DIFFERENT from this keypress AND
                        # we're past the gripper time threshold, to avoid high frequency toggling
                        # i.e.: holding down the spacebar shouldn't result in rapid toggling of the gripper
                        if keypress != self.last_keypress:
                            self.gripper_direction *= -1.0

                        # Modify the gripper value
                        val *= self.gripper_direction
                        if self.persistent_gripper_action is not None:
                            self.persistent_gripper_action = val

                    # If there is no index, the user is controlling a joint with "[" and "]". Set the idx to self.current_joint
                    if idx is None and self.current_joint != -1:
                        idx = self.current_joint

                    if idx is not None:
                        action[idx] = val

        sys.stdout.write("\033[K")
        print("Pressed {}. Action: {}".format(keypress, action))
        sys.stdout.write("\033[F")

        # Update last keypress
        self.last_keypress = keypress

        # Possibly set the persistent gripper action
        if self.persistent_gripper_action is not None and self.keypress_mapping[" "]["val"] is not None:
            action[self.keypress_mapping[" "]["idx"]] = self.persistent_gripper_action

        # Return action
        return action

    def get_keyboard_input(self):
        """
        Checks for newly received user inputs and returns the first received input, if any
        :return None or str: User input in string form. Note that only the characters mentioned in
        @self.print_keyboard_teleop_info are explicitly supported
        """
        global gui

        # Getting current time
        current_time = time.time()
        if gui == "pb":
            kbe = p.getKeyboardEvents()
            # Record the first keypress if any was detected
            keypress = -1 if len(kbe.keys()) == 0 else list(kbe.keys())[0]
        else:
            # Record the last keypress if it's pressed after the last check
            keypress = (
                -1
                if self.simulator.viewer.time_last_pressed_key is None
                or self.simulator.viewer.time_last_pressed_key < self.time_last_keyboard_input
                else self.simulator.viewer.last_pressed_key
            )
        # Updating the time of the last check
        self.time_last_keyboard_input = current_time

        if keypress in ARROWS:
            # Handle special case of arrow keys, which are mapped differently between pybullet and cv2
            keypress = ARROWS[keypress]
        else:
            # Handle general case where a key was actually pressed (value > -1)
            keypress = chr(keypress) if keypress > -1 else None

        return keypress

    @staticmethod
    def print_keyboard_teleop_info():
        """
        Prints out relevant information for teleop controlling a robot
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print()
        print("*" * 30)
        print("Controlling the Robot Using the Keyboard")
        print("*" * 30)
        print()
        print("Joint Control")
        print_command("0-9", "specify the joint to control")
        print_command("[, ]", "move the joint backwards, forwards, respectively")
        print()
        print("Differential Drive Control")
        print_command("i, k", "turn left, right")
        print_command("l, j", "move forward, backwards")
        print()
        print("Inverse Kinematics Control")
        print_command(u"\u2190, \u2192", "translate arm eef along x-axis")
        print_command(u"\u2191, \u2193", "translate arm eef along y-axis")
        print_command("p, ;", "translate arm eef along z-axis")
        print_command("n, b", "rotate arm eef about x-axis")
        print_command("o, u", "rotate arm eef about y-axis")
        print_command("v, c", "rotate arm eef about z-axis")
        print()
        print("Boolean Gripper Control")
        print_command("space", "toggle gripper (open/close)")
        print()
        print("*" * 30)
        print()


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Create an initial headless dummy scene so we can load the requested robot and extract useful info
    s = Simulator(mode="headless", use_pb_gui=False)
    scene = EmptyScene()
    s.import_scene(scene)

    # Get robot to create
    robot_name = choose_from_options(
        options=list(sorted(REGISTERED_ROBOTS.keys())), name="robot", random_selection=random_selection
    )
    robot = REGISTERED_ROBOTS[robot_name](action_type="continuous")
    s.import_object(robot)

    # Get controller choice
    controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # Choose control mode
    if random_selection:
        control_mode = "random"
    else:
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # Choose scene to load
    scene_id = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)

    # Choose GUI
    global gui
    gui = choose_from_options(options=GUIS, name="gui", random_selection=random_selection)

    if (
        gui == "ig"
        and platform.system() != "Darwin"
        and control_mode == "teleop"
        and isinstance(robot, ManipulationRobot)
        and controller_choices["arm"] == "InverseKinematicsController"
    ):
        message = "Warning: iG GUI does not support arrow keys for your OS (needed to control the arm with an IK Controller). Falling back to PyBullet (pb) GUI."
        logging.warning(message)
        gui = "pb"

    # Infer what GUI(s) to use
    render_mode, use_pb_gui = None, None
    if gui == "ig":
        render_mode, use_pb_gui = "gui_interactive", False
    elif gui == "pb":
        render_mode, use_pb_gui = "headless", True
    else:
        raise ValueError("Unknown GUI: {}".format(gui))

    if headless:
        render_mode, use_pb_gui = "headless", False

    # Shut down dummy simulator and re-create actual simulator
    s.disconnect()
    del s
    s = Simulator(mode=render_mode, use_pb_gui=use_pb_gui, image_width=512, image_height=512)

    # Load scene
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1]) if scene_id == "empty" else InteractiveIndoorScene(scene_id)
    s.import_scene(scene)

    # Load robot
    robot = REGISTERED_ROBOTS[robot_name](
        action_type="continuous",
        action_normalize=True,
        controller_config={
            component: {"name": controller_name} for component, controller_name in controller_choices.items()
        },
    )
    s.import_object(robot)

    # Reset the robot
    robot.set_position([0, 0, 0])
    robot.reset()
    robot.keep_still()

    # Set initial viewer if using IG GUI
    if gui != "pb" and not headless:
        s.viewer.initial_pos = [1.6, 0, 1.3]
        s.viewer.initial_view_direction = [-0.7, 0, -0.7]
        s.viewer.reset_viewer()

    # Create teleop controller
    action_generator = KeyboardController(robot=robot, simulator=s)

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo. Switch to the viewer windows")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0
    while step != max_steps:
        action = (
            action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        )
        robot.apply_action(action)
        for _ in range(10):
            s.step()
            step += 1

    s.disconnect()


if __name__ == "__main__":
    main()
