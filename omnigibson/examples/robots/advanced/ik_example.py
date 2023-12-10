import argparse
import time

import numpy as np

import omnigibson as og
from omnigibson.objects import PrimitiveObject
from omnigibson.robots import Fetch
from omnigibson.scenes import Scene
from omnigibson.utils.control_utils import IKSolver

import carb
import omni


def main(random_selection=False, headless=False, short_exec=False):
    """
    Minimal example of usage of inverse kinematics solver

    This example showcases how to construct your own IK functionality using omniverse's native lula library
    without explicitly utilizing all of OmniGibson's class abstractions, and also showcases how to manipulate
    the simulator at a lower-level than the main Environment entry point.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Assuming that if random_selection=True, headless=True, short_exec=True, we are calling it from tests and we
    # do not want to parse args (it would fail because the calling function is pytest "testfile.py")
    if not (random_selection and headless and short_exec):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--programmatic",
            "-p",
            dest="programmatic_pos",
            action="store_true",
            help="if the IK solvers should be used with the GUI or programmatically",
        )
        args = parser.parse_args()
        programmatic_pos = args.programmatic_pos
    else:
        programmatic_pos = True

    # Import scene and robot (Fetch)
    scene = Scene()
    og.sim.import_scene(scene)

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([4.32248, -5.74338, 6.85436]),
        orientation=np.array([0.39592, 0.13485, 0.29286, 0.85982]),
    )

    # Create Fetch robot
    # Note that since we only care about IK functionality, we fix the base (this also makes the robot more stable)
    # (any object can also have its fixed_base attribute set to True!)
    # Note that since we're going to be setting joint position targets, we also need to make sure the robot's arm joints
    # (which includes the trunk) are being controlled using joint positions
    robot = Fetch(
        prim_path="/World/robot",
        name="robot",
        fixed_base=True,
        controller_config={
            "arm_0": {
                "name": "JointController",
                "motor_type": "position",
            }
        }
    )
    og.sim.import_object(robot)

    # Set robot base at the origin
    robot.set_position_orientation(np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
    # At least one simulation step while the simulator is playing must occur for the robot (or in general, any object)
    # to be fully initialized after it is imported into the simulator
    og.sim.play()
    og.sim.step()
    # Make sure none of the joints are moving
    robot.keep_still()

    # Create the IK solver -- note that we are controlling both the trunk and the arm since both are part of the
    # controllable kinematic chain for the end-effector!
    control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    ik_solver = IKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
        default_joint_pos=robot.get_joint_positions()[control_idx],
        eef_name=robot.eef_link_names[robot.default_arm],
    )

    # Define a helper function for executing specific end-effector commands using the ik solver
    def execute_ik(pos, quat=None, max_iter=100):
        og.log.info("Querying joint configuration to current marker position")
        # Grab the joint positions in order to reach the desired pose target
        joint_pos = ik_solver.solve(
            target_pos=pos,
            target_quat=quat,
            max_iterations=max_iter,
        )
        if joint_pos is not None:
            og.log.info("Solution found. Setting new arm configuration.")
            robot.set_joint_positions(joint_pos, indices=control_idx, drive=True)
        else:
            og.log.info("EE position not reachable.")
        og.sim.step()

    if programmatic_pos or headless:
        # Sanity check IK using pre-defined hardcoded positions
        query_positions = [[1, 0, 0.8], [1, 1, 1], [0.5, 0.5, 0], [0.5, 0.5, 0.5]]
        for query_pos in query_positions:
            execute_ik(query_pos)
            time.sleep(2)
    else:
        # Create a visual marker to be moved by the user, representing desired end-effector position
        marker = PrimitiveObject(
            prim_path=f"/World/marker",
            name="marker",
            primitive_type="Sphere",
            radius=0.03,
            visual_only=True,
            rgba=[1.0, 0, 0, 1.0],
        )
        og.sim.import_object(marker)

        # Get initial EE position and set marker to that location
        command = robot.get_eef_position()
        marker.set_position(command)
        og.sim.step()

        # Setup callbacks for grabbing keyboard inputs from omni
        exit_now = False

        def keyboard_event_handler(event, *args, **kwargs):
            nonlocal command, exit_now
            # Check if we've received a key press or repeat
            if event.type == carb.input.KeyboardEventType.KEY_PRESS \
                    or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                if event.input == carb.input.KeyboardInput.ENTER:
                    # Execute the command
                    execute_ik(pos=command)
                elif event.input == carb.input.KeyboardInput.ESCAPE:
                    # Quit
                    og.log.info("Quit.")
                    exit_now = True
                else:
                    # We see if we received a valid delta command, and if so, we update our command and visualized
                    # marker position
                    delta_cmd = input_to_xyz_delta_command(inp=event.input)
                    if delta_cmd is not None:
                        command = command + delta_cmd
                        marker.set_position(command)
                        og.sim.step()

            # Callback must return True if valid
            return True

        # Hook up the callback function with omni's user interface
        appwindow = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)

        # Print out helpful information to the user
        print_message()

        # Loop until the user requests an exit
        while not exit_now:
            og.sim.step()

    # Always shut the simulation down cleanly at the end
    og.app.close()


def input_to_xyz_delta_command(inp, delta=0.01):
    mapping = {
        carb.input.KeyboardInput.W: np.array([delta, 0, 0]),
        carb.input.KeyboardInput.S: np.array([-delta, 0, 0]),
        carb.input.KeyboardInput.DOWN: np.array([0, 0, -delta]),
        carb.input.KeyboardInput.UP: np.array([0, 0, delta]),
        carb.input.KeyboardInput.A: np.array([0, delta, 0]),
        carb.input.KeyboardInput.D: np.array([0, -delta, 0]),
    }

    return mapping.get(inp)


def print_message():
    print("*" * 80)
    print("Move the marker to a desired position to query IK and press ENTER")
    print("W/S: move marker further away or closer to the robot")
    print("A/D: move marker to the left or the right of the robot")
    print("T/G: move marker up and down")
    print("ESC: quit")


if __name__ == "__main__":
    main()
