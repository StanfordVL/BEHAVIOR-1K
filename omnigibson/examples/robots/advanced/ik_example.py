import argparse
import time

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.objects import PrimitiveObject
from omnigibson.robots import Fetch
from omnigibson.scenes import Scene
from omnigibson.utils.control_utils import IKSolver


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
    scene_cfg = {"type": "Scene"}
    # Create Fetch robot
    # Note that since we only care about IK functionality, we fix the base (this also makes the robot more stable)
    # (any object can also have its fixed_base attribute set to True!)
    # Note that since we're going to be setting joint position targets, we also need to make sure the robot's arm joints
    # (which includes the trunk) are being controlled using joint positions
    robot_cfg = {
        "type": "Fetch",
        "fixed_base": True,
        "controller_config": {
            "arm_0": {
                "name": "NullJointController",
                "motor_type": "position",
            }
        },
    }
    cfg = dict(scene=scene_cfg, robots=[robot_cfg])
    env = og.Environment(configs=cfg)

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([4.32248, -5.74338, 6.85436]),
        orientation=th.tensor([0.39592, 0.13485, 0.29286, 0.85982]),
    )

    robot = env.robots[0]

    # Set robot base at the origin
    robot.set_position_orientation(position=th.tensor([0, 0, 0]), orientation=th.tensor([0, 0, 0, 1]))
    # At least one simulation step while the simulator is playing must occur for the robot (or in general, any object)
    # to be fully initialized after it is imported into the simulator
    og.sim.play()
    og.sim.step()
    # Make sure none of the joints are moving
    robot.keep_still()
    # Since this demo aims to showcase how users can directly control the robot with IK,
    # we will need to disable the built-in controllers in OmniGibson
    robot.control_enabled = False

    # Create the IK solver -- note that we are controlling both the trunk and the arm since both are part of the
    # controllable kinematic chain for the end-effector!
    control_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    ik_solver = IKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
        reset_joint_pos=robot.get_joint_positions()[control_idx],
        eef_name=robot.eef_link_names[robot.default_arm],
    )

    # Define a helper function for executing specific end-effector commands using the ik solver
    def execute_ik(pos, quat=None, max_iter=100):
        og.log.info("Querying joint configuration to current marker position")
        # Grab the joint positions in order to reach the desired pose target
        joint_pos = ik_solver.solve(
            target_pos=pos,
            target_quat=quat,
            tolerance_pos=0.002,
            tolerance_quat=0.01,
            weight_pos=20.0,
            weight_quat=0.05,
            max_iterations=max_iter,
            initial_joint_pos=robot.get_joint_positions()[control_idx],
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
            relative_prim_path=f"/marker",
            name="marker",
            primitive_type="Sphere",
            radius=0.03,
            visual_only=True,
            rgba=[1.0, 0, 0, 1.0],
        )
        env.scene.add_object(marker)

        # Get initial EE position and set marker to that location
        command = robot.get_eef_position()
        marker.set_position_orientation(position=command)
        og.sim.step()

        # Setup callbacks for grabbing keyboard inputs from omni
        exit_now = False

        def keyboard_event_handler(event, *args, **kwargs):
            nonlocal command, exit_now
            # Check if we've received a key press or repeat
            if (
                event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS
                or event.type == lazy.carb.input.KeyboardEventType.KEY_REPEAT
            ):
                if event.input == lazy.carb.input.KeyboardInput.ENTER:
                    # Execute the command
                    execute_ik(pos=command)
                elif event.input == lazy.carb.input.KeyboardInput.ESCAPE:
                    # Quit
                    og.log.info("Quit.")
                    exit_now = True
                else:
                    # We see if we received a valid delta command, and if so, we update our command and visualized
                    # marker position
                    delta_cmd = input_to_xyz_delta_command(inp=event.input)
                    if delta_cmd is not None:
                        command = command + delta_cmd
                        marker.set_position_orientation(position=command)
                        og.sim.step()

            # Callback must return True if valid
            return True

        # Hook up the callback function with omni's user interface
        appwindow = lazy.omni.appwindow.get_default_app_window()
        input_interface = lazy.carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)

        # Print out helpful information to the user
        print_message()

        # Loop until the user requests an exit
        while not exit_now:
            og.sim.step()

    # Always shut the simulation down cleanly at the end
    og.clear()


def input_to_xyz_delta_command(inp, delta=0.01):
    mapping = {
        lazy.carb.input.KeyboardInput.W: th.tensor([delta, 0, 0]),
        lazy.carb.input.KeyboardInput.S: th.tensor([-delta, 0, 0]),
        lazy.carb.input.KeyboardInput.DOWN: th.tensor([0, 0, -delta]),
        lazy.carb.input.KeyboardInput.UP: th.tensor([0, 0, delta]),
        lazy.carb.input.KeyboardInput.A: th.tensor([0, delta, 0]),
        lazy.carb.input.KeyboardInput.D: th.tensor([0, -delta, 0]),
    }

    return mapping.get(inp)


def print_message():
    print("*" * 80)
    print("Move the marker to a desired position to query IK and press ENTER")
    print("W/S: move marker further away or closer to the robot")
    print("A/D: move marker to the left or the right of the robot")
    print("UP/DOWN: move marker up and down")
    print("ESC: quit")


if __name__ == "__main__":
    main()
