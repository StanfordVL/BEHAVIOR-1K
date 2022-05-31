import argparse
import logging
import os
import time

import numpy as np


import igibson
# from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
)
from igibson.objects.visual_marker import VisualMarker
from igibson.robots.fetch import Fetch
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import l2_distance, parse_config, restoreState


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of usage of inverse kinematics solver
    This is a pybullet functionality but we keep an example because it can be useful and we do not provide a direct
    API from iGibson
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

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

    # Create simulator, scene and robot (Fetch)
    config = parse_config(os.path.join(igibson.example_config_path, "fetch_reaching.yaml"))
    s = Simulator(mode="headless", use_pb_gui=True if not headless else False)
    scene = EmptyScene()
    s.import_scene(scene)

    robot_config = config["robot"]
    robot_config.pop("name")

    fetch = Fetch(**robot_config)
    s.import_object(fetch)

    body_ids = fetch.get_body_ids()
    assert len(body_ids) == 1, "Fetch robot is expected to be single-body."
    robot_id = body_ids[0]

    arm_default_joint_positions = (
        0.10322468280792236,
        -1.414019864768982,
        1.5178184935241699,
        0.8189625336474915,
        2.200358942909668,
        2.9631312579803466,
        -1.2862852996643066,
        0.0008453550418615341,
    )

    robot_default_joint_positions = (
        [0.0, 0.0]
        + [arm_default_joint_positions[0]]
        + [0.0, 0.0]
        + list(arm_default_joint_positions[1:])
        + [0.01, 0.01]
    )

    robot_joint_names = [
        "r_wheel_joint",
        "l_wheel_joint",
        "torso_lift_joint",
        "head_pan_joint",
        "head_tilt_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
        "r_gripper_finger_joint",
        "l_gripper_finger_joint",
    ]
    arm_joints_names = [
        "torso_lift_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]

    # Indices of the joints of the arm in the vectors returned by IK and motion planning (excluding wheels, head, fingers)
    robot_arm_indices = [robot_joint_names.index(arm_joint_name) for arm_joint_name in arm_joints_names]

    # PyBullet ids of the joints corresponding to the joints of the arm
    arm_joint_ids = joints_from_names(robot_id, arm_joints_names)
    all_joint_ids = joints_from_names(robot_id, robot_joint_names)

    set_joint_positions(robot_id, arm_joint_ids, arm_default_joint_positions)

    # Set robot base
    fetch.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
    fetch.keep_still()

    # Get initial EE position
    x, y, z = fetch.get_eef_position()

    # Define the limits (max and min, range), and resting position for the joints, including the two joints of the
    # wheels of the base (indices 0 and 1), the two joints for the head (indices 3 and 4) and the two joints of the
    # fingers (indices 12 and 13)
    max_limits = get_max_limits(robot_id, all_joint_ids)
    min_limits = get_min_limits(robot_id, all_joint_ids)
    rest_position = robot_default_joint_positions
    joint_range = list(np.array(max_limits) - np.array(min_limits))
    joint_range = [item + 1 for item in joint_range]
    joint_damping = [0.1 for _ in joint_range]

    def accurate_calculate_inverse_kinematics(robot_id, eef_link_id, target_pos, threshold, max_iter):
        logging.info("IK solution to end effector position {}".format(target_pos))
        # Save initial robot pose
        state_id = p.saveState()

        max_attempts = 5
        solution_found = False
        joint_poses = None
        for attempt in range(1, max_attempts + 1):
            logging.info("Attempt {} of {}".format(attempt, max_attempts))
            # Get a random robot pose to start the IK solver iterative process
            # We attempt from max_attempt different initial random poses
            sample_fn = get_sample_fn(robot_id, arm_joint_ids)
            sample = np.array(sample_fn())
            # Set the pose of the robot there
            set_joint_positions(robot_id, arm_joint_ids, sample)

            it = 0
            # Query IK, set the pose to the solution, check if it is good enough repeat if not
            while it < max_iter:

                joint_poses = p.calculateInverseKinematics(
                    robot_id,
                    eef_link_id,
                    target_pos,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                )
                joint_poses = np.array(joint_poses)[robot_arm_indices]

                set_joint_positions(robot_id, arm_joint_ids, joint_poses)

                dist = l2_distance(fetch.get_eef_position(), target_pos)
                if dist < threshold:
                    solution_found = True
                    break
                logging.debug("Dist: " + str(dist))
                it += 1

            if solution_found:
                logging.info("Solution found at iter: " + str(it) + ", residual: " + str(dist))
                break
            else:
                logging.info("Attempt failed. Retry")
                joint_poses = None

        restoreState(state_id)
        p.removeState(state_id)
        return joint_poses

    threshold = 0.03
    max_iter = 100
    if programmatic_pos or headless:
        query_positions = [[1, 0, 0.8], [1, 1, 1], [0.5, 0.5, 0], [0.5, 0.5, 0.5]]
        for pos in query_positions:
            logging.info("Querying joint configuration to current marker position")
            joint_pos = accurate_calculate_inverse_kinematics(
                robot_id, fetch.eef_links[fetch.default_arm].link_id, pos, threshold, max_iter
            )
            if joint_pos is not None and len(joint_pos) > 0:
                logging.info("Solution found. Setting new arm configuration.")
                set_joint_positions(robot_id, arm_joint_ids, joint_pos)
            else:
                logging.info("EE position not reachable.")
            fetch.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
            fetch.keep_still()
            time.sleep(10)
    else:
        marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
        s.import_object(marker)
        marker.set_position([x, y, z])

        print_message()
        quit_now = False
        while True:
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN):
                    y -= 0.01
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN):
                    y += 0.01
                if k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN):
                    x += 0.01
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN):
                    x -= 0.01
                if k == ord("z") and (v & p.KEY_IS_DOWN):
                    z += 0.01
                if k == ord("x") and (v & p.KEY_IS_DOWN):
                    z -= 0.01
                if k == ord(" "):
                    logging.info("Querying joint configuration to current marker position")
                    joint_pos = accurate_calculate_inverse_kinematics(
                        robot_id, fetch.eef_links[fetch.default_arm].link_id, [x, y, z], threshold, max_iter
                    )
                    if joint_pos is not None and len(joint_pos) > 0:
                        logging.info("Solution found. Setting new arm configuration.")
                        set_joint_positions(robot_id, arm_joint_ids, joint_pos)
                        print_message()
                    else:
                        logging.info(
                            "No configuration to reach that point. Move the marker to a different configuration and try again."
                        )
                if k == ord("q"):
                    logging.info("Quit.")
                    quit_now = True
                    break

            if quit_now:
                break

            marker.set_position([x, y, z])
            fetch.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
            fetch.keep_still()
            s.step()

    s.disconnect()


def print_message():
    print("*" * 80)
    print("Move the marker to a desired position to query IK and press SPACE")
    print("Up/Down arrows: move marker further away or closer to the robot")
    print("Left/Right arrows: move marker to the left or the right of the robot")
    print("z/x: move marker up and down")
    print("q: quit")


if __name__ == "__main__":
    main()
