# Script for automatically calibrating a JoyLo set
# Refer to the README for the correct arm positions for calibration

from gello.dynamixel.driver import DynamixelDriver
import numpy as np
import sys
from dataclasses import dataclass
import yaml
import tyro
import os

CONFIG_DIR = f"{os.path.dirname(__file__)}/../configs"

@dataclass
class Args:
    gello_name: str
    """The name of the gello (used to determine which file to write to)"""

    port: str = "/dev/ttyUSB0"
    """The port that GELLO is connected to."""

    baudrate: int = 2000000
    """The baudrate of the connected GELLO's dynamixel board."""
    
    robot: str = "R1Pro" # "R1" or "R1Pro"
    """The robot type. This is used to determine the number of joints."""
    

def pretty_print_list(items: list[float]) -> None:
    for x in items:
        print(f"{x:>10.4f}", end=' ')
    print('')


def get_joint_angles_L(driver: DynamixelDriver, num_joints_per_arm: int) -> np.ndarray:
    return np.rad2deg(driver.get_joints()[:num_joints_per_arm])


def get_joint_angles_R(driver: DynamixelDriver, num_joints_per_arm: int) -> np.ndarray:
    return np.rad2deg(driver.get_joints()[num_joints_per_arm:])


def compute_joint_offsets_and_signs(joints_1: np.ndarray,
                                    joints_2: np.ndarray,
                                    robot_name: str)-> tuple[np.ndarray, np.ndarray]:
    """
    Computes the joints signs and offsets given the measured angles at two known positions 
    (for now, fixed in code to be a cannonical "zero" position, and a second "calibration" 
    position)

    Returns:
        signs (np.ndarray)
        offsets (np.ndarray)
    """
    if robot_name == "R1":
        # Hard-coded for now, could make an argument in the future
        expected_positions_1 = np.array([0, 0, 45, 45, -45, 0, 0, 0,
                                        0, 0, 45, 45, -45, 0, 0, 0])
        expected_positions_2 = np.array([90, 90, 180, 180, -180, 90, 90, -90,
                                        -90, -90, 180, 180, -180, -90, -90, 90])
    elif robot_name == "R1Pro":
        expected_positions_1 = np.zeros(18)
        expected_positions_2 = np.array([-90, -90, 90, 90, -90, -90, 90, 60, 90,
                                        -90, -90, -90, -90, 90, -90, -90, 60, -90])
    else:
        raise ValueError("Robot name must be either R1 or R1Pro")

    # Compute signs by comparing measured and expected delta between positions
    delta = joints_2 - joints_1
    expected_delta = expected_positions_2 - expected_positions_1
    
    signs = np.sign(delta / expected_delta)
    
    # Compute offsets by subtracting expected position and accounting for sign
    def round_to_90(x: np.ndarray) -> np.ndarray:
        """ Round all elements to the nearest multiple of 90 degrees """
        return 90 * np.rint(x / 90)
    
    offsets_1 = round_to_90(joints_1 - signs * expected_positions_1)
    offsets_2 = round_to_90(joints_2 - signs * expected_positions_2)
    
    # Exit if these offsets are not equal
    if not np.all(offsets_1 == offsets_2):
        print("WARNING: The joint offsets at the two positions don't seem to match!")
        print("         Are you sure that you positioned the robot correctly?")
        print("")
        print("         Try re-running this script and placing the arms at the correct")
        print("         positions for calibration!")
        sys.exit(1)
    
    return signs, offsets_1
            

def main(args: Args) -> None:
    assert args.robot in ["R1", "R1Pro"], "Robot type must be either R1 or R1Pro"
    
    num_joints = 18 if args.robot == "R1Pro" else 16
    num_joints_per_arm = num_joints // 2

    joint_ids = list(range(num_joints))
    driver = DynamixelDriver(ids=joint_ids, port=args.port, baudrate=args.baudrate)

    # Initialize arrays to store joint angles
    joint_angles_pos_1 = np.zeros(num_joints)
    joint_angles_pos_2 = np.zeros(num_joints)

    # Warm up dynamixel driver
    for _ in range(10):
        driver.get_joints()

    print("=======================")
    print("GELLO Joint Calibration")
    print("=======================")
    print('')

    # Gather data for the left arm
    print("Now Calibrating LEFT arm...")
    print("Place the LEFT arm in the zero position and press enter!", end ='')
    input()
    joint_angles_pos_1[:num_joints_per_arm] = get_joint_angles_L(driver, num_joints_per_arm)

    print("Place the LEFT arm in the calibration position and press enter!", end ='')
    input()
    joint_angles_pos_2[:num_joints_per_arm] = get_joint_angles_L(driver, num_joints_per_arm)
    
    # Gather data for the right arm
    print("\n\nNow Calibrating RIGHT arm...")
    print("\nPlace the RIGHT arm in the zero position and press enter...", end ='')
    input()
    joint_angles_pos_1[num_joints_per_arm:] = get_joint_angles_R(driver, num_joints_per_arm)

    print("Place the RIGHT arm in the calibration position and press enter!", end ='')
    input()
    joint_angles_pos_2[num_joints_per_arm:] = get_joint_angles_R(driver, num_joints_per_arm)

    # Compute offsets and angles 
    signs, offsets = compute_joint_offsets_and_signs(joint_angles_pos_1, joint_angles_pos_2, args.robot)

    print("")
    print("Successfully calibrated your GELLO:")
    print("Signs:")
    pretty_print_list(signs)
    print("Offsets:")
    pretty_print_list(offsets)
    
    # Write to output_file
    joint_data = {
        "joints": {
            "offsets": [float(x) for x in offsets], # Needed so we have default float type
            "signs": [float(x) for x in signs], # Needed so we have default float type
        }
    }
    
    output_filename = f"{CONFIG_DIR}/joint_config_{args.gello_name}.yaml"
    
    with open(output_filename, "w+") as file:
        yaml.dump(joint_data, file, default_flow_style=False)
    
    print("")
    print(f"Joint offsets and signs have been saved to configs/joint_config_{args.gello_name}.yaml!")
    print("")


if __name__ == "__main__":
    main(tyro.cli(Args))