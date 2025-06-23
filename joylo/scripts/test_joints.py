# Script for automatically testing the joint offsets and signs of a JoyLo set

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
    
    robot: str = "R1" # "R1" or "R1Pro"
    """The robot type. This is used to determine the number of joints."""
    

def pretty_print_joints(joints: np.ndarray, num_joints_per_arm: int) -> None:
    
    print("L:   ", end='')
    for x in joints[:num_joints_per_arm]:
        print(f"{x:>8.2f}", end=' ')
    
    print("   | R:   ", end='')
    for x in joints[num_joints_per_arm:]:
        print(f"{x:>8.2f}", end=' ')
    
    print("")


def main(args: Args) -> None:
    assert args.robot in ["R1", "R1Pro"], "Robot type must be either R1 or R1Pro"
    
    num_joints = 18 if args.robot == "R1Pro" else 16
    num_joints_per_arm = num_joints // 2

    joint_ids = list(range(num_joints))
    driver = DynamixelDriver(ids=joint_ids, port=args.port, baudrate=args.baudrate)
    
    # Load offsets and signs from file
    config_filename = f"{CONFIG_DIR}/joint_config_{args.gello_name}.yaml"
    
    with open(config_filename, "r") as file:
        joint_config = yaml.safe_load(file)
    
    offsets = joint_config["joints"]["offsets"]
    signs = joint_config["joints"]["signs"]
    
    assert len(offsets) == num_joints, "Number of joints in config file does not match selected robot!"
    assert len(signs) == num_joints, "Number of joints in config file does not match selected robot!"


    # Warm up dynamixel driver
    for _ in range(10):
        driver.get_joints()
    
    # Repeatedly get and print joints
    while True:
        j_raw = np.rad2deg(driver.get_joints())
        j_adjusted = signs * (j_raw - offsets)

        pretty_print_joints(j_adjusted, num_joints_per_arm)

if __name__ == "__main__":
    main(tyro.cli(Args))