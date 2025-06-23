import atexit
import glob
import time
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

from gello.agents.agent import DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.agents.spacemouse_agent import SpacemouseAgent
from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot


@dataclass
class Args:
    hz: int = 100

    agent: str = "gello"
    robot: str = "ur5"
    gello_port: Optional[str] = None
    mock: bool = False
    verbose: bool = False

    hostname: str = "127.0.0.1"
    robot_port: int = 6001


def launch_robot_server(port: int, args: Args):
    if args.robot == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        from gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_panda":
        from gello.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()

    else:
        if args.robot == "xarm":
            from gello.robots.xarm_robot import XArmRobot

            robot = XArmRobot()
        elif args.robot == "ur5":
            from gello.robots.ur import URRobot

            robot = URRobot(robot_ip=args.robot_ip)
        else:
            raise NotImplementedError(
                f"Robot {args.robot} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
            )
        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        print(f"Starting robot server on port {port}")
        server.serve()


def start_robot_process(args: Args):
    process = Process(target=launch_robot_server, args=(args.robot_port, args))

    # Function to kill the child process
    def kill_child_process(process):
        print("Killing child process...")
        process.terminate()

    # Register the kill_child_process function to be called at exit
    atexit.register(kill_child_process, process)
    process.start()


def main(args: Args):
    start_robot_process(args)

    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz)

    if args.agent == "gello":
        gello_port = args.gello_port
        if gello_port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                print(f"using port {gello_port}")
            else:
                raise ValueError(
                    "No gello port found, please specify one or plug in gello"
                )
        agent = GelloAgent(port=gello_port)

        reset_joints = np.array([0, 0, 0, -np.pi, 0, np.pi, 0, 0])
        curr_joints = env.get_obs()["joint_positions"]
        if reset_joints.shape == curr_joints.shape:
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 100)

            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)
                time.sleep(0.001)

    elif args.agent == "quest":
        from gello.agents.quest_agent import SingleArmQuestAgent

        agent = SingleArmQuestAgent(robot_type=args.robot, which_hand="l")
    elif args.agent == "spacemouse":
        agent = SpacemouseAgent(robot_type=args.robot, verbose=args.verbose)
    elif args.agent == "dummy" or args.agent == "none":
        agent = DummyAgent(num_dofs=robot_client.num_dofs())
    else:
        raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    while True:
        action = agent.act(obs)
        obs = env.step(action)


if __name__ == "__main__":
    main(tyro.cli(Args))
