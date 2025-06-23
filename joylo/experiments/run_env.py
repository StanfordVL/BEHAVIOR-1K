import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch as th
import numpy as np
import tyro

from gello.agents.agent import BimanualAgent, DummyAgent, MultiControllerAgent
from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig
from gello.agents.joycon_agent import JoyconAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot


from tqdm import tqdm


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    damping_motor_kp: float = 0.0
    use_joycons: bool = True
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    multi: bool = False
    verbose: bool = False


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)

    active_arm = "right"
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients, active_arm=active_arm)

    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISZZV-if00-port0"
            left_agent = GelloAgent(port=left)
            right_agent = GelloAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = th.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = th.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = th.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (th.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in th.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    elif args.multi:
        assert args.agent == "gello"        # Only supported mode for now
        gello_port = args.gello_port
        if gello_port is None:
            import platform

            if platform.system().lower() == "linux":
                usb_ports = glob.glob("/dev/serial/by-id/*")
            elif platform.system().lower() == "darwin":
                usb_ports = glob.glob("/dev/cu.usbserial-*")
            else:
                raise ValueError(f"Unsupported platform {platform.system()}")
            print(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                print(f"using port {gello_port}")
            else:
                raise ValueError(
                    "No gello port found, please specify one or plug in gello"
                )

        # dynamixel_config_left = DynamixelRobotConfig(
        #     joint_ids=(0, 1, 2, 3, 4, 5),
        #     joint_offsets=[3*np.pi/2, 0.5*np.pi/2, 4*np.pi/2, 2*np.pi/2, 1*np.pi/2, 3*np.pi/2],
        #     joint_signs=(1, 1, 1, 1, 1, 1),
        #     gripper_config=None, #(8, 202, 152),
        # )
        # dynamixel_config_right = DynamixelRobotConfig(
        #     joint_ids=(6, 7, 8, 9, 10, 11),
        #     joint_offsets=[np.pi/2, 0 * np.pi/2, 2*np.pi/2, 3*np.pi/2, 2*np.pi/2, 3*np.pi/2],
        #     joint_signs=(1, 1, 1, 1, 1, 1),
        #     gripper_config=None, #(8, 202, 152),
        # )
        dynamixel_config = DynamixelRobotConfig(
            joint_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
            joint_offsets=[2*np.pi/2, 3*np.pi/2, -1*np.pi/2, 4*np.pi/2, 1*np.pi/2, 0*np.pi/2,
                           3 * np.pi/2, 0 * np.pi/2, 2*np.pi/2, -1*np.pi/2, 2*np.pi/2, 1*np.pi/2],
            joint_signs=(1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            gripper_config=None,  # (8, 202, 152),
        )

        # arm_left_agent = GelloAgent(
        #     port=gello_port,
        #     dynamixel_config=dynamixel_config_left,
        #     start_joints=args.start_joints,
        #     damping_motor_kv=args.damping_motor_kv,
        # )
        # arm_right_agent = GelloAgent(
        #     port=gello_port,
        #     dynamixel_config=dynamixel_config_right,
        #     start_joints=args.start_joints,
        #     damping_motor_kv=args.damping_motor_kv,
        # )
        arm_agent = GelloAgent(
            port=gello_port,
            dynamixel_config=dynamixel_config,
            start_joints=args.start_joints,
            damping_motor_kp=args.damping_motor_kp,
        )

        agent_parts = [arm_agent]

        if args.use_joycons:
            base_trunk_gripper_agent = JoyconAgent(
                calibration_dir=f"{REPO_DIR}/configs",
                deadzone_threshold=0.2,
                max_translation=0.5,
                max_rotation=0.8,
                max_trunk_translate=0.2,
                max_trunk_tilt=0.4,
            )
            agent_parts.append(base_trunk_gripper_agent)

        agent = MultiControllerAgent(agents=agent_parts)

    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                import platform

                if platform.system().lower() == "linux":
                    usb_ports = glob.glob("/dev/serial/by-id/*")
                elif platform.system().lower() == "darwin":
                    usb_ports = glob.glob("/dev/cu.usbserial-*")
                else:
                    raise ValueError(f"Unsupported platform {platform.system()}")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            reset_joints = th.tensor(
                [
                    0.09162008114028396,
                    -0.19826458111314524,
                    -0.01990020486871322,
                    -2.4732269941140346,
                    -0.01307073642274261,
                    2.30396583422025,
                    0.8480939705504309,
                ], dtype=th.float
            )
            agent = GelloAgent(
                port=gello_port,
                start_joints=args.start_joints,
                damping_motor_kp=args.damping_motor_kp,
            )
            # curr_joints = env.get_obs()["joint_positions"]
            # reset_joints[-1] = curr_joints[-1]
            # if reset_joints.shape == curr_joints.shape:
            #     max_delta = (th.abs(curr_joints - reset_joints)).max()
            #     steps = min(int(max_delta / 0.01), 100)
            #
            #     for jnt in tqdm(
            #         th.linspace(curr_joints, reset_joints, steps),
            #         desc="Reset to default joint...",
            #     ):
            #         env.step(jnt)
            #         time.sleep(0.001)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()

    # # TODO: Make less hacky
    # joints = obs[f"arm_{active_arm}_joint_positions"]
    #
    # abs_deltas = th.abs(start_pos - joints)
    # id_max_joint_delta = th.argmax(abs_deltas)

    # max_joint_delta = 0.8
    # if abs_deltas[id_max_joint_delta] > max_joint_delta:
    #     id_mask = abs_deltas > max_joint_delta
    #     print()
    #     ids = np.arange(len(id_mask))[id_mask]
    #     for i, delta, joint, current_j in zip(
    #         ids,
    #         abs_deltas[id_mask],
    #         start_pos[id_mask],
    #         joints[id_mask],
    #     ):
    #         print(
    #             f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
    #         )
    #     return

    # print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    # assert len(start_pos) == len(
    #     joints
    # ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    # max_delta = 0.05
    # for _ in tqdm(range(25), desc="Reset to start position..."):
    #     obs = env.get_obs()
    #     command_joints = agent.act(obs)
    #     current_joints = obs[f"arm_{active_arm}_joint_positions"]
    #     delta = command_joints - current_joints
    #     max_joint_delta = th.abs(delta).max()
    #     if max_joint_delta > max_delta:
    #         delta = delta / max_joint_delta * max_delta
    #
    #     # env_action = th.zeros(env.robot().num_dofs())
    #     # env_action[obs[f"arm_{active_arm}_control_idx"]] = current_joints + delta
    #     # env.step(env_action)
    #     env.step(current_joints + delta)
    #
    # obs = env.get_obs()
    # joints = obs["joint_positions"]
    # action = agent.act(obs)
    # if (action - joints > 0.5).any():
    #     print("Action is too big")
    #
    #     # print which joints are too big
    #     joint_index = np.where(action - joints > 0.5)
    #     for j in joint_index:
    #         print(
    #             f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
    #         )
    #     exit()

    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset

        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    save_path = None
    start_time = time.time()
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        action = agent.act(obs)
        dt = datetime.datetime.now()
        if args.use_save_interface:
            state = kb_interface.update()
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = (
                    Path(args.data_dir).expanduser()
                    / args.agent
                    / dt_time.strftime("%m%d_%H%M%S")
                )
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Saving to {save_path}")
            elif state == "save":
                assert save_path is not None, "something went wrong"
                save_frame(save_path, dt, obs, action)
            elif state == "normal":
                save_path = None
            else:
                raise ValueError(f"Invalid state {state}")
        obs = env.step(action)


if __name__ == "__main__":
    main(tyro.cli(Args))
