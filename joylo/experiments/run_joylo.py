import glob
import time
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import tyro

from gello.agents.bimanual_agent import (
    ROBOT_TELEOP_CONFIGS,
    BimanualAgent,
    MotorFeedbackConfig,
)
from gello.agents.dynamixel_arm_agent import DynamixelArmAgent, DynamixelRobotConfig
from gello.agents.joycon_agent import JoyconAgent
from gello.env import RobotEnv
from gello.robots.base_robot import PrintRobot
from gello.utils.zmq_utils import ZMQRobotClient
from gello import REPO_DIR


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    gello_model: str = "r1pro"
    gello_name: str = "default"
    gello_port: Optional[str] = None
    mock: bool = False
    damping_motor_kp: float = 0.3
    motor_feedback_type: str = "NONE"
    use_joycons: bool = True


def main(args):
    assert args.gello_model in ROBOT_TELEOP_CONFIGS, (
        f"Unsupported gello model: {args.gello_model}"
    )
    bimanual_config = ROBOT_TELEOP_CONFIGS[args.gello_model]

    if args.mock:
        robot_client = PrintRobot(bimanual_config.joints_per_arm * 2, dont_print=True)
    else:
        robot_client = ZMQRobotClient(port=args.robot_port, host=args.hostname)

    env = RobotEnv(robot_client, control_rate_hz=args.hz)

    # Find gello port
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
            raise ValueError("No gello port found, please specify one or plug in gello")

    # Read joint config from yaml
    with open(f"{REPO_DIR}/configs/joint_config_{args.gello_name}.yaml", "r") as file:
        joint_config = yaml.load(file, Loader=yaml.SafeLoader)

    num_motors = bimanual_config.motors_per_arm * 2
    dynamixel_config = DynamixelRobotConfig(
        joint_ids=tuple(np.arange(num_motors).tolist()),
        joint_offsets=[np.deg2rad(x) for x in joint_config["joints"]["offsets"]],
        joint_signs=joint_config["joints"]["signs"],
        gripper_config=None,
    )

    # Default start joints
    start_joints = args.start_joints
    if start_joints is None:
        start_joints = bimanual_config.start_joints.copy()

    # Create JoyCon agent
    joycon_agent = None
    if args.use_joycons:
        joycon_agent = JoyconAgent(
            calibration_dir=f"{REPO_DIR}/configs",
            deadzone_threshold=0.2,
            max_translation=0.35,
            max_rotation=0.3,
            max_trunk_translate=0.1,
            max_trunk_tilt=0.05,
            enable_rumble=False,
        )

    # Create arm agent and bimanual agent
    arm_agent = DynamixelArmAgent(
        port=gello_port,
        dynamixel_config=dynamixel_config,
        start_joints=start_joints,
        damping_motor_kp=args.damping_motor_kp,
    )

    agent = BimanualAgent(
        config=bimanual_config,
        arm_agent=arm_agent,
        joycon_agent=joycon_agent,
        motor_feedback_type=MotorFeedbackConfig[args.motor_feedback_type],
    )

    agent.start()

    print("Going to start position")
    agent.reset()

    print_color("*" * 40, color="magenta", attrs=("bold",))
    print_color(
        f"\nWelcome to JoyLo ({args.gello_model.upper()})!\n",
        color="magenta",
        attrs=("bold",),
    )
    print_color(
        f"{args.gello_model.upper()} Teleoperation Commands:\n",
        color="magenta",
        attrs=("bold",),
    )
    print_color("\t ZL / ZR: Toggle grasping", color="magenta", attrs=("bold",))
    print_color(
        "\t Left Joystick (not pressed): Translate the robot base",
        color="magenta",
        attrs=("bold",),
    )
    print_color(
        "\t Right Joystick: Rotate the robot base + tilt the trunk torso",
        color="magenta",
        attrs=("bold",),
    )
    print_color(
        "\t Up / Down Button: Raise / Lower the trunk torso",
        color="magenta",
        attrs=("bold",),
    )
    print_color(
        "\t Left / Right Button: Toggle gripper light",
        color="magenta",
        attrs=("bold",),
    )
    if args.gello_model == "r1":
        print_color(
            "\t L / R: Lock the lower two wrist joints while leaving the upper joints free",
            color="magenta",
            attrs=("bold",),
        )
    elif args.gello_model == "r1pro":
        print_color(
            "\t L / R: Lock the lower three wrist joints while leaving the upper joints free",
            color="magenta",
            attrs=("bold",),
        )
    print_color(
        "\t - / +: Lock the upper joints while leaving the lower wrist roll joint free.\n\t\tThe wrist pose will NOT be tracked while held.",
        color="magenta",
        attrs=("bold",),
    )
    print_color(
        "\t Y: Move the robot towards its reset pose", color="magenta", attrs=("bold",)
    )
    print_color("\t B: Toggle camera", color="magenta", attrs=("bold",))
    print_color("\t A: Toggle X-ray", color="magenta", attrs=("bold",))
    print_color("\t Home: Reset the environment\n", color="magenta", attrs=("bold",))
    print_color("*" * 40, color="magenta", attrs=("bold",))

    obs = env.get_obs()
    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))
    start_time = time.time()
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(message, color="white", attrs=("bold",), end="", flush=True)
        action = agent.act(obs)
        obs = env.step(action)


if __name__ == "__main__":
    main(tyro.cli(Args))
