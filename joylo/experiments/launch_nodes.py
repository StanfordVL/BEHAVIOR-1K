import tyro
from dataclasses import dataclass
from gello.robots.og_robot import OGRobotServer
from typing import Optional


@dataclass
class Args:
    robot: str = "r1pro" # Robot type
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    recording_path: Optional[str] = None
    task_name: Optional[str] = None
    partial_load: Optional[bool] = True
    instance_id: Optional[int] = None
    ghosting: Optional[bool] = True


def launch_robot_server(args: Args):

    server = OGRobotServer(
        robot=args.robot,
        port=args.robot_port,
        host=args.hostname,
        recording_path=args.recording_path,
        task_name=args.task_name,
        partial_load=args.partial_load,
        instance_id=args.instance_id,
        ghosting=args.ghosting
    )
    server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
