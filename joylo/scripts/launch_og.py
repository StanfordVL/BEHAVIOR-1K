import tyro
from dataclasses import dataclass
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
    attachment_joint_visuals: Optional[bool] = None
    # "physx" or "newton"
    physics_backend: str = "physx"
    # Render backend when --physics-backend newton: "newton" or "none" (no camera capture at all).
    # Ignored for physx.
    renderer: Optional[str] = None
    # Renderer for the live, on-screen Newton viewer window: "gl" or "rtx"
    viewer_renderer: str = "gl"
    # Show the secondary cameras in NewtonGL
    viewer_panels: bool = True


def launch_robot_server(args: Args):
    assert args.physics_backend in ("physx", "newton"), f"Invalid physics backend: {args.physics_backend}"
    assert args.viewer_renderer in ("gl", "rtx"), f"Invalid viewer renderer: {args.viewer_renderer}"
    if args.physics_backend == "newton":
        from omnigibson.macros import gm

        gm.PHYSICS_BACKEND = "newton"
        gm.RENDER_BACKEND = args.renderer if args.renderer is not None else "newton"

    from gello.robots.og_robot import OGRobotServer

    server = OGRobotServer(
        robot=args.robot,
        port=args.robot_port,
        host=args.hostname,
        recording_path=args.recording_path,
        task_name=args.task_name,
        partial_load=args.partial_load,
        instance_id=args.instance_id,
        ghosting=args.ghosting,
        attachment_joint_visuals=args.attachment_joint_visuals,
        viewer_renderer=args.viewer_renderer,
        viewer_panels=args.viewer_panels,
    )
    server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
