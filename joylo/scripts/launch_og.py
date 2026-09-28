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
    # Physics backend: "physx" (default; Kit/Isaac Sim rendering + full JoyLo UI) or "newton"
    # (Kit-free MuJoCo-Warp physics; rendering through Newton's own GL viewer window, with Kit-only
    # UI extras -- viewports, overlay UI, ghost robot, VR -- disabled).
    physics_backend: str = "physx"
    # Render backend when --physics-backend newton: "newton" (default; VisionSensor rgb capture
    # available through Newton's GL renderer) or "none" (no camera capture at all). Ignored for physx.
    renderer: Optional[str] = None
    # Renderer for the live, on-screen Newton viewer window (only used when --physics-backend newton;
    # this is the window you actually watch/drive the robot through, separate from --renderer above,
    # which controls the robot's own onboard VisionSensor rgb capture): "gl" (default; rasterizer, no
    # known visual issues) or "rtx" (real ray tracer, higher fidelity but renders an enclosed interior
    # scene near-black due to Newton's default lighting rig -- fine for the default open-air testing
    # scene, degrades for a full household InteractiveTraversableScene).
    viewer_renderer: str = "gl"
    # Show the secondary cameras (left/right shoulder + wrist, i.e. what JoyLo docks as extra Kit
    # viewports) as docked image panels inside the Newton viewer window. GL viewer only -- Newton's
    # log_image() is implemented by ViewerGL and is a no-op on ViewerRTX -- and needs --renderer
    # newton (the "none" render backend cannot capture images). Each panel is an extra off-screen
    # render, so pass --no-viewer-panels if you want every frame spent on the main view.
    viewer_panels: bool = True
    # Grasping mode for the robot: "assisted" (default), "sticky", or "physical". Only "physical"
    # skips Robot.post_step()'s assisted-grasping handling, whose raycasts dominate the teleop tick
    # under --physics-backend newton (measured on r1pro: 103 ms of a 186 ms tick -> ~5.4 fps; with
    # "physical" the tick drops to ~84 ms, ~12 fps). Pass --grasping-mode physical to trade real
    # grasping (objects then only stay in hand by friction) for frame rate. Defaults to
    # og_teleop_cfg.GRASPING_MODE when not given.
    grasping_mode: Optional[str] = None


def launch_robot_server(args: Args):
    # gm must be configured before any og.Environment is constructed (the app/backend choice is read
    # at launch time). Importing omnigibson (via og_robot) does NOT launch the app, so setting the
    # macros here, before OGRobotServer is constructed, is early enough.
    assert args.physics_backend in ("physx", "newton"), f"Invalid physics backend: {args.physics_backend}"
    assert args.viewer_renderer in ("gl", "rtx"), f"Invalid viewer renderer: {args.viewer_renderer}"
    assert args.grasping_mode in (None, "physical", "assisted", "sticky"), (
        f"Invalid grasping mode: {args.grasping_mode}"
    )
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
        grasping_mode=args.grasping_mode,
    )
    server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
