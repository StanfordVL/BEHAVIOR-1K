"""
Test script to load and visualize the Franka+SharpaWave robot
using OmniGibson's proper robot loading pipeline (with full physics).

Usage:
    conda run -n behavior python test_sharpa_robot.py                # Right hand (default)
    conda run -n behavior python test_sharpa_robot.py --hand right   # Right hand
    conda run -n behavior python test_sharpa_robot.py --hand left    # Left hand
"""
import argparse
import torch as th
import omnigibson as og
from omnigibson.macros import gm

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False  # Flatcache can desync visual transforms for complex articulations
gm.USE_PBR_MATERIALS = True

HAND_TO_ROBOT_CLASS = {
    "right": "franka_mounted_sharpa_right",
    "left": "franka_mounted_sharpa_left",
}


def main():
    parser = argparse.ArgumentParser(description="Test Franka+SharpaWave robot in OmniGibson")
    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
        help="Which hand variant to test (default: right)",
    )
    args = parser.parse_args()

    robot_class = HAND_TO_ROBOT_CLASS[args.hand]
    print(f"\n*** Loading {robot_class} ({args.hand} hand) ***\n")

    # Create empty scene with the robot
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "model": robot_class,
                "name": f"franka_sharpa_{args.hand}",
                "obs_modalities": [],
                "fixed_base": True,
                "self_collisions": False,
                "grasping_direction": "upper",
                "load_config": {
                    "xform_props_pre_loaded": False,
                },
            }
        ],
    }

    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    # right/left_hand_C_MC is the physical palm — restore visibility hidden by robot._initialize()
    for arm in robot.arm_names:
        robot.links[robot.eef_link_names[arm]].visible = True

    # Switch to RTX-Real Time renderer (avoids path-tracing noise)
    try:
        import carb
        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", "RaytracedLighting")
    except Exception as e:
        print(f"Could not switch renderer: {e}")

    # Debug: print articulation info
    print(f"\n--- ROBOT DEBUG INFO ---")
    print(f"Robot class: {robot.__class__.__name__}")
    print(f"fixed_base: {robot.fixed_base}")
    print(f"kinematic_only: {robot.kinematic_only}")
    print(f"Root link: {robot.root_link_name}")
    print(f"n_joints: {robot.n_joints}")
    print(f"n_dof: {robot.n_dof}")
    print(f"Articulated: {robot.articulated}")
    print(f"Articulation root path: {robot.articulation_root_path}")
    print(f"prim_path: {robot.prim_path}")

    # Check root link position
    root_pos = robot.root_link.get_position_orientation()[0]
    print(f"Root link position BEFORE step: {root_pos}")

    # Step physics a few times
    for _ in range(5):
        og.sim.step()

    root_pos_after = robot.root_link.get_position_orientation()[0]
    print(f"Root link position AFTER 5 steps: {root_pos_after}")
    print(f"Position changed (falling?): {not th.allclose(root_pos, root_pos_after, atol=1e-4)}")

    # Print ALL link positions to see if they form a kinematic chain
    print(f"\n  --- ALL LINK POSITIONS ---")
    for link_name, link_obj in robot.links.items():
        pos = link_obj.get_position_orientation()[0]
        print(f"  {link_name:30s} pos: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]")
    print(f"  --- END ALL LINKS ---")
    print(f"--- END DEBUG INFO ---\n")

    # Debug: check visual/collision meshes for every link
    print(f"\n  --- VISUAL/COLLISION MESH DEBUG ---")
    for link_name, link_obj in robot.links.items():
        vis = link_obj.visual_meshes
        col = link_obj.collision_meshes
        vis_names = list(vis.keys()) if vis else []
        col_names = list(col.keys()) if col else []
        marker = "  " if vis_names else "!!"
        print(f"  {marker} {link_name:30s} visuals={vis_names}  collisions={col_names}")
    print(f"  --- END MESH DEBUG ---\n")

    # Reset robot to default pose
    robot.reset()
    robot.keep_still()

    # Position camera further back to see the full robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.5, 1.5, 1.5]),
        orientation=th.tensor([0.2706, 0.2706, 0.6533, 0.6533]),
    )
    og.sim.enable_viewer_camera_teleoperation()

    print("\n" + "=" * 60)
    print(f"Franka + SharpaWave {args.hand.upper()} loaded with full physics!")
    print("  Alt + Left-click  = Orbit")
    print("  Alt + Right-click = Zoom")
    print("  Alt + Middle-click = Pan")
    print("  Ctrl+C to exit")
    print("=" * 60 + "\n")

    # Give renderer time to converge (200 warmup frames)
    print("Warming up renderer (200 frames)...")
    for _ in range(200):
        og.sim.render()
    print("Warmup done. Robot should be visible now.")

    while True:
        og.sim.step()


if __name__ == "__main__":
    main()
