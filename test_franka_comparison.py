"""
Quick comparison test: load the WORKING FrankaMounted robot to see if rendering
looks correct with the same settings. If it looks connected, our USD is the problem.
If it also looks like disconnected blobs, the issue is rendering settings.

Usage:
    conda run -n behavior python test_franka_comparison.py
"""
import torch as th
import omnigibson as og
from omnigibson.macros import gm

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False
gm.USE_PBR_MATERIALS = True


def main():
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "type": "FrankaMounted",
                "name": "franka_standard",
                "obs_modalities": [],
                "fixed_base": True,
                "self_collisions": False,
            }
        ],
    }

    env = og.Environment(configs=cfg)
    robot = env.robots[0]

    print(f"\n--- STANDARD FRANKA DEBUG ---")
    print(f"fixed_base: {robot.fixed_base}")
    print(f"Root link: {robot.root_link_name}")
    print(f"n_joints: {robot.n_joints}")

    for _ in range(5):
        og.sim.step()

    for link_name, link_obj in robot.links.items():
        pos = link_obj.get_position_orientation()[0]
        print(f"  {link_name:30s} pos: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]")
    print(f"--- END ---\n")

    robot.reset()
    robot.keep_still()

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.5, 1.5, 1.5]),
        orientation=th.tensor([0.2706, 0.2706, 0.6533, 0.6533]),
    )
    og.sim.enable_viewer_camera_teleoperation()

    print("Standard FrankaMounted loaded. Compare visually with the SharpaWave version.")
    print("Press Ctrl+C to exit.")

    # Warmup
    for _ in range(200):
        og.sim.render()

    while True:
        og.sim.step()


if __name__ == "__main__":
    main()
