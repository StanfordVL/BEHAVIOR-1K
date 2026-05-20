"""
Load the Franka Panda robot, step the simulator, record frames, save as mp4.
Usage:
    conda run -n behavior python visualize_franka.py
"""
import os
import imageio
import numpy as np
import torch as th

import omnigibson as og
from omnigibson.macros import gm

gm.HEADLESS = True

# Filter out system directories that are missing metadata.json (dataset is incomplete).
import omnigibson.scenes.scene_base as _scene_base
from omnigibson.utils.asset_utils import get_dataset_path as _get_dataset_path

def _filtered_system_names():
    system_dir = os.path.join(_get_dataset_path("behavior-1k-assets"), "systems")
    if not os.path.exists(system_dir):
        return set()
    return {d for d in os.listdir(system_dir)
            if os.path.isfile(os.path.join(system_dir, d, "metadata.json"))}

_scene_base.get_all_system_names = _filtered_system_names

OUTPUT_VIDEO = "/tmp/franka_visualization.mp4"
N_FRAMES = 60  # ~2 seconds at 30 fps

cfg = {
    "scene": {"type": "Scene"},
    "robots": [
        {
            "name": "franka",
            "model": "franka",
            "obs_modalities": ["rgb"],
        }
    ],
}

env = og.Environment(configs=cfg)

og.sim.viewer_camera.set_position_orientation(
    position=th.tensor([2.0, -2.0, 1.5]),
    orientation=th.tensor([0.39592411, 0.1348514, 0.29286304, 0.85982]),
)

robot = env.robots[0]
robot.reset()
robot.keep_still()

for _ in range(20):
    og.sim.step()

frames = []
for i in range(N_FRAMES):
    action = robot.action_space.sample() * 0.1
    env.step(action)
    rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
    if isinstance(rgb, th.Tensor):
        rgb = rgb.cpu().numpy()
    frames.append(rgb[..., :3].astype(np.uint8))

imageio.mimsave(OUTPUT_VIDEO, frames, fps=30)
print(f"Saved video to {OUTPUT_VIDEO}")
og.shutdown()
