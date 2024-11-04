"""
Example script for interacting with OmniGibson scenes with VR.
"""
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.teleop_utils import OVXRSystem
from omnigibson.utils.asset_utils import get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options

def main():
    """
    Users can navigate around and interact with a selected scene using VR.
    """

    # Choose the scene model to load
    scenes = get_available_og_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model")

    # Create the config for generating the environment we want
    scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": scene_model}
    robot0_cfg = {
        "type": "Fetch",
        "obs_modalities": [],
    }
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    # start vrsys
    vrsys = OVXRSystem(
        robot=env.robots[0], show_control_marker=True, system="SteamVR", eef_tracking_mode="disabled", align_anchor_to="touchpad"
    )
    vrsys.start()
    # set headset position to be 1m above ground and facing +x
    vrsys.set_initial_transform(pos=th.tensor([0.0, 0.0, 1.0]), orn=th.tensor([0.0, 0.0, 0.0, 1.0]))

    # main simulation loop
    while True:
        # step the VR system to get the latest data from VR runtime
        vrsys.update()
        og.sim.render()

    # Shut down the environment cleanly at the end
    vrsys.stop()
    og.clear()


if __name__ == "__main__":
    main()
