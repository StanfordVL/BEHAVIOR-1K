"""
Example script for interacting with OmniGibson scenes with VR.
"""

import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_og_scenes
from omnigibson.utils.teleop_utils import OVXRSystem
from omnigibson.utils.ui_utils import choose_from_options

gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False


def main():
    """
    Users can navigate around and interact with a selected scene using VR.
    """

    # Choose the scene model to load
    scenes = get_available_og_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model")

    # Create the config for generating the environment we want
    scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": scene_model}
    cfg = dict(scene=scene_cfg)

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    # start vrsys
    vrsys = OVXRSystem(
        robot=None,
        show_control_marker=True,
        system="SteamVR",
        eef_tracking_mode="disabled",
        align_anchor_to="touchpad",
    )
    vrsys.start()
    # set headset position to be 1m above ground and facing +x
    vrsys.xr_core.schedule_set_camera(
        vrsys.og2xr(th.tensor([0.0, 0.0, 1.0]), th.tensor([-0.5, 0.5, 0.5, -0.5])).numpy()
    )

    # main simulation loop
    for _ in range(3000):
        # step the VR system to get the latest data from VR runtime
        vrsys.update(optimized_for_tour=True)
        og.sim.render()

    # Shut down the environment cleanly at the end
    print("Cleaning up...")
    vrsys.stop()
    og.clear()


if __name__ == "__main__":
    main()
