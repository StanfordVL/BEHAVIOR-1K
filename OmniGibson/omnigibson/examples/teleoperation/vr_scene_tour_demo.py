"""
Example script for interacting with OmniGibson scenes with VR.
"""

import sys
from typing import Optional
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_behavior_1k_scenes
from omnigibson.utils.teleop_utils import OVXRSystem
from omnigibson.utils.ui_utils import KeyboardEventHandler, choose_from_options

gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False


def main(scene_model: Optional[str] = None):
    """
    Users can navigate around and interact with a selected scene using VR.
    """

    # Choose the scene model to load
    if not scene_model:
        scenes = get_available_behavior_1k_scenes()
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
    is_done_with_sim = False

    def _exit():
        nonlocal is_done_with_sim
        is_done_with_sim = True

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.SPACE,
        callback_fn=_exit,
    )
    stepping = False

    def _toggle_stepping():
        nonlocal stepping
        stepping = not stepping

    KeyboardEventHandler.add_keyboard_callback(
        key="right_a",
        callback_fn=_toggle_stepping,
    )
    should_reset = False

    def _queue_reset():
        nonlocal should_reset
        should_reset = True

    KeyboardEventHandler.add_keyboard_callback(
        key="right_b",
        callback_fn=_queue_reset,
    )
    while not is_done_with_sim:
        # step the VR system to get the latest data from VR runtime
        vrsys.update(optimized_for_tour=True)
        if stepping:
            og.sim.step()
        else:
            og.sim.render()
        if should_reset:
            env.reset()
            should_reset = False

    # Shut down the environment cleanly at the end
    vrsys.stop()
    og.clear()


if __name__ == "__main__":
    scene_model = None
    if len(sys.argv) > 1:
        scene_model = sys.argv[1]
    main(scene_model=scene_model)
