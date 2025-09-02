import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_behavior_1k_scenes
from omnigibson.utils.ui_utils import KeyboardEventHandler, choose_from_options
from omnigibson.utils.constants import STRUCTURE_CATEGORIES


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads it.

    It sets the camera to various poses and records images, and then generates a trajectory from a set of waypoints
    and records the resulting video.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Make sure the example is not being run headless. If so, terminate early
    if gm.HEADLESS:
        print("This demo should only be run not headless! Exiting early.")
        og.shutdown()

    # Choose the scene model to load
    scenes = get_available_behavior_1k_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    print(f"scene model: {scene_model}")

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
        },
    }

    # Check if we want to quick load or full load the scene
    load_options = {
        "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
        "Full": "Load all interactive objects in the scene",
    }
    load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
    if load_mode == "Quick":
        cfg["scene"]["load_object_categories"] = list(STRUCTURE_CATEGORIES)

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to teleoperate the camera
    cam_mover = og.sim.enable_viewer_camera_teleoperation()

    # Create a keyboard event handler for generating waypoints
    waypoints = []

    def add_waypoint():
        nonlocal waypoints
        pos = cam_mover.cam.get_position_orientation()[0]
        print(f"Added waypoint at {pos}")
        waypoints.append(pos)

    def clear_waypoints():
        nonlocal waypoints
        print("Cleared all waypoints!")
        waypoints = []

    KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.X,
        callback_fn=add_waypoint,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.C,
        callback_fn=clear_waypoints,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.J,
        callback_fn=lambda: cam_mover.record_trajectory_from_waypoints(
            waypoints=th.tensor(waypoints),
            per_step_distance=0.02,
            fps=30,
            steps_per_frame=1,
            fpath=None,  # This corresponds to the default path inferred from cam_mover.save_dir
        ),
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=lambda: og.shutdown(),
    )

    # Print out additional keyboard commands
    print("\t X: Save the current camera pose as a waypoint")
    print("\t C: Clear all waypoints")
    print("\t J: Record the camera trajectory from the current set of waypoints")
    print("\t ESC: Terminate the demo")

    # Loop indefinitely
    steps = 0
    max_steps = -1 if not short_exec else 100
    while steps != max_steps:
        env.step([])
        steps += 1

    og.shutdown()


if __name__ == "__main__":
    main()
