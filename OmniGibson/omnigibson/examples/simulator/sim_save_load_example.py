import os

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardEventHandler

TEST_OUT_PATH = ""  # Define output directory here.


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select whether they are saving or loading an environment, and interactively
    shows how an environment can be saved or restored.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "walls", "bed", "bottom_cabinet", "chair"],
        },
        "robots": [
            {
                "type": "Turtlebot",
                "obs_modalities": ["rgb", "depth"],
            },
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Set the camera to a good angle
    def set_camera_pose():
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([-0.229375, -3.40576, 7.26143]),
            orientation=th.tensor([0.27619733, -0.00230233, -0.00801152, 0.9610648]),
        )

    set_camera_pose()

    # Give user instructions, and then loop until completed
    completed = short_exec
    if not short_exec and not random_selection:
        # Notify user to manipulate environment until ready, then press Z to exit
        print()
        print("Modify the scene by SHIFT + left clicking objects and dragging them. Once finished, press Z.")

        # Register callback so user knows to press space once they're done manipulating the scene
        def complete_loop():
            nonlocal completed
            completed = True

        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.Z, complete_loop)
    while not completed:
        action_lo, action_hi = -1, 1
        env.step(th.rand(env.robots[0].action_dim) * (action_hi - action_lo) + action_lo)

    print("Completed scene modification, saving scene...")
    save_path = os.path.join(TEST_OUT_PATH, "saved_stage.json")
    og.sim.save(json_paths=[save_path])

    print("Re-loading scene...")
    og.clear()
    og.sim.restore(scene_files=[save_path])

    # env is no longer valid after og.clear()
    del env

    # Take a sim step and play
    og.sim.step()
    og.sim.play()
    set_camera_pose()

    # Loop until user terminates
    completed = short_exec
    if not short_exec and not random_selection:
        # Notify user to manipulate environment until ready, then press Z to exit
        print()
        print("View reloaded scene. Once finished, press Z.")
        # Register callback so user knows to press space once they're done manipulating the scene
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.Z, complete_loop)
    while not completed:
        og.sim.step()

    # Always shut down the environment cleanly at the end
    og.shutdown()


if __name__ == "__main__":
    main()
