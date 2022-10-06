import logging
import igibson as ig
from igibson.utils.asset_utils import get_available_g_scenes, get_available_ig_scenes
from igibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available non-interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Choose the scene type to load
    scene_options = {
        "InteractiveTraversableScene": "Procedurally generated scene with fully interactive objects",
        "StaticTraversableScene": "Monolithic scene mesh with no interactive objects",
    }
    scene_type = choose_from_options(options=scene_options, name="scene type", random_selection=random_selection)

    # Choose the scene model to load
    scenes = get_available_ig_scenes() if scene_type == "InteractiveTraversableScene" else get_available_g_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    print(f"scene model: {scene_model}")

    cfg = {
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
        },
        "robots": [
            {
                "type": "Turtlebot",
                "obs_modalities": ["scan", "rgb", "depth"],
                "action_type": "continuous",
                "action_normalize": True,
            }
        ],
    }

    # If the scene type is interactive, also check if we want to quick load or full load the scene
    if scene_type == "InteractiveTraversableScene":
        load_options = {
            "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
            "Full": "Load all interactive objects in the scene",
        }
        load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
        if load_mode == "Quick":
            cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load the environment
    env = ig.Environment(configs=cfg)

    # Allow user to move camera more easily
    ig.sim.enable_viewer_camera_teleoperation()

    # Run a simple loop and reset periodically
    max_iterations = 10 if not short_exec else 1
    for j in range(max_iterations):
        logging.info("Resetting environment")
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                logging.info("Episode finished after {} timesteps".format(i + 1))
                break

    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()
