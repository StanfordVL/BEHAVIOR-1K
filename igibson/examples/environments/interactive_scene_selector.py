import logging
import os

import yaml

import igibson as ig
from igibson.utils.asset_utils import get_available_ig_scenes
from igibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = os.path.join(ig.example_config_path, "turtlebot_nav.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Uncomment the following line to accelerate loading with only the building
    # config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Choose the scene to load, modify the config appropriately, and create the environment
    ig_scenes = get_available_ig_scenes()
    scene_model = choose_from_options(options=ig_scenes, name="scene model", random_selection=random_selection)
    print(f"scene model: {scene_model}")
    config["scene"]["scene_model"] = scene_model
    env = ig.Environment(configs=config)

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
