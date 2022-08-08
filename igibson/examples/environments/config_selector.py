import logging
import os
from sys import platform

import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.asset_utils import folder_is_hidden
from igibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    ig_config_path = igibson.example_config_path
    available_configs = sorted(
        [
            f
            for f in os.listdir(ig_config_path)
            if (not folder_is_hidden(f) and os.path.isfile(os.path.join(ig_config_path, f)))
        ]
    )
    config_id = choose_from_options(options=available_configs, name="config file", random_selection=random_selection)
    logging.info("Using config file " + config_id)
    config_filename = os.path.join(igibson.example_config_path, config_id)
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5
    config_data["image_width"] = 512
    config_data["image_height"] = 512
    config_data["vertical_fov"] = 60
    # config_data["load_object_categories"] = []  # Uncomment this line to accelerate loading with only the building
    env = iGibsonEnv(config_file=config_data, mode="gui_interactive" if not headless else "headless")
    max_iterations = 10 if not short_exec else 1
    for j in range(max_iterations):
        logging.info("Resetting environment")
        env.reset()
        for i in range(100):
            with Profiler("Environment action step"):
                action = env.action_space.sample() * 0.05
                state, reward, done, info = env.step(action)
                if done:
                    logging.info("Episode finished after {} timesteps".format(i + 1))
                    break
    env.close()


if __name__ == "__main__":
    main()
