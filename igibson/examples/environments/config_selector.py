import logging
import os

import yaml

import igibson as ig
from igibson.utils.asset_utils import folder_is_hidden
from igibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Grab all configs and choose one to load
    ig_config_path = ig.example_config_path
    available_configs = sorted(
        [
            f
            for f in os.listdir(ig_config_path)
            if (not folder_is_hidden(f) and os.path.isfile(os.path.join(ig_config_path, f)))
        ]
    )
    config_id = choose_from_options(options=available_configs, name="config file", random_selection=random_selection)
    logging.info("Using config file " + config_id)
    config_filename = os.path.join(ig.example_config_path, config_id)
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Uncomment the following line to accelerate loading with only the building
    # config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load the environment
    env = ig.Environment(configs=config)

    max_iterations = 10 if not short_exec else 1
    for j in range(max_iterations):
        logging.info("Resetting environment")
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            for robot_name in action.keys():
                action[robot_name] = action[robot_name] * 0.05
            state, reward, done, info = env.step(action)
            if done:
                logging.info("Episode finished after {} timesteps".format(i + 1))
                break

    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()
