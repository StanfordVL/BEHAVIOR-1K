import logging
import os

import yaml

import omnigibson as og
from omnigibson.utils.asset_utils import folder_is_hidden
from omnigibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Grab all configs and choose one to load
    og_config_path = og.example_config_path
    available_configs = sorted(
        [
            f
            for f in os.listdir(og_config_path)
            if (not folder_is_hidden(f) and os.path.isfile(os.path.join(og_config_path, f)))
        ]
    )
    config_id = choose_from_options(options=available_configs, name="config file", random_selection=random_selection)
    logging.info("Using config file " + config_id)
    config_filename = os.path.join(og.example_config_path, config_id)
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    load_options = {
        "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
        "Full": "Load all interactive objects in the scene",
    }
    load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
    if load_mode == "Quick":
        config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load the environment
    env = og.Environment(configs=config)

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
