import logging
import os

import yaml

import igibson as ig


def main(random_selection=False, headless=False, short_exec=False):
    """
    Generates a BEHAVIOR Task environment from a pre-defined configuration file.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Load the pre-selected configuration
    config_filename = os.path.join(ig.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

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
            import numpy as np
            action = np.zeros(env.robots[0].action_dim)
            state, reward, done, info = env.step(action)
            if done:
                logging.info("Episode finished after {} timesteps".format(i + 1))
                break

    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()
