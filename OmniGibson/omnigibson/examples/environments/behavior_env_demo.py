import os

import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Generates a BEHAVIOR Task environment in an online fashion.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Ask the user whether they want online object sampling or not
    sampling_options = {
        False: "Use a pre-sampled cached BEHAVIOR activity scene",
        True: "Sample the BEHAVIOR activity in an online fashion",
    }
    should_sample = choose_from_options(
        options=sampling_options, name="online object sampling", random_selection=random_selection
    )

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["task"]["online_object_sampling"] = should_sample

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Run a simple loop and reset periodically
    max_iterations = 10 if not short_exec else 1
    for j in range(max_iterations):
        og.log.info("Resetting environment")
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                og.log.info("Episode finished after {} timesteps".format(i + 1))
                break

    # Always close the environment at the end
    og.clear()


if __name__ == "__main__":
    main()
