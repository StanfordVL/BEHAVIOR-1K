import logging

import numpy as np

import omnigibson as og
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.scenes import EmptyScene


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot demo
    Loads all robots in an empty scene, generate random actions
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # Create empty scene with no robots in it initially
    cfg = {
        "scene": {
            "type": "EmptyScene",
        }
    }

    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Iterate over all robots and demo their motion
    for robot_name, robot_cls in REGISTERED_ROBOTS.items():
        # Create and import robot
        robot = robot_cls(
            prim_path=f"/World/{robot_name}",
            name=robot_name,
            obs_modalities=[],              # We're just moving robots around so don't load any observation modalities
        )
        og.sim.import_object(robot)

        # At least one step is always needed while sim is playing for any imported object to be fully initialized
        og.sim.play()
        og.sim.step()

        # Reset robot and make sure it's not moving
        robot.reset()
        robot.keep_still()

        # Log information
        logging.info("Loaded " + robot_name)
        logging.info("Moving " + robot_name)

        if not headless:
            # Set viewer in front facing robot
            og.sim.viewer_camera.set_position_orientation(
                position=np.array([4.32248, -5.74338, 6.85436]),
                orientation=np.array([0.39592, 0.13485, 0.29286, 0.85982]),
            )

        # Hold still briefly so viewer can see robot
        for _ in range(100):
            og.sim.step()

        # Then apply random actions for a bit
        for _ in range(30):
            action = np.random.uniform(-1, 1, robot.action_dim)
            for _ in range(10):
                env.step(action)

        # Re-import the scene
        og.sim.stop()
        og.sim.import_scene(EmptyScene())

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
