import numpy as np

import omnigibson as og
from omnigibson.robots import REGISTERED_ROBOTS


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot demo
    Loads all robots in an empty scene, generate random actions
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
    # Create empty scene with no robots in it initially
    cfg = {
        "scene": {
            "type": "Scene",
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
        og.log.info(f"Loaded {robot_name}")
        og.log.info(f"Moving {robot_name}")

        if not headless:
            # Set viewer in front facing robot
            og.sim.viewer_camera.set_position_orientation(
                position=np.array([ 2.69918369, -3.63686664,  4.57894564]),
                orientation=np.array([0.39592411, 0.1348514 , 0.29286304, 0.85982   ]),
            )

        og.sim.enable_viewer_camera_teleoperation()

        # Hold still briefly so viewer can see robot
        for _ in range(100):
            og.sim.step()

        # Then apply random actions for a bit
        for _ in range(30):
            action = np.random.uniform(-1, 1, robot.action_dim)
            for _ in range(10):
                env.step(action)

        # Stop the simulator and remove the robot
        og.sim.stop()
        og.sim.remove_object(obj=robot)

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
