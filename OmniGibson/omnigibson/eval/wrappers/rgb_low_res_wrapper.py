from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.eval.utils.eval_utils import ROBOT_CAMERA_NAMES


# Create module logger
logger = create_module_logger("RGBLowResWrapper")


class RGBLowResWrapper(EnvironmentWrapper):
    """
    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        # Here, we modify the robot observation to use 224 * 224 resolution
        # For a complete list of available modalities, see VisionSensor.ALL_MODALITIES
        # env.robots is list[list[Robot]] (one inner list per scene); flatten to every robot.
        for robot in [r for scene_robots in env.robots for r in scene_robots]:
            for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
                sensor_name = camera_name.split("::")[1]
                sensor = robot.sensors[sensor_name]
                if camera_id == "head":
                    sensor.horizontal_aperture = 40.0  # this is what we used in data collection
                sensor.image_height = 224
                sensor.image_width = 224
        # Full reload is num_envs-safe (the per-sensor obs_space patch assumed a single-env obs space).
        env.load_observation_space()
        logger.info("Reloaded observation space!")
