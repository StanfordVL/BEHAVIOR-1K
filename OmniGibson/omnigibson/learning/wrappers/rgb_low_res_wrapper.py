from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES


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
        robot = env.robots[0]
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            if camera_id == "head":
                robot.sensors[sensor_name].horizontal_aperture = 40.0  # this is what we used in data collection
            robot.sensors[sensor_name].image_height = 224
            robot.sensors[sensor_name].image_width = 224
        # reload observation space
        env.load_observation_space()
        logger.info("Reloaded observation space!")
