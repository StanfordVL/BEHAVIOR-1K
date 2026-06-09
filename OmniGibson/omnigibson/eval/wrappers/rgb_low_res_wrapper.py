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
        for robot in [r for scene_robots in env.robots for r in scene_robots]:
            for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
                sensor_name = camera_name.split("::")[1]
                sensor = robot.sensors[sensor_name]
                if camera_id == "head":
                    sensor.horizontal_aperture = 40.0  # this is what we used in data collection
                sensor.image_height = 224
                sensor.image_width = 224
        robot0 = env.robots[0][0]
        if env.observation_space is not None:
            for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
                sensor_name = camera_name.split("::")[1]
                sensor_space = robot0.sensors[sensor_name].load_observation_space()
                env.observation_space.spaces[robot0.name].spaces[sensor_name] = sensor_space
        logger.info("Reloaded camera observation spaces!")
