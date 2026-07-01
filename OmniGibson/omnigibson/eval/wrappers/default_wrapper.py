from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import set_sensor_modalities
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger(module_name=__name__)


class DefaultWrapper(EnvironmentWrapper):
    """
    Default eval wrapper: low-resolution RGB observations.

    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        robot = env.robots[0]
        for sensor_name, sensor in robot.sensors.items():
            if not hasattr(sensor, "image_height") or not hasattr(sensor, "image_width"):
                continue
            set_sensor_modalities(sensor, {"rgb"})
            sensor.image_height = 224
            sensor.image_width = 224
            sensor_space = sensor.load_observation_space()
            if env.observation_space is not None:
                env.observation_space.spaces[robot.name].spaces[sensor_name] = sensor_space
        logger.info("Reloaded camera observation spaces!")
