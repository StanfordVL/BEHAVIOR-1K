from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.eval.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    get_robot_camera_names,
    set_sensor_modalities,
)

logger = create_module_logger(module_name=__name__)


class RGBDFullResWrapper(EnvironmentWrapper):
    """
    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        # Note that from eval.py we only set rgb modality, here we include depth.
        # Here, we change the camera resolution to match the one we used in data collection
        robot = env.robots[0]
        robot_eval_config = getattr(env, "_eval_robot_config", {})
        camera_roles_by_sensor_name = {
            camera_name.split("::")[1]: camera_id
            for camera_id, camera_name in get_robot_camera_names(robot.name, robot_eval_config).items()
        }
        for sensor_name, sensor in robot.sensors.items():
            if not hasattr(sensor, "image_height") or not hasattr(sensor, "image_width"):
                continue
            set_sensor_modalities(sensor, {"rgb", "depth_linear"})
            camera_id = camera_roles_by_sensor_name.get(sensor_name)
            if camera_id == "head":
                sensor.image_height = HEAD_RESOLUTION[0]
                sensor.image_width = HEAD_RESOLUTION[1]
            else:
                sensor.image_height = WRIST_RESOLUTION[0]
                sensor.image_width = WRIST_RESOLUTION[1]
            # Reload only this sensor's observation space; a full env.load_observation_space()
            # would also query proprio while physics views are not yet valid.
            sensor_space = sensor.load_observation_space()
            if env.observation_space is not None:
                env.observation_space.spaces[robot.name].spaces[sensor_name] = sensor_space
        logger.info("Reloaded camera observation spaces!")
