from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.eval.utils.eval_utils import ROBOT_CAMERA_NAMES, HEAD_RESOLUTION, WRIST_RESOLUTION

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
        # Update robot sensors:
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            if camera_id == "head":
                robot.sensors[sensor_name].image_height = HEAD_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = HEAD_RESOLUTION[1]
            else:
                robot.sensors[sensor_name].image_height = WRIST_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = WRIST_RESOLUTION[1]
            # add depth
            robot.sensors[sensor_name].add_modality("depth_linear")
        # reload observation space
        env.load_observation_space()
        logger.info("Reloaded observation space!")
