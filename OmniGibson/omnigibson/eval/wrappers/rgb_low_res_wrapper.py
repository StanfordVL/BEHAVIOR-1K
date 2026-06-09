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
        # Set the camera resolution (and head aperture) on EVERY robot so all scenes render to match.
        # env.robots is list[list[Robot]] (one inner list per scene); flatten to every robot.
        for robot in [r for scene_robots in env.robots for r in scene_robots]:
            for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
                sensor_name = camera_name.split("::")[1]
                sensor = robot.sensors[sensor_name]
                if camera_id == "head":
                    sensor.horizontal_aperture = 40.0  # this is what we used in data collection
                sensor.image_height = 224
                sensor.image_width = 224
        # Patch ONLY the camera sensor obs spaces. A full env.load_observation_space() would also rebuild
        # the proprio space, which queries live joint positions -- not available at wrapper-construction
        # time (returns None). The env obs space is built from scene 0's robot
        # (EnvironmentBase._load_observation_space), so update that robot's camera spaces, keeping every
        # scene's resolution change above. The obs space is nested (raw obs are flattened later in
        # _preprocess_obs), keyed robot.name -> sensor_name.
        robot0 = env.robots[0][0]
        if env.observation_space is not None:
            for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
                sensor_name = camera_name.split("::")[1]
                sensor_space = robot0.sensors[sensor_name].load_observation_space()
                env.observation_space.spaces[robot0.name].spaces[sensor_name] = sensor_space
        logger.info("Reloaded camera observation spaces!")
