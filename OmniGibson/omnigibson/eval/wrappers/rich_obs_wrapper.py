from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.eval.utils.eval_utils import ROBOT_CAMERA_NAMES, HEAD_RESOLUTION, WRIST_RESOLUTION


# Create module logger
logger = create_module_logger("RichObservationWrapper")


class RichObservationWrapper(EnvironmentWrapper):
    """
    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        # This wrapper is single-env / single-robot only: it mutates exactly one robot's
        # camera resolutions and observation modalities.
        assert env.num_envs == 1, f"RichObservationWrapper requires num_envs=1; got num_envs={env.num_envs}."
        assert len(env.scene.robots) == 1, (
            f"RichObservationWrapper requires exactly one robot per scene; " f"got {len(env.scene.robots)}."
        )
        # Note that from eval.py we already set the robot to include rgb + depth + seg_instance_id modalities
        robot = env.scene.robots[0]
        # Here, we change the camera resolution and head camera aperture to match the one we used in data collection
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            if camera_id == "head":
                robot.sensors[sensor_name].horizontal_aperture = 40.0
                robot.sensors[sensor_name].image_height = HEAD_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = HEAD_RESOLUTION[1]
            else:
                robot.sensors[sensor_name].image_height = WRIST_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = WRIST_RESOLUTION[1]
            # Here, we modify the robot observation to include normal and flow modalities
            # For a complete list of available modalities, see VisionSensor.ALL_MODALITIES
            robot.sensors[sensor_name].add_modality("normal")
            robot.sensors[sensor_name].add_modality("flow")
        # we also set task to include obs
        env.task._include_obs = True
        # reload observation space
        env.load_observation_space()
        logger.info("Reloaded observation space!")
