from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.sensors import VisionSensor
from omnigibson.utils.ui_utils import create_module_logger


# Create module logger
log = create_module_logger("RichObservationWrapper")


class RichObservationWrapper(EnvironmentWrapper):
    """
    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        # Here, we modify the robot observation to include more modalities
        # For a complete list of available modalities, see VisionSensor.ALL_MODALITIES
        robot = env.robots[0]
        for sensor in robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                sensor.add_modality("normal")
                sensor.add_modality("flow")
                sensor.add_modality("bbox_3d")
        # Here, we include privileged task observations (i.e. object poses for task-relevant objects)
        env.task._include_obs = True

    def step(self, action, n_render_iterations=1):
        """
        By default, run the normal environment step() function

        Args:
            action (th.tensor): action to take in environment
            n_render_iterations (int): Number of rendering iterations to use before returning observations

        Returns:
            4-tuple:
                - (dict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is terminated
                - (bool) whether the current episode is truncated
                - (dict) misc information
        """
        obs, reward, terminated, truncated, info = self.env.step(action, n_render_iterations=n_render_iterations)
        # Now, query the environment for some additional info
        return obs, reward, terminated, truncated, info
