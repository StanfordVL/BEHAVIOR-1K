from omnigibson.scenes.scene_base import Scene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.utils.python_utils import classproperty


class DummyTask(BaseTask):
    """
    Dummy task
    """

    def _load(self, env):
        # Do nothing here
        pass

    def _create_termination_conditions(self):
        # Do nothing
        return dict()

    def _create_reward_functions(self):
        # Do nothing
        return dict()

    def _get_obs(self, env):
        # No task-specific obs of any kind
        return dict(), dict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    @classproperty
    def valid_scene_types(cls):
        # Any scene works
        return {Scene}

    @classproperty
    def default_termination_config(cls):
        # Empty dict
        return {}

    @classproperty
    def default_reward_config(cls):
        # Empty dict
        return {}
