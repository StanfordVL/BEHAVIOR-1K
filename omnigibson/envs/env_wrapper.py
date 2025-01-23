from copy import deepcopy

from omnigibson.utils.python_utils import Registerable, Wrapper, classproperty, create_class_from_registry_and_config
from omnigibson.utils.ui_utils import create_module_logger

# Global dicts that will contain mappings
REGISTERED_ENV_WRAPPERS = dict()

# Create module logger
log = create_module_logger(module_name=__name__)


def create_wrapper(env, wrapper_cfg=None):
    """
    Wraps environment @env with wrapper defined @wrapper_cfg

    Args:
        env (og.Environment): environment to wrap
        wrapper_cfg (None or dict): Specified, configuration to wrap environment with
            If not specified, will default to env.wrapper_config
    """
    wrapper_cfg = deepcopy(env.wrapper_config if wrapper_cfg is None else wrapper_cfg)
    wrapper_type = wrapper_cfg.pop("type")
    wrapper_cfg["env"] = env

    return create_class_from_registry_and_config(
        cls_name=wrapper_type,
        cls_registry=REGISTERED_ENV_WRAPPERS,
        cfg=wrapper_cfg,
        cls_type_descriptor="wrapper",
    )


class EnvironmentWrapper(Wrapper, Registerable):
    """
    Base class for all environment wrappers in OmniGibson. In general, reset(), step(), and observation_spec() should
    be overwritten

    Args:
        env (OmniGibsonEnv): The environment to wrap.
    """

    def __init__(self, env):
        self.env = env

        # Run super
        super().__init__(obj=env)

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
        return self.env.step(action, n_render_iterations=n_render_iterations)

    def reset(self):
        """
        By default, run the normal environment reset() function

        Returns:
            dict: Environment observation space after reset occurs
        """
        return self.env.reset()

    def observation_spec(self):
        """
        By default, grabs the normal environment observation_spec

        Returns:
            dict: Observations from the environment
        """
        return self.env.observation_spec()

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("EnvironmentWrapper")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global robot registry
        global REGISTERED_ENV_WRAPPERS
        return REGISTERED_ENV_WRAPPERS
