from omnigibson.utils.python_utils import Wrapper


class EnvironmentWrapper(Wrapper):
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

    def step(self, action):
        """
        By default, run the normal environment step() function

        Args:
            action (np.array): action to take in environment

        Returns:
            4-tuple:
                - (dict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        return self.env.step(action)

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

