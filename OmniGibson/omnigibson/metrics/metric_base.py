class MetricBase:
    """
    Class for defining a programmatic environment metric that can be tracked over the course of
    each environment episode
    """

    def __init__(self):
        self.state = dict()

    @classmethod
    def is_compatible(cls, env):
        """
        Checks if this metric class is compatible with @env

        Args:
            env (og.Environment or EnvironmentWrapper): Environment to check compatibility

        Returns:
            bool: Whether this metric is compatible or not
        """
        return True

    @classmethod
    def validate_episode(cls, episode_metrics, **kwargs):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            kwargs (Any): Any keyword arguments relevant to this specific MetricBase

        Returns:
            dict: Keyword-mapped dictionary mapping each validation test name to {"success": bool, "feedback": str} dict
                where "success" is True if the given @episode_metrics pass that specific test; if False, "feedback"
                provides information as to why the test failed
        """
        raise NotImplementedError

    def step(self, env, action, obs, reward, terminated, truncated, info):
        """
        Steps this metric, updating any internal values being tracked.

        Args:
            env (EnvironmentWrapper): Environment being tracked
            action (th.Tensor): action deployed resulting in @obs
            obs (dict): state, i.e. observation
            reward (float): reward, i.e. reward at this current timestep
            terminated (bool): terminated, i.e. whether this episode ended due to a failure or success
            truncated (bool): truncated, i.e. whether this episode ended due to a time limit etc.
            info (dict): info, i.e. dictionary with any useful information
        """
        step_metrics = self._compute_step_metrics(env, action, obs, reward, terminated, truncated, info)
        assert (
            env.scene in self.state
        ), f"Environment {env} is not being tracked, please call 'self.reset(env)' to track!"
        state = self.state[env.scene]
        for k, v in step_metrics.items():
            if k not in state:
                state[k] = []
            state[k].append(v)

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        """
        Compute any step-wise metrics at the current environment step that just occurred

        Args:
            env (EnvironmentWrapper): Environment being tracked
            action (th.Tensor): action deployed resulting in @obs
            obs (dict): state, i.e. observation
            reward (float): reward, i.e. reward at this current timestep
            terminated (bool): terminated, i.e. whether this episode ended due to a failure or success
            truncated (bool): truncated, i.e. whether this episode ended due to a time limit etc.
            info (dict): info, i.e. dictionary with any useful information

        Returns:
            dict: Any per-step information that should be internally tracked
        """
        raise NotImplementedError

    def _compute_episode_metrics(self, env, episode_info):
        """
        Computes the aggregated metrics over the current trajectory episode in @env

        Args:
            env (EnvironmentWrapper): Environment being tracked
            episode_info (dict): Internal information that was tracked using @_compute_episode metrics. This
                information is is the same key-mapped dict as @_compute_step_metrics mapped to the
                list of values aggregated over the current trajectory episode

        Returns:
            dict: Any per-step information that should be internally tracked
        """
        raise NotImplementedError

    def aggregate(self, env):
        """
        Aggregates information over the current trajectory being tracked in @env

        Args:
            env (EnvironmentWrapper): Environment being tracked

        Returns:
            dict: Any relevant aggregated metric information
        """
        if env.scene in self.state:
            if self.state[env.scene] == dict():
                return dict()
            else:
                return self._compute_episode_metrics(env=env, episode_info=self.state[env.scene])
        else:
            print("Environment not yet tracked, skipping metric aggregation!")
            return dict()

    def reset(self, env):
        """
        Resets this metric with respect to @env

        Args:
            env (EnvironmentWrapper): Environment being tracked
        """
        self.state[env.scene] = dict()
