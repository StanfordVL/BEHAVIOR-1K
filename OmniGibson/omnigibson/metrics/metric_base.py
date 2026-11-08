class MetricBase:
    """
    Class for defining a programmatic environment metric that can be tracked over the course of
    each environment episode
    """

    def __init__(self, env_idx=0, env_accessor=None):
        # Batched evaluation binds a metric to an accessor, which owns the mapping into the shared
        # environment. env_idx remains as a compatibility path for MetricsWrapper and external users.
        self.env_idx = env_idx
        self.env_accessor = env_accessor
        self.state = dict()

    def _resolve_env(self, env):
        if env is not None:
            return env
        if self.env_accessor is None:
            raise ValueError("Metric is not bound to an environment accessor; pass env explicitly.")
        return self.env_accessor

    def _scene(self, env=None):
        """Return the scene tracked by this metric for bound and legacy callers."""
        env = self._resolve_env(env)
        return env.scene if env is self.env_accessor else env.scenes[self.env_idx]

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

    def step(self, env=None, action=None, obs=None, reward=None, terminated=None, truncated=None, info=None):
        """
        Steps this metric, updating any internal values being tracked.

        Args:
            action (th.Tensor): action deployed resulting in @obs
            obs (dict): state, i.e. observation
            reward (float): reward, i.e. reward at this current timestep
            terminated (bool): terminated, i.e. whether this episode ended due to a failure or success
            truncated (bool): truncated, i.e. whether this episode ended due to a time limit etc.
            info (dict): info, i.e. dictionary with any useful information
        """
        env = self._resolve_env(env)
        step_metrics = self._compute_step_metrics(env, action, obs, reward, terminated, truncated, info)
        scene = self._scene(env)
        assert scene in self.state, f"Environment {scene} is not being tracked, please call 'self.reset()' to track!"
        state = self.state[scene]
        for k, v in step_metrics.items():
            if k not in state:
                state[k] = []
            state[k].append(v)

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        """
        Compute any step-wise metrics at the current environment step that just occurred

        Args:
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
            episode_info (dict): Internal information that was tracked using @_compute_episode metrics. This
                information is is the same key-mapped dict as @_compute_step_metrics mapped to the
                list of values aggregated over the current trajectory episode

        Returns:
            dict: Any per-step information that should be internally tracked
        """
        raise NotImplementedError

    def aggregate(self, env=None):
        """
        Aggregates information over the current trajectory tracked by this metric.

        Returns:
            dict: Any relevant aggregated metric information
        """
        env = self._resolve_env(env)
        scene = self._scene(env)
        if scene in self.state:
            if self.state[scene] == dict():
                return dict()
            else:
                return self._compute_episode_metrics(env=env, episode_info=self.state[scene])
        else:
            print("Environment not yet tracked, skipping metric aggregation!")
            return dict()

    def reset(self, env=None):
        """
        Resets this metric for its bound logical environment.
        """
        self.state[self._scene(env)] = dict()
