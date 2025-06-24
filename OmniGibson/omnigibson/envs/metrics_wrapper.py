from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.utils.gym_utils import recursively_generate_flat_dict


class EnvMetric:
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
        # Return true by default
        return True

    @classmethod
    def validate_episode(cls, episode_metrics, **kwargs):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            kwargs (Any): Any keyword arguments relevant to this specific EnvMetric

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


class MetricsWrapper(EnvironmentWrapper):
    """
    Wrapper for running programmatic metric checks during env stepping
    """

    def __init__(
        self,
        env,
    ):
        """
        Args:
            env (Environment): The environment to wrap
        """
        # Store variable for tracking QA metrics
        self.metrics = dict()

        # Run super init
        super().__init__(env=env)

    def add_metric(self, name, metric):
        """
        Adds a data metric to track

        Args:
            name (str): Name of the metric. This will be the name printed out when displaying the aggregated results
            metric (EnvMetric): Metric to add
        """
        # Validate the metric is compatible, then add
        assert metric.is_compatible(
            self
        ), f"Metric {metric.__class__.__name__} is not compatible with this environment!"
        self.metrics[name] = metric

    def remove_metric(self, name):
        """
        Removes a metric from the internally tracked ones

        Args:
            name (str): Name of the metric to remove
        """
        self.metrics.pop(name)

    def reset(self):
        # Call super first
        ret = super().reset()

        # Reset all owned metrics
        for name, metric in self.metrics.items():
            metric.reset(self)

        return ret

    def aggregate_metrics(self, flatten=True):
        """
        Aggregates metrics information

        Args:
            flatten (bool): Whether to flatten the metrics dictionary or not

        Returns:
            dict: Keyword-mapped aggregated metrics information
        """
        results = dict()
        for name, metric in self.metrics.items():
            results[name] = metric.aggregate(self)

        if flatten:
            results = recursively_generate_flat_dict(dic=results)

        return results

    def step(self, action, n_render_iterations=1):
        # Run super first
        obs, reward, terminated, truncated, info = super().step(action, n_render_iterations=n_render_iterations)

        # Run all step-wise QA checks
        for name, metric in self.metrics.items():
            metric.step(self.env, action, obs, reward, terminated, truncated, info)

        return obs, reward, terminated, truncated, info
