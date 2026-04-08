from typing import Iterable

from omnigibson.envs.env_base import Environment
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.metrics.metric_base import MetricBase
from omnigibson.utils.gym_utils import recursively_generate_flat_dict


class MetricsWrapper(EnvironmentWrapper):
    """
    Wrapper for running programmatic metric checks during env stepping
    """

    def __init__(self, env: Environment) -> None:
        """
        Args:
            env (Environment): The environment to wrap
        """
        # Store variable for tracking QA metrics
        self.metrics = dict()

        # Run super init
        super().__init__(env=env)

    def add_metric(self, name: str, metric: MetricBase) -> None:
        """
        Adds a data metric to track

        Args:
            name (str): Name of the metric. This will be the name printed out when displaying the aggregated results
            metric (MetricBase): Metric to add
        """
        # Validate the metric is compatible, then add
        assert metric.is_compatible(
            self
        ), f"Metric {metric.__class__.__name__} is not compatible with this environment!"
        self.metrics[name] = metric

    def remove_metric(self, name: str) -> None:
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

    def aggregate_metrics(self, flatten: bool = True) -> dict:
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

    def step(self, action: dict | Iterable, n_render_iterations: int = 1) -> tuple:
        # Run super first
        obs, reward, terminated, truncated, info = super().step(action, n_render_iterations=n_render_iterations)

        # Run all step-wise QA checks
        for name, metric in self.metrics.items():
            metric.step(self.env, action, obs, reward, terminated, truncated, info)

        return obs, reward, terminated, truncated, info
