import numpy as np
import omnigibson as og
from omnigibson.metrics.metric_base import MetricBase


class TaskMetric(MetricBase):
    def __init__(self):
        self.timesteps = 0

    def start_callback(self, env):
        self.timesteps = 0
        self.render_timestep = og.sim.get_rendering_dt()

    def step_callback(self, env):
        self.timesteps += 1

    def end_callback(self, env):
        candidate_q_score = []
        for option in env.task.ground_goal_state_options:
            predicate_truth_values = []
            for predicate in option:
                predicate_truth_values.append(predicate.evaluate())
            candidate_q_score.append(np.mean(predicate_truth_values))
        self.final_q_score = float(np.max(candidate_q_score))

    def gather_results(self):
        return {
            "q_score": {"final": self.final_q_score},
            "time": {
                "simulator_steps": self.timesteps,
                "simulator_time": self.timesteps * self.render_timestep,
            },
        }
