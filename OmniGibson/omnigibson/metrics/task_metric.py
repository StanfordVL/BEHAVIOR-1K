import numpy as np
import omnigibson as og
from omnigibson.metrics.metric_base import MetricBase
from typing import Optional


class TaskMetric(MetricBase):
    def __init__(self, human_stats: Optional[dict] = None):
        self.timesteps = 0
        self.human_stats = human_stats
        if human_stats is None:
            print("No human stats provided.")
        else:
            self.human_stats = {
                "steps": self.human_stats["length"],
            }

    def start_callback(self, env):
        self.timesteps = 0
        self.render_timestep = og.sim.get_rendering_dt()

        # Store the initial state (true/false) of each predicate for each option
        self.initial_predicate_states = [
            [pred.evaluate() for pred in option]
            for option in env.task.ground_goal_state_options
        ]

    def step_callback(self, env):
        self.timesteps += 1

    def end_callback(self, env):
        # If task is fully complete, return perfect score
        if env.task.success:
            self.final_q_score = 1.0
            return

        # Otherwise calculate partial credit based on newly satisfied predicates. The partial credit is the maximum progress
        # made, across any of the groundings, from the initial state of the task.
        self.final_q_score = max(
            sum(
                int(not initially_true and pred.evaluate())
                for pred, initially_true in zip(option, option_previous_state)
            )
            / len(option)
            for option, option_previous_state in zip(env.task.ground_goal_state_options, self.initial_predicate_states)
        )

    def gather_results(self):
        return {
            "q_score": {"final": self.final_q_score},
            "time": {
                "simulator_steps": self.timesteps,
                "simulator_time": self.timesteps * self.render_timestep,
                "normalized_time": self.human_stats["steps"] / self.timesteps,
            },
        }
