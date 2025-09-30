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

        # Calculate initially satisfied predicates for each option
        self.initial_satisfied_per_option = []
        for option in env.task.ground_goal_state_options:
            initial_satisfied_count = 0
            for predicate in option:
                if predicate.evaluate():
                    initial_satisfied_count += 1
            self.initial_satisfied_per_option.append(initial_satisfied_count)

    def step_callback(self, env):
        self.timesteps += 1

    def end_callback(self, env):
        candidate_q_score = []
        for i, option in enumerate(env.task.ground_goal_state_options):
            # Count currently satisfied predicates
            currently_satisfied_count = 0
            for predicate in option:
                if predicate.evaluate():
                    currently_satisfied_count += 1

            # Calculate score: newly satisfied predicates / total predicates
            # Offset by initially satisfied predicates and cap at 0
            total_predicates = len(option)
            newly_satisfied = max(0, currently_satisfied_count - self.initial_satisfied_per_option[i])

            # Calculate the partial credit score for this option
            if total_predicates > 0:
                # Only count predicates that weren't initially satisfied
                predicates_to_complete = total_predicates - self.initial_satisfied_per_option[i]
                if predicates_to_complete > 0:
                    option_score = newly_satisfied / predicates_to_complete
                else:
                    # All predicates were already satisfied initially
                    option_score = 0.0
            else:
                option_score = 0.0

            candidate_q_score.append(option_score)

        self.final_q_score = float(np.max(candidate_q_score))

    def gather_results(self):
        return {
            "q_score": {"final": self.final_q_score},
            "time": {
                "simulator_steps": self.timesteps,
                "simulator_time": self.timesteps * self.render_timestep,
                "normalized_time": self.human_stats["steps"] / self.timesteps,
            },
        }
