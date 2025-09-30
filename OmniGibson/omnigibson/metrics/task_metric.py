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
        self.initial_predicate_states = []
        for option in env.task.ground_goal_state_options:
            option_states = []
            for predicate in option:
                option_states.append(predicate.evaluate())
            self.initial_predicate_states.append(option_states)

    def step_callback(self, env):
        self.timesteps += 1

    def end_callback(self, env):
        candidate_q_score = []
        for i, option in enumerate(env.task.ground_goal_state_options):
            # Compare current predicate states with initial states
            initial_states = self.initial_predicate_states[i]

            # Count predicates that weren't initially satisfied (these are the ones we need to complete)
            predicates_to_complete = sum(1 for was_satisfied in initial_states if not was_satisfied)

            if predicates_to_complete > 0:
                # Count predicates that are now satisfied but weren't initially
                newly_satisfied_count = 0
                for j, predicate in enumerate(option):
                    current_state = predicate.evaluate()
                    initial_state = initial_states[j]
                    # Only count if it went from False to True
                    if current_state and not initial_state:
                        newly_satisfied_count += 1

                # Calculate score as proportion of initially unsatisfied predicates that are now satisfied
                option_score = newly_satisfied_count / predicates_to_complete
            else:
                # All predicates were already satisfied initially, no progress to measure
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
