import omnigibson as og
from omnigibson.metrics.metric_base import MetricBase
from typing import Optional


class TaskMetric(MetricBase):
    def __init__(self, human_stats: Optional[dict] = None):
        super().__init__()
        self.timesteps = 0
        self.human_stats = human_stats
        if human_stats is None:
            print("No human stats provided.")
        else:
            self.human_stats = {
                "steps": self.human_stats["length"],
            }

    def reset(self, env):
        self.state[env.scene] = dict()
        self.timesteps = 0
        self.render_timestep = og.sim.get_rendering_dt()
        self.initial_predicate_states = [
            [pred.evaluate() for pred in option] for option in env.task.ground_goal_state_options
        ]

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        self.timesteps += 1
        return {"timesteps": self.timesteps}

    def _compute_episode_metrics(self, env, episode_info):
        # Use the accumulated state from episode_info
        timesteps = episode_info.get("timesteps", [])[-1] if episode_info.get("timesteps") else self.timesteps

        if env.task.success:
            final_q_score = 1.0
        else:
            final_q_score = max(
                sum(
                    int(not initially_true and pred.evaluate())
                    for pred, initially_true in zip(option, option_previous_state)
                )
                / len(option)
                for option, option_previous_state in zip(
                    env.task.ground_goal_state_options, self.initial_predicate_states
                )
            )

        return {
            "q_score": {"final": final_q_score},
            "time": {
                "simulator_steps": timesteps,
                "simulator_time": timesteps * self.render_timestep,
                "normalized_time": self.human_stats["steps"] / timesteps if timesteps > 0 else float("inf"),
            },
        }
