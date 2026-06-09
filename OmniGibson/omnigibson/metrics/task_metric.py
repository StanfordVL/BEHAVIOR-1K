import omnigibson as og
from omnigibson.eval.utils.vec_eval_scheduler import compute_q_score
from omnigibson.metrics.metric_base import MetricBase
from typing import Optional


class TaskMetric(MetricBase):
    def __init__(self, human_stats: Optional[dict] = None, env_idx: int = 0):
        super().__init__(env_idx=env_idx)
        self.timesteps = 0
        self.human_stats = human_stats
        if human_stats is None:
            print("No human stats provided.")
        else:
            self.human_stats = {
                "steps": self.human_stats["length"],
            }

    def reset(self, env):
        # Tracks env.scenes[env_idx]. Partial-success (Q-score) is computed via the env-aware
        # BehaviorTask.get_goal_option_satisfaction(env_idx) so each env reports its OWN goal state;
        # reading ground_goal_state_options[*].evaluate() would bind the shared scope (env 0).
        self.state[self._scene(env)] = dict()
        self.timesteps = 0
        self.render_timestep = og.sim.get_rendering_dt()
        self.initial_predicate_states = env.task.get_goal_option_satisfaction(self.env_idx)

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        self.timesteps += 1
        return {"timesteps": self.timesteps}

    def _compute_episode_metrics(self, env, episode_info):
        # Use the accumulated state from episode_info
        timesteps = episode_info.get("timesteps", [])[-1] if episode_info.get("timesteps") else self.timesteps

        # task.success is a (num_envs,) bool tensor; read THIS env's slot. Partial credit (when not a
        # full success) counts newly-satisfied goal predicates per option, max over options.
        final_q_score = compute_q_score(
            success=bool(env.task.success[self.env_idx]),
            now_satisfied_options=env.task.get_goal_option_satisfaction(self.env_idx),
            initial_satisfied_options=self.initial_predicate_states,
        )

        return {
            "q_score": {"final": final_q_score},
            "time": {
                "simulator_steps": timesteps,
                "simulator_time": timesteps * self.render_timestep,
                "normalized_time": self.human_stats["steps"] / timesteps if timesteps > 0 else float("inf"),
            },
        }
