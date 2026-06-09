import omnigibson as og
from omnigibson.metrics.metric_base import MetricBase
from typing import Optional, Sequence


def compute_q_score(
    success: bool,
    now_satisfied_options: Sequence[Sequence[bool]],
    initial_satisfied_options: Sequence[Sequence[bool]],
) -> float:
    """
    Partial-success (Q-score) for one episode/env: a full success scores 1.0; otherwise the fraction
    of goal predicates that were NOT satisfied at episode start but ARE satisfied now, maximized over
    the alternative goal-state options. Mirrors the pre-refactor inline formula (lives here next to its
    only caller, TaskMetric). Empty options/no options return 0.0 instead of raising.
    """
    if success:
        return 1.0
    if not now_satisfied_options:
        return 0.0
    option_scores = []
    for now_opt, init_opt in zip(now_satisfied_options, initial_satisfied_options):
        if len(now_opt) == 0:
            option_scores.append(0.0)
            continue
        newly_satisfied = sum(int((not init) and now) for now, init in zip(now_opt, init_opt))
        option_scores.append(newly_satisfied / len(now_opt))
    return max(option_scores) if option_scores else 0.0


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
