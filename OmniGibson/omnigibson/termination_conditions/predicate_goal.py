import torch as th


from omnigibson.termination_conditions.termination_condition_base import SuccessCondition


class PredicateGoal(SuccessCondition):
    """
    PredicateGoal (success condition) used for BehaviorTask
    Episode terminates if all the predicates are satisfied

    Args:
        check_goal_fn (method): function that checks goal satisfaction. Function signature should be:

            (all_satisfied, results) = check_goal_fn()

            where @all_satisfied is a bool and @results maps "satisfied"/"unsatisfied" to lists of indices.
    """

    def __init__(self, check_goal_fn):
        # Store internal vars
        self._check_goal_fn = check_goal_fn
        self._goal_status = None

        # Run super
        super().__init__()

    def reset(self, task, env, env_indices):
        # Run super first
        super().reset(task, env, env_indices)

        # Reset status
        if self._goal_status is None:
            self._goal_status = [{"satisfied": [], "unsatisfied": []} for _ in range(env.num_envs)]
        for idx in env_indices:
            self._goal_status[idx] = {"satisfied": [], "unsatisfied": []}

    def _step(self, task, env, action):
        # Terminate if all goal conditions are met in the task
        results = th.zeros(env.num_envs, dtype=th.bool)
        for env_idx in range(env.num_envs):
            done, self._goal_status[env_idx] = self._check_goal_fn(env_idx)
            results[env_idx] = done
        return results

    @property
    def goal_status(self):
        """
        Returns:
            list[dict]: Current goal status for the active predicate(s), mapping "satisfied" and "unsatisfied" to a list
                of the predicates matching either of those conditions for each env
        """
        return self._goal_status
