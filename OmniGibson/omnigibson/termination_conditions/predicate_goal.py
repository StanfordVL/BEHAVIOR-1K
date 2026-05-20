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

    def reset(self, task, env):
        # Run super first
        super().reset(task, env)

        # Reset status
        self._goal_status = {"satisfied": [], "unsatisfied": []}

    def _step(self, task, env, action):
        # Terminate if all goal conditions are met in the task
        done, self._goal_status = self._check_goal_fn()
        return done

    @property
    def goal_status(self):
        """
        Returns:
            dict: Current goal status for the active predicate(s), mapping "satisfied" and "unsatisfied" to a list
                of the predicates matching either of those conditions
        """
        return self._goal_status
