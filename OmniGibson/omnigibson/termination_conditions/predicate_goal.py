from bddl.activity import evaluate_goal_conditions

from omnigibson.termination_conditions.termination_condition_base import SuccessCondition


class PredicateGoal(SuccessCondition):
    """
    PredicateGoal (success condition) used for BehaviorTask
    Episode terminates if all the predicates are satisfied

    Args:
        goal_fcn (method): function for calculating goal(s). Function signature should be:

            goals = goal_fcn()

            where @goals is a list of bddl.condition_evaluation.HEAD -- compiled BDDL goal conditions
    """

    def __init__(self, goal_fcn):
        # Store internal vars
        self._goal_fcn = goal_fcn
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
        done, self._goal_status = evaluate_goal_conditions(self._goal_fcn())
        return done

    @property
    def goal_status(self):
        """
        Returns:
            dict: Current goal status for the active predicate(s), mapping "satisfied" and "unsatisfied" to a list
                of the predicates matching either of those conditions
        """
        return self._goal_status
