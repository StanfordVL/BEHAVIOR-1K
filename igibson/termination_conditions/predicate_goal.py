from bddl.activity import evaluate_goal_conditions

from igibson.termination_conditions.termination_condition_base import SuccessCondition


class PredicateGoal(SuccessCondition):
    """
    PredicateGoal (success condition) used for BehaviorTask
    Episode terminates if all the predicates are satisfied
    """

    def _step(self, task, env, action):
        # Terminate if all goal conditions are met in the task
        done, _ = evaluate_goal_conditions(task.goal_conditions)
        return done
