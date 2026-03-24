from bddl.activity import Conditions, get_object_scope, get_initial_conditions, get_goal_conditions, get_ground_goal_state_options
from bddl.condition_evaluation import evaluate_state

class Task(object):
    def __init__(self, behavior_activity, activity_definition, simulator_name="behavior-1k", predefined_problem=None):
        self.behavior_activity = behavior_activity
        self.activity_definition = activity_definition
        self.simulator_name = simulator_name

        self.conditions = Conditions(
            behavior_activity,
            activity_definition,
            simulator_name,
            predefined_problem=predefined_problem,
        )
        self.object_scope = get_object_scope(self.conditions)
        self.parsed_objects = self.conditions.parsed_objects

        self.initial_conditions = get_initial_conditions(self.conditions, self.object_scope)
        self.goal_conditions = get_goal_conditions(self.conditions, self.object_scope)
        self.ground_goal_state_options = get_ground_goal_state_options(self.conditions, self.object_scope, self.goal_conditions)

    def check_goal(self, evaluate_fn):
        """
        Check if the goal conditions are met.
        evaluate_fn: callback function `evaluate_fn(predicate_name, *objects)`
        Returns: True if goal is met, otherwise False, plus a dict of satisfied vs unsatisfied conditions.
        """
        return evaluate_state(self.goal_conditions, evaluate_fn)
    
    def check_initial_conditions(self, evaluate_fn):
        """
        Check if the initial conditions are met.
        evaluate_fn: callback function `evaluate_fn(predicate_name, *objects)`
        Returns: True if initial conditions are met, otherwise False, plus a dict of satisfied vs unsatisfied conditions.
        """
        return evaluate_state(self.initial_conditions, evaluate_fn)
