import os
import re
from bddl.condition_evaluation import (
    compile_state,
    create_scope,
    evaluate_state,
    get_ground_state_options,
)
from bddl.config import ACTIVITY_CONFIGS_PATH
from bddl.object_taxonomy import ObjectTaxonomy
from bddl.parsing import (
    gen_natural_language_condition,
    gen_natural_language_conditions,
    parse_domain,
    parse_problem,
)

INSTANCE_EXPR = re.compile(r"problem(\d+).bddl")


class Conditions(object):
    def __init__(
        self,
        behavior_activity,
        activity_definition,
        simulator_name,
        predefined_problem=None,
    ):
        """Object to store behavior activity content and compile conditions for checking and
            simulator use

        Args:
            behavior_activity (str): behavior activity being used
            activity_definition (int): specific definition of behavior_activity
            simulator_name (str): simulator that BEHAVIOR is being used with
            predefined_problem (str): a pre-defined problem that is not in the activity_definitions folder
        """
        self.behavior_activity = behavior_activity
        self.activity_definition = activity_definition
        domain_name, *__ = parse_domain(simulator_name)
        (
            __,
            self.parsed_objects,
            self.parsed_initial_conditions,
            self.parsed_goal_conditions,
        ) = parse_problem(
            self.behavior_activity,
            self.activity_definition,
            domain_name,
            predefined_problem=predefined_problem,
        )


######## API ########


def get_object_scope(conds):
    """Create unpopulated object scope to populate for generating goal and
        ground goal conditions.

    Args:
        conds (Conditions): conditions for the particular activity and definition

    Returns:
        dict<str: None>: unpopulated scope with string keys to be mapped to
                            simulator object values
    """
    return create_scope(conds.parsed_objects)


def get_initial_conditions(conds, backend, scope, generate_ground_options=True):
    """Create compiled initial conditions that can be checked and sampled

    Args:
        conds (Conditions): conditions for the particular activity and definition

    Returns:
        list<bddl.condition_evaluation.HEAD>: compiled conditions if initial
                                                condition definition is not
                                                empty else None
    """
    if bool(conds.parsed_initial_conditions[0]):
        initial_conditions = compile_state(
            [
                cond
                for cond in conds.parsed_initial_conditions
                if cond[0] not in ["inroom"]
            ],
            backend,
            scope=scope,
            object_map=conds.parsed_objects,
            generate_ground_options=generate_ground_options,
        )
        return initial_conditions


def get_goal_conditions(conds, backend, scope, generate_ground_options=True):
    """Create compiled goal conditions with a populated object scope for checking

    Args:
        conds (Conditions): conditions for the particular activity and definition
        populated_object_scope (dict<str: simulator object>): scope mapping object
                                                                terms in BDDL to
                                                                simulator objects

    Returns:
        list<bddl.condition_evaluation.HEAD>: compiled conditions if goal condition
                                                definition is not empty else None
    """
    if bool(conds.parsed_goal_conditions[0]):
        goal_conditions = compile_state(
            conds.parsed_goal_conditions,
            backend,
            scope=scope,
            object_map=conds.parsed_objects,
            generate_ground_options=generate_ground_options,
        )
        return goal_conditions


def get_ground_goal_state_options(conds, backend, scope, goal_conditions):
    """Create compiled ground solutions to goal state with a populated object scope
        for checking progress on specific solutions

    Args:
        conds (Conditions): conditions for the particular activity and definition
        populated_object_scope (dict<str: simulator object>): scope mapping object
                                                                terms in BDDL to
                                                                simulator objects

    Returns:
        list<bddl.condition_evaluation.HEAD>: compiled goal solutions

    Raises:
        AssertionError if there are no ground solutions
    """
    ground_goal_state_options = get_ground_state_options(
        goal_conditions, backend, scope=scope, object_map=conds.parsed_objects
    )
    assert len(ground_goal_state_options) > 0
    return ground_goal_state_options


def evaluate_goal_conditions(goal_conditions):
    """Evaluate compiled goal state to see if current simulator state has been met

    Args:
        goal_conditions (list<bddl.condition_evaluation.HEAD>): list of compiled
                                                                goal conditions with
                                                                populated scope

    Returns:
        bool, dict<str: list<int>>: [description]
    """
    return evaluate_state(goal_conditions)


def get_natural_initial_conditions(conds):
    """Return natural language translation of init of given conditions

    Args:
        conditions (list): conditions being translated

    Returns:
        list<str>: natural language translations, one per condition in conditions
    """
    return gen_natural_language_conditions(conds.parsed_initial_conditions)


def get_natural_goal_conditions(conds):
    """Return natural language translation of goal of given conditions

    Args:
        conditions (list): conditions being translated

    Returns:
        list<str>: natural language translations, one per condition in conditions
    """
    return gen_natural_language_conditions(conds.parsed_goal_conditions)


def get_all_activities():
    """Return a list of all activities included in this version of BDDL.

    Returns:
        list<str>: list containing the name of each included activity
    """
    return [
        x
        for x in os.listdir(ACTIVITY_CONFIGS_PATH)
        if os.path.isdir(os.path.join(ACTIVITY_CONFIGS_PATH, x))
    ]


def get_instance_count(act):
    """Return the number of instances of a given activity that are included in this version of BDDL.

    Args:
        act (str): name of the activity to check

    Returns:
        int: number of instances of the given activity
    """
    problem_files = [
        INSTANCE_EXPR.fullmatch(x)
        for x in os.listdir(os.path.join(ACTIVITY_CONFIGS_PATH, act))
    ]
    ids = set(int(x.group(1)) for x in problem_files if x is not None)
    assert ids == set(
        range(len(ids))
    ), f"Non-contiguous instance IDs found for problem {act}"
    return len(ids)


def get_reward(ground_goal_state_options):
    """Return reward given ground goal state options.
       Reward formulated as max(<percent literals that are satisfied in the option> for option in ground_goal_state_options)

    Args:
        ground_goal_state_options (list<list<HEAD>>): list of compiled ground goal state options

    Returns:
        float: reward
    """
    return max(
        len(evaluate_state(option)[-1]["satisfied"]) / float(len(option))
        for option in ground_goal_state_options
    )
