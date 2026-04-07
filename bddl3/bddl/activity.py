"""Public API for loading and working with BDDL activity definitions.

This module is the main entry point for consumers who want to:

* Parse an activity's BDDL problem file into structured conditions.
* Create an object scope and compile conditions for evaluation.
* Evaluate compiled goal conditions against a simulator.
* Translate conditions into natural language.

Most users should prefer the higher-level
:class:`~bddl.knowledge_base.models.Task` class (accessible via
``KnowledgeBase.get_task()``), which bundles all of the above into a single
object.  The functions here are the lower-level building blocks that ``Task``
calls internally.
"""

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
    """Container for a parsed BDDL activity definition.

    On construction the domain file is parsed (to discover the available
    predicates and their arities), and the problem file is parsed to extract
    the declared objects, initial-state literals, and goal-state expression.

    The parsed data is stored as raw nested-list structures suitable for
    later compilation by :func:`~bddl.condition_evaluation.compile_state`.

    Args:
        behavior_activity (str): Activity name (e.g. ``"cleaning_up_after_a_meal"``).
        activity_definition (int): Which numbered problem variant to load.
        simulator_name (str): Name of the BDDL domain file (e.g. ``"behavior-1k"``).
            This selects which ``domain_<name>.bddl`` file is used to validate
            predicate names.
        predefined_problem (str | None): If provided, a raw BDDL problem string
            used instead of loading from the filesystem.

    Attributes:
        parsed_objects (dict[str, list[str]]): ``{synset_category: [instance_names]}``.
        parsed_initial_conditions (list): Nested-list representation of initial-state
            literals.
        parsed_goal_conditions (list): Nested-list representation of the goal
            expression.
    """

    def __init__(
        self,
        behavior_activity,
        activity_definition,
        simulator_name,
        predefined_problem=None,
    ):
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
    """Create the object scope for an activity definition.

    The scope maps every declared object instance name to itself (a string).
    This dict is later passed to :func:`compile_state` so that compiled
    expression nodes can resolve object references.

    Args:
        conds (Conditions): Parsed activity conditions.

    Returns:
        dict[str, str]: ``{instance_name: instance_name}`` for each declared
        object instance.
    """
    return create_scope(conds.parsed_objects)


def get_initial_conditions(conds, scope, generate_ground_options=True):
    """Create compiled initial conditions that can be checked and sampled

    Args:
        conds (Conditions): conditions for the particular activity and definition
        scope (dict): object scope mapping
        generate_ground_options (bool): whether to generate ground goal options

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
            scope=scope,
            object_map=conds.parsed_objects,
            generate_ground_options=generate_ground_options,
        )
        return initial_conditions


def get_goal_conditions(conds, scope, generate_ground_options=True):
    """Create compiled goal conditions with a populated object scope for checking

    Args:
        conds (Conditions): conditions for the particular activity and definition
        scope (dict<str: str>): scope mapping object terms in BDDL to strings

    Returns:
        list<bddl.condition_evaluation.HEAD>: compiled conditions if goal condition
                                                definition is not empty else None
    """
    if bool(conds.parsed_goal_conditions[0]):
        goal_conditions = compile_state(
            conds.parsed_goal_conditions,
            scope=scope,
            object_map=conds.parsed_objects,
            generate_ground_options=generate_ground_options,
        )
        return goal_conditions


def get_ground_goal_state_options(conds, scope, goal_conditions):
    """Enumerate all grounded solutions to the goal state.

    A *grounded* (or *ground*) solution is a specific, fully-instantiated
    set of atomic predicates (possibly negated) that, if all satisfied,
    would make the entire goal expression true.  When the goal contains
    disjunctions or quantifiers there may be many such solutions.

    This is useful for tracking partial progress: you can evaluate each
    ground option independently and report which fraction of its literals
    are already satisfied.

    Args:
        conds (Conditions): Parsed activity conditions (used for the object map).
        scope (dict[str, str]): Object scope mapping instance names to strings.
        goal_conditions (list[HEAD]): Pre-compiled goal conditions (from
            :func:`get_goal_conditions`).

    Returns:
        list[list[HEAD]]: Each inner list is an independently evaluable set of
        grounded conditions.

    Raises:
        AssertionError: If no consistent ground solutions exist.
    """
    ground_goal_state_options = get_ground_state_options(
        goal_conditions, scope=scope, object_map=conds.parsed_objects
    )
    assert len(ground_goal_state_options) > 0
    return ground_goal_state_options


def evaluate_goal_conditions(goal_conditions, evaluate_fn):
    """Evaluate compiled goal state to see if current simulator state has been met

    Args:
        goal_conditions (list<bddl.condition_evaluation.HEAD>): list of compiled
                                                                goal conditions with
                                                                populated scope
        evaluate_fn (function): callback function to evaluate condition predicates

    Returns:
        tuple[bool, dict[str, list[int]]]: ``(all_satisfied, results)`` where
            *results* maps ``"satisfied"`` / ``"unsatisfied"`` to lists of
            integer indices into *goal_conditions*.
    """
    return evaluate_state(goal_conditions, evaluate_fn)


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


def get_reward(ground_goal_state_options, evaluate_fn):
    """Return reward given ground goal state options.
       Reward formulated as max(<percent literals that are satisfied in the option> for option in ground_goal_state_options)

    Args:
        ground_goal_state_options (list<list<HEAD>>): list of compiled ground goal state options
        evaluate_fn (function): callback function evaluated over primitives

    Returns:
        float: reward
    """
    return max(
        len(evaluate_state(option, evaluate_fn)[-1]["satisfied"]) / float(len(option))
        for option in ground_goal_state_options
    )
