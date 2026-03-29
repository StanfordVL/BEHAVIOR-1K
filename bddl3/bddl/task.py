"""
High-level Task interface for BDDL activities.

A Task bundles together the parsed activity definition, the object scope, and
pre-compiled initial/goal conditions. Simulators interact with BDDL primarily
through this class.

Key concepts
------------
- **Activity**: A household task described in a ``.bddl`` problem file (e.g.
  "cleaning_up_after_a_meal"). Each activity has one or more *definitions*
  (numbered variants with different objects or goals).
- **Object scope**: A dictionary mapping BDDL object instance names (e.g.
  ``"bowl.n.01_1"``) to themselves (strings). During evaluation the simulator
  provides an ``evaluate_fn`` callback that maps these names to whatever
  entity representation it uses internally.
- **Conditions**: Parsed and compiled logical expressions (trees of
  :class:`~bddl.condition_evaluation.Expression` nodes) representing initial
  or goal states. They are evaluated by passing an ``evaluate_fn`` callback
  that the simulator supplies.
- **Grounding**: Conditions may contain disjunctions or quantifiers, meaning
  multiple concrete (ground) assignments of objects could satisfy them. A
  *ground goal state option* is one specific, fully-instantiated set of
  atomic predicates that, if all satisfied simultaneously, fulfills the goal.
  The list ``ground_goal_state_options`` enumerates all such satisfying
  assignments so progress can be tracked per-option.
"""

from bddl.activity import Conditions, get_object_scope, get_initial_conditions, get_goal_conditions, get_ground_goal_state_options
from bddl.condition_evaluation import evaluate_state


class Task:
    """Encapsulates a single BDDL activity definition and its compiled conditions.

    On construction the problem file is parsed, the object scope is created,
    and initial / goal conditions are compiled into expression trees ready for
    evaluation.

    Args:
        behavior_activity: Name of the activity (e.g. ``"assembling_gift_baskets"``).
        activity_definition: Integer index selecting which problem definition to
            load for the activity (``0`` for the first variant, etc.).
        simulator_name: Name used to locate the BDDL domain file that lists
            available predicates. Defaults to ``"behavior-1k"``.
        predefined_problem: If given, a raw BDDL problem string to use instead
            of loading from the activity definitions directory.

    Attributes:
        conditions (Conditions): The parsed BDDL conditions container holding
            ``parsed_objects``, ``parsed_initial_conditions``, and
            ``parsed_goal_conditions``.
        object_scope (dict[str, str]): Mapping from each object instance name
            to itself. The simulator should use these names as keys to look up
            its own entity representations inside ``evaluate_fn``.
        parsed_objects (dict[str, list[str]]): Mapping from synset category to
            the list of instance names declared in the problem file.
        initial_conditions (list[HEAD]): Compiled initial-state condition trees.
        goal_conditions (list[HEAD]): Compiled goal-state condition trees.
        ground_goal_state_options (list[list[HEAD]]): Each element is one fully
            grounded (variable-free) assignment of atomic predicates that would
            satisfy the goal. Useful for measuring partial progress toward a
            specific solution path.
    """

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
        """Check whether the goal conditions are currently satisfied.

        Args:
            evaluate_fn: Callback with signature
                ``evaluate_fn(predicate_name: str, *entity_names: str) -> bool``.
                Called once per leaf predicate in the goal tree.  The entity
                names are the object-scope strings (e.g. ``"bowl.n.01_1"``).

        Returns:
            tuple[bool, dict[str, list[int]]]: A pair of ``(all_satisfied,
            results)`` where *results* maps ``"satisfied"`` and
            ``"unsatisfied"`` to lists of integer indices into
            ``self.goal_conditions``.
        """
        return evaluate_state(self.goal_conditions, evaluate_fn)

    def check_initial_conditions(self, evaluate_fn):
        """Check whether the initial conditions are currently satisfied.

        Same interface as :meth:`check_goal` but operates on
        ``self.initial_conditions``.

        Args:
            evaluate_fn: Callback with the same signature as in
                :meth:`check_goal`.

        Returns:
            tuple[bool, dict[str, list[int]]]: See :meth:`check_goal`.
        """
        return evaluate_state(self.initial_conditions, evaluate_fn)
