"""BDDL predicate class definitions.

Every predicate declared in the BDDL domain file has a corresponding class
here.  These classes are the **leaf nodes** of the compiled expression tree --
each instance binds a predicate to specific object arguments resolved through
the scope.

The two base classes, :class:`UnaryPredicate` and :class:`BinaryPredicate`,
indicate whether the predicate takes one or two object arguments.  Both
inherit from :class:`Predicate`, which inherits from
:class:`~bddl.logic_base.Expression` so that predicate instances can sit
directly in the expression tree alongside logical connectives like
``Conjunction`` and ``Negation``.

Usage in a simulator callback::

    from bddl.predicates import OnTop, Cooked

    def my_evaluate(predicate_cls, *entities):
        if predicate_cls is OnTop:
            return check_on_top(entities[0], entities[1])
        elif predicate_cls is Cooked:
            return check_cooked(entities[0])
        ...
"""

from bddl.logic_base import Expression
from bddl.utils import UncontrolledCategoryError


class Predicate(Expression):
    """Base class for all BDDL predicates.

    A Predicate instance is a leaf node in the expression tree.  It binds a
    predicate to specific object arguments resolved through the scope.

    On evaluation, it looks up the resolved arguments from the scope and
    passes them (along with its own class) to the user-supplied
    ``evaluate_fn`` callback.

    Attributes:
        arity (int): Number of object arguments the predicate takes.
        STATE_NAME (str): The BDDL token string (e.g. ``"ontop"``).  Set at
            construction by :func:`~bddl.condition_evaluation.get_predicate_for_token`.
            Used by the sampler for string-based comparisons.
        inputs (list[str]): Resolved argument names after stripping ``?``
            and following any scope indirections (for quantifier-bound variables).
    """

    arity: int

    def __init__(self, token, scope, body, object_map, generate_ground_options=True):
        """
        Args:
            token: The BDDL predicate name string (e.g. ``"ontop"``).
            scope: Object scope -- a set of instance names, or a dict with
                variable bindings from a quantifier.
            body: List of argument tokens (e.g. ``["bowl.n.01_1", "table.n.02_1"]``).
            object_map: Category-to-instances mapping.
            generate_ground_options: Whether to compute ground options.
        """
        super().__init__(scope, body, object_map)
        self.STATE_NAME = token
        self.inputs = [inp.strip("?") for inp in body]
        # Resolve quantifier-bound variables through the scope dict
        if isinstance(scope, dict):
            for i, inp in enumerate(self.inputs):
                if inp in scope:
                    self.inputs[i] = scope[inp]
                elif inp not in object_map and not any(inp in insts for insts in object_map.values()):
                    raise UncontrolledCategoryError(inp)
        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        """Evaluate this predicate by calling *evaluate_fn*.

        Passes the resolved input instance names directly to the callback.

        Args:
            evaluate_fn: ``(predicate_cls, *entity_names) -> bool``.

        Returns:
            bool: Result of the callback.
        """
        return evaluate_fn(type(self), *self.inputs, **self.kwargs)

    def sample(self, sample_fn, binary_state, **kwargs):
        """Request the simulator to set this predicate to *binary_state*.

        Args:
            sample_fn: ``(predicate_cls, *entity_names, binary_state, **kw) -> bool``.
            binary_state: Desired truth value.
            **kwargs: Extra arguments forwarded to *sample_fn*.

        Returns:
            bool: Whether sampling succeeded.
        """
        return sample_fn(type(self), *self.inputs, binary_state, **kwargs, **self.kwargs)

    def get_ground_options(self):
        """A single predicate has exactly one ground option: itself."""
        self.flattened_condition_options = [[[self.STATE_NAME] + self.inputs]]


class UnaryPredicate(Predicate):
    """A predicate that takes exactly one object argument."""

    arity = 1


class BinaryPredicate(Predicate):
    """A predicate that takes exactly two object arguments."""

    arity = 2


# ---------------------------------------------------------------------------
# Unary predicates
# ---------------------------------------------------------------------------

class Cooked(UnaryPredicate):
    pass


class Frozen(UnaryPredicate):
    pass


class Open(UnaryPredicate):
    pass


class Folded(UnaryPredicate):
    pass


class Unfolded(UnaryPredicate):
    pass


class ToggledOn(UnaryPredicate):
    pass


class Hot(UnaryPredicate):
    pass


class OnFire(UnaryPredicate):
    pass


class Future(UnaryPredicate):
    pass


class Real(UnaryPredicate):
    pass


class Broken(UnaryPredicate):
    pass


# ---------------------------------------------------------------------------
# Binary predicates
# ---------------------------------------------------------------------------

class Saturated(BinaryPredicate):
    pass


class Covered(BinaryPredicate):
    pass


class Filled(BinaryPredicate):
    pass


class Contains(BinaryPredicate):
    pass


class OnTop(BinaryPredicate):
    pass


class NextTo(BinaryPredicate):
    pass


class Under(BinaryPredicate):
    pass


class Touching(BinaryPredicate):
    pass


class Inside(BinaryPredicate):
    pass


class Overlaid(BinaryPredicate):
    pass


class Attached(BinaryPredicate):
    pass


class Draped(BinaryPredicate):
    pass


class InSource(BinaryPredicate):
    pass


class InRoom(BinaryPredicate):
    pass


class Grasped(BinaryPredicate):
    pass


# ---------------------------------------------------------------------------
# Token-to-predicate mapping: BDDL token string -> predicate class
# ---------------------------------------------------------------------------

TOKEN_TO_PREDICATE = {
    # Unary
    "cooked": Cooked,
    "frozen": Frozen,
    "open": Open,
    "folded": Folded,
    "unfolded": Unfolded,
    "toggled_on": ToggledOn,
    "hot": Hot,
    "on_fire": OnFire,
    "future": Future,
    "real": Real,
    "broken": Broken,
    # Binary
    "saturated": Saturated,
    "covered": Covered,
    "filled": Filled,
    "contains": Contains,
    "ontop": OnTop,
    "nextto": NextTo,
    "under": Under,
    "touching": Touching,
    "inside": Inside,
    "overlaid": Overlaid,
    "attached": Attached,
    "draped": Draped,
    "insource": InSource,
    "inroom": InRoom,
    "grasped": Grasped,
}
