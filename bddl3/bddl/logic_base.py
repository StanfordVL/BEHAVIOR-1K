"""Abstract base classes for the BDDL expression tree.

Every compiled BDDL condition is represented as a tree of :class:`Expression`
nodes.  Interior nodes are logical connectives (conjunction, negation, etc.)
defined in :mod:`bddl.condition_evaluation`.  Leaf nodes are predicates --
concrete :class:`~bddl.predicates.Predicate` subclasses (e.g.
:class:`~bddl.predicates.OnTop`, :class:`~bddl.predicates.Cooked`), or the
legacy :class:`UnaryAtomicFormula` / :class:`BinaryAtomicFormula` retained
here for backwards compatibility.

Terminology
-----------
- **scope**: ``dict[str, Any]`` mapping each object instance name (e.g.
  ``"bowl.n.01_1"``) to a value.  In the standard pipeline the value is the
  instance name itself (a string); a simulator may later mutate the dict to
  store its own entity objects.  Quantifiers create shallow copies of the
  scope with the bound variable added.
- **body**: The raw parsed sub-expression list coming from the BDDL parser.
  For a leaf predicate like ``(ontop bowl.n.01_1 table.n.02_1)`` the body is
  ``["bowl.n.01_1", "table.n.02_1"]``.  For compound expressions it is a
  nested list structure.
- **object_map**: ``dict[str, list[str]]`` mapping each synset category to
  its list of declared object instance names.  Used by quantifiers to iterate
  over all instances of a category.
- **evaluate_fn**: A callback supplied at evaluation time with signature
  ``(predicate_name, *entity_values) -> bool``.  The expression tree never
  calls the simulator directly; instead it delegates to this callback.
- **sample_fn**: A callback supplied at sampling time with signature
  ``(predicate_name, *entity_values, binary_state, **kwargs) -> bool``.
  Used to request that the simulator set a predicate to a desired state.
- **ground options** (``flattened_condition_options``): Each expression node
  can produce a list of *ground options* -- fully instantiated lists of
  atomic predicates (with ``"not"`` wrappers for negations) that, if all
  true, would satisfy the expression.  This is used downstream to enumerate
  concrete solution paths for a goal that contains disjunctions or
  quantifiers.  See :func:`~bddl.condition_evaluation.get_ground_state_options`.
"""

from abc import abstractmethod, ABCMeta

from future.utils import with_metaclass
from bddl.utils import UncontrolledCategoryError


class Expression(with_metaclass(ABCMeta)):
    """Base class for all nodes in a compiled BDDL expression tree.

    Subclasses must implement :meth:`evaluate`.  Most subclasses also implement
    ``get_ground_options`` to populate ``flattened_condition_options``.

    Args:
        scope: Object scope dictionary (see module docstring).
        body: Raw parsed sub-expression from the BDDL parser.
        object_map: Category-to-instances mapping (see module docstring).

    Attributes:
        children (list[Expression]): Child expression nodes.
        child_values (list): Cached boolean results from the last evaluation.
        kwargs (dict): Extra keyword arguments forwarded to ``evaluate_fn``
            or ``sample_fn``.
        body: The raw parsed body stored for later inspection (e.g. by the
            sampler).
        scope: Reference to the shared scope dict.
        object_map: Reference to the category-to-instances dict.
    """

    def __init__(self, scope, body, object_map):
        self.children = []
        self.child_values = []
        self.kwargs = {}
        self.body = body
        self.scope = scope
        self.object_map = object_map

    @abstractmethod
    def evaluate(self, evaluate_fn):
        """Evaluate this expression against the current simulator state.

        Args:
            evaluate_fn: Callback ``(predicate_name, *entities) -> bool``.

        Returns:
            bool: Whether this expression is satisfied.
        """
        pass


# ---------------------------------------------------------------------------
# Legacy atomic formula classes
# ---------------------------------------------------------------------------
# These were the original leaf-node classes whose subclasses implemented
# ``_evaluate`` / ``_sample`` directly.  The current pipeline uses
# concrete ``Predicate`` subclasses (in predicates.py) instead, which
# delegate to user-supplied callbacks.  The classes below are retained for any
# external code that may still reference them, but are not instantiated by
# the standard compilation path.
# ---------------------------------------------------------------------------


class AtomicFormula(Expression):
    """Base class for leaf-level predicate formulas (legacy)."""

    def __init__(self, scope, body, object_map):
        super().__init__(scope, body, object_map)


class BinaryAtomicFormula(AtomicFormula):
    """A predicate that takes exactly two object arguments (legacy).

    At construction time the two input names are resolved through the scope:
    if a scope value is a string (e.g. a quantifier-bound variable), the
    input is replaced with the resolved name.

    Attributes:
        STATE_NAME (str | None): Predicate name (set by subclasses).
        input1 (str): Resolved first argument name.
        input2 (str): Resolved second argument name.
    """

    STATE_NAME = None

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)
        assert (
            len(body) == 2
        ), f"Param list for predicate {self.STATE_NAME} should have 2 args"
        self.input1, self.input2 = [inp.strip("?") for inp in body]
        self.scope = scope
        try:
            if isinstance(self.scope[self.input1], str):
                self.input1 = self.scope[self.input1]
        except KeyError as e:
            raise UncontrolledCategoryError(e)
        try:
            if isinstance(self.scope[self.input2], str):
                self.input2 = self.scope[self.input2]
        except KeyError as e:
            raise UncontrolledCategoryError(e)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        """Evaluate by delegating to *evaluate_fn* with both resolved inputs."""
        if (self.scope[self.input1] is not None) and (
            self.scope[self.input2] is not None
        ):
            return evaluate_fn(self.STATE_NAME, self.scope[self.input1], self.scope[self.input2], **self.kwargs)
        else:
            print(
                "%s and/or %s are not mapped to simulator objects in scope"
                % (self.input1, self.input2)
            )

    def sample(self, sample_fn, binary_state, **kwargs):
        """Request the simulator to set this predicate to *binary_state*."""
        if (self.scope[self.input1] is not None) and (
            self.scope[self.input2] is not None
        ):
            return sample_fn(self.STATE_NAME, self.scope[self.input1], self.scope[self.input2], binary_state, **kwargs, **self.kwargs)
        else:
            print(
                "%s and/or %s are not mapped to simulator objects in scope"
                % (self.input1, self.input2)
            )

    def get_ground_options(self):
        """A binary predicate has exactly one ground option: itself."""
        self.flattened_condition_options = [
            [[self.STATE_NAME, self.input1, self.input2]]
        ]


class UnaryAtomicFormula(AtomicFormula):
    """A predicate that takes exactly one object argument (legacy).

    Attributes:
        STATE_NAME (str | None): Predicate name (set by subclasses).
        input (str): Resolved argument name.
    """

    STATE_NAME = None

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)
        assert (
            len(body) == 1
        ), f"Param list for predicate {self.STATE_NAME} should have 1 arg"
        self.input = body[0].strip("?")
        self.scope = scope
        try:
            if isinstance(self.scope[self.input], str):
                self.input = self.scope[self.input]
        except KeyError as e:
            raise UncontrolledCategoryError(e)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        """Evaluate by delegating to *evaluate_fn* with the resolved input."""
        if self.scope[self.input] is not None:
            return evaluate_fn(self.STATE_NAME, self.scope[self.input], **self.kwargs)
        else:
            print("%s is not mapped to a simulator object in scope" % self.input)
            return False

    def sample(self, sample_fn, binary_state, **kwargs):
        """Request the simulator to set this predicate to *binary_state*."""
        if self.scope[self.input] is not None:
            return sample_fn(self.STATE_NAME, self.scope[self.input], binary_state, **kwargs, **self.kwargs)
        else:
            print("%s is not mapped to a simulator object in scope" % self.input)
            return False

    def get_ground_options(self):
        """A unary predicate has exactly one ground option: itself."""
        self.flattened_condition_options = [[[self.STATE_NAME, self.input]]]
