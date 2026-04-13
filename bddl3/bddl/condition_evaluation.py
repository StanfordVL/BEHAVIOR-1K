"""Compile, evaluate, and ground BDDL conditions.

This module turns parsed BDDL expressions (nested Python lists produced by
:func:`~bddl.parsing.parse_problem`) into a tree of :class:`Expression` nodes
that can be **evaluated** against a simulator and **grounded** into concrete
solution paths.

Overview
--------
1. **Compilation** -- :func:`compile_state` walks a list of parsed conditions
   and wraps each one in a :class:`HEAD` node whose single child is the
   recursively-compiled sub-expression.
2. **Evaluation** -- Calling ``head.evaluate(evaluate_fn)`` propagates the
   ``evaluate_fn`` callback down the tree.  Leaf
   :class:`~bddl.predicates.Predicate` nodes call
   ``evaluate_fn(predicate_cls, *entity_names)`` and logical
   connectives combine the results.
3. **Grounding** -- Each node produces ``flattened_condition_options``: a list
   of *ground options*, where each option is a list of atomic predicates
   (possibly wrapped with ``["not", ...]``) that would satisfy the node.
   :func:`get_ground_state_options` enumerates all consistent combinations
   across a set of compiled conditions and re-compiles each one so it can be
   independently evaluated.

Key terminology
~~~~~~~~~~~~~~~
- **Scope**: ``dict[str, str]`` mapping object instance names to themselves
  (or, after simulator population, to entity objects).  Quantifiers create
  shallow copies with the bound variable added.
- **Object map**: ``dict[str, list[str]]`` -- category to instance-name list.
- **Ground / grounded**: A condition is *grounded* when all quantified
  variables have been replaced by specific object instances and all
  disjunctions have been resolved to a single branch.  A grounded condition
  contains only atomic predicates (possibly negated).
- **Ground option**: One specific set of grounded atomic predicates that, if
  all true simultaneously, satisfies the parent expression.
- **evaluate_fn**: User-supplied callback
  ``(predicate_name: str, *entities) -> bool`` evaluated at leaf nodes.
- **sample_fn**: User-supplied callback
  ``(predicate_name: str, *entities, binary_state: bool, **kw) -> bool``
  used to request the simulator set a predicate.
"""

import copy
import itertools

import numpy as np

import bddl
from bddl.logic_base import Expression
from bddl.predicates import TOKEN_TO_PREDICATE, Predicate
from bddl.utils import UncontrolledCategoryError

#################### SCOPE HELPERS ####################


def _iter_scope(scope):
    """Iterate over instance names in a scope (set or dict)."""
    if isinstance(scope, dict):
        return scope.keys()
    return scope


def _bind_variable(scope, param_label, obj_name):
    """Create a new scope dict with a quantifier variable binding.

    If scope is a set, converts to a dict first. If already a dict
    (from a parent quantifier), shallow-copies and adds the binding.
    """
    if isinstance(scope, set):
        new_scope = {name: name for name in scope}
    else:
        new_scope = copy.copy(scope)
    new_scope[param_label] = obj_name
    return new_scope


#################### RECURSIVE PREDICATES ####################

# -JUNCTIONS


class Conjunction(Expression):
    """Logical AND over child sub-expressions.

    Satisfied when **all** children evaluate to True.

    Ground options are the Cartesian product of each child's options
    (every child must be satisfied simultaneously).
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)

        new_scope = copy.copy(scope)
        child_predicates = [
            get_predicate_for_token(subexpression[0])(
                scope,
                subexpression[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
            for subexpression in body
        ]
        self.children.extend(child_predicates)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = [child.evaluate(evaluate_fn) for child in self.children]
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return all(self.child_values)

    def get_ground_options(self):
        options = list(
            itertools.product(
                *[child.flattened_condition_options for child in self.children]
            )
        )
        self.flattened_condition_options = []
        for option in options:
            self.flattened_condition_options.append(list(itertools.chain(*option)))


class Disjunction(Expression):
    """Logical OR over child sub-expressions.

    Satisfied when **at least one** child evaluates to True.

    Ground options are the union of each child's options (any single child
    being satisfied is enough).
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)

        # body = [[predicate1], [predicate2], ..., [predicateN]]
        new_scope = copy.copy(scope)
        child_predicates = [
            get_predicate_for_token(subexpression[0])(
                scope,
                subexpression[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
            for subexpression in body
        ]
        self.children.extend(child_predicates)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = [child.evaluate(evaluate_fn) for child in self.children]
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return any(self.child_values)

    def get_ground_options(self):
        self.flattened_condition_options = []
        for child in self.children:
            self.flattened_condition_options.extend(child.flattened_condition_options)


# QUANTIFIERS


class Universal(Expression):
    """Universal quantifier (``forall``).

    ``(forall (?x - category) (predicate ...))``

    Creates one child sub-expression per object instance of the given
    *category*, with the quantified variable bound in a fresh scope copy.

    Satisfied when **all** children are satisfied.  Ground options are the
    Cartesian product (like :class:`Conjunction`).
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)
        iterable, subexpression = body
        param_label, __, category = iterable
        param_label = param_label.strip("?")
        assert __ == "-", "Middle was not a hyphen"
        for obj_name in _iter_scope(scope):
            if obj_name in object_map[category]:
                new_scope = _bind_variable(scope, param_label, obj_name)
                self.children.append(
                    get_predicate_for_token(subexpression[0])(
                        new_scope,
                        subexpression[1:],
                        object_map,
                        generate_ground_options=generate_ground_options,
                    )
                )

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = [child.evaluate(evaluate_fn) for child in self.children]
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return all(self.child_values)

    def get_ground_options(self):
        options = list(
            itertools.product(
                *[child.flattened_condition_options for child in self.children]
            )
        )
        self.flattened_condition_options = []
        for option in options:
            self.flattened_condition_options.append(list(itertools.chain(*option)))


class Existential(Expression):
    """Existential quantifier (``exists``).

    ``(exists (?x - category) (predicate ...))``

    Creates one child per instance of *category*.  Satisfied when **at least
    one** child is satisfied.  Ground options are the union (like
    :class:`Disjunction`).
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)
        iterable, subexpression = body
        param_label, __, category = iterable
        param_label = param_label.strip("?")
        assert __ == "-", "Middle was not a hyphen"
        for obj_name in _iter_scope(scope):
            if obj_name in object_map[category]:
                new_scope = _bind_variable(scope, param_label, obj_name)
                self.children.append(
                    get_predicate_for_token(subexpression[0])(
                        new_scope,
                        subexpression[1:],
                        object_map,
                        generate_ground_options=generate_ground_options,
                    )
                )

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = [child.evaluate(evaluate_fn) for child in self.children]
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return any(self.child_values)

    def get_ground_options(self):
        self.flattened_condition_options = []
        for child in self.children:
            self.flattened_condition_options.extend(child.flattened_condition_options)


class NQuantifier(Expression):
    """Exact-count quantifier (``forn``).

    ``(forn (N) (?x - category) (predicate ...))``

    Satisfied when **exactly N** of the children are satisfied.

    Ground options enumerate all combinations of exactly *N* children.
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)

        N, iterable, subexpression = body
        self.N = int(N[0])
        param_label, __, category = iterable
        param_label = param_label.strip("?")
        assert __ == "-", "Middle was not a hyphen"
        for obj_name in _iter_scope(scope):
            if obj_name in object_map[category]:
                new_scope = _bind_variable(scope, param_label, obj_name)
                self.children.append(
                    get_predicate_for_token(subexpression[0])(
                        new_scope,
                        subexpression[1:],
                        object_map,
                        generate_ground_options=generate_ground_options,
                    )
                )

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = [child.evaluate(evaluate_fn) for child in self.children]
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return sum(self.child_values) == self.N

    def get_ground_options(self):
        options = list(
            itertools.product(
                *[child.flattened_condition_options for child in self.children]
            )
        )
        self.flattened_condition_options = []
        for option in options:
            # Use a minimal solution (exactly N fulfilled, rather than >=N fulfilled)
            for combination in itertools.combinations(option, self.N):
                self.flattened_condition_options.append(
                    list(itertools.chain(*combination))
                )


class ForPairs(Expression):
    """Pair-wise quantifier (``forpairs``).

    ``(forpairs (?x - cat1) (?y - cat2) (predicate ...))``

    Creates a 2-D matrix of children: one sub-expression for every
    ``(x, y)`` pair where ``x != y``.  Satisfied when a perfect matching
    (bipartite) of size ``min(|cat1|, |cat2|)`` exists such that each
    matched pair's predicate holds.

    Children are stored as a list of lists: ``children[i][j]`` is the
    sub-expression for the *i*-th instance of *cat1* paired with the
    *j*-th instance of *cat2*.
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)

        iterable1, iterable2, subexpression = body
        param_label1, __, category1 = iterable1
        param_label2, __, category2 = iterable2
        param_label1 = param_label1.strip("?")
        param_label2 = param_label2.strip("?")
        for obj_name_1 in _iter_scope(scope):
            if obj_name_1 in object_map[category1]:
                sub = []
                for obj_name_2 in _iter_scope(scope):
                    if obj_name_2 in object_map[category2] and obj_name_1 != obj_name_2:
                        new_scope = _bind_variable(scope, param_label1, obj_name_1)
                        new_scope[param_label2] = obj_name_2
                        sub.append(
                            get_predicate_for_token(subexpression[0])(
                                new_scope,
                                subexpression[1:],
                                object_map,
                                generate_ground_options=generate_ground_options,
                            )
                        )
                self.children.append(sub)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = np.array(
            [
                np.array([subchild.evaluate(evaluate_fn) for subchild in child])
                for child in self.children
            ]
        )

        L = min(len(self.children), len(self.children[0]))
        return (np.sum(np.any(self.child_values, axis=1), axis=0) >= L) and (
            np.sum(np.any(self.child_values, axis=0), axis=0) >= L
        )

    def get_ground_options(self):
        self.flattened_condition_options = []
        M, N = len(self.children), len(self.children[0])
        L, G = min(M, N), max(M, N)
        all_L_choices = itertools.permutations(range(L))
        all_G_choices = itertools.permutations(range(G), r=L)
        for lchoice in all_L_choices:
            for gchoice in all_G_choices:
                if M < N:
                    all_child_options = [
                        self.children[lchoice[l]][
                            gchoice[l]
                        ].flattened_condition_options
                        for l in range(L)
                    ]
                else:
                    all_child_options = [
                        self.children[gchoice[l]][
                            lchoice[l]
                        ].flattened_condition_options
                        for l in range(L)
                    ]
                choice_options = itertools.product(*all_child_options)
                unpacked_choice_options = []
                for choice_option in choice_options:
                    unpacked_choice_options.append(
                        list(itertools.chain(*choice_option))
                    )
                self.flattened_condition_options.extend(unpacked_choice_options)


class ForNPairs(Expression):
    """N-pair quantifier (``fornpairs``).

    ``(fornpairs (N) (?x - cat1) (?y - cat2) (predicate ...))``

    Like :class:`ForPairs` but requires exactly *N* matched pairs instead of
    a full matching.
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)

        N, iterable1, iterable2, subexpression = body
        self.N = int(N[0])
        param_label1, __, category1 = iterable1
        param_label2, __, category2 = iterable2
        param_label1 = param_label1.strip("?")
        param_label2 = param_label2.strip("?")
        for obj_name_1 in _iter_scope(scope):
            if obj_name_1 in object_map[category1]:
                sub = []
                for obj_name_2 in _iter_scope(scope):
                    if obj_name_2 in object_map[category2] and obj_name_1 != obj_name_2:
                        new_scope = _bind_variable(scope, param_label1, obj_name_1)
                        new_scope[param_label2] = obj_name_2
                        sub.append(
                            get_predicate_for_token(subexpression[0])(
                                new_scope,
                                subexpression[1:],
                                object_map,
                                generate_ground_options=generate_ground_options,
                            )
                        )
                self.children.append(sub)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = np.array(
            [
                np.array([subchild.evaluate(evaluate_fn) for subchild in child])
                for child in self.children
            ]
        )
        return (np.sum(np.any(self.child_values, axis=1), axis=0) >= self.N) and (
            np.sum(np.any(self.child_values, axis=0), axis=0) >= self.N
        )

    def get_ground_options(self):
        self.flattened_condition_options = []
        P, Q = len(self.children), len(self.children[0])
        L = min(P, Q)
        assert self.N <= L, "ForNPairs asks for more pairs than instances available"
        all_P_choices = itertools.permutations(range(P), r=self.N)
        all_Q_choices = itertools.permutations(range(Q), r=self.N)
        for pchoice in all_P_choices:
            for qchoice in all_Q_choices:
                all_child_options = [
                    self.children[pchoice[n]][qchoice[n]].flattened_condition_options
                    for n in range(self.N)
                ]
                choice_options = itertools.product(*all_child_options)
                unpacked_choice_options = []
                for choice_option in choice_options:
                    unpacked_choice_options.append(
                        list(itertools.chain(*choice_option))
                    )
                self.flattened_condition_options.extend(unpacked_choice_options)


# NEGATION


class Negation(Expression):
    """Logical NOT wrapping a single child expression.

    Ground options negate each atomic predicate in the child's options using
    De Morgan's law.
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)

        # body = [[predicate]]
        subexpression = body[0]
        self.children.append(
            get_predicate_for_token(subexpression[0])(
                scope,
                subexpression[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )
        assert len(self.children) == 1, "More than one child."

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = [child.evaluate(evaluate_fn) for child in self.children]
        assert len(self.child_values) == 1, "More than one child value"
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return not self.child_values[0]

    def get_ground_options(self):
        # De Morgan's law: NOT(a OR b) = (NOT a) AND (NOT b)
        self.flattened_condition_options = []
        child = self.children[0]
        negated_options = []
        for option in child.flattened_condition_options:
            negated_conds = []
            for cond in option:
                negated_conds.append(["not", cond])
            negated_options.append(negated_conds)
        # Pick one negated condition from each set of disjuncts
        for negated_option_selections in itertools.product(*negated_options):
            self.flattened_condition_options.append(
                list(itertools.chain(negated_option_selections))
            )


# IMPLICATION


class Implication(Expression):
    """Material implication: ``(imply antecedent consequent)``.

    Equivalent to ``(not antecedent) OR consequent``.

    Ground options are the union of the negated antecedent options and the
    consequent options.
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)

        # body = [[antecedent], [consequent]]
        antecedent, consequent = body
        self.children.append(
            get_predicate_for_token(antecedent[0])(
                scope,
                antecedent[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )
        self.children.append(
            get_predicate_for_token(consequent[0])(
                scope,
                consequent[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = [child.evaluate(evaluate_fn) for child in self.children]
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        ante, cons = self.child_values
        return (not ante) or cons

    def get_ground_options(self):
        # (not antecedent) or consequent
        flattened_neg_antecedent_options = []
        antecedent = self.children[0]
        negated_options = []
        for option in antecedent.flattened_condition_options:
            negated_conds = []
            for cond in option:
                negated_conds.append(["not", cond])
            negated_options.append(negated_conds)
        for negated_option_selections in itertools.product(*negated_options):
            flattened_neg_antecedent_options.append(
                list(itertools.chain(negated_option_selections))
            )

        flattened_consequent_options = self.children[1].flattened_condition_options

        self.flattened_condition_options = (
            flattened_neg_antecedent_options + flattened_consequent_options
        )


# HEAD


class HEAD(Expression):
    """Root wrapper for a single top-level condition.

    Every compiled condition is wrapped in a HEAD, which has exactly one
    child (the actual expression tree).  HEAD also extracts ``terms`` -- the
    flat list of all tokens in the body -- for use by higher-level code
    (e.g. to find which objects are relevant to a condition).

    Attributes:
        terms (list[str]): All tokens from the parsed body with leading ``?``
            stripped.  Includes predicate names, object instances, and
            category names.
        currently_satisfied (bool | None): Cached result of the last
            :meth:`evaluate` call.
    """

    def __init__(self, scope, body, object_map, generate_ground_options=True):
        super().__init__(scope, body, object_map)

        subexpression = body
        self.children.append(
            get_predicate_for_token(subexpression[0])(
                scope,
                subexpression[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )

        self.terms = [term.lstrip("?") for term in list(flatten_list(self.body))]

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self, evaluate_fn):
        self.child_values = [child.evaluate(evaluate_fn) for child in self.children]
        assert len(self.child_values) == 1, "More than one child value"
        self.currently_satisfied = self.child_values[0]
        return self.currently_satisfied

    def get_relevant_objects(self):
        """Return all scope entries referenced by this condition.

        Includes direct object-instance references and, for quantified
        conditions, every instance in the quantified category.

        Returns:
            list: Scope values (strings or simulator entities) referenced by
            this condition.
        """
        scope_names = self.scope if isinstance(self.scope, set) else set(self.scope.keys())
        objects = {name for name in self.terms if name in scope_names}

        # For quantifiers, the category-relevant objects won't all be caught
        # by the above, so add them here.
        for term in self.terms:
            if term in self.object_map:
                for name in scope_names:
                    if name in self.object_map[term]:
                        objects.add(name)

        return list(objects)

    def get_ground_options(self):
        self.flattened_condition_options = self.children[0].flattened_condition_options


#################### CHECKING ####################


def create_scope(object_terms):
    """Create an object scope as a set of all declared instance names.

    Args:
        object_terms: ``dict[str, list[str]]`` mapping synset categories to
            their declared instance names.

    Returns:
        set[str]: All instance names across all categories.
    """
    scope = set()
    for object_cat in object_terms:
        for object_inst in object_terms[object_cat]:
            scope.add(object_inst)
    return scope


def compile_state(parsed_state, scope=None, object_map=None, generate_ground_options=True):
    """Compile a list of parsed BDDL conditions into expression trees.

    Each parsed condition (a nested list from the parser) is wrapped in a
    :class:`HEAD` node.

    Args:
        parsed_state: List of parsed conditions (each a nested list).
        scope: Object scope dict.  Defaults to an empty dict if None.
        object_map: Category-to-instances mapping.
        generate_ground_options: Whether to eagerly compute
            ``flattened_condition_options`` on each node.

    Returns:
        list[HEAD]: One HEAD per parsed condition.
    """
    compiled_state = []
    for parsed_condition in parsed_state:
        scope = scope if scope is not None else {}
        compiled_state.append(
            HEAD(
                scope,
                parsed_condition,
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )
    return compiled_state


def evaluate_state(compiled_state, evaluate_fn):
    """Evaluate a list of compiled conditions and report which are satisfied.

    Args:
        compiled_state: List of :class:`HEAD` nodes to evaluate.
        evaluate_fn: Callback ``(predicate_name, *entities) -> bool``.

    Returns:
        tuple[bool, dict[str, list[int]]]: ``(all_satisfied, results)`` where
        *results* maps ``"satisfied"`` and ``"unsatisfied"`` to lists of
        integer indices into *compiled_state*.
    """
    results = {"satisfied": [], "unsatisfied": []}
    for i, compiled_condition in enumerate(compiled_state):
        if compiled_condition.evaluate(evaluate_fn):
            results["satisfied"].append(i)
        else:
            results["unsatisfied"].append(i)
    return not bool(results["unsatisfied"]), results


def get_ground_state_options(compiled_state, scope=None, object_map=None):
    """Enumerate all grounded solution paths for a set of compiled conditions.

    Takes the Cartesian product of each condition's
    ``flattened_condition_options``, filters out self-contradictory
    combinations (where both ``P`` and ``NOT P`` appear), and re-compiles
    each surviving option into its own list of :class:`HEAD` nodes.

    This is used to turn a goal that contains disjunctions / quantifiers into
    a list of concrete "if you achieve exactly these atomic predicates, the
    goal is met" sets.

    Args:
        compiled_state: List of :class:`HEAD` nodes (with ground options
            already computed).
        scope: Object scope dict for re-compilation.
        object_map: Category-to-instances mapping for re-compilation.

    Returns:
        list[list[HEAD]]: Each inner list is an independently evaluable set
        of grounded conditions.  Sorted shortest-first.
    """
    all_options = list(
        itertools.product(
            *[
                compiled_condition.flattened_condition_options
                for compiled_condition in compiled_state
            ]
        )
    )
    all_unpacked_options = [list(itertools.chain(*option)) for option in all_options]

    # Remove all unsatisfiable options (those that contain some (cond1 and not cond1))
    consistent_unpacked_options = []
    for option in all_unpacked_options:
        consistent = True
        for cond1, cond2 in itertools.combinations(option, 2):
            if (cond1[0] == "not" and cond1[1] == cond2) or (
                cond2[0] == "not" and cond2[1] == cond1
            ):
                consistent = False
                break
        if not consistent:
            continue
        consistent_unpacked_options.append(option)

    consistent_unpacked_options = [
        compile_state(option, scope=scope, object_map=object_map)
        for option in sorted(consistent_unpacked_options, key=len)
    ]
    return consistent_unpacked_options


#################### UTIL ######################


def flatten_list(li):
    """Recursively yield all non-list elements from a nested list."""
    for elem in li:
        if isinstance(elem, list):
            yield from flatten_list(elem)
        else:
            yield elem


#################### TOKEN MAPPING ####################


TOKEN_MAPPING = {
    # Standard logical connectives
    "forall": Universal,
    "exists": Existential,
    "and": Conjunction,
    "or": Disjunction,
    "not": Negation,
    "imply": Implication,
    # BDDL extensions
    "forn": NQuantifier,
    "forpairs": ForPairs,
    "fornpairs": ForNPairs,
}


def get_predicate_for_token(token):
    """Return a constructor for the expression tree node matching *token*.

    If *token* is a logical connective (e.g. ``"and"``, ``"forall"``), the
    corresponding class from :data:`TOKEN_MAPPING` is returned directly.

    Otherwise *token* is looked up in :data:`~bddl.predicates.TOKEN_TO_PREDICATE`
    and a factory is returned that instantiates the predicate class with the
    token string baked in.

    Args:
        token: The first element of a parsed sub-expression.

    Returns:
        callable: A constructor with signature
        ``(scope, body, object_map, generate_ground_options=True)``.

    Raises:
        KeyError: If *token* is neither a logical connective nor a known
        predicate.
    """
    if token in TOKEN_MAPPING:
        return TOKEN_MAPPING[token]
    else:
        predicate_class = TOKEN_TO_PREDICATE[token]
        return lambda scope, body, object_map, generate_ground_options=True: predicate_class(
            token, scope, body, object_map, generate_ground_options=generate_ground_options
        )
