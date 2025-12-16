import copy
import itertools

import numpy as np

import bddl
from bddl.logic_base import Expression
from bddl.utils import UnsupportedPredicateError

#################### RECURSIVE PREDICATES ####################

# -JUNCTIONS


class Conjunction(Expression):
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)

        new_scope = copy.copy(scope)
        child_predicates = [
            get_predicate_for_token(subexpression[0], backend)(
                scope,
                backend,
                subexpression[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
            for subexpression in body
        ]
        self.children.extend(child_predicates)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
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
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)

        # body = [[predicate1], [predicate2], ..., [predicateN]]
        new_scope = copy.copy(scope)
        child_predicates = [
            get_predicate_for_token(subexpression[0], backend)(
                scope,
                backend,
                subexpression[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
            for subexpression in body
        ]
        self.children.extend(child_predicates)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
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
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)
        iterable, subexpression = body
        param_label, __, category = iterable
        param_label = param_label.strip("?")
        assert __ == "-", "Middle was not a hyphen"
        for obj_name, obj in scope.items():
            if obj_name in object_map[category]:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj_name
                self.children.append(
                    get_predicate_for_token(subexpression[0], backend)(
                        new_scope,
                        backend,
                        subexpression[1:],
                        object_map,
                        generate_ground_options=generate_ground_options,
                    )
                )

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return all(self.child_values)

    def get_ground_options(self):
        # Accept just a few possible options
        options = list(
            itertools.product(
                *[child.flattened_condition_options for child in self.children]
            )
        )
        self.flattened_condition_options = []
        for option in options:
            self.flattened_condition_options.append(list(itertools.chain(*option)))


class Existential(Expression):
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)
        iterable, subexpression = body
        param_label, __, category = iterable
        param_label = param_label.strip("?")
        assert __ == "-", "Middle was not a hyphen"
        for obj_name, obj in scope.items():
            if obj_name in object_map[category]:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj_name
                # body = [["param_label", "-", "category"], [predicate]]
                self.children.append(
                    get_predicate_for_token(subexpression[0], backend)(
                        new_scope,
                        backend,
                        subexpression[1:],
                        object_map,
                        generate_ground_options=generate_ground_options,
                    )
                )

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return any(self.child_values)

    def get_ground_options(self):
        self.flattened_condition_options = []
        for child in self.children:
            self.flattened_condition_options.extend(child.flattened_condition_options)


class NQuantifier(Expression):
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)

        N, iterable, subexpression = body
        self.N = int(N[0])
        param_label, __, category = iterable
        param_label = param_label.strip("?")
        assert __ == "-", "Middle was not a hyphen"
        for obj_name, obj in scope.items():
            if obj_name in object_map[category]:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj_name
                self.children.append(
                    get_predicate_for_token(subexpression[0], backend)(
                        new_scope,
                        backend,
                        subexpression[1:],
                        object_map,
                        generate_ground_options=generate_ground_options,
                    )
                )

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return sum(self.child_values) == self.N

    def get_ground_options(self):
        # Accept just a few possible options
        options = list(
            itertools.product(
                *[child.flattened_condition_options for child in self.children]
            )
        )
        self.flattened_condition_options = []
        for option in options:
            # for combination in [combo for num_el in range(self.N - 1, len(option)) for combo in itertools.combinations(option, num_el + 1)]:
            # Use a minimal solution (exactly N fulfilled, rather than >=N fulfilled)
            for combination in itertools.combinations(option, self.N):
                self.flattened_condition_options.append(
                    list(itertools.chain(*combination))
                )


class ForPairs(Expression):
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)

        iterable1, iterable2, subexpression = body
        param_label1, __, category1 = iterable1
        param_label2, __, category2 = iterable2
        param_label1 = param_label1.strip("?")
        param_label2 = param_label2.strip("?")
        for obj_name_1, obj_1 in scope.items():
            if obj_name_1 in object_map[category1]:
                sub = []
                for obj_name_2, obj_2 in scope.items():
                    if obj_name_2 in object_map[category2] and obj_name_1 != obj_name_2:
                        new_scope = copy.copy(scope)
                        new_scope[param_label1] = obj_name_1
                        new_scope[param_label2] = obj_name_2
                        sub.append(
                            get_predicate_for_token(subexpression[0], backend)(
                                new_scope,
                                backend,
                                subexpression[1:],
                                object_map,
                                generate_ground_options=generate_ground_options,
                            )
                        )
                self.children.append(sub)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = np.array(
            [
                np.array([subchild.evaluate() for subchild in child])
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
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)

        N, iterable1, iterable2, subexpression = body
        self.N = int(N[0])
        param_label1, __, category1 = iterable1
        param_label2, __, category2 = iterable2
        param_label1 = param_label1.strip("?")
        param_label2 = param_label2.strip("?")
        for obj_name_1, obj_1 in scope.items():
            if obj_name_1 in object_map[category1]:
                sub = []
                for obj_name_2, obj_2 in scope.items():
                    if obj_name_2 in object_map[category2] and obj_name_1 != obj_name_2:
                        new_scope = copy.copy(scope)
                        new_scope[param_label1] = obj_name_1
                        new_scope[param_label2] = obj_name_2
                        sub.append(
                            get_predicate_for_token(subexpression[0], backend)(
                                new_scope,
                                backend,
                                subexpression[1:],
                                object_map,
                                generate_ground_options=generate_ground_options,
                            )
                        )
                self.children.append(sub)

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = np.array(
            [
                np.array([subchild.evaluate() for subchild in child])
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
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)

        # body = [[predicate]]
        subexpression = body[0]
        self.children.append(
            get_predicate_for_token(subexpression[0], backend)(
                scope,
                backend,
                subexpression[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )
        assert len(self.children) == 1, "More than one child."

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert len(self.child_values) == 1, "More than one child value"
        assert all(
            [val is not None for val in self.child_values]
        ), "child_values has NoneTypes"
        return not self.child_values[0]

    def get_ground_options(self):
        # demorgan's law
        self.flattened_condition_options = []
        child = self.children[0]
        negated_options = []
        for option in child.flattened_condition_options:
            negated_conds = []
            for cond in option:
                negated_conds.append(["not", cond])
            negated_options.append(negated_conds)
        # only picking one condition from each set of disjuncts
        for negated_option_selections in itertools.product(*negated_options):
            self.flattened_condition_options.append(
                list(itertools.chain(negated_option_selections))
            )


# IMPLICATION
class Implication(Expression):
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)

        # body = [[antecedent], [consequent]]
        antecedent, consequent = body
        self.children.append(
            get_predicate_for_token(antecedent[0], backend)(
                scope,
                backend,
                antecedent[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )
        self.children.append(
            get_predicate_for_token(consequent[0], backend)(
                scope,
                backend,
                consequent[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
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
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        super().__init__(scope, backend, body, object_map)

        subexpression = body
        self.children.append(
            get_predicate_for_token(subexpression[0], backend)(
                scope,
                backend,
                subexpression[1:],
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )

        self.terms = [term.lstrip("?") for term in list(flatten_list(self.body))]

        if generate_ground_options:
            self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert len(self.child_values) == 1, "More than one child value"
        self.currently_satisfied = self.child_values[0]
        return self.currently_satisfied

    def get_relevant_objects(self):
        # All object instances and categories that are in the scope will be collected
        objects = set(
            [self.scope[obj_name] for obj_name in self.terms if obj_name in self.scope]
        )

        # If this has a quantifier, the category-relevant objects won't all be caught, so adding them here
        # No matter what the quantifier, every object of the category/ies is relevant
        for term in self.terms:
            if term in self.object_map:
                for obj_name, obj in self.scope.items():
                    if obj_name in self.object_map[term]:
                        objects.add(obj)

        return list(objects)

    def get_ground_options(self):
        self.flattened_condition_options = self.children[0].flattened_condition_options


#################### CHECKING ####################


def create_scope(object_terms):
    """
    Creates degenerate scope mapping all object parameters to None
    :param objects: (list of strings) BDDL terms for objects
    """
    scope = {}
    for object_cat in object_terms:
        for object_inst in object_terms[object_cat]:
            scope[object_inst] = None
    return scope


def compile_state(
    parsed_state, backend, scope=None, object_map=None, generate_ground_options=True
):
    compiled_state = []
    for parsed_condition in parsed_state:
        scope = scope if scope is not None else {}
        compiled_state.append(
            HEAD(
                scope,
                backend,
                parsed_condition,
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )
    return compiled_state


def evaluate_state(compiled_state):
    results = {"satisfied": [], "unsatisfied": []}
    for i, compiled_condition in enumerate(compiled_state):
        if compiled_condition.evaluate():
            results["satisfied"].append(i)
        else:
            results["unsatisfied"].append(i)
    return not bool(results["unsatisfied"]), results


def get_ground_state_options(compiled_state, backend, scope=None, object_map=None):
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
        compile_state(option, backend, scope=scope, object_map=object_map)
        for option in sorted(consistent_unpacked_options, key=len)
    ]
    return consistent_unpacked_options


#################### UTIL ######################


def flatten_list(li):
    for elem in li:
        if isinstance(elem, list):
            yield from flatten_list(elem)
        else:
            yield elem


#################### TOKEN MAPPING ####################

TOKEN_MAPPING = {
    # BDDL
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


def get_predicate_for_token(token, backend):
    if token in TOKEN_MAPPING:
        return TOKEN_MAPPING[token]
    else:
        try:
            return backend.get_predicate_class(token)
        except KeyError as e:
            raise UnsupportedPredicateError(e)
