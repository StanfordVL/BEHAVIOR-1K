import re
from nltk.corpus import wordnet as wn
from typing import Tuple, List, Set
from bddl.activity import get_initial_conditions, get_goal_conditions, get_object_scope
from bddl.logic_base import UnaryAtomicFormula, BinaryAtomicFormula, Expression
from bddl.backend_abc import BDDLBackend
from bddl.parsing import parse_domain

from enum import Enum, auto


class SynsetState(Enum):
    MATCHED = "Matched"
    PLANNED = "Planned"
    UNMATCHED = "Unmatched"
    ILLEGAL = "Illegal"
    NONE = "No State Info"

    def __str__(self):
        return str(self.value)


# predicates that can only be used for substances
SUBSTANCE_PREDICATES = {
    "filled",
    "insource",
    "empty",
    "saturated",
    "contains",
    "covered",
}

# predicates that indicate the need for a fillable volume
FILLABLE_PREDICATES = {"filled", "contains", "empty"}


def canonicalize(s):
    try:
        return wn.synset(s).name()
    except:
        return s


def wn_synset_exists(synset):
    try:
        wn.synset(synset)
        return True
    except:
        return False


*__, domain_predicates = parse_domain("omnigibson")
UNARIES = [
    predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 1
]
BINARIES = [
    predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 2
]


class TrivialUnaryFormula(UnaryAtomicFormula):
    def _evaluate():
        return True

    def _sample():
        return True


class TrivialBinaryFormula(BinaryAtomicFormula):
    def _evaluate():
        return True

    def _sample():
        return True


def gen_unary_token(predicate_name, generate_ground_options=True):
    return type(
        f"{predicate_name}StateUnaryPredicate",
        (TrivialUnaryFormula,),
        {"STATE_CLASS": "HowDoesItMatter", "STATE_NAME": predicate_name},
    )


def gen_binary_token(predicate_name, generate_ground_options=True):
    return type(
        f"{predicate_name}StateBinaryPredicate",
        (TrivialBinaryFormula,),
        {"STATE_CLASS": "HowDoesItMatter", "STATE_NAME": predicate_name},
    )


class TrivialBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        if predicate_name in UNARIES:
            return gen_unary_token(predicate_name)
        elif predicate_name in BINARIES:
            return gen_binary_token(predicate_name)
        else:
            raise KeyError(predicate_name)


class TrivialGenericObject(object):
    def __init__(self, name):
        self.name = name


def get_initial_and_goal_conditions(conds) -> Tuple[List, List]:
    scope = get_object_scope(conds)
    # Pretend scope has been filled
    for name in scope:
        scope[name] = TrivialGenericObject(name)
    initial_conds = get_initial_conditions(
        conds, TrivialBackend(), scope, generate_ground_options=False
    )
    goal_conds = get_goal_conditions(
        conds, TrivialBackend(), scope, generate_ground_options=False
    )
    return initial_conds, goal_conds


def get_leaf_conditions(cond) -> List:
    if isinstance(cond, list):
        return [leaf_cond for child in cond for leaf_cond in get_leaf_conditions(child)]
    # elif isinstance(cond, (UnaryAtomicFormula, BinaryAtomicFormula)):   # This is too slow.
    if hasattr(cond, "input") or hasattr(cond, "input1"):
        if cond.children:
            raise ValueError(f"Found an atomic formula {cond} with children.")
        else:
            return [cond]
    elif isinstance(cond, Expression):
        if not cond.children:
            raise ValueError(f"Found empty expression {cond} in tree.")

        return [
            leaf_cond
            for child in cond.children
            for leaf_cond in get_leaf_conditions(child)
        ]
    else:
        raise ValueError(f"Found unexpected item {cond} in tree.")


def get_synsets(cond):
    def get_synset_from_scope_name(scope_name):
        lemma, n, number = scope_name.split(".")
        number = number.rsplit("_", 1)[0]
        synset = f"{lemma}.{n}.{number}"
        assert re.fullmatch(
            r"^[A-Za-z-_]+\.n\.[0-9]+$", synset
        ), f"Invalid synset name: {synset}"
        return synset

    # TODO: Too slow!
    # assert isinstance(cond, (UnaryAtomicFormula, BinaryAtomicFormula)), "This only works with atomic formulae"
    if hasattr(cond, "input"):
        return [get_synset_from_scope_name(cond.input)]
    else:
        return [
            get_synset_from_scope_name(cond.input1),
            get_synset_from_scope_name(cond.input2),
        ]


def object_substance_match(cond, synset) -> Tuple[bool, bool]:
    """
    Return two bools corresponding to whether synset is used as a non-substance and as a substance, respectively, in this condition subtree
    """
    leafs = get_leaf_conditions(cond)

    # It's used as a substance if it shows up as the last argument of any substance predicate
    is_used_as_substance = any(
        synset == get_synsets(leaf)[-1]
        for leaf in leafs
        if leaf.STATE_NAME in SUBSTANCE_PREDICATES
    )

    # It's used as a non-substance if it shows up as any argument of a non-substance predicate
    is_used_as_non_substance_in_non_substance_predicate = any(
        synset in get_synsets(leaf)
        for leaf in leafs
        if leaf.STATE_NAME not in SUBSTANCE_PREDICATES | {"future", "real"}
    )
    # or the first argument of a two-argument substance predicate
    is_used_as_non_substance_in_substance_predicate = any(
        synset == get_synsets(leaf)[0]
        for leaf in leafs
        if leaf.STATE_NAME in SUBSTANCE_PREDICATES and hasattr(leaf, "input2")
    )
    is_used_as_non_substance = (
        is_used_as_non_substance_in_non_substance_predicate
        or is_used_as_non_substance_in_substance_predicate
    )
    return is_used_as_non_substance, is_used_as_substance


def object_used_as_fillable(cond, synset) -> Tuple[bool, bool]:
    """
    Return a bool corresponding to whether the synset is used as a fillable at any point
    """

    # Looking for the first argument of one of the fillable predicates.
    leafs = get_leaf_conditions(cond)
    return any(
        synset == get_synsets(leaf)[0]
        for leaf in leafs
        if leaf.STATE_NAME in FILLABLE_PREDICATES
    )


def object_used_predicates(cond, synset) -> Tuple[bool, bool]:
    leafs = get_leaf_conditions(cond)
    return {leaf.STATE_NAME for leaf in leafs if synset in get_synsets(leaf)}


def all_task_predicates(cond) -> Set[str]:
    return {leaf.STATE_NAME for leaf in get_leaf_conditions(cond)}


def leaf_inroom_conds(raw_cond, synsets: Set[str]) -> List[Tuple[str, str]]:
    """
    Return a list of all inroom conditions in the subtree of raw_cond
    """
    ret = []
    if isinstance(raw_cond, list):
        for child in raw_cond:
            ret.extend(leaf_inroom_conds(child, synsets))
        if raw_cond[0] == "inroom":
            synset = raw_cond[1].split("?")[-1].rsplit("_", 1)[0]
            assert synset in synsets, f"{synset} not in valid format"
            ret.append((canonicalize(synset), raw_cond[2]))
    return ret
