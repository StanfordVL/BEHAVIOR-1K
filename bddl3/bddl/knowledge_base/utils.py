import re
from typing import Tuple, List, Set
from bddl.activity import get_initial_conditions, get_goal_conditions, get_object_scope
from bddl.predicates import Predicate
from bddl.logic_base import Expression
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
    """Assert that a synset name is already in canonical WordNet form.

    Raises AssertionError if the synset exists in WordNet but under a
    different canonical name, indicating a data error in the BDDL files.
    """
    from nltk.corpus import wordnet as wn

    try:
        canonical = wn.synset(s).name()
    except Exception:
        return s  # Not in WordNet (custom synset) -- pass through
    assert canonical == s, f"Synset '{s}' is not canonical (expected '{canonical}')"
    return s


def wn_synset_exists(synset):
    from nltk.corpus import wordnet as wn

    try:
        wn.synset(synset)
        return True
    except Exception:
        return False


*__, domain_predicates = parse_domain("behavior-1k")
UNARIES = [
    predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 1
]
BINARIES = [
    predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 2
]


def get_initial_and_goal_conditions(conds) -> Tuple[List, List]:
    scope = get_object_scope(conds)
    initial_conds = get_initial_conditions(
        conds, scope, generate_ground_options=False
    )
    goal_conds = get_goal_conditions(
        conds, scope, generate_ground_options=False
    )
    return initial_conds, goal_conds


def get_leaf_conditions(cond) -> List:
    if isinstance(cond, list):
        return [leaf_cond for child in cond for leaf_cond in get_leaf_conditions(child)]
    if isinstance(cond, Predicate):
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

SYNSET_NAME_REGEX = re.compile(r"^[A-Za-z-_]+\.n\.[0-9]+$")
def get_synsets(cond):
    def get_synset_from_scope_name(scope_name):
        lemma, n, number = scope_name.split(".")
        number = number.rsplit("_", 1)[0]
        synset = f"{lemma}.{n}.{number}"
        assert SYNSET_NAME_REGEX.fullmatch(synset), f"Invalid synset name: {synset}"
        return synset

    return [get_synset_from_scope_name(inp) for inp in cond.inputs]


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
        if leaf.STATE_NAME in SUBSTANCE_PREDICATES and len(leaf.inputs) == 2
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
            ret.append((synset, raw_cond[2]))
    return ret
