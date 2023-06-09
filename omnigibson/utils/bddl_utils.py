import json
import bddl
import os
from bddl.backend_abc import BDDLBackend
from bddl.condition_evaluation import Negation
from bddl.logic_base import BinaryAtomicFormula, UnaryAtomicFormula, AtomicFormula
from bddl.object_taxonomy import ObjectTaxonomy
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.python_utils import Wrapper
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.robots import BaseRobot
from omnigibson import object_states
from omnigibson.object_states.factory import _KINEMATIC_STATE_SET
from omnigibson.systems.system_base import REGISTERED_SYSTEMS, is_system_active, get_system

# Create module logger
log = create_module_logger(module_name=__name__)


class ObjectStateFuturePredicate(UnaryAtomicFormula):
    STATE_NAME = "future"

    def _evaluate(self, entity, **kwargs):
        return not entity.exists()

    def _sample(self, entity, **kwargs):
        raise NotImplementedError()


class ObjectStateRealPredicate(UnaryAtomicFormula):
    STATE_NAME = "real"

    def _evaluate(self, entity, **kwargs):
        return entity.exists()

    def _sample(self, entity, **kwargs):
        raise NotImplementedError()


class ObjectStateUnaryPredicate(UnaryAtomicFormula):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, obj, **kwargs):
        return obj.states[self.STATE_CLASS].get_value(**kwargs)

    def _sample(self, obj, binary_state, **kwargs):
        return obj.states[self.STATE_CLASS].set_value(binary_state, **kwargs)


class ObjectStateBinaryPredicate(BinaryAtomicFormula):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, obj1, obj2, **kwargs):
        return obj1.states[self.STATE_CLASS].get_value(obj2, **kwargs)

    def _sample(self, obj1, obj2, binary_state, **kwargs):
        return obj1.states[self.STATE_CLASS].set_value(obj2, binary_state, **kwargs)


def get_unary_predicate_for_state(state_class, state_name):
    return type(
        state_class.__name__ + "StateUnaryPredicate",
        (ObjectStateUnaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )


def get_binary_predicate_for_state(state_class, state_name):
    return type(
        state_class.__name__ + "StateBinaryPredicate",
        (ObjectStateBinaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )


def process_single_condition(condition):
    """
    Processes a single BDDL condition

    Args:
        condition (Condition): Condition to process

    Returns:
        2-tuple:
            - Expression: Condition's expression
            - bool: Whether this evaluated condition is positive or negative
    """
    if not isinstance(condition.children[0], Negation) and not isinstance(condition.children[0], AtomicFormula):
        log.debug(("Skipping over sampling of predicate that is not a negation or an atomic formula"))
        return None, None

    if isinstance(condition.children[0], Negation):
        condition = condition.children[0].children[0]
        positive = False
    else:
        condition = condition.children[0]
        positive = True

    return condition, positive


# TODO: Add remaining predicates.
SUPPORTED_PREDICATES = {
    "inside": get_binary_predicate_for_state(object_states.Inside, "inside"),
    "nextto": get_binary_predicate_for_state(object_states.NextTo, "nextto"),
    "ontop": get_binary_predicate_for_state(object_states.OnTop, "ontop"),
    "under": get_binary_predicate_for_state(object_states.Under, "under"),
    "touching": get_binary_predicate_for_state(object_states.Touching, "touching"),
    "covered": get_binary_predicate_for_state(object_states.Covered, "covered"),
    "contains": get_binary_predicate_for_state(object_states.Contains, "contains"),
    "saturated": get_binary_predicate_for_state(object_states.Saturated, "saturated"),
    "filled": get_binary_predicate_for_state(object_states.Filled, "filled"),
    "cooked": get_unary_predicate_for_state(object_states.Cooked, "cooked"),
    "burnt": get_unary_predicate_for_state(object_states.Burnt, "burnt"),
    "open": get_unary_predicate_for_state(object_states.Open, "open"),
    "toggled_on": get_unary_predicate_for_state(object_states.ToggledOn, "toggled_on"),
    "frozen": get_unary_predicate_for_state(object_states.Frozen, "frozen"),
    "future": ObjectStateFuturePredicate,
    "real": ObjectStateRealPredicate,
}

KINEMATIC_STATES_BDDL = frozenset([state.__name__.lower() for state in _KINEMATIC_STATE_SET])

# Load substance and object mapping
with open(f"{bddl.__path__[0]}/../substance_synset_mapping.json", "r") as f:
    SUBSTANCE_SYNSET_MAPPING = json.load(f)


# BEHAVIOR-related
OBJECT_TAXONOMY = ObjectTaxonomy() #hierarchy_type="b1k")
FLOOR_SYNSET = "floor.n.01"
NON_SAMPLEABLE_SYNSETS = set()
non_sampleable_category_txt = os.path.join(gm.DATASET_PATH, "metadata/non_sampleable_categories.txt")
if os.path.isfile(non_sampleable_category_txt):
    with open(non_sampleable_category_txt) as f:
        NON_SAMPLEABLE_SYNSETS = set([FLOOR_SYNSET] + [line.strip() for line in f.readlines()])


class OmniGibsonBDDLBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        return SUPPORTED_PREDICATES[predicate_name]
