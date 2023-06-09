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

    def _evaluate(self, entity, **kwargs):
        return entity.get_state(self.STATE_CLASS, **kwargs)

    def _sample(self, entity, binary_state, **kwargs):
        return entity.set_state(self.STATE_CLASS, binary_state, **kwargs)


class ObjectStateBinaryPredicate(BinaryAtomicFormula):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, entity1, entity2, **kwargs):
        return entity1.get_state(self.STATE_CLASS, entity2.wrapped_obj, **kwargs) if entity2.exists() else None

    def _sample(self, entity1, entity2, binary_state, **kwargs):
        return entity1.set_state(self.STATE_CLASS, entity2.wrapped_obj, binary_state, **kwargs) if entity2.exists() else None


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


class BDDLEntity(Wrapper):
    """
    Thin wrapper class that wraps an object or system if it exists, or nothing if it does not exist. Will
    dynamically reference an object / system as they become real in the sim
    """
    def __init__(
        self,
        object_scope,
        entity=None,
    ):
        """
        Args:
            object_scope (str): BDDL synset instance of the entity, e.g.: "almond.n.01_1"
            entity (None or DatasetObject or BaseSystem): If specified, the BDDL entity to wrap. If not
                specified, will initially wrap nothing, but may dynamically reference an actual object or system
                if it exists in the future
        """
        # Store synset and other info, and pass entity internally
        self.object_scope = object_scope
        self.synset = "_".join(self.object_scope.split("_")[:-1])
        self.is_system = self.synset in SUBSTANCE_SYNSET_MAPPING

        # Infer the correct category to assign, special casing agents
        if entity is not None and isinstance(entity, BaseRobot):
            self.og_categories = ["agent"]
        else:
            self.og_categories = [SUBSTANCE_SYNSET_MAPPING[self.synset]] if self.is_system else \
                OBJECT_TAXONOMY.get_subtree_igibson_categories(self.synset)

        super().__init__(obj=entity)

    def _find_valid_object(self):
        """
        Internal helper function to find the first valid simulator object whose category is one of @self.og_categories,
        and has not been mapped to a BDDLEntity yet

        Returns:
            None or DatasetObject: If found, the valid object matching a category from @self.og_categories and not
                mapped
        """
        for category in self.og_categories:
            for obj in og.sim.scene.object_registry("category", category, default_val=[]):
                if isinstance(obj, DatasetObject) and obj.bddl_object_scope is None:
                    # Found valid one, return it
                    return obj

    def exists(self):
        """
        Checks whether the entity referenced by @synset exists. Note: this dynamically mutates self.wrapped_obj, and
        potentially removes it or adds a reference if the entity no longer / now exists.

        Returns:
            bool: Whether the entity referenced by @synset exists
        """
        if self.wrapped_obj is None:
            # If system, check to see if active or not and grab it if so
            if self.is_system:
                if is_system_active(self.og_categories[0]):
                    self.wrapped_obj = get_system(self.og_categories[0])
            # Otherwise, is object, check to see if any valid one exists and grab it if so
            else:
                found_obj = self._find_valid_object()
                if found_obj is not None:
                    found_obj.bddl_object_scope = self.object_scope
                    self.wrapped_obj = found_obj
        else:
            # Check to see if entity no longer exists
            if self.is_system:
                if not is_system_active(self.og_categories[0]):
                    self.wrapped_obj = None
            # Otherwise, is object, check to see if there are no valid ones
            else:
                if og.sim.scene.object_registry("name", self.wrapped_obj.name) is None:
                    self.wrapped_obj = None

        return self.wrapped_obj is not None

    def get_state(self, state, *args, **kwargs):
        """
        Helper function to grab wrapped entity's state @state

        Args:
            state (BaseObjectState): State whose get_value() should be called
            *args (tuple): Any arguments to pass to getter, in order
            **kwargs (dict): Any keyword arguments to pass to getter, in order

        Returns:
            None or any: Returned value(s) from @state if self.wrapped_obj exists (i.e.: not None), else None
        """
        return self.wrapped_obj.states[state].get_value(*args, **kwargs) if self.exists() else None

    def set_state(self, state, *args, **kwargs):
        """
        Helper function to set wrapped entity's state @state. Note: Should only be called if the entity exists!

        Args:
            state (BaseObjectState): State whose set_value() should be called
            *args (tuple): Any arguments to pass to getter, in order
            **kwargs (dict): Any keyword arguments to pass to getter, in order

        Returns:
            any: Returned value(s) from @state if self.wrapped_obj exists (i.e.: not None)
        """
        assert self.exists(), \
            f"Cannot call set_state() for BDDLEntity {self.synset} when the entity does not exist!"
        return self.wrapped_obj.states[state].set_value(*args, **kwargs)

    def __getattr__(self, attr):
        # Sanity check to make sure wrapped obj is not None -- if so, raise error
        assert self.wrapped_obj is not None, f"Cannot access attribute {attr}, since no valid entity currently " \
                                             f"wrapped for BDDLEntity synset {self.synset}!"

        # Call super
        return super().__getattr__(attr=attr)
