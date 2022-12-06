from bddl.backend_abc import BDDLBackend
from bddl.logic_base import BinaryAtomicFormula, UnaryAtomicFormula

from omnigibson import object_states


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


# TODO: Add remaining predicates.
SUPPORTED_PREDICATES = {
    "inside": get_binary_predicate_for_state(object_states.Inside, "inside"),
    "nextto": get_binary_predicate_for_state(object_states.NextTo, "nextto"),
    "ontop": get_binary_predicate_for_state(object_states.OnTop, "ontop"),
    "under": get_binary_predicate_for_state(object_states.Under, "under"),
    "touching": get_binary_predicate_for_state(object_states.Touching, "touching"),
    "cooked": get_unary_predicate_for_state(object_states.Cooked, "cooked"),
    "burnt": get_unary_predicate_for_state(object_states.Burnt, "burnt"),
    "open": get_unary_predicate_for_state(object_states.Open, "open"),
    "sliced": get_unary_predicate_for_state(object_states.Sliced, "sliced"),
    "toggled_on": get_unary_predicate_for_state(object_states.ToggledOn, "toggled_on"),
    "frozen": get_unary_predicate_for_state(object_states.Frozen, "frozen"),
}


class IGibsonBDDLBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        return SUPPORTED_PREDICATES[predicate_name]
