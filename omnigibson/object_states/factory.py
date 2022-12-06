import networkx as nx

from omnigibson.object_states import *
from omnigibson.object_states.object_state_base import BaseObjectState
from omnigibson.object_states.fluid_source import FluidSource

_ALL_STATES = frozenset(
    [
        AABB,
        Burnt,
        ContactBodies,
        Cooked,
        Covered,
        Heated,
        Attached,
        Frozen,
        HeatSourceOrSink,
        HorizontalAdjacency,
        InFOVOfRobot,
        InHandOfRobot,
        InReachOfRobot,
        InSameRoomAsRobot,
        Inside,
        InsideRoomTypes,
        MaxTemperature,
        NextTo,
        ObjectsInFOVOfRobot,
        OnTop,
        Open,
        ParticleApplier,
        ParticleRemover,
        Pose,
        Saturated,
        Sliced,
        Slicer,
        Temperature,
        ToggledOn,
        Touching,
        Under,
        VerticalAdjacency,
        WaterSource,
        WaterSink,
        Filled,
        Folded,
    ]
    + ROOM_STATES
)

_ABILITY_TO_STATE_MAPPING = {
    "attachable": [Attached],
    "burnable": [Burnt],
    "particleApplier": [ParticleApplier],
    "particleRemover": [ParticleRemover],
    "coldSource": [HeatSourceOrSink],
    "cookable": [Cooked],
    "coverable": [Covered],
    "freezable": [Frozen],
    "heatable": [Heated],
    "heatSource": [HeatSourceOrSink],
    "openable": [Open],
    "robot": ROOM_STATES + [ObjectsInFOVOfRobot],
    "saturable": [Saturated],
    "sliceable": [Sliced],
    "slicer": [Slicer],
    "toggleable": [ToggledOn],
    "waterSource": [WaterSource],
    "waterSink": [WaterSink],
    "fillable": [Filled],
    "foldable": [Folded],
}

_DEFAULT_STATE_SET = frozenset(
    [
        InFOVOfRobot,
        InHandOfRobot,
        InReachOfRobot,
        InSameRoomAsRobot,
        Inside,
        NextTo,
        OnTop,
        Touching,
        Under,
        Covered,
    ]
)

_FIRE_STATE_SET = frozenset(
    [
        HeatSourceOrSink,
    ]
)

_STEAM_STATE_SET = frozenset(
    [
        Heated,
    ]
)

_TEXTURE_CHANGE_STATE_SET = frozenset(
    [
        Burnt,
        Cooked,
        Frozen,
        ToggledOn,
    ]
)

_TEXTURE_CHANGE_PRIORITY = {
    Frozen: 4,
    Burnt: 3,
    Cooked: 2,
    Saturated: 1,
    ToggledOn: 0,
}


def get_fire_states():
    return _FIRE_STATE_SET


def get_steam_states():
    return _STEAM_STATE_SET


def get_texture_change_states():
    return _TEXTURE_CHANGE_STATE_SET


def get_texture_change_priority():
    return _TEXTURE_CHANGE_PRIORITY


def get_default_states():
    return _DEFAULT_STATE_SET

def get_fluid_source_states():
    return [state for state in _ALL_STATES if issubclass(state, FluidSource)]

def get_all_states():
    return _ALL_STATES


def get_state_name(state):
    # Get the name of the class.
    return state.__name__


def get_state_from_name(name):
    return next(state for state in _ALL_STATES if get_state_name(state) == name)


def get_states_for_ability(ability):
    if ability not in _ABILITY_TO_STATE_MAPPING:
        return []
    return _ABILITY_TO_STATE_MAPPING[ability]


def get_object_state_instance(state_class, obj, params=None):
    """
    Create an BaseObjectState child class instance for a given object & state.

    The parameters passed in as a dictionary through params are passed as
    kwargs to the object state class constructor.

    :param state_class: The state name from the state name dictionary.
    :param obj: The object for which the state is being constructed.
    :param params: Dict of {param: value} corresponding to the state's params.
    :return: The constructed state object, an instance of a child of
        BaseObjectState.
    """
    if not issubclass(state_class, BaseObjectState):
        assert False, "unknown state class: {}".format(state_class)

    if params is None:
        params = {}

    return state_class(obj, **params)


def get_state_dependency_graph():
    """
    Produce dependency graph of supported object states.
    """
    dependencies = {state: state.get_dependencies() + state.get_optional_dependencies() for state in get_all_states()}
    return nx.DiGraph(dependencies)


def get_states_by_dependency_order():
    """
    Produce a list of all states in topological order of dependency.
    """
    return list(reversed(list(nx.algorithms.topological_sort(get_state_dependency_graph()))))
