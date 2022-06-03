import networkx as nx

from igibson.object_states import *
from igibson.object_states.object_state_base import BaseObjectState

_ALL_STATES = frozenset(
    [
        AABB,
        Burnt,
        CleaningTool,
        ContactBodies,
        Cooked,
        Dusty,
        Heated,
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
        OnFloor,
        OnTop,
        Open,
        Pose,
        Sliced,
        Slicer,
        Soaked,
        Stained,
        Temperature,
        ToggledOn,
        Touching,
        Under,
        VerticalAdjacency,
        WaterSource,
        WaterSink,
        Filled,
    ]
    + ROOM_STATES
)

_ABILITY_TO_STATE_MAPPING = {
    "burnable": [Burnt],
    "cleaningTool": [CleaningTool],
    "coldSource": [HeatSourceOrSink],
    "cookable": [Cooked],
    "dustyable": [Dusty],
    "freezable": [Frozen],
    "heatable": [Heated],
    "heatSource": [HeatSourceOrSink],
    "openable": [Open],
    "robot": ROOM_STATES + [ObjectsInFOVOfRobot],
    "sliceable": [Sliced],
    "slicer": [Slicer],
    "soakable": [Soaked],
    "stainable": [Stained],
    "toggleable": [ToggledOn],
    "waterSource": [WaterSource],
    "waterSink": [WaterSink],
    "filled": [Filled],
}

_DEFAULT_STATE_SET = frozenset(
    [
        InFOVOfRobot,
        InHandOfRobot,
        InReachOfRobot,
        InSameRoomAsRobot,
        Inside,
        NextTo,
        OnFloor,
        OnTop,
        Touching,
        Under,
    ]
)

_STEAM_STATE_SET = frozenset(
    [
        Heated,
    ]
)


def get_steam_states():
    return _STEAM_STATE_SET


def get_default_states():
    return _DEFAULT_STATE_SET


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
