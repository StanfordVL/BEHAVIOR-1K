from collections import namedtuple

import networkx as nx

from omnigibson.object_states import (
    REGISTERED_OBJECT_STATES,
    AttachedTo,
    Burnt,
    Contains,
    Cooked,
    Covered,
    Draped,
    Filled,
    Folded,
    Frozen,
    Heated,
    HeatSourceOrSink,
    Inside,
    IsGrasping,
    MaxTemperature,
    NextTo,
    ObjectsInFOVOfRobot,
    OnFire,
    OnTop,
    Open,
    Overlaid,
    ParticleApplier,
    ParticleRemover,
    ParticleRequirement,
    ParticleSink,
    ParticleSource,
    Saturated,
    SliceableRequirement,
    SlicerActive,
    ToggledOn,
    Touching,
    Under,
    Unfolded,
)
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin

# states: list of ObjectBaseState
# requirements: list of ObjectBaseRequirement
AbilityDependencies = namedtuple("AbilityDependencies", ("states", "requirements"))

# Maps ability name to list of Object States and / or Ability Requirements that determine
# whether the given ability can be instantiated for a requested object
_ABILITY_DEPENDENCIES = {
    "robot": AbilityDependencies(states=[IsGrasping, ObjectsInFOVOfRobot], requirements=[]),
    "attachable": AbilityDependencies(states=[AttachedTo], requirements=[]),
    "particleApplier": AbilityDependencies(states=[ParticleApplier], requirements=[]),
    "particleRemover": AbilityDependencies(states=[ParticleRemover], requirements=[]),
    "particleSource": AbilityDependencies(states=[ParticleSource], requirements=[ParticleRequirement]),
    "particleSink": AbilityDependencies(states=[ParticleSink], requirements=[ParticleRequirement]),
    "coldSource": AbilityDependencies(states=[HeatSourceOrSink], requirements=[]),
    "cookable": AbilityDependencies(states=[Cooked, Burnt], requirements=[]),
    "coverable": AbilityDependencies(states=[Covered], requirements=[]),
    "freezable": AbilityDependencies(states=[Frozen], requirements=[]),
    "heatable": AbilityDependencies(states=[Heated], requirements=[]),
    "heatSource": AbilityDependencies(states=[HeatSourceOrSink], requirements=[]),
    "meltable": AbilityDependencies(states=[MaxTemperature], requirements=[]),
    "mixingTool": AbilityDependencies(states=[], requirements=[]),
    "openable": AbilityDependencies(states=[Open], requirements=[]),
    "flammable": AbilityDependencies(states=[OnFire], requirements=[]),
    "saturable": AbilityDependencies(states=[Saturated], requirements=[]),
    "sliceable": AbilityDependencies(states=[], requirements=[SliceableRequirement]),
    "slicer": AbilityDependencies(states=[SlicerActive], requirements=[]),
    "toggleable": AbilityDependencies(states=[ToggledOn], requirements=[]),
    "cloth": AbilityDependencies(states=[Folded, Unfolded, Overlaid, Draped], requirements=[]),
    "fillable": AbilityDependencies(states=[Filled, Contains], requirements=[]),
}

_DEFAULT_STATE_SET = frozenset(
    [
        Inside,
        NextTo,
        OnTop,
        Touching,
        Under,
        Covered,
    ]
)

_KINEMATIC_STATE_SET = frozenset(
    [state for state in REGISTERED_OBJECT_STATES.values() if issubclass(state, KinematicsMixin)]
)

_FIRE_STATE_SET = frozenset(
    [
        HeatSourceOrSink,
        OnFire,
    ]
)

_STEAM_STATE_SET = frozenset(
    [
        Heated,
    ]
)

_TEXTURE_CHANGE_STATE_SET = frozenset(
    [
        Frozen,
        Burnt,
        Cooked,
        Saturated,
        ToggledOn,
    ]
)

_SYSTEM_STATE_SET = frozenset(
    [
        Covered,
        Saturated,
        Filled,
        Contains,
    ]
)

_VISUAL_STATE_SET = frozenset(_FIRE_STATE_SET | _STEAM_STATE_SET | _TEXTURE_CHANGE_STATE_SET)

_TEXTURE_CHANGE_PRIORITY = {
    Frozen: 4,
    Burnt: 3,
    Cooked: 2,
    Saturated: 1,
    ToggledOn: 0,
}


def get_system_states():
    return _SYSTEM_STATE_SET


def get_fire_states():
    return _FIRE_STATE_SET


def get_steam_states():
    return _STEAM_STATE_SET


def get_texture_change_states():
    return _TEXTURE_CHANGE_STATE_SET


def get_texture_change_priority():
    return _TEXTURE_CHANGE_PRIORITY


def get_visual_states():
    return _VISUAL_STATE_SET


def get_default_states():
    return _DEFAULT_STATE_SET


def get_state_name(state):
    # Get the name of the class.
    return state.__name__


def get_states_for_ability(ability):
    if ability not in _ABILITY_DEPENDENCIES:
        return []
    return _ABILITY_DEPENDENCIES[ability].states


def get_requirements_for_ability(ability):
    if ability not in _ABILITY_DEPENDENCIES:
        return []
    return _ABILITY_DEPENDENCIES[ability].requirements


def get_state_dependency_graph(states=None):
    """
    Args:
        states (None or Iterable): If specified, specific state(s) to sort. Otherwise, will generate dependency graph
            over all states

    Returns:
        nx.DiGraph: State dependency graph of supported object states
    """
    states = REGISTERED_OBJECT_STATES.values() if states is None else states
    dependencies = {state: set.union(state.get_dependencies(), state.get_optional_dependencies()) for state in states}
    return nx.DiGraph(dependencies)


def get_states_by_dependency_order(states=None):
    """
    Args:
        states (None or Iterable): If specified, specific state(s) to sort. Otherwise, will generate dependency graph
            over all states

    Returns:
        list: all states in topological order of dependency
    """
    return list(
        reversed(
            list(
                nx.algorithms.lexicographical_topological_sort(
                    get_state_dependency_graph(states), key=lambda n: n.__name__
                )
            )
        )
    )


# Define all meta links
META_LINK_TYPES = set()
for state in get_states_by_dependency_order():
    if issubclass(state, LinkBasedStateMixin):
        try:
            for meta_link_type in state.meta_link_types:
                META_LINK_TYPES.add(meta_link_type)
        except NotImplementedError:
            pass
