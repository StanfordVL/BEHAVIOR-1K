import sys
import logging
from igibson.object_states.factory import get_state_name, get_default_states, get_states_for_ability, get_object_state_instance
from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.object_states.object_state_base import CachingEnabledObjectState
from igibson.objects.object_base import BaseObject

# Optionally import bddl for object taxonomy.
try:
    from bddl.object_taxonomy import ObjectTaxonomy

    OBJECT_TAXONOMY = ObjectTaxonomy()
except ImportError:
    print("BDDL could not be imported - object taxonomy / abilities will be unavailable.", file=sys.stderr)
    OBJECT_TAXONOMY = None


class StatefulObject(BaseObject):
    """Objects that support object states."""

    def __init__(
            self,
            prim_path,
            name=None,
            category="object",
            class_id=None,
            scale=None,
            rendering_params=None,
            visible=True,
            fixed_base=False,
            load_config=None,
            abilities=None,
    ):
        # Values that will be filled later
        self._states = None

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            scale=scale,
            rendering_params=rendering_params,
            visible=visible,
            fixed_base=fixed_base,
            load_config=load_config,
        )

        # Load abilities from taxonomy if needed & possible
        if abilities is None:
            if OBJECT_TAXONOMY is not None:
                taxonomy_class = OBJECT_TAXONOMY.get_class_name_from_igibson_category(self.category)
                if taxonomy_class is not None:
                    abilities = OBJECT_TAXONOMY.get_abilities(taxonomy_class)
                else:
                    abilities = {}
            else:
                abilities = {}
        assert isinstance(abilities, dict), "Object abilities must be in dictionary form."

        self.prepare_object_states(abilities=abilities)

    def _post_load(self, simulator=None):
        # Run super method first
        super()._post_load(simulator=simulator)

        # Initialize any states created
        for state in self._states.values():
            state.initialize(simulator)

    def initialize_states(self):
        """
        Initializes states for this object, and also clears any states that were existing beforehand.
        """
        self._states = dict()

    def add_state(self, name, state):
        """
        Adds state @state with name @name to self.states.

        Args:
            name (str): name of the state to add to this object.
            state (any): state to add
        """

    @property
    def states(self):
        """
        Get the current states of this object.

        Returns:
            dict: Keyword-mapped states for this object
        """
        return self._states

    def prepare_object_states(self, abilities=None):
        """
        Prepare the state dictionary for an object by generating the appropriate
        object state instances.

        This uses the abilities of the object and the state dependency graph to
        find & instantiate all relevant states.

        :param abilities: dict in the form of {ability: {param: value}} containing
            object abilities and parameters.
        """
        if abilities is None:
            abilities = {}

        state_types_and_params = [(state, {}) for state in get_default_states()]

        # Map the ability params to the states immediately imported by the abilities
        for ability, params in abilities.items():
            state_types_and_params.extend((state_name, params) for state_name in get_states_for_ability(ability))

        # Add the dependencies into the list, too.
        for state_type, _ in state_types_and_params:
            # Add each state's dependencies, too. Note that only required dependencies are added.
            for dependency in state_type.get_dependencies():
                if all(other_state != dependency for other_state, _ in state_types_and_params):
                    state_types_and_params.append((dependency, {}))

        # Now generate the states in topological order.
        self.initialize_states()
        for state_type, params in reversed(state_types_and_params):
            self._states[state_type] = get_object_state_instance(state_type, self, params)

    def dump_state(self):
        return {
            get_state_name(state_type): state_instance.dump()
            for state_type, state_instance in self._states.items()
            if issubclass(state_type, AbsoluteObjectState)
        }

    def load_state(self, dump):
        for state_type, state_instance in self._states.items():
            state_name = get_state_name(state_type)
            if issubclass(state_type, AbsoluteObjectState):
                if state_name in dump:
                    state_instance.load(dump[state_name])
                else:
                    logging.warning("Missing object state [{}] in the state dump".format(state_name))

    def clear_cached_states(self):
        for _, obj_state in self._states.items():
            if isinstance(obj_state, CachingEnabledObjectState):
                obj_state.clear_cached_value()

    def set_position_orientation(self, position=None, orientation=None):
        super().set_position_orientation(position=position, orientation=orientation)
        self.clear_cached_states()

    # TODO: Redundant?
    def set_base_link_position_orientation(self, position, orientation):
        super().set_position_orientation(position=position, orientation=orientation)
        self.clear_cached_states()
