import sys
import logging
import numpy as np
from collections import OrderedDict
from igibson.object_states.factory import (
    get_state_name,
    get_default_states,
    get_states_for_ability,
    get_object_state_instance,
)
from igibson.object_states.object_state_base import CachingEnabledObjectState, REGISTERED_OBJECT_STATES
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
        visual_only=False,
        self_collisions=False,
        load_config=None,
        abilities=None,
        **kwargs,
    ):
        """
        @param prim_path: str, global path in the stage to this object
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
        self_collisions (bool): Whether to enable self collisions for this object
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        @param abilities: dict in the form of {ability: {param: value}} containing
            object abilities and parameters.
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
            Note that this base object does NOT pass kwargs down into the Prim-type super() classes, and we assume
            that kwargs are only shared between all SUBclasses (children), not SUPERclasses (parents).
        """
        # Values that will be filled later
        self._states = None

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

        self._abilities = abilities
        self.prepare_object_states(abilities=abilities)

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
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            **kwargs,
        )

    def load(self, simulator=None):
        # Run super first
        prim = super().load(simulator=simulator)

        # Make sure the simulator is not None
        assert simulator is not None, "Simulator must be specified when loading StatefulObject!"

        # Store temporary simulator reference just so we can load states later
        # We use this very opaque method to generate the attribute to denote that this should NOT
        # be referenced like a normal variable
        setattr(self, "_tmp_sim", simulator)

        return prim

    def _post_load(self):
        # Run super method first
        super()._post_load()

        # Initialize any states created
        tmp_sim = getattr(self, "_tmp_sim")
        for state in self._states.values():
            state.initialize(tmp_sim)

        # Delete the temporary simulator reference
        delattr(self, "_tmp_sim")

    def initialize_states(self):
        """
        Initializes states for this object, and also clears any states that were existing beforehand.
        """
        self._states = OrderedDict()

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
            OrderedDict: Keyword-mapped states for this object
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

    def _dump_state(self):
        # Grab state from super class
        state = super()._dump_state()

        # Also add non-kinematic states
        non_kin_states = OrderedDict()
        for state_type, state_instance in self._states.items():
            if state_instance.settable:
                non_kin_states[get_state_name(state_type)] = state_instance.dump_state(serialized=False)

        state["non_kin"] = non_kin_states

        return state

    def _load_state(self, state):
        # Call super method first
        super()._load_state(state=state)

        # Load all states that are settable
        for state_type, state_instance in self._states.items():
            state_name = get_state_name(state_type)
            if state_instance.settable:
                if state_name in state["non_kin"]:
                    state_instance.load_state(state=state["non_kin"][state_name], serialized=False)
                else:
                    logging.warning("Missing object state [{}] in the state dump".format(state_name))

    def _serialize(self, state):
        # Call super method first
        state_flat = super()._serialize(state=state)

        # Iterate over all states and serialize them individually
        non_kin_state_flat = (
            np.concatenate(
                [
                    self._states[REGISTERED_OBJECT_STATES[state_name]].serialize(state_dict)
                    for state_name, state_dict in state["non_kin"].items()
                ]
            )
            if len(state["non_kin"]) > 0
            else np.array([])
        )

        # Combine these two arrays
        return np.concatenate([state_flat, non_kin_state_flat])

    def _deserialize(self, state):
        # TODO: Need to check that self._state_size is accurate at the end of initialize()
        # Call super method first
        state_dic, idx = super()._deserialize(state=state)

        # Iterate over all states and deserialize their states if they're settable
        non_kin_state_dic = OrderedDict()
        for state_type, state_instance in self._states.items():
            state_name = get_state_name(state_type)
            if state_instance.settable:
                non_kin_state_dic[state_name] = state_instance.deserialize(state[idx : idx + state_instance.state_size])
                idx += state_instance.state_size
        state_dic["non_kin"] = non_kin_state_dic

        return state_dic, idx

    def clear_cached_states(self):
        if not self._states:
            return
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
