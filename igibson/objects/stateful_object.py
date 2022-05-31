import logging
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import omni
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.usd import get_shader_from_material
from pxr.Sdf import ValueTypeNames as VT

from igibson.object_states.factory import (
    get_default_states,
    get_object_state_instance,
    get_state_name,
    get_states_for_ability,
    get_steam_states,
)
from igibson.object_states.object_state_base import REGISTERED_OBJECT_STATES, CachingEnabledObjectState
from igibson.objects.object_base import BaseObject
from igibson.renderer_settings.renderer_settings import RendererSettings

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
        self._texture_states = []
        self._extra_prims_flags = defaultdict(bool)
        self._emitter = None

        # Load abilities from taxonomy if needed & possible
        if abilities is None:
            if OBJECT_TAXONOMY is not None:
                taxonomy_class = OBJECT_TAXONOMY.get_class_name_from_igibson_category(category)
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

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Initialize all states
        for state in self._states.values():
            state.initialize(self._simulator)
            
        # Create steam prims if this object has states with steam effects.
        if self._extra_prims_flags["has_steam_prims"]:
            self._create_steam_prims()

    def initialize_states(self):
        """
        Initializes states for this object, and also clears any states that were existing beforehand.
        """
        self._states = OrderedDict()

    def add_state(self, state):
        """
        Adds state @state with name @name to self.states.

        Args:
            state (ObjectStateBase): Object state instance to add to this object
        """
        assert self._states is not None, "Cannot add state since states have not been initialized yet!"
        assert state.__class__ not in self._states, f"State {state.__class__.__name__} " \
                                                    f"has already been added to this object!"
        self._states[state.__class__] = state

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
            if state_type in get_steam_states():
                self._extra_prims_flags["has_steam_prims"] = True
            self._states[state_type] = get_object_state_instance(state_type, self, params)

    def _create_steam_prims(self):
        """
        Create necessary prims for steam effects.
        """
        # Make sure that flow setting is enabled.
        renderer_setting = RendererSettings()
        renderer_setting.common_settings.flow_settings.enable()

        # Define prim paths.
        flowEmitterBox_prim_path = self._prim_path + "/flowEmitterBox"
        flowSimulate_prim_path = self._prim_path+ "/flowSimulate"
        flowOffscreen_prim_path = self._prim_path + "/flowOffscreen"
        flowRender_prim_path = self._prim_path+ "/flowRender"

        # Define prims.
        stage = omni.usd.get_context().get_stage()
        emitter = stage.DefinePrim(flowEmitterBox_prim_path, "FlowEmitterBox")
        simulate = stage.DefinePrim(flowSimulate_prim_path, "FlowSimulate")
        offscreen = stage.DefinePrim(flowOffscreen_prim_path, "FlowOffscreen")
        renderer = stage.DefinePrim(flowRender_prim_path, "FlowRender")
        advection = stage.DefinePrim(flowSimulate_prim_path + "/advection", "FlowAdvectionCombustionParams")
        vorticity = stage.DefinePrim(flowSimulate_prim_path + "/vorticity", "FlowVorticityParams")
        rayMarch = stage.DefinePrim(flowRender_prim_path + "/rayMarch", "FlowRayMarchParams")

        self._emitter = emitter

        # Update settings.
        bbox = self.bbox
        
        emitter.CreateAttribute("enabled", VT.Bool, False).Set(False)
        emitter.CreateAttribute("fuel", VT.Float, False).Set(1.0)
        emitter.CreateAttribute("coupleRateFuel", VT.Float, False).Set(0.5)
        emitter.CreateAttribute("coupleRateVelocity", VT.Float, False).Set(2.0)
        emitter.CreateAttribute("halfSize", VT.Float3, False).Set((bbox[0]*0.4, bbox[1]*0.4, bbox[2]*0.15))
        emitter.CreateAttribute("position", VT.Float3, False).Set((0, 0, bbox[2]*1.0))
        emitter.CreateAttribute("velocity", VT.Float3, False).Set((0, 0, 0))

        simulate.CreateAttribute("densityCellSize", VT.Float, False).Set(bbox[2]*0.1)

        vorticity.CreateAttribute("constantMask", VT.Float, False).Set(10.0)

        advection.CreateAttribute("buoyancyPerTemp", VT.Float, False).Set(0.05)
        advection.CreateAttribute("burnPerTemp", VT.Float, False).Set(0.5)
        advection.CreateAttribute("gravity", VT.Float3, False).Set((0, 0, -50.0))

        rayMarch.CreateAttribute("attenuation", VT.Float, False).Set(1.5)

    def set_emitter_enabled(self, value):
        """
        Enable/disable the emitter prim for steam effect.

        Args:
            value (bool): Value to set
        """
        self._emitter.CreateAttribute("enabled", VT.Bool, False).Set(value)

    def get_textures(self):
        """Gets prim's texture files.

        Returns:
            list of (str): List of texture file paths
        """
        textures = []
        looks_prim_path = f"{str(self._prim_path)}/Looks"
        looks_prim = get_prim_at_path(looks_prim_path)
        if not looks_prim:
            return
        for subprim in looks_prim.GetChildren():
            if subprim.GetPrimTypeInfo().GetTypeName() != "Material":
                continue
            shader = get_shader_from_material(subprim)
            texture_path = shader.GetInput("diffuse_texture").Get()
            if texture_path:
                textures.append(texture_path.path)
        return textures

    def update_textures_for_state(self, state, value):
        """Update prim's textures to represent @state.

        Args:
            state (BaseObjectState): State to represent
            value (bool): Whether or not changing to the state
        """
        # Determine the state to set.
        TEXTURE_CHANGE_PRIORITY = {
            "Frozen": 4,
            "Burnt": 3,
            "Cooked": 2,
            "Soaked": 1,
            "ToggledOn": 0,
        }
        if value and state.__name__ not in self._texture_states:
            self._texture_states.append(state.__name__)
            self._texture_states.sort(key=lambda s: TEXTURE_CHANGE_PRIORITY[s])
        if not value and state.__name__ in self._texture_states:
            self._texture_states.remove(state.__name__)
        
        state_to_set = self._texture_states[-1] if self._texture_states else None

        # Find the material prims to update.
        looks_prim_path = f"{str(self._prim_path)}/Looks"
        looks_prim = get_prim_at_path(looks_prim_path)
        if not looks_prim:
            return
        for subprim in looks_prim.GetChildren():
            if subprim.GetPrimTypeInfo().GetTypeName() != "Material":
                continue
            shader = get_shader_from_material(subprim)
            texture_path = shader.GetInput("diffuse_texture").Get()
            if texture_path is None:
                continue
            # Get updated texture file path for state.
            texture_path_split = texture_path.path.split("/")
            filedir, filename = "/".join(texture_path_split[:-1]), texture_path_split[-1]
            assert filename[-4:] == ".png", f"Texture file {filename} does not end with .png"
            filename_split = filename[:-4].split("_")
            # Check both file names for backward compatibility.
            if len(filename_split) > 0 and filename_split[-1] not in ("DIFFUSE", "albedo"):
                filename_split.pop()
            target_texture_path = f"{filedir}/{'_'.join(filename_split)}"
            target_texture_path += f"_{state_to_set}.png" if state_to_set else ".png"
            if not os.path.exists(target_texture_path):
                print(f"Warning: get texture path failed because {target_texture_path} does not exist")
                continue
            shader.GetInput("diffuse_texture").Set(target_texture_path)

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
        non_kin_state_flat = np.concatenate([
            self._states[REGISTERED_OBJECT_STATES[state_name]].serialize(state_dict)
            for state_name, state_dict in state["non_kin"].items()
        ]) if len(state["non_kin"]) > 0 else np.array([])

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
                non_kin_state_dic[state_name] = state_instance.deserialize(state[idx:idx+state_instance.state_size])
                idx += state_instance.state_size
        state_dic["non_kin"] = non_kin_state_dic

        return state_dic, idx

    def clear_cached_states(self):
        # Check self._states just in case states have not been initialized yet.
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
