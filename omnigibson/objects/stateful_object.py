import logging
import sys
from collections import OrderedDict, defaultdict

import numpy as np
from pxr.Sdf import ValueTypeNames as VT
from pxr import Sdf, Gf

from omnigibson.macros import create_module_macros
from omnigibson.object_states.factory import (
    get_default_states,
    get_object_state_instance,
    get_state_name,
    get_states_for_ability,
    get_texture_change_states,
    get_fire_states,
    get_steam_states,
    get_texture_change_priority,
)
from omnigibson.object_states.object_state_base import REGISTERED_OBJECT_STATES
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.objects.object_base import BaseObject
from omnigibson.renderer_settings.renderer_settings import RendererSettings
from omnigibson.systems.micro_particle_system import get_fluid_systems
from omnigibson.utils.constants import PrimType, EmitterType
from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.utils.python_utils import classproperty
from omnigibson.object_states import Saturated


# Optionally import bddl for object taxonomy.
try:
    from bddl.object_taxonomy import ObjectTaxonomy

    OBJECT_TAXONOMY = ObjectTaxonomy()
except ImportError:
    print("BDDL could not be imported - object taxonomy / abilities will be unavailable.", file=sys.stderr)
    OBJECT_TAXONOMY = None


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.STEAM_EMITTER_SIZE_RATIO = [0.8, 0.8, 0.4]    # (x,y,z) scale of generated steam relative to its object, range [0, inf)
m.STEAM_EMITTER_DENSITY_CELL_RATIO = 0.1        # scale of steam density relative to its object, range [0, inf)
m.STEAM_EMITTER_HEIGHT_RATIO = 0.6              # z-height of generated steam relative to its object's native height, range [0, inf)


class StatefulObject(BaseObject):
    """Objects that support object states."""

    def __init__(
            self,
            prim_path,
            name=None,
            category="object",
            class_id=None,
            uuid=None,
            scale=None,
            visible=True,
            fixed_base=False,
            visual_only=False,
            self_collisions=False,
            prim_type=PrimType.RIGID,
            load_config=None,
            abilities=None,
            include_default_states=True,
            **kwargs,
    ):
        """
        Args:
            prim_path (str): global path in the stage to this object
            name (None or str): Name for the object. Names need to be unique per scene. If None, a name will be
                generated at the time the object is added to the scene, using the object's category.
            category (str): Category for the object. Defaults to "object".
            class_id (None or int): What class ID the object should be assigned in semantic segmentation rendering mode.
                If None, the ID will be inferred from this object's category.
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            include_default_states (bool): whether to include the default object states from @get_default_states
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Values that will be filled later
        self._states = None
        self._emitters = OrderedDict()
        self._current_texture_state = None

        # Load abilities from taxonomy if needed & possible
        if abilities is None:
            abilities = {}
            if OBJECT_TAXONOMY is not None:
                # TODO! Update!!
                taxonomy_class = OBJECT_TAXONOMY.get_class_name_from_igibson_category(category)
                if taxonomy_class is not None:
                    abilities = OBJECT_TAXONOMY.get_abilities(taxonomy_class)
        assert isinstance(abilities, dict), "Object abilities must be in dictionary form."

        self._abilities = abilities
        self.prepare_object_states(abilities=abilities, include_default_states=include_default_states)

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            load_config=load_config,
            **kwargs,
        )

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Initialize all states
        for state in self._states.values():
            state.initialize(self._simulator)

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

    def prepare_object_states(self, abilities=None, include_default_states=True):
        """
        Prepare the state dictionary for an object by generating the appropriate
        object state instances.

        This uses the abilities of the object and the state dependency graph to
        find & instantiate all relevant states.

        Args:
            abilities (None or dict): If specified, dict in the form of {ability: {param: value}} containing
                object abilities and parameters.
            include_default_states (bool): whether to include the default object states from @get_default_states
        """
        if abilities is None:
            abilities = {}

        state_types_and_params = [(state, {}) for state in get_default_states()] if include_default_states else []

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
        self._states = OrderedDict()
        for state_type, params in reversed(state_types_and_params):
            self._states[state_type] = get_object_state_instance(state_type, self, params)

    def _post_load(self):
        super()._post_load()

        if len(set(self.states) & set(get_steam_states())) > 0:
            self._create_emitter_apis(EmitterType.STEAM)

        if len(set(self.states) & set(get_fire_states())) > 0 and self.states[HeatSourceOrSink].get_state_link_name() in self._links:
            self._create_emitter_apis(EmitterType.FIRE)

    def _create_emitter_apis(self, emitter_type):
        """
        Create necessary prims and apis for steam effects.

        Args:
            emitter_type (EmitterType): Emitter to create
        """
        # Make sure that flow setting is enabled.
        renderer_setting = RendererSettings()
        renderer_setting.common_settings.flow_settings.enable()

        # Specify emitter config.
        emitter_config = {}
        link_name = self.root_link_name
        if emitter_type == EmitterType.FIRE:
            link_name = self.states[HeatSourceOrSink].get_state_link_name()
            emitter_config["name"] = "flowEmitterSphere"
            emitter_config["type"] = "FlowEmitterSphere"
            emitter_config["position"] = (0.0, 0.0, 0.0)
            emitter_config["fuel"] = 0.6
            emitter_config["coupleRateFuel"] = 1.2
            emitter_config["buoyancyPerTemp"] = 0.04
            emitter_config["burnPerTemp"] = 4
            emitter_config["gravity"] = (0, 0, -60.0)
            emitter_config["constantMask"] = 5.0
            emitter_config["attenuation"] = 0.5
        elif emitter_type == EmitterType.STEAM:
            bbox_extent_local = self.native_bbox if hasattr(self, "native_bbox") else self.aabb_extent / self.scale
            emitter_config["name"] = "flowEmitterBox"
            emitter_config["type"] = "FlowEmitterBox"
            emitter_config["position"] = (0.0, 0.0, bbox_extent_local[2] * m.STEAM_EMITTER_HEIGHT_RATIO)
            emitter_config["fuel"] = 1.0
            emitter_config["coupleRateFuel"] = 0.5
            emitter_config["buoyancyPerTemp"] = 0.05
            emitter_config["burnPerTemp"] = 0.5
            emitter_config["gravity"] = (0, 0, -50.0)
            emitter_config["constantMask"] = 10.0
            emitter_config["attenuation"] = 1.5
        else:
            raise ValueError("Currently, only EmitterTypes FIRE and STEAM are supported!")

        # Define prim paths.
        # The flow system is created under the root link so that it automatically updates its pose as the object moves
        flowEmitter_prim_path = f"{self._prim_path}/{link_name}/{emitter_config['name']}"
        flowSimulate_prim_path = f"{self._prim_path}/{link_name}/flowSimulate"
        flowOffscreen_prim_path = f"{self._prim_path}/{link_name}/flowOffscreen"
        flowRender_prim_path = f"{self._prim_path}/{link_name}/flowRender"

        # Define prims.
        stage = self._simulator.stage
        emitter = stage.DefinePrim(flowEmitter_prim_path, emitter_config["type"])
        simulate = stage.DefinePrim(flowSimulate_prim_path, "FlowSimulate")
        offscreen = stage.DefinePrim(flowOffscreen_prim_path, "FlowOffscreen")
        renderer = stage.DefinePrim(flowRender_prim_path, "FlowRender")
        advection = stage.DefinePrim(flowSimulate_prim_path + "/advection", "FlowAdvectionCombustionParams")
        smoke = stage.DefinePrim(flowSimulate_prim_path + "/advection/smoke", "FlowAdvectionCombustionParams")
        vorticity = stage.DefinePrim(flowSimulate_prim_path + "/vorticity", "FlowVorticityParams")
        rayMarch = stage.DefinePrim(flowRender_prim_path + "/rayMarch", "FlowRayMarchParams")
        colormap = stage.DefinePrim(flowOffscreen_prim_path + "/colormap", "FlowRayMarchColormapParams")

        self._emitters[emitter_type] = emitter

        # Update emitter general settings.
        emitter.CreateAttribute("enabled", VT.Bool, False).Set(False)
        emitter.CreateAttribute("position", VT.Float3, False).Set(emitter_config["position"])
        emitter.CreateAttribute("fuel", VT.Float, False).Set(emitter_config["fuel"])
        emitter.CreateAttribute("coupleRateFuel", VT.Float, False).Set(emitter_config["coupleRateFuel"])
        emitter.CreateAttribute("coupleRateVelocity", VT.Float, False).Set(2.0)
        emitter.CreateAttribute("velocity", VT.Float3, False).Set((0, 0, 0))
        advection.CreateAttribute("buoyancyPerTemp", VT.Float, False).Set(emitter_config["buoyancyPerTemp"])
        advection.CreateAttribute("burnPerTemp", VT.Float, False).Set(emitter_config["burnPerTemp"])
        advection.CreateAttribute("gravity", VT.Float3, False).Set(emitter_config["gravity"])
        vorticity.CreateAttribute("constantMask", VT.Float, False).Set(emitter_config["constantMask"])
        rayMarch.CreateAttribute("attenuation", VT.Float, False).Set(emitter_config["attenuation"])

        # Update emitter unique settings.
        if emitter_type == EmitterType.FIRE:
            # TODO: get radius of heat_source_link from metadata.
            radius = 0.05
            emitter.CreateAttribute("radius", VT.Float, False).Set(radius)
            simulate.CreateAttribute("densityCellSize", VT.Float, False).Set(radius*0.2)
            smoke.CreateAttribute("fade", Sdf.ValueTypeNames.Float, False).Set(2.0)
            # Set fire colormap.
            rgbaPoints = []
            rgbaPoints.append(Gf.Vec4f(0.0154, 0.0177, 0.0154, 0.004902))
            rgbaPoints.append(Gf.Vec4f(0.03575, 0.03575, 0.03575, 0.504902))
            rgbaPoints.append(Gf.Vec4f(0.03575, 0.03575, 0.03575, 0.504902))
            rgbaPoints.append(Gf.Vec4f(1, 0.1594, 0.0134, 0.8))
            rgbaPoints.append(Gf.Vec4f(13.53, 2.99, 0.12599, 0.8))
            rgbaPoints.append(Gf.Vec4f(78, 39, 6.1, 0.7))
            colormap.CreateAttribute("rgbaPoints", Sdf.ValueTypeNames.Float4Array, False).Set(rgbaPoints)
        elif emitter_type == EmitterType.STEAM:
            emitter.CreateAttribute("halfSize", VT.Float3, False).Set(
                tuple(bbox_extent_local * np.array(m.STEAM_EMITTER_SIZE_RATIO) / 2.0))
            simulate.CreateAttribute("densityCellSize", VT.Float, False).Set(bbox_extent_local[2] * m.STEAM_EMITTER_DENSITY_CELL_RATIO)

    def set_emitter_enabled(self, emitter_type, value):
        """
        Enable/disable the emitter prim for fire/steam effect.

        Args:
            emitter_type (EmitterType): Emitter to set
            value (bool): Value to set
        """
        if emitter_type not in self._emitters:
            return
        if value != self._emitters[emitter_type].GetAttribute("enabled").Get():
            self._emitters[emitter_type].GetAttribute("enabled").Set(value)

    def get_textures(self):
        """
        Gets prim's texture files.

        Returns:
            list of str: List of texture file paths
        """
        return [material.diffuse_texture for material in self.materials if material.diffuse_texture is not None]

    def update_visuals(self):
        """
        Update the prim's visuals (texture change, steam/fire effects, etc).
        Should be called after all the states are updated.
        """
        texture_change_states = []
        emitter_enabled = defaultdict(bool)
        for state_type, state in self.states.items():
            if state_type in get_texture_change_states():
                if state_type == Saturated:
                    for fluid_system in get_fluid_systems.values():
                        if state.get_value(fluid_system):
                            texture_change_states.append(state)
                            # Only need to do this once, since soaked handles all fluid systems
                            break
                elif state.get_value():
                    texture_change_states.append(state)
            if state_type in get_steam_states():
                emitter_enabled[EmitterType.STEAM] |= state.get_value()
            if state_type in get_fire_states():
                # Currently, the only state that uses fire is HeatSourceOrSink, whose get_value()
                # returns (heat_source_state, heat_source_position).
                emitter_enabled[EmitterType.FIRE] |= state.get_value()[0]

            for emitter_type in emitter_enabled:
                self.set_emitter_enabled(emitter_type, emitter_enabled[emitter_type])

        texture_change_states.sort(key=lambda s: get_texture_change_priority()[s.__class__])
        object_state = texture_change_states[-1] if len(texture_change_states) > 0 else None

        # Only update our texture change if it's a different object state than the one we already have
        if object_state != self._current_texture_state:
            self._update_texture_change(object_state)
            self._current_texture_state = object_state

    def _update_texture_change(self, object_state):
        """
        Update the texture based on the given object_state. E.g. if object_state is Frozen, update the diffuse color
        to match the frozen state. If object_state is None, update the diffuse color to the default value. It modifies
        the current albedo map by adding and scaling the values. See @self._update_albedo_value for details.

        Args:
            object_state (BooleanState or None): the object state that the diffuse color should match to
        """
        for material in self.materials:
            self._update_albedo_value(object_state, material)

    @staticmethod
    def _update_albedo_value(object_state, material):
        """
        Update the albedo value based on the given object_state. The final albedo value is
        albedo_value = diffuse_tint * (albedo_value + albedo_add)

        Args:
            object_state (BooleanState or None): the object state that the diffuse color should match to
            material (MaterialPrim): the material to use to update the albedo value
        """
        if object_state is None:
            # This restore the albedo map to its original value
            albedo_add = 0.0
            diffuse_tint = (1.0, 1.0, 1.0)
        else:
            # Query the object state for the parameters
            albedo_add, diffuse_tint = object_state.get_texture_change_params()

        if material.albedo_add != albedo_add:
            material.albedo_add = albedo_add

        if not np.allclose(material.diffuse_tint, diffuse_tint):
            material.diffuse_tint = diffuse_tint

    def remove(self, simulator=None):
        """
        Removes this prim from omniverse stage

        Args:
            simulator (None or SimulationContext): If specified, should be simulator into which this prim will be
                removed. Otherwise, it will be removed from the default stage
        """
        # Iterate over all states and run their remove call
        for state_instance in self._states.values():
            state_instance.remove()

        # Run super
        super().remove(simulator=simulator)

    def _dump_state(self):
        # Grab state from super class
        state = super()._dump_state()

        # Also add non-kinematic states
        non_kin_states = OrderedDict()
        for state_type, state_instance in self._states.items():
            if state_instance.stateful:
                non_kin_states[get_state_name(state_type)] = state_instance.dump_state(serialized=False)

        state["non_kin"] = non_kin_states

        return state

    def _load_state(self, state):
        # Call super method first
        super()._load_state(state=state)

        # Load all states that are stateful
        for state_type, state_instance in self._states.items():
            state_name = get_state_name(state_type)
            if state_instance.stateful:
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
        return np.concatenate([state_flat, non_kin_state_flat]).astype(float)

    def _deserialize(self, state):
        # Call super method first
        state_dic, idx = super()._deserialize(state=state)

        # Iterate over all states and deserialize their states if they're stateful
        non_kin_state_dic = OrderedDict()
        for state_type, state_instance in self._states.items():
            state_name = get_state_name(state_type)
            if state_instance.stateful:
                non_kin_state_dic[state_name] = state_instance.deserialize(state[idx:idx+state_instance.state_size])
                idx += state_instance.state_size
        state_dic["non_kin"] = non_kin_state_dic

        return state_dic, idx

    def clear_cached_states(self):
        """
        Clears the internal cache from all owned states
        """
        # Check self._states just in case states have not been initialized yet.
        if not self._states:
            return
        for _, obj_state in self._states.items():
            obj_state.clear_cache()
        BoundingBoxAPI.clear()

    def reset_states(self):
        """
        Resets all object states' internal values
        """
        # Check self._states just in case states have not been initialized yet.
        if not self._states:
            return
        for _, obj_state in self._states.items():
            obj_state.reset()

    def reset(self):
        # Call super first
        super().reset()

        # Reset all states
        self.reset_states()

    def set_position_orientation(self, position=None, orientation=None):
        super().set_position_orientation(position=position, orientation=orientation)
        self.clear_cached_states()

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("StatefulObject")
        return classes
