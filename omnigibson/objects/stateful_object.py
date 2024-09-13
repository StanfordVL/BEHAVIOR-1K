import sys
from collections import defaultdict
from typing import Literal

import torch as th
from bddl.object_taxonomy import ObjectTaxonomy

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros, gm
from omnigibson.object_states import Saturated
from omnigibson.object_states.factory import (
    get_default_states,
    get_fire_states,
    get_requirements_for_ability,
    get_state_name,
    get_states_by_dependency_order,
    get_states_for_ability,
    get_steam_states,
    get_texture_change_priority,
    get_texture_change_states,
    get_visual_states,
)
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.object_state_base import REGISTERED_OBJECT_STATES
from omnigibson.object_states.on_fire import OnFire
from omnigibson.object_states.particle_modifier import ParticleRemover
from omnigibson.objects.object_base import BaseObject
from omnigibson.renderer_settings.renderer_settings import RendererSettings
from omnigibson.utils.constants import EmitterType, PrimType
from omnigibson.utils.python_utils import classproperty, extract_class_init_kwargs_from_dict
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

OBJECT_TAXONOMY = ObjectTaxonomy()

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.STEAM_EMITTER_SIZE_RATIO = [0.8, 0.8, 0.4]  # (x,y,z) scale of generated steam relative to its object, range [0, inf)
m.STEAM_EMITTER_DENSITY_CELL_RATIO = 0.1  # scale of steam density relative to its object, range [0, inf)
m.STEAM_EMITTER_HEIGHT_RATIO = 0.6  # z-height of generated steam relative to its object's native height, range [0, inf)
m.FIRE_EMITTER_HEIGHT_RATIO = 0.4  # z-height of generated fire relative to its object's native height, range [0, inf)


class FlowEmitterLayerRegistry:
    """
    Registry for flow emitter layers. This is used to ensure that all flow emitters are placed on unique layers, so that
    they do not interfere with each other.
    """

    def __init__(self):
        self._layer = 0

    def __call__(self):
        self._layer += 1
        return self._layer


LAYER_REGISTRY = FlowEmitterLayerRegistry()


class StatefulObject(BaseObject):
    """Objects that support object states."""

    def __init__(
        self,
        name,
        relative_prim_path=None,
        category="object",
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        kinematic_only=None,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        load_config=None,
        abilities=None,
        include_default_states=True,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (None or str): The path relative to its scene prim for this object. If not specified, it defaults to /<name>.
            category (str): Category for the object. Defaults to "object".
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            kinematic_only (None or bool): Whether this object should be kinematic only (and not get affected by any
                collisions). If None, then this value will be set to True if @fixed_base is True and some other criteria
                are satisfied (see object_base.py post_load function), else False.
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
        self._emitters = dict()
        self._visual_states = None
        self._current_texture_state = None
        self._include_default_states = include_default_states

        # Load abilities from taxonomy if needed & possible
        if abilities is None:
            abilities = {}
            taxonomy_class = OBJECT_TAXONOMY.get_synset_from_category(category)
            if taxonomy_class is not None:
                abilities = OBJECT_TAXONOMY.get_abilities(taxonomy_class)
        assert isinstance(abilities, dict), "Object abilities must be in dictionary form."
        self._abilities = abilities

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            category=category,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            kinematic_only=kinematic_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            load_config=load_config,
            **kwargs,
        )

    def _post_load(self):
        # Run super first
        super()._post_load()

        # Prepare the object states
        self._states = {}
        self.prepare_object_states()

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Initialize all states
        for state in self._states.values():
            state.initialize()

        # Check whether this object requires any visual updates
        states_set = set(self.states)
        self._visual_states = states_set & get_visual_states()

        # If we require visual updates, possibly create additional APIs
        if len(self._visual_states) > 0:
            if len(states_set & get_steam_states()) > 0:
                self._create_emitter_apis(EmitterType.STEAM)

            if len(states_set & get_fire_states()) > 0:
                self._create_emitter_apis(EmitterType.FIRE)

    def add_state(self, state):
        """
        Adds state @state with name @name to self.states.

        Args:
            state (ObjectStateBase): Object state instance to add to this object
        """
        assert self._states is not None, "Cannot add state since states have not been initialized yet!"
        assert state.__class__ not in self._states, (
            f"State {state.__class__.__name__} " f"has already been added to this object!"
        )
        self._states[state.__class__] = state

    @property
    def states(self):
        """
        Get the current states of this object.

        Returns:
            dict: Keyword-mapped states for this object
        """
        return self._states

    @property
    def abilities(self):
        """
        Returns:
            dict: Dictionary mapping ability name to ability arguments for this object
        """
        return self._abilities

    @property
    def is_active(self):
        """
        Returns:
            bool: True if this object is currently considered active -- e.g.: if this object is currently awake
        """
        return super().is_active or self in self.scene.updated_state_objects

    def state_updated(self):
        """
        Adds this object to this object's scene's updated_state_objects set -- generally called externally
        by owned object state instances when its state is updated. This is useful for tracking when this object
        has had its state updated within the last simulation step
        """
        self.scene.updated_state_objects.add(self)

    def prepare_object_states(self):
        """
        Prepare the state dictionary for an object by generating the appropriate
        object state instances.

        This uses the abilities of the object and the state dependency graph to
        find & instantiate all relevant states.
        """
        states_info = (
            {state_type: {"ability": None, "params": dict()} for state_type in get_default_states()}
            if self._include_default_states
            else dict()
        )

        # Map the state type (class) to ability name and params
        if gm.ENABLE_OBJECT_STATES:
            for ability in tuple(self._abilities.keys()):
                # First, sanity check all ability requirements
                compatible = True
                for requirement in get_requirements_for_ability(ability):
                    compatible, reason = requirement.is_compatible(obj=self)
                    if not compatible:
                        # Print out warning and pop ability
                        log.warning(
                            f"Ability '{ability}' is incompatible with obj {self.name}, "
                            f"because requirement {requirement.__name__} was not met. Reason: {reason}"
                        )
                        self._abilities.pop(ability)
                        break
                if compatible:
                    params = self._abilities[ability]
                    for state_type in get_states_for_ability(ability):
                        states_info[state_type] = {
                            "ability": ability,
                            "params": state_type.postprocess_ability_params(params, self.scene),
                        }

        # Add the dependencies into the list, too, and sort based on the dependency chain
        # Must iterate over explicit tuple since dictionary changes size mid-iteration
        for state_type in tuple(states_info.keys()):
            # Add each state's dependencies, too. Note that only required dependencies are explicitly added, but both
            # required AND optional dependencies are checked / sorted
            for dependency in state_type.get_dependencies():
                if dependency not in states_info:
                    states_info[dependency] = {"ability": None, "params": dict()}

        # Iterate over all sorted state types, generating the states in topological order.
        self._states = dict()
        for state_type in get_states_by_dependency_order(states=states_info):
            # Skip over any types that are not in our info dict -- these correspond to optional dependencies
            if state_type not in states_info:
                continue

            relevant_params = extract_class_init_kwargs_from_dict(
                cls=state_type, dic=states_info[state_type]["params"], copy=False
            )
            compatible, reason = state_type.is_compatible(obj=self, **relevant_params)
            if compatible:
                self._states[state_type] = state_type(obj=self, **relevant_params)
            else:
                log.warning(f"State {state_type.__name__} is incompatible with obj {self.name}. Reason: {reason}")
                # Remove the ability if it exists
                # Note that the object may still have some of the states related to the desired ability. In this way,
                # we guarantee that the existence of a certain ability in self.abilities means at ALL corresponding
                # object state dependencies are met by the underlying object asset
                ability = states_info[state_type]["ability"]
                if ability in self._abilities:
                    self._abilities.pop(ability)

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
        bbox_extent_local = self.native_bbox if hasattr(self, "native_bbox") else self.aabb_extent / self.scale
        if emitter_type == EmitterType.FIRE:
            fire_at_metalink = True
            if OnFire in self.states:
                # Note whether the heat source link is explicitly set
                link = self.states[OnFire].link
                fire_at_metalink = link != self.root_link
            elif HeatSourceOrSink in self.states:
                # Only apply fire to non-root-link (i.e.: explicitly specified) heat source links
                # Otherwise, immediately return
                link = self.states[HeatSourceOrSink].link
                if link == self.root_link:
                    return
            else:
                raise ValueError("Unknown fire state")

            emitter_config["name"] = "flowEmitterSphere"
            emitter_config["type"] = "FlowEmitterSphere"
            emitter_config["position"] = (
                (0.0, 0.0, 0.0) if fire_at_metalink else (0.0, 0.0, bbox_extent_local[2] * m.FIRE_EMITTER_HEIGHT_RATIO)
            )
            emitter_config["fuel"] = 0.6
            emitter_config["coupleRateFuel"] = 1.2
            emitter_config["buoyancyPerTemp"] = 0.04
            emitter_config["burnPerTemp"] = 4
            emitter_config["gravity"] = (0, 0, -60.0)
            emitter_config["constantMask"] = 5.0
            emitter_config["attenuation"] = 0.5
        elif emitter_type == EmitterType.STEAM:
            link = self.root_link
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
        flowEmitter_prim_path = f"{link.prim_path}/{emitter_config['name']}"
        flowSimulate_prim_path = f"{link.prim_path}/flowSimulate"
        flowOffscreen_prim_path = f"{link.prim_path}/flowOffscreen"
        flowRender_prim_path = f"{link.prim_path}/flowRender"

        # Define prims.
        stage = og.sim.stage
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

        layer_number = LAYER_REGISTRY()

        # Update emitter general settings.
        emitter.CreateAttribute("enabled", lazy.pxr.Sdf.ValueTypeNames.Bool, False).Set(False)
        emitter.CreateAttribute("position", lazy.pxr.Sdf.ValueTypeNames.Float3, False).Set(emitter_config["position"])
        emitter.CreateAttribute("fuel", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(emitter_config["fuel"])
        emitter.CreateAttribute("coupleRateFuel", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(
            emitter_config["coupleRateFuel"]
        )
        emitter.CreateAttribute("coupleRateVelocity", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(2.0)
        emitter.CreateAttribute("velocity", lazy.pxr.Sdf.ValueTypeNames.Float3, False).Set((0, 0, 0))
        emitter.CreateAttribute("layer", lazy.pxr.Sdf.ValueTypeNames.Int, False).Set(layer_number)
        simulate.CreateAttribute("layer", lazy.pxr.Sdf.ValueTypeNames.Int, False).Set(layer_number)
        offscreen.CreateAttribute("layer", lazy.pxr.Sdf.ValueTypeNames.Int, False).Set(layer_number)
        renderer.CreateAttribute("layer", lazy.pxr.Sdf.ValueTypeNames.Int, False).Set(layer_number)
        advection.CreateAttribute("buoyancyPerTemp", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(
            emitter_config["buoyancyPerTemp"]
        )
        advection.CreateAttribute("burnPerTemp", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(
            emitter_config["burnPerTemp"]
        )
        advection.CreateAttribute("gravity", lazy.pxr.Sdf.ValueTypeNames.Float3, False).Set(emitter_config["gravity"])
        vorticity.CreateAttribute("constantMask", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(
            emitter_config["constantMask"]
        )
        rayMarch.CreateAttribute("attenuation", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(
            emitter_config["attenuation"]
        )

        # Update emitter unique settings.
        if emitter_type == EmitterType.FIRE:
            # Radius is in the absolute world coordinate even though the fire is under the link frame.
            # In other words, scaling the object doesn't change the fire radius.
            if fire_at_metalink:
                # TODO: get radius of heat_source_link from metadata.
                radius = 0.05
            else:
                bbox_extent_world = self.native_bbox * self.scale if hasattr(self, "native_bbox") else self.aabb_extent
                # Radius is the average x-y half-extent of the object
                radius = float(th.mean(bbox_extent_world[:2]) / 2.0)
            emitter.CreateAttribute("radius", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(radius)
            simulate.CreateAttribute("densityCellSize", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(radius * 0.2)
            smoke.CreateAttribute("fade", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(2.0)
            # Set fire colormap.
            rgbaPoints = []
            rgbaPoints.append(lazy.pxr.Gf.Vec4f(0.0154, 0.0177, 0.0154, 0.004902))
            rgbaPoints.append(lazy.pxr.Gf.Vec4f(0.03575, 0.03575, 0.03575, 0.504902))
            rgbaPoints.append(lazy.pxr.Gf.Vec4f(0.03575, 0.03575, 0.03575, 0.504902))
            rgbaPoints.append(lazy.pxr.Gf.Vec4f(1, 0.1594, 0.0134, 0.8))
            rgbaPoints.append(lazy.pxr.Gf.Vec4f(13.53, 2.99, 0.12599, 0.8))
            rgbaPoints.append(lazy.pxr.Gf.Vec4f(78, 39, 6.1, 0.7))
            colormap.CreateAttribute("rgbaPoints", lazy.pxr.Sdf.ValueTypeNames.Float4Array, False).Set(rgbaPoints)
        elif emitter_type == EmitterType.STEAM:
            emitter.CreateAttribute("halfSize", lazy.pxr.Sdf.ValueTypeNames.Float3, False).Set(
                tuple(bbox_extent_local * th.tensor(m.STEAM_EMITTER_SIZE_RATIO) / 2.0)
            )
            simulate.CreateAttribute("densityCellSize", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(
                bbox_extent_local[2].item() * m.STEAM_EMITTER_DENSITY_CELL_RATIO
            )

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
        if len(self._visual_states) > 0:
            texture_change_states = []
            emitter_enabled = defaultdict(bool)
            for state_type in self._visual_states:
                state = self.states[state_type]
                if state_type in get_texture_change_states():
                    if state_type == Saturated:
                        for particle_system in self.scene.active_systems.values():
                            if state.get_value(particle_system):
                                texture_change_states.append(state)
                                # Only need to do this once, since soaked handles all fluid systems
                                break
                    elif state.get_value():
                        texture_change_states.append(state)
                if state_type in get_steam_states():
                    emitter_enabled[EmitterType.STEAM] |= state.get_value()
                if state_type in get_fire_states():
                    emitter_enabled[EmitterType.FIRE] |= state.get_value()

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
            object_state (BooleanStateMixin or None): the object state that the diffuse color should match to
        """
        for material in self.materials:
            self._update_albedo_value(object_state, material)

    @staticmethod
    def _update_albedo_value(object_state, material):
        """
        Update the albedo value based on the given object_state. The final albedo value is
        albedo_value = diffuse_tint * (albedo_value + albedo_add)

        Args:
            object_state (BooleanStateMixin or None): the object state that the diffuse color should match to
            material (MaterialPrim): the material to use to update the albedo value
        """
        if object_state is None:
            # This restore the albedo map to its original value
            albedo_add = 0.0
            diffuse_tint = th.tensor([1.0, 1.0, 1.0])
        else:
            # Query the object state for the parameters
            albedo_add, diffuse_tint = object_state.get_texture_change_params()

        if material.is_glass:
            if not th.allclose(material.glass_color, diffuse_tint):
                material.glass_color = diffuse_tint

        else:
            if material.albedo_add != albedo_add:
                material.albedo_add = albedo_add

            if not th.allclose(material.diffuse_tint, diffuse_tint):
                material.diffuse_tint = diffuse_tint

    def remove(self):
        # Run super
        super().remove()

        # Iterate over all states and run their remove call
        for state_instance in self._states.values():
            state_instance.remove()

    def _dump_state(self):
        # Grab state from super class
        state = super()._dump_state()

        # Also add non-kinematic states
        non_kin_states = dict()
        for state_type, state_instance in self._states.items():
            if state_instance.stateful:
                non_kin_states[get_state_name(state_type)] = state_instance.dump_state(serialized=False)

        state["non_kin"] = non_kin_states

        return state

    def _load_state(self, state):
        # Call super method first
        super()._load_state(state=state)

        # Load non-kinematic states
        self.load_non_kin_state(state)

    def load_non_kin_state(self, state):
        # Load all states that are stateful
        for state_type, state_instance in self._states.items():
            state_name = get_state_name(state_type)
            if state_instance.stateful:
                if state_name in state["non_kin"]:
                    state_instance.load_state(state=state["non_kin"][state_name], serialized=False)
                else:
                    log.debug(f"Missing object state [{state_name}] in the state dump for obj {self.name}")

        # Clear cache after loading state
        self.clear_states_cache()

    def serialize(self, state):
        # Call super method first
        state_flat = super().serialize(state=state)

        # Iterate over all states and serialize them individually
        non_kin_state_flat = (
            th.cat(
                [
                    self._states[REGISTERED_OBJECT_STATES[state_name]].serialize(state_dict)
                    for state_name, state_dict in state["non_kin"].items()
                ]
            )
            if len(state["non_kin"]) > 0
            else th.empty(0)
        )

        # Combine these two arrays
        return th.cat([state_flat, non_kin_state_flat])

    def deserialize(self, state):
        # Call super method first
        state_dic, idx = super().deserialize(state=state)

        # Iterate over all states and deserialize their states if they're stateful
        non_kin_state_dic = dict()
        for state_type, state_instance in self._states.items():
            state_name = get_state_name(state_type)
            if state_instance.stateful:
                non_kin_state_dic[state_name], deserialized_items = state_instance.deserialize(state[idx:])
                idx += deserialized_items
        state_dic["non_kin"] = non_kin_state_dic

        return state_dic, idx

    def clear_states_cache(self):
        """
        Clears the internal cache from all owned states
        """
        # Check self._states just in case states have not been initialized yet.
        if not self._states:
            return
        for _, obj_state in self._states.items():
            obj_state.clear_cache()

    def set_position_orientation(
        self, position=None, orientation=None, frame: Literal["world", "parent", "scene"] = "world"
    ):
        """
        Set the position and orientation of stateful object.

        Args:
            position (None or 3-array): The position to set the object to. If None, the position is not changed.
            orientation (None or 4-array): The orientation to set the object to. If None, the orientation is not changed.
            frame (Literal): The frame in which to set the position and orientation. Defaults to world. parent frame
            set position relative to the object parent. scene frame set position relative to the scene.
        """
        super().set_position_orientation(position=position, orientation=orientation, frame=frame)
        self.clear_states_cache()

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("StatefulObject")
        return classes
