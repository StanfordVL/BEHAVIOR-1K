import hashlib
import os
import tempfile
from abc import ABCMeta
from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property
from typing import Literal

import torch as th
from omnigibson.utils.bddl_utils import get_knowledge_base

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
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
from omnigibson.prims.entity_prim import EntityPrim
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.constants import EmitterType, PrimType
from omnigibson.utils.python_utils import Registerable, classproperty, extract_class_init_kwargs_from_dict, get_uuid
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import (
    absolute_prim_path_to_scene_relative,
    add_asset_to_stage,
    compute_kinematic_only,
    count_joints,
    create_joint,
)
from omnigibson.utils.vision_utils import add_semantic_label

# Global dicts that will contain mappings
REGISTERED_OBJECTS = dict()

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Settings for highlighting objects
m.HIGHLIGHT_RGB = [1.0, 0.1, 0.92]  # Default highlighting (R,G,B) color when highlighting objects
m.HIGHLIGHT_INTENSITY = 10000.0  # Highlight intensity to apply, range [0, 10000)

# Physics settings for objects -- see https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/RigidBodyDynamics.html?highlight=velocity%20iteration#solver-iterations
m.DEFAULT_SOLVER_POSITION_ITERATIONS = 32
m.DEFAULT_SOLVER_VELOCITY_ITERATIONS = 1

m.STEAM_EMITTER_SIZE_RATIO = [0.8, 0.8, 0.4]  # (x,y,z) scale of generated steam relative to its object, range [0, inf)
m.STEAM_EMITTER_DENSITY_CELL_RATIO = 0.1  # scale of steam density relative to its object, range [0, inf)
m.STEAM_EMITTER_HEIGHT_RATIO = 0.6  # z-height of generated steam relative to its object's native height, range [0, inf)
m.FIRE_EMITTER_HEIGHT_RATIO = 0.4  # z-height of generated fire relative to its object's native height, range [0, inf)


# Counter that assigns each flow emitter a unique layer number so emitters don't interfere.
_EMITTER_LAYER_COUNTER = 1


class USDObject(EntityPrim, Registerable, metaclass=ABCMeta):
    """
    USDObject is the interface that all OmniGibson objects must implement.
    Objects are instantiated from a USD file and can be composed of one or more links and joints.
    """

    def __init__(
        self,
        name,
        usd_path,
        encrypted=False,
        relative_prim_path=None,
        category="object",
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        kinematic_only=None,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        link_physics_materials=None,
        load_config=None,
        abilities=None,
        include_default_states=True,
        expected_file_hash=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            usd_path (str): global path to the USD file to load
            encrypted (bool): whether this file is encrypted (and should therefore be decrypted) or not
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
                are satisfied (see usd_object.py post_load function), else False.
            self_collisions (bool): Whether to enable self collisions for this object
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            include_default_states (bool): whether to include the default object states from @get_default_states
            expected_file_hash (str): The expected hash of the file to load. This is used to check if the file has changed. None to disable check.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject).
                Note that this base object does NOT pass kwargs down into the Prim-type super() classes, and we assume
                that kwargs are only shared between all SUBclasses (children), not SUPERclasses (parents).
        """
        self._usd_path = usd_path
        self._encrypted = encrypted
        self._expected_file_hash = expected_file_hash
        # Generate default prim path if none is specified
        relative_prim_path = f"/{name}" if relative_prim_path is None else relative_prim_path

        # Store values
        self._uuid = get_uuid(name, deterministic=True)
        self.category = category
        self.fixed_base = fixed_base
        self._link_physics_materials = dict() if link_physics_materials is None else link_physics_materials

        # Values to be created at runtime
        self._highlighted = False
        self._highlight_color = m.HIGHLIGHT_RGB
        self._highlight_intensity = m.HIGHLIGHT_INTENSITY

        # Object state values
        self._states = None
        self._emitters = dict()
        self._visual_states = None
        self._current_texture_state = None
        self._include_default_states = include_default_states

        # Load abilities from taxonomy if needed & possible
        # TODO: Move this to dataset object? Loads B1K abilities for non-B1K objects.
        if abilities is None:
            abilities = {}
            kb_category = get_knowledge_base().get_category(category)
            if kb_category is not None and kb_category.synset is not None:
                abilities = kb_category.synset.abilities
        assert isinstance(abilities, dict), "Object abilities must be in dictionary form."
        self._abilities = abilities

        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["scale"] = (
            scale
            if isinstance(scale, th.Tensor)
            else th.tensor(scale, dtype=th.float32)
            if isinstance(scale, Iterable)
            else scale
        )
        load_config["visible"] = visible
        load_config["visual_only"] = visual_only
        load_config["kinematic_only"] = kinematic_only
        load_config["self_collisions"] = self_collisions
        load_config["prim_type"] = prim_type

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

        # TODO: Super hacky, think of a better way to preserve this info
        # Update init info for this
        self._init_info["args"]["name"] = self.name

    def _prepare_to_load(self):
        """Prepare to load the USD by decrypting, correcting paths, checking hashes, and
        pre-applying ArticulationRootAPI at the correct location."""
        usd_path = self._usd_path

        if self._encrypted:
            # Create a temporary file to store the decrytped asset, load it, and then delete it
            encrypted_filename = self._usd_path.replace(".usd", ".encrypted.usd")
            self.check_hash(encrypted_filename)
            basename = os.path.basename(self._usd_path)
            tempdir_path = tempfile.mkdtemp(basename, dir=og.tempdir)
            usd_path = os.path.join(tempdir_path, f"{basename}.usd")
            decrypt_file(encrypted_filename, usd_path)

            # Update the paths of all assets to be the absolute path. This is important because the
            # relative paths are relative to the encrypted file and not the decrypted file in the
            # tempdir.
            side_stage = lazy.pxr.Usd.Stage.Open(usd_path)

            def _update_path(asset_path):
                if ".mdl" in asset_path:
                    # MDL paths are searched for in a different search space, so we don't modify them
                    return asset_path
                return os.path.join(os.path.dirname(encrypted_filename), asset_path)

            lazy.pxr.UsdUtils.ModifyAssetPaths(side_stage.GetRootLayer(), _update_path)
            side_stage.Save()
            del side_stage
        else:
            self.check_hash(usd_path)

        usd_path = self._preapply_articulation_root(usd_path)

        return usd_path

    def _get_preapply_scale(self, default_prim):
        """
        Returns the scale tensor to use when computing kinematic_only inside _preapply_articulation_root.
        Subclasses can override to derive scale from USD-side data (e.g. ig:nativeBB for dataset objects).

        Args:
            default_prim (Usd.Prim): The default prim of the side stage opened by _preapply_articulation_root.

        Returns:
            th.Tensor: 3-element float scale tensor.
        """
        raw_scale = self._load_config.get("scale", None)
        if raw_scale is not None:
            scale = raw_scale if isinstance(raw_scale, th.Tensor) else th.tensor(raw_scale, dtype=th.float32)
            if scale.dim() == 0:
                scale = scale.expand(3)
            return scale.float()
        return th.ones(3)

    def _preapply_articulation_root(self, usd_path):
        """
        Opens @usd_path with the pxr library, strips any existing ArticulationRootAPI, determines the correct prim to
        carry it, applies it there, and returns a path to the modified USD written to a temp file.
        """
        stage = lazy.pxr.Usd.Stage.Open(usd_path)
        default_prim = stage.GetDefaultPrim()

        for p in stage.Traverse():
            p.RemoveAPI(lazy.pxr.UsdPhysics.ArticulationRootAPI)
            p.RemoveAPI(lazy.pxr.PhysxSchema.PhysxArticulationAPI)

        n_joints, n_fixed_joints, has_attachment = count_joints(default_prim)

        scale = self._get_preapply_scale(default_prim)
        # Only persist scale to _load_config if the user already provided one, or if a non-trivial
        # scale was derived (e.g. from bounding_box).  Avoid overwriting None with the default
        # ones(3) so that PrimitiveObjects using radius/height/size are not affected.
        if self._load_config.get("scale", None) is not None or not th.allclose(scale, th.ones_like(scale)):
            self._load_config["scale"] = scale

        kinematic_only = compute_kinematic_only(
            self.fixed_base,
            scale,
            n_joints,
            n_fixed_joints,
            self._load_config.get("kinematic_only", None),
            has_attachment,
        )
        self._load_config["kinematic_only"] = kinematic_only

        # Find root link: the Xform child that is not body1 of any joint that also has a body0.
        joint_children = set()
        link_names = []
        for prim in default_prim.GetChildren():
            if prim.GetTypeName() != "Xform":
                continue
            link_names.append(prim.GetName())
            for child in prim.GetChildren():
                if "joint" not in child.GetTypeName().lower():
                    continue
                rels = {r.GetName(): r for r in child.GetRelationships()}
                body0_rel = rels.get("physics:body0")
                body1_rel = rels.get("physics:body1")
                if body0_rel is None or body1_rel is None:
                    continue
                if len(body0_rel.GetTargets()) > 0 and len(body1_rel.GetTargets()) > 0:
                    joint_children.add(body1_rel.GetTargets()[0].pathString.split("/")[-1])
        valid_roots = list(set(link_names) - joint_children)
        assert len(valid_roots) == 1, (
            f"Exactly one root link should have been found for {default_prim.GetName()}, "
            f"but found none/multiple instead: {valid_roots}"
        )
        root_link = default_prim.GetPrimAtPath(valid_roots[0])

        if self.fixed_base and not kinematic_only:
            create_joint(
                prim_path=f"{default_prim.GetPath()}/rootJoint",
                joint_type="FixedJoint",
                body1=f"{root_link.GetPath()}",
                stage=stage,
            )
            n_fixed_joints += 1

        # Determine which prim should carry ArticulationRootAPI
        articulation_root_prim = None
        if not kinematic_only and (n_joints > 0 or n_fixed_joints > 0):
            if not self.fixed_base and n_joints > 0:
                articulation_root_prim = root_link
            else:
                articulation_root_prim = default_prim

        if articulation_root_prim is not None:
            lazy.pxr.UsdPhysics.ArticulationRootAPI.Apply(articulation_root_prim)
            lazy.pxr.PhysxSchema.PhysxArticulationAPI.Apply(articulation_root_prim)
            articulation_root_prim.GetAttribute("physxArticulation:enabledSelfCollisions").Set(
                bool(self._load_config.get("self_collisions", False))
            )

        # Export to a temp file
        basename = os.path.basename(usd_path)
        tempdir_path = tempfile.mkdtemp(basename, dir=og.tempdir)
        temp_usd_path = os.path.join(tempdir_path, basename)
        stage.Export(temp_usd_path)
        return temp_usd_path

    def prebuild(self, stage):
        """
        Pre-build this object on an USD stage that is not loaded into Isaac Sim.
        This is useful for pre-compiling scene USDs, speeding up load times especially for parallel envs.
        """
        # The /World in the scene USD will be mapped to /World/scene_i in Isaac Sim.
        prim_path = "/World" + self._relative_prim_path
        usd_path = self._prepare_to_load()
        prim = stage.GetPrimAtPath(prim_path)
        assert not prim.IsValid(), f"Prim path {prim_path} already exists in the stage!"
        prim = stage.DefinePrim(prim_path, "Xform")
        assert prim.GetReferences().AddReference(usd_path)

    def _load(self):
        return add_asset_to_stage(asset_path=self._prepared_usd_path, prim_path=self.prim_path)

    def load(self, scene):
        # Always run _prepare_to_load (which calls _preapply_articulation_root) so that
        # _load_config["kinematic_only"] and _load_config["scale"] are set correctly before
        # _post_load runs, even when the prim already exists in the stage (e.g. from prebuild).
        self._prepared_usd_path = self._prepare_to_load()
        prim = super().load(scene)
        log.info(f"Loaded {self.name} at {self.prim_path}")
        return prim

    def remove(self):
        # Run super first
        super().remove()

        # Notify user that the object was removed
        log.info(f"Removed {self.name} from {self.prim_path}")

        # Iterate over all states and run their remove call
        for state_instance in self._states.values():
            state_instance.remove()

    def _post_load(self):
        # Run super first
        super()._post_load()

        # Set visibility
        if "visible" in self._load_config and self._load_config["visible"] is not None:
            self.visible = self._load_config["visible"]

        # Set position / velocity solver iterations if we're not cloth and not kinematic only
        if self._prim_type != PrimType.CLOTH and not self.kinematic_only:
            self.solver_position_iteration_count = m.DEFAULT_SOLVER_POSITION_ITERATIONS
            self.solver_velocity_iteration_count = m.DEFAULT_SOLVER_VELOCITY_ITERATIONS

        # Add link materials if specified
        if self._link_physics_materials is not None:
            for link_name, material_info in self._link_physics_materials.items():
                # We will permute the link materials dict in place to now point to the created material
                mat_name = f"{link_name}_physics_mat"
                physics_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
                    prim_path=f"{self.prim_path}/Looks/{mat_name}",
                    name=mat_name,
                    **material_info,
                )
                for msh in self.links[link_name].collision_meshes.values():
                    msh.apply_physics_material(physics_mat)
                self._link_physics_materials[link_name] = physics_mat

        # Add semantics
        add_semantic_label(prim=self._prim, label=self.category)

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
        assert (
            state.__class__ not in self._states
        ), f"State {state.__class__.__name__} has already been added to this object!"
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
    def usd_path(self):
        """
        Returns:
            str: absolute path to this model's USD file
        """
        return self._usd_path

    def check_hash(self, usd_path):
        """
        Check if the hash of the file matches the expected hash.

        Args:
            usd_path (str): The path to the USD file.
        """
        # Hash the file to record the loaded asset's version
        hash_md5 = hashlib.md5()
        with open(usd_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()

        # If there is a file hash already in the init info, compare against it to see if the file has changed
        if self._expected_file_hash is not None:
            if file_hash != self._expected_file_hash:
                log.warn(
                    f"Object {self.name} was expected to have USD file hash {self._expected_file_hash} but loaded with {file_hash}. The saved state might be incompatible."
                )
        else:
            # If there is no expected file hash, set the expected file hash to the loaded one
            self._expected_file_hash = file_hash

            # Update the init info too so that the information gets saved with the scene.
            # TODO: Super hacky, think of a better way to preserve this info
            self._init_info["args"]["expected_file_hash"] = file_hash

    @property
    def is_active(self):
        """
        Returns:
            bool: True if this object is currently considered active -- e.g.: if this object is currently awake
        """
        return not self.kinematic_only and not self.is_asleep or self in self.scene.updated_state_objects

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
                        log.debug(
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
                log.debug(f"State {state_type.__name__} is incompatible with obj {self.name}. Reason: {reason}")
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
        # Specify emitter config.
        emitter_config = {}
        bbox_extent_local = self.native_bbox if hasattr(self, "native_bbox") else self.aabb_extent / self.scale
        if emitter_type == EmitterType.FIRE:
            fire_at_meta_link = True
            if OnFire in self.states:
                # Note whether the heat source link is explicitly set
                link = self.states[OnFire].link
                fire_at_meta_link = link != self.root_link
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
                (0.0, 0.0, 0.0) if fire_at_meta_link else (0.0, 0.0, bbox_extent_local[2] * m.FIRE_EMITTER_HEIGHT_RATIO)
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
        # The flow system is created under the root link (under a dummy mesh) so that it automatically updates its pose
        # as the object moves. We put it under a dummy mesh so as not to force write synchronization to the actual
        # physx-tracked links (required when using Fabric), which causes physics issues
        dummy_mesh_path = f"{link.prim_path}/emitter"
        lazy.pxr.UsdGeom.Sphere.Define(og.sim.stage, dummy_mesh_path)
        relative_dummy_mesh_path = absolute_prim_path_to_scene_relative(self._scene, dummy_mesh_path)
        mesh = GeomPrim(relative_prim_path=relative_dummy_mesh_path, name=f"{self.name}_emitter")
        mesh.load(self._scene)
        mesh.visible = False

        flowEmitter_prim_path = f"{mesh.prim_path}/{emitter_config['name']}"
        flowSimulate_prim_path = f"{mesh.prim_path}/flowSimulate"
        flowOffscreen_prim_path = f"{mesh.prim_path}/flowOffscreen"
        flowRender_prim_path = f"{mesh.prim_path}/flowRender"

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

        self._emitters[emitter_type] = {
            "emitter": emitter,
            "mesh": mesh,
            "link": link,
            "canonical_pose": mesh.get_position_orientation(),
        }

        global _EMITTER_LAYER_COUNTER
        layer_number = _EMITTER_LAYER_COUNTER
        _EMITTER_LAYER_COUNTER += 1

        # Update emitter general settings.
        emitter.CreateAttribute("enabled", lazy.pxr.Sdf.ValueTypeNames.Bool, False).Set(False)
        emitter.CreateAttribute("position", lazy.pxr.Sdf.ValueTypeNames.Float3, False).Set(emitter_config["position"])
        emitter.CreateAttribute("fuel", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(emitter_config["fuel"])
        emitter.CreateAttribute("coupleRateFuel", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(
            emitter_config["coupleRateFuel"]
        )
        emitter.CreateAttribute("coupleRateVelocity", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(2.0)
        emitter.CreateAttribute("velocity", lazy.pxr.Sdf.ValueTypeNames.Float3, False).Set((0, 0, 0))
        emitter.CreateAttribute("physicsVelocityScale", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(1.0)
        emitter.CreateAttribute("layer", lazy.pxr.Sdf.ValueTypeNames.Int, False).Set(layer_number)
        simulate.CreateAttribute("layer", lazy.pxr.Sdf.ValueTypeNames.Int, False).Set(layer_number)
        simulate.CreateAttribute("stepsPerSecond", lazy.pxr.Sdf.ValueTypeNames.Float, False).Set(
            1 / og.sim.get_sim_step_dt()
        )
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
            if fire_at_meta_link:
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
        # If we're running fabric and the value is active, we need to manually update the pose in the USD
        # to ensure the rendering is updated properly at the correct pose
        # TODO(#2082): Verify if this is still needed.
        if value:
            self._sync_emitter_mesh_on_usd(emitter_type=emitter_type)
        if value != self._emitters[emitter_type]["emitter"].GetAttribute("enabled").Get():
            self._emitters[emitter_type]["emitter"].GetAttribute("enabled").Set(value)

    def _sync_emitter_mesh_on_usd(self, emitter_type):
        """
        Synchronizes the emitter's pose corresponding to @emitter_type on the USD

        Args:
            emitter_type (EmitterType): Emitter to synchronize
        """
        emitter_info = self._emitters[emitter_type]
        mesh = emitter_info["mesh"]
        link_pose = emitter_info["link"].get_position_orientation()
        position, orientation = T.relative_pose_transform(*link_pose, *emitter_info["canonical_pose"])

        # Actually set the local pose now.
        position = lazy.pxr.Gf.Vec3d(*position.tolist())
        mesh.set_attribute("xformOp:translate", position)
        orientation = orientation[[3, 0, 1, 2]].tolist()
        xform_op = mesh.prim.GetAttribute("xformOp:orient")
        if xform_op.GetTypeName() == "quatf":
            rotq = lazy.pxr.Gf.Quatf(*orientation)
        else:
            rotq = lazy.pxr.Gf.Quatd(*orientation)
        xform_op.Set(rotq)

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
        the current albedo map by adding and scaling the values. The final albedo value is
        albedo_value = diffuse_tint * (albedo_value + albedo_add)

        Args:
            object_state (BooleanStateMixin or None): the object state that the diffuse color should match to
        """
        # Compute the add and tint values
        if object_state is None:
            # This restore the albedo map to its original value
            albedo_add = 0.0
            diffuse_tint = th.tensor([1.0, 1.0, 1.0])
        else:
            # Query the object state for the parameters
            albedo_add, diffuse_tint = object_state.get_texture_change_params()

        # Apply the add and tint values
        for material in self.materials:
            if material.albedo_add != albedo_add:
                material.albedo_add = albedo_add

            if not th.allclose(material.diffuse_tint, diffuse_tint):
                material.diffuse_tint = diffuse_tint

    @cached_property
    def articulation_root_path(self):
        has_articulated_joints, has_fixed_joints = self.n_joints > 0, self.n_fixed_joints > 0
        if self.kinematic_only or (not has_articulated_joints and not has_fixed_joints):
            # Kinematic only, or non-jointed single body objects
            return None
        elif not self.fixed_base and has_articulated_joints:
            # This is all remaining non-fixed objects
            # This is a bit hacky because omniverse is buggy
            # Articulation roots mess up the joint order if it's on a non-fixed base robot, e.g. a
            # mobile manipulator. So if we have to move it to the actual root link of the robot instead.
            # See https://forums.developer.nvidia.com/t/inconsistent-values-from-isaacsims-dc-get-joint-parent-child-body/201452/2
            # for more info
            return f"{self.prim_path}/{self.root_link_name}"
        else:
            # Fixed objects that are not kinematic only, or non-fixed objects that have no articulated joints but do
            # have fixed joints
            return self.prim_path

    @property
    def uuid(self):
        """
        Returns:
            int: 8-digit unique identifier for this object. It is randomly generated from this object's name
                but deterministic
        """
        return self._uuid

    @property
    def mass(self):
        """
        Returns:
             float: Cumulative mass of this potentially articulated object.
        """
        mass = 0.0
        for link in self._links.values():
            if isinstance(link, RigidDynamicPrim):
                mass += link.mass

        return mass

    @mass.setter
    def mass(self, mass):
        raise NotImplementedError("Cannot set mass directly for an object!")

    @property
    def volume(self):
        """
        Returns:
             float: Cumulative volume of this potentially articulated object.
        """
        return sum(link.volume for link in self._links.values())

    @volume.setter
    def volume(self, volume):
        raise NotImplementedError("Cannot set volume directly for an object!")

    @property
    def scale(self):
        # Just super call
        return super().scale

    @scale.setter
    def scale(self, scale):
        # call super first
        # A bit esoteric -- see https://gist.github.com/Susensio/979259559e2bebcd0273f1a95d7c1e79
        super(USDObject, type(self)).scale.fset(self, scale)

        # Update init info for scale
        self._init_info["args"]["scale"] = scale

    @property
    def highlighted(self):
        """
        Returns:
            bool: Whether the object is highlighted or not
        """
        return self._highlighted

    @highlighted.setter
    def highlighted(self, enabled):
        """
        Iterates over all owned links, and modifies their materials with emissive colors so that the object is
        highlighted (magenta by default)

        Args:
            enabled (bool): whether the object should be highlighted or not
        """
        # Return early if the set value matches the internal value
        if enabled == self._highlighted:
            return

        for material in self.materials:
            if enabled:
                material.enable_highlight(self._highlight_color, self._highlight_intensity)
            else:
                material.disable_highlight()

        # Update internal value
        self._highlighted = enabled

    def set_highlight_properties(self, color=m.HIGHLIGHT_RGB, intensity=m.HIGHLIGHT_INTENSITY):
        """
        Sets the highlight properties for this object

        Args:
            color (3-array): RGB color for the highlight
            intensity (float): Intensity for the highlight
        """
        self._highlight_color = color
        self._highlight_intensity = intensity

        # Update the highlight properties if the object is currently highlighted
        if self._highlighted:
            self.highlighted = False
            self.highlighted = True

    def get_base_aligned_bbox(self, link_name=None, visual=False, xy_aligned=False):
        """
        Get a bounding box for this object that's axis-aligned in the object's base frame.

        Args:
            link_name (None or str): If specified, only get the bbox for the given link
            visual (bool): Whether to aggregate the bounding boxes from the visual meshes. Otherwise, will use
                collision meshes
            xy_aligned (bool): Whether to align the bounding box to the global XY-plane

        Returns:
            4-tuple:
                - 3-array: (x,y,z) bbox center position in world frame
                - 3-array: (x,y,z,w) bbox quaternion orientation in world frame
                - 3-array: (x,y,z) bbox extent in desired frame
                - 3-array: (x,y,z) bbox center in desired frame
        """
        # Get the base position transform.
        pos, orn = self.get_position_orientation()
        base_frame_to_world = T.pose2mat((pos, orn))

        # Prepare the desired frame.
        if xy_aligned:
            # If the user requested an XY-plane aligned bbox, convert everything to that frame.
            # The desired frame is same as the base_com frame with its X/Y rotations removed.
            translate = base_frame_to_world[:3, 3]

            # To find the rotation that this transform does around the Z axis, we rotate the [1, 0, 0] vector by it
            # and then take the arctangent of its projection onto the XY plane.
            rotated_X_axis = base_frame_to_world[:3, 0]
            rotation_around_Z_axis = th.arctan2(rotated_X_axis[1], rotated_X_axis[0])
            xy_aligned_base_com_to_world = th.eye(4, dtype=th.float32)
            xy_aligned_base_com_to_world[:3, 3] = translate
            xy_aligned_base_com_to_world[:3, :3] = T.euler2mat(
                th.tensor([0, 0, rotation_around_Z_axis], dtype=th.float32)
            )

            # Finally update our desired frame.
            desired_frame_to_world = xy_aligned_base_com_to_world
        else:
            # Default desired frame is base CoM frame.
            desired_frame_to_world = base_frame_to_world

        # Compute the world-to-base frame transform.
        world_to_desired_frame = th.linalg.inv_ex(desired_frame_to_world).inverse

        # Grab all the world-frame points corresponding to the object's visual or collision hulls.
        points_in_world = []
        if self.prim_type == PrimType.CLOTH:
            particle_contact_offset = self.root_link.cloth_system.particle_contact_offset
            particle_positions = self.root_link.compute_particle_positions()
            particles_in_world_frame = th.cat(
                [particle_positions - particle_contact_offset, particle_positions + particle_contact_offset], dim=0
            )
            points_in_world.extend(particles_in_world_frame.tolist())
        else:
            links = {link_name: self._links[link_name]} if link_name is not None else self._links
            for link_name, link in links.items():
                if visual:
                    hull_points = link.visual_boundary_points_world
                else:
                    hull_points = link.collision_boundary_points_world

                if hull_points is not None:
                    points_in_world.extend(hull_points.tolist())

        # Move the points to the desired frame
        points = T.transform_points(th.tensor(points_in_world, dtype=th.float32), world_to_desired_frame)

        # All points are now in the desired frame: either the base CoM or the xy-plane-aligned base CoM.
        # Now fit a bounding box to all the points by taking the minimum/maximum in the desired frame.
        aabb_min_in_desired_frame = th.amin(points, dim=0)
        aabb_max_in_desired_frame = th.amax(points, dim=0)
        bbox_center_in_desired_frame = (aabb_min_in_desired_frame + aabb_max_in_desired_frame) / 2
        bbox_extent_in_desired_frame = aabb_max_in_desired_frame - aabb_min_in_desired_frame

        # Transform the center to the world frame.
        bbox_center_in_world = T.transform_points(
            bbox_center_in_desired_frame.unsqueeze(0), desired_frame_to_world
        ).squeeze(0)
        bbox_orn_in_world = T.mat2quat(desired_frame_to_world[:3, :3])

        return bbox_center_in_world, bbox_orn_in_world, bbox_extent_in_desired_frame, bbox_center_in_desired_frame

    def dump_state(self, serialized=False):
        """
        Dumps the state of this object in either dictionary of flattened numerical form.

        Args:
            serialized (bool): If True, will return the state of this object as a 1D numpy array. Otherewise, will return
                a (potentially nested) dictionary of states for this object

        Returns:
            dict or n-array: Either:
                - Keyword-mapped states of this object, or
                - encoded + serialized, 1D numerical th.Tensor capturing this object's state
        """
        assert self._initialized, "Object must be initialized before dumping state!"
        return super().dump_state(serialized=serialized)

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
        Set the position and orientation of the object.

        Args:
            position (None or 3-array): The position to set the object to. If None, the position is not changed.
            orientation (None or 4-array): The orientation to set the object to. If None, the orientation is not changed.
            frame (Literal): The frame in which to set the position and orientation. Defaults to world.
                parent frame: set position relative to the object parent.
                scene frame: set position relative to the scene.
        """
        super().set_position_orientation(position=position, orientation=orientation, frame=frame)
        self.clear_states_cache()

    @classproperty
    def _do_not_register_classes(cls):
        classes = super()._do_not_register_classes
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global robot registry
        global REGISTERED_OBJECTS
        return REGISTERED_OBJECTS
