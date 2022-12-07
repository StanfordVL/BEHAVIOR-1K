from abc import ABCMeta, abstractmethod
from collections import Iterable, OrderedDict
import logging

from omnigibson.macros import create_module_macros
from omnigibson.utils.constants import (
    ALL_COLLISION_GROUPS_MASK,
    DEFAULT_COLLISION_GROUP,
    SPECIAL_COLLISION_GROUPS,
    SemanticClass,
)
from pxr import UsdPhysics, PhysxSchema
from omnigibson.utils.usd_utils import get_prim_nested_children, create_joint, CollisionAPI
from omnigibson.prims.entity_prim import EntityPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.prims.rigid_prim import RigidPrim
from omnigibson.utils.python_utils import Registerable, classproperty
from omnigibson.utils.constants import PrimType, CLASS_NAME_TO_CLASS_ID

from omni.isaac.core.utils.semantics import add_update_semantics
from pxr import Gf

# Global dicts that will contain mappings
REGISTERED_OBJECTS = OrderedDict()

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Settings for highlighting objects
m.HIGHLIGHT_RGB = [1.0, 0.1, 0.92]          # Default highlighting (R,G,B) color when highlighting objects
m.HIGHLIGHT_INTENSITY = 10000.0             # Highlight intensity to apply, range [0, 10000)


class BaseObject(EntityPrim, Registerable, metaclass=ABCMeta):
    """This is the interface that all OmniGibson objects must implement."""

    def __init__(
            self,
            prim_path,
            name=None,
            category="object",
            class_id=None,
            uuid=None,
            scale=None,
            rendering_params=None,
            visible=True,
            fixed_base=False,
            visual_only=False,
            self_collisions=False,
            prim_type=PrimType.RIGID,
            load_config=None,
            **kwargs,
    ):
        """
        Create an object instance with the minimum information of class ID and rendering parameters.

        @param prim_path: str, global path in the stage to this object
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param uuid: Unique unsigned-integer identifier to assign to this object (max 8-numbers).
            If None is specified, then it will be auto-generated
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
        self_collisions (bool): Whether to enable self collisions for this object
        prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
            Note that this base object does NOT pass kwargs down into the Prim-type super() classes, and we assume
            that kwargs are only shared between all SUBclasses (children), not SUPERclasses (parents).
        """
        # Generate a name if necessary. Note that the generation order & set of these names is not deterministic.
        if name is None:
            address = "%08X" % id(self)
            name = "{}_{}".format(category, address)

        # Store values
        self.uuid = int(str(id(self))[-8:]) if uuid is None else uuid
        assert len(str(self.uuid)) <= 8, f"UUID for this object must be at max 8-digits, got: {self.uuid}"
        self.category = category
        self.fixed_base = fixed_base

        logging.info(f"Category: {self.category}")

        # TODO
        # This sets the collision group of the object. In omnigibson, objects are only permitted to be part of a single
        # collision group, e.g. collisions are only enabled within a single group
        self.collision_group = SPECIAL_COLLISION_GROUPS.get(self.category, DEFAULT_COLLISION_GROUP)

        # category_based_rendering_params = {}
        # if category in ["walls", "floors", "ceilings"]:
        #     category_based_rendering_params["use_pbr"] = False
        #     category_based_rendering_params["use_pbr_mapping"] = False
        # if category == "ceilings":
        #     category_based_rendering_params["shadow_caster"] = False
        #
        # if rendering_params:  # Use the input rendering params as an override.
        #     category_based_rendering_params.update(rendering_params)

        if class_id is None:
            class_id = CLASS_NAME_TO_CLASS_ID.get(category, SemanticClass.USER_ADDED_OBJS)

        self.class_id = class_id
        self.rendering_params = rendering_params
        # self._rendering_params = dict(self.DEFAULT_RENDERING_PARAMS)
        # self._rendering_params.update(category_based_rendering_params)

        # Values to be created at runtime
        self._simulator = None
        self._highlight_cached_values = None
        self._highlighted = None

        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["scale"] = scale
        load_config["visible"] = visible
        load_config["visual_only"] = visual_only
        load_config["self_collisions"] = self_collisions
        load_config["prim_type"] = prim_type

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

        # TODO: Super hacky, think of a better way to preserve this info
        # Update init info for this
        self._init_info["args"]["name"] = self.name
        self._init_info["args"]["uuid"] = self.uuid

    def load(self, simulator=None):
        # Run sanity check, any of these objects REQUIRE a simulator to be specified
        assert simulator is not None, "Simulator must be specified for loading any object subclassed from BaseObject!"

        # Save simulator reference
        self._simulator = simulator

        # Run super method ONLY if we're not loaded yet
        return super().load(simulator=simulator) if not self.loaded else self._prim

    def _post_load(self):
        # Run super first
        super()._post_load()

        # Set visibility
        if "visible" in self._load_config and self._load_config["visible"] is not None:
            self.visible = self._load_config["visible"]

        # Add fixed joint if we're fixing the base
        if self.fixed_base:
            # Create fixed joint, and set Body0 to be this object's root prim
            create_joint(
                prim_path=f"{self._prim_path}/rootJoint",
                joint_type="FixedJoint",
                body1=f"{self._prim_path}/{self._root_link_name}",
            )
            # Also set the articulation root to be the object head if it doesn't already exist
            if not self._prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                UsdPhysics.ArticulationRootAPI.Apply(self.prim)
                PhysxSchema.PhysxArticulationAPI.Apply(self.prim)
        else:
            if self._prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                # If we only have a link, remove the articulation root API
                if self.n_links == 1:
                    self._prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    self._prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                else:
                    # We need to fix (change) the articulation root
                    # We have to do something very hacky because omniverse is buggy
                    # Articulation roots mess up the joint order if it's on a non-fixed base robot, e.g. a
                    # mobile manipulator. So if we have to move it to the actual root link of the robot instead.
                    # See https://forums.developer.nvidia.com/t/inconsistent-values-from-isaacsims-dc-get-joint-parent-child-body/201452/2
                    # for more info
                    self._prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    self._prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                    UsdPhysics.ArticulationRootAPI.Apply(self.root_prim)
                    PhysxSchema.PhysxArticulationAPI.Apply(self.root_prim)

        # Set self collisions if we have articulation API to set
        if self._prim.HasAPI(UsdPhysics.ArticulationRootAPI) or self.root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self.self_collisions = self._load_config["self_collisions"]

        # TODO: Do we need to explicitly add all links? or is adding articulation root itself sufficient?
        # Set the collision group
        CollisionAPI.add_to_collision_group(
            col_group=self.collision_group,
            prim_path=self.prim_path,
            create_if_not_exist=True,
        )

        # Update semantics
        add_update_semantics(
            prim=self._prim,
            semantic_label=self.category,
            type_label="class",
        )

        # Force populate inputs and outputs of the shaders of all materials
        for material in self.materials:
            material.shader_force_populate()
            material.shader_update_asset_paths()

        # Iterate over all links and grab their relevant material info for highlighting (i.e.: emissivity info)
        self._highlighted = False
        self._highlight_cached_values = OrderedDict()

        for material in self.materials:
            self._highlight_cached_values[material] = {
                "enable_emission": material.enable_emission,
                "emissive_color": material.emissive_color,
                "emissive_intensity": material.emissive_intensity,
            }

    @property
    def articulation_root_path(self):
        # We override this because omniverse is buggy ):
        # For non-fixed base objects (e.g.: mobile manipulators), using the default articulation root breaks the
        # kinematic chain for some reason. So, the current workaround is to set the articulation root to be the
        # actual base link of the robot instead.
        # See https://forums.developer.nvidia.com/t/inconsistent-values-from-isaacsims-dc-get-joint-parent-child-body/201452/2
        # for more info
        return f"{self._prim_path}/{self.root_link_name}" if \
            (not self.fixed_base) and (self.n_links > 1) else super().articulation_root_path

    @property
    def mass(self):
        """
        Returns:
             float: Cumulative mass of this potentially articulated object.
        """
        mass = 0.0
        for link in self._links.values():
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
        volume = 0.0
        for link in self._links.values():
            volume += link.volume

        return volume

    @volume.setter
    def volume(self, volume):
        raise NotImplementedError("Cannot set volume directly for an object!")

    @property
    def link_prim_paths(self):
        return [link.prim_path for link in self._links.values()]

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
                # Store values before swapping
                self._highlight_cached_values[material] = {
                    "enable_emission": material.enable_emission,
                    "emissive_color": material.emissive_color,
                    "emissive_intensity": material.emissive_intensity,
                }
            material.enable_emission = True if enabled else self._highlight_cached_values[material]["enable_emission"]
            material.emissive_color = m.HIGHLIGHT_RGB if enabled else self._highlight_cached_values[material]["emissive_color"]
            material.emissive_intensity = m.HIGHLIGHT_INTENSITY if enabled else self._highlight_cached_values[material]["emissive_intensity"]

        # Update internal value
        self._highlighted = enabled

    def get_velocities(self):
        """Get this object's root body velocity in the format of Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]"""
        return self.get_linear_velocity(), self.get_angular_velocity()

    def set_velocities(self, velocities):
        """Set this object's root body velocity in the format of Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]"""
        lin_vel, ang_vel = velocities

        self.set_linear_velocity(velocity=lin_vel)
        self.set_angular_velocity(velocity=ang_vel)

    def dump_config(self):
        """
        Dumps relevant configuration for this object.

        Returns:
            OrderedDict: Object configuration.
        """
        return OrderedDict(
            category=self.category,
            class_id=self.class_id,
            scale=self.scale,
            self_collisions=self.self_collisions,
            rendering_params=self.rendering_params,
        )

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseObject")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global robot registry
        global REGISTERED_OBJECTS
        return REGISTERED_OBJECTS

