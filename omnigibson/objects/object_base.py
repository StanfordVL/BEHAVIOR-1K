from abc import ABCMeta
import numpy as np
from collections.abc import Iterable

import omnigibson as og
from omnigibson.macros import create_module_macros, gm
from omnigibson.utils.constants import (
    DEFAULT_COLLISION_GROUP,
    SPECIAL_COLLISION_GROUPS,
    SemanticClass,
)
from pxr import UsdPhysics, PhysxSchema
from omnigibson.utils.usd_utils import create_joint, CollisionAPI
from omnigibson.prims.entity_prim import EntityPrim
from omnigibson.utils.python_utils import Registerable, classproperty, get_uuid
from omnigibson.utils.constants import PrimType, CLASS_NAME_TO_CLASS_ID
from omnigibson.utils.ui_utils import create_module_logger, suppress_omni_log

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.semantics import add_update_semantics

# Global dicts that will contain mappings
REGISTERED_OBJECTS = dict()

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Settings for highlighting objects
m.HIGHLIGHT_RGB = [1.0, 0.1, 0.92]          # Default highlighting (R,G,B) color when highlighting objects
m.HIGHLIGHT_INTENSITY = 10000.0             # Highlight intensity to apply, range [0, 10000)


class BaseObject(EntityPrim, Registerable, metaclass=ABCMeta):
    """This is the interface that all OmniGibson objects must implement."""

    def __init__(
            self,
            name,
            prim_path=None,
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
            **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
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
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
                Note that this base object does NOT pass kwargs down into the Prim-type super() classes, and we assume
                that kwargs are only shared between all SUBclasses (children), not SUPERclasses (parents).
        """
        # Generate default prim path if none is specified
        prim_path = f"/World/{name}" if prim_path is None else prim_path

        # Store values
        self.uuid = get_uuid(name) if uuid is None else uuid
        assert len(str(self.uuid)) <= 8, f"UUID for this object must be at max 8-digits, got: {self.uuid}"
        self.category = category
        self.fixed_base = fixed_base

        # This sets the collision group of the object. In omnigibson, objects are only permitted to be part of a single
        # collision group, e.g. collisions are only enabled within a single group
        self.collision_group = SPECIAL_COLLISION_GROUPS.get(self.category, DEFAULT_COLLISION_GROUP)

        # Infer class ID if not specified
        if class_id is None:
            class_id = CLASS_NAME_TO_CLASS_ID.get(category, SemanticClass.USER_ADDED_OBJS)
        self.class_id = class_id

        # Values to be created at runtime
        self._highlight_cached_values = None
        self._highlighted = None

        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["scale"] = np.array(scale) if isinstance(scale, Iterable) else scale
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

    def load(self):
        # Run super method ONLY if we're not loaded yet
        if self.loaded:
            prim = self._prim
        else:
            prim = super().load()
            log.info(f"Loaded {self.name} at {self.prim_path}")
        return prim

    def remove(self):
        # Run super first
        super().remove()

        # Notify user that the object was removed
        log.info(f"Removed {self.name} from {self.prim_path}")

    def _post_load(self):
        # Run super first
        super()._post_load()

        # Set visibility
        if "visible" in self._load_config and self._load_config["visible"] is not None:
            self.visible = self._load_config["visible"]

        # First, remove any articulation root API that already exists at the object-level prim
        if self._prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            self._prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        # Add fixed joint if we're fixing the base
        if self.fixed_base:
            # For optimization purposes, if we only have a single rigid body that has either
            # (no custom scaling OR no fixed joints), we assume this is not an articulated object so we
            # merely set this to be a static collider, i.e.: kinematic-only
            # The custom scaling / fixed joints requirement is needed because omniverse complains about scaling that
            # occurs with respect to fixed joints, as omni will "snap" bodies together otherwise
            if self.n_joints == 0 and (np.all(np.isclose(self.scale, 1.0, atol=1e-3)) or self.n_fixed_joints == 0):
                self.kinematic_only = True
            else:
                # Create fixed joint, and set Body0 to be this object's root prim
                create_joint(
                    prim_path=f"{self._prim_path}/rootJoint",
                    joint_type="FixedJoint",
                    body1=f"{self._prim_path}/{self._root_link_name}",
                )

        # Potentially add articulation root APIs and also set self collisions
        root_prim = None if self.articulation_root_path is None else get_prim_at_path(self.articulation_root_path)
        if root_prim is not None:
            UsdPhysics.ArticulationRootAPI.Apply(root_prim)
            PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
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

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Iterate over all links and grab their relevant material info for highlighting (i.e.: emissivity info)
        self._highlighted = False
        self._highlight_cached_values = dict()

        for material in self.materials:
            self._highlight_cached_values[material] = {
                "enable_emission": material.enable_emission,
                "emissive_color": material.emissive_color,
                "emissive_intensity": material.emissive_intensity,
            }

    @property
    def articulation_root_path(self):
        has_articulated_joints, has_fixed_joints = self.n_joints > 0, self.n_fixed_joints > 0
        if self.kinematic_only or ((not has_articulated_joints) and (not has_fixed_joints)):
            # Kinematic only, or non-jointed single body objects
            return None
        elif not self.fixed_base and has_articulated_joints:
            # This is all remaining non-fixed objects
            # This is a bit hacky because omniverse is buggy
            # Articulation roots mess up the joint order if it's on a non-fixed base robot, e.g. a
            # mobile manipulator. So if we have to move it to the actual root link of the robot instead.
            # See https://forums.developer.nvidia.com/t/inconsistent-values-from-isaacsims-dc-get-joint-parent-child-body/201452/2
            # for more info
            return f"{self._prim_path}/{self.root_link_name}"
        else:
            # Fixed objects that are not kinematic only, or non-fixed objects that have no articulated joints but do
            # have fixed joints
            return self._prim_path

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
    def scale(self):
        # Just super call
        return super().scale

    @scale.setter
    def scale(self, scale):
        # call super first
        # A bit esoteric -- see https://gist.github.com/Susensio/979259559e2bebcd0273f1a95d7c1e79
        super(BaseObject, type(self)).scale.fset(self, scale)

        # Update init info for scale
        self._init_info["args"]["scale"] = scale

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

