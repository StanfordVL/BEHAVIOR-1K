from abc import ABCMeta
import numpy as np
from collections.abc import Iterable
import trimesh
from scipy.spatial.transform import Rotation

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros, gm
from omnigibson.utils.usd_utils import create_joint, CollisionAPI
from omnigibson.prims.entity_prim import EntityPrim
from omnigibson.utils.python_utils import Registerable, classproperty, get_uuid
from omnigibson.utils.constants import PrimType, semantic_class_name_to_id
from omnigibson.utils.ui_utils import create_module_logger, suppress_omni_log
import omnigibson.utils.transform_utils as T


# Global dicts that will contain mappings
REGISTERED_OBJECTS = dict()

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Settings for highlighting objects
m.HIGHLIGHT_RGB = [1.0, 0.1, 0.92]          # Default highlighting (R,G,B) color when highlighting objects
m.HIGHLIGHT_INTENSITY = 10000.0             # Highlight intensity to apply, range [0, 10000)

# Physics settings for objects -- see https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/RigidBodyDynamics.html?highlight=velocity%20iteration#solver-iterations
m.DEFAULT_SOLVER_POSITION_ITERATIONS = 32
m.DEFAULT_SOLVER_VELOCITY_ITERATIONS = 1

class BaseObject(EntityPrim, Registerable, metaclass=ABCMeta):
    """This is the interface that all OmniGibson objects must implement."""

    def __init__(
            self,
            name,
            prim_path=None,
            category="object",
            uuid=None,
            scale=None,
            visible=True,
            fixed_base=False,
            visual_only=False,
            kinematic_only=None,
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
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
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

        # Values to be created at runtime
        self._highlight_cached_values = None
        self._highlighted = None

        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["scale"] = np.array(scale) if isinstance(scale, Iterable) else scale
        load_config["visible"] = visible
        load_config["visual_only"] = visual_only
        load_config["kinematic_only"] = kinematic_only
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
        # Add fixed joint or make object kinematic only if we're fixing the base
        kinematic_only = False
        if self.fixed_base:
            # For optimization purposes, if we only have a single rigid body that has either
            # (no custom scaling OR no fixed joints), we assume this is not an articulated object so we
            # merely set this to be a static collider, i.e.: kinematic-only
            # The custom scaling / fixed joints requirement is needed because omniverse complains about scaling that
            # occurs with respect to fixed joints, as omni will "snap" bodies together otherwise
            scale = np.ones(3) if self._load_config["scale"] is None else np.array(self._load_config["scale"])
            if self.n_joints == 0 and (np.all(np.isclose(scale, 1.0, atol=1e-3)) or self.n_fixed_joints == 0) and (self._load_config["kinematic_only"] != False) and not self.has_attachment_points:
                kinematic_only = True
        
        # Validate that we didn't make a kinematic-only decision that does not match
        assert self._load_config["kinematic_only"] is None or kinematic_only == self._load_config["kinematic_only"], \
            f"Kinematic only decision does not match! Got: {kinematic_only}, expected: {self._load_config['kinematic_only']}"
        
        # Actually apply the kinematic-only decision
        self._load_config["kinematic_only"] = kinematic_only

        # Run super first
        super()._post_load()

        # If the object is fixed_base but kinematic only is false, create the joint
        if self.fixed_base and not self.kinematic_only:
            # Create fixed joint, and set Body0 to be this object's root prim
            # This renders, which causes a material lookup error since we're creating a temp file, so we suppress
            # the error explicitly here
            with suppress_omni_log(channels=["omni.hydra"]):
                create_joint(
                    prim_path=f"{self._prim_path}/rootJoint",
                    joint_type="FixedJoint",
                    body1=f"{self._prim_path}/{self._root_link_name}",
                )

            # Delete n_fixed_joints cached property if it exists since the number of fixed joints has now changed
            # See https://stackoverflow.com/questions/59899732/python-cached-property-how-to-delete and
            # https://docs.python.org/3/library/functools.html#functools.cached_property
            if "n_fixed_joints" in self.__dict__:
                del self.n_fixed_joints

        # Set visibility
        if "visible" in self._load_config and self._load_config["visible"] is not None:
            self.visible = self._load_config["visible"]

        # First, remove any articulation root API that already exists at the object-level or root link level prim
        if self._prim.HasAPI(lazy.pxr.UsdPhysics.ArticulationRootAPI):
            self._prim.RemoveAPI(lazy.pxr.UsdPhysics.ArticulationRootAPI)
            self._prim.RemoveAPI(lazy.pxr.PhysxSchema.PhysxArticulationAPI)

        if self.root_prim.HasAPI(lazy.pxr.UsdPhysics.ArticulationRootAPI):
            self.root_prim.RemoveAPI(lazy.pxr.UsdPhysics.ArticulationRootAPI)
            self.root_prim.RemoveAPI(lazy.pxr.PhysxSchema.PhysxArticulationAPI)

        # Potentially add articulation root APIs and also set self collisions
        root_prim = None if self.articulation_root_path is None else lazy.omni.isaac.core.utils.prims.get_prim_at_path(self.articulation_root_path)
        if root_prim is not None:
            lazy.pxr.UsdPhysics.ArticulationRootAPI.Apply(root_prim)
            lazy.pxr.PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
            self.self_collisions = self._load_config["self_collisions"]

        # Set position / velocity solver iterations if we're not cloth
        if self._prim_type != PrimType.CLOTH:
            self.solver_position_iteration_count = m.DEFAULT_SOLVER_POSITION_ITERATIONS
            self.solver_velocity_iteration_count = m.DEFAULT_SOLVER_VELOCITY_ITERATIONS

        # Add semantics
        lazy.omni.isaac.core.utils.semantics.add_update_semantics(
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
            translate = trimesh.transformations.translation_from_matrix(base_frame_to_world)

            # To find the rotation that this transform does around the Z axis, we rotate the [1, 0, 0] vector by it
            # and then take the arctangent of its projection onto the XY plane.
            rotated_X_axis = base_frame_to_world[:3, 0]
            rotation_around_Z_axis = np.arctan2(rotated_X_axis[1], rotated_X_axis[0])
            xy_aligned_base_com_to_world = trimesh.transformations.compose_matrix(
                translate=translate, angles=[0, 0, rotation_around_Z_axis]
            )

            # Finally update our desired frame.
            desired_frame_to_world = xy_aligned_base_com_to_world
        else:
            # Default desired frame is base CoM frame.
            desired_frame_to_world = base_frame_to_world

        # Compute the world-to-base frame transform.
        world_to_desired_frame = np.linalg.inv(desired_frame_to_world)

        # Grab all the world-frame points corresponding to the object's visual or collision hulls.
        points_in_world = []
        if self.prim_type == PrimType.CLOTH:
            particle_contact_offset = self.root_link.cloth_system.particle_contact_offset
            particle_positions = self.root_link.compute_particle_positions()
            particles_in_world_frame = np.concatenate([
                particle_positions - particle_contact_offset,
                particle_positions + particle_contact_offset
            ], axis=0)
            points_in_world.extend(particles_in_world_frame)
        else:
            links = {link_name: self._links[link_name]} if link_name is not None else self._links
            for link_name, link in links.items():
                if visual:
                    hull_points = link.visual_boundary_points_world
                else:
                    hull_points = link.collision_boundary_points_world

                if hull_points is not None:
                    points_in_world.extend(hull_points)

        # Move the points to the desired frame
        points = trimesh.transformations.transform_points(points_in_world, world_to_desired_frame)

        # All points are now in the desired frame: either the base CoM or the xy-plane-aligned base CoM.
        # Now fit a bounding box to all the points by taking the minimum/maximum in the desired frame.
        aabb_min_in_desired_frame = np.amin(points, axis=0)
        aabb_max_in_desired_frame = np.amax(points, axis=0)
        bbox_center_in_desired_frame = (aabb_min_in_desired_frame + aabb_max_in_desired_frame) / 2
        bbox_extent_in_desired_frame = aabb_max_in_desired_frame - aabb_min_in_desired_frame

        # Transform the center to the world frame.
        bbox_center_in_world = trimesh.transformations.transform_points(
            [bbox_center_in_desired_frame], desired_frame_to_world
        )[0]
        bbox_orn_in_world = Rotation.from_matrix(desired_frame_to_world[:3, :3]).as_quat()

        return bbox_center_in_world, bbox_orn_in_world, bbox_extent_in_desired_frame, bbox_center_in_desired_frame

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

