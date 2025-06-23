from abc import ABCMeta
from collections.abc import Iterable
from functools import cached_property

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.prims.entity_prim import EntityPrim
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import Registerable, classproperty, get_uuid
from omnigibson.utils.ui_utils import create_module_logger, suppress_omni_log
from omnigibson.utils.usd_utils import create_joint

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


class BaseObject(EntityPrim, Registerable, metaclass=ABCMeta):
    """This is the interface that all OmniGibson objects must implement."""

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
        link_physics_materials=None,
        load_config=None,
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
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
                Note that this base object does NOT pass kwargs down into the Prim-type super() classes, and we assume
                that kwargs are only shared between all SUBclasses (children), not SUPERclasses (parents).
        """
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

    def prebuild(self, stage):
        """
        Implement this function to provide pre-building functionality on an USD stage
        that is not loaded into Isaac Sim. This is useful for pre-compiling scene USDs,
        speeding up load times especially for parallel envs.
        """
        pass

    def load(self, scene):
        prim = super().load(scene)
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
            scale = th.ones(3) if self._load_config["scale"] is None else self._load_config["scale"]
            if (
                # no articulated joints
                self.n_joints == 0
                # no fixed joints or scaling is [1, 1, 1] (TODO verify [1, 1, 1] is still needed)
                and (th.all(th.isclose(scale, th.ones_like(scale), atol=1e-3)).item() or self.n_fixed_joints == 0)
                # users force the object to not have kinematic_only
                and self._load_config["kinematic_only"] is not False  # if can be True or None
                and not self.has_attachment_points
            ):
                kinematic_only = True

        # Validate that we didn't make a kinematic-only decision that does not match
        assert (
            self._load_config["kinematic_only"] is None or kinematic_only == self._load_config["kinematic_only"]
        ), f"Kinematic only decision does not match! Got: {kinematic_only}, expected: {self._load_config['kinematic_only']}"

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
                    prim_path=f"{self.prim_path}/rootJoint",
                    joint_type="FixedJoint",
                    body1=f"{self.prim_path}/{self._root_link_name}",
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

        if og.sim.is_playing():
            log.warning(
                "An object's articulation root API was changed while simulation is playing. This may cause issues."
            )

        # Potentially add articulation root APIs and also set self collisions
        root_prim = (
            None
            if self.articulation_root_path is None
            else lazy.isaacsim.core.utils.prims.get_prim_at_path(self.articulation_root_path)
        )
        if root_prim is not None:
            lazy.pxr.UsdPhysics.ArticulationRootAPI.Apply(root_prim)
            lazy.pxr.PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
            self.self_collisions = self._load_config["self_collisions"]

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
        lazy.isaacsim.core.utils.semantics.add_update_semantics(
            prim=self._prim,
            semantic_label=self.category,
            type_label="class",
        )

    def _initialize(self):
        # Run super first
        super()._initialize()

    @cached_property
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
            return f"{self.prim_path}/{self.root_link_name}"
        else:
            # Fixed objects that are not kinematic only, or non-fixed objects that have no articulated joints but do
            # have fixed joints
            return self.prim_path

    @property
    def is_active(self):
        """
        Returns:
            bool: True if this object is currently considered active -- e.g.: if this object is currently awake
        """
        return not self.kinematic_only and not self.is_asleep

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
        super(BaseObject, type(self)).scale.fset(self, scale)

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
