import math
from collections import Iterable
import os

import omni.usd
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage, add_reference_to_stage
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache, compute_combined_aabb
from omni.syntheticdata import helpers
from omni.kit.primitive.mesh.evaluators.sphere import SphereEvaluator
from omni.kit.primitive.mesh.evaluators.disk import DiskEvaluator
from omni.kit.primitive.mesh.evaluators.plane import PlaneEvaluator
from omni.kit.primitive.mesh.evaluators.cylinder import CylinderEvaluator
from omni.kit.primitive.mesh.evaluators.torus import TorusEvaluator
from omni.kit.primitive.mesh.evaluators.cone import ConeEvaluator
from omni.kit.primitive.mesh.evaluators.cube import CubeEvaluator

from pxr import Gf, Vt, Usd, Sdf, UsdGeom, UsdShade, UsdPhysics, PhysxSchema
import carb
import numpy as np
import trimesh

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.constants import JointType, PRIMITIVE_MESH_TYPES, PrimType, GEOM_TYPES
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.ui_utils import suppress_omni_log

import omnigibson.utils.transform_utils as T

GF_TO_VT_MAPPING = {
    Gf.Vec3d: Vt.Vec3dArray,
    Gf.Vec3f: Vt.Vec3fArray,
    Gf.Vec3h: Vt.Vec3hArray,
    Gf.Quatd: Vt.QuatdArray,
    Gf.Quatf: Vt.QuatfArray,
    Gf.Quath: Vt.QuathArray,
    int: Vt.IntArray,
    float: Vt.FloatArray,
    bool: Vt.BoolArray,
    str: Vt.StringArray,
    chr: Vt.CharArray,
}

MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING = {
    "Sphere": SphereEvaluator,
    "Disk": DiskEvaluator,
    "Plane": PlaneEvaluator,
    "Cylinder": CylinderEvaluator,
    "Torus": TorusEvaluator,
    "Cone": ConeEvaluator,
    "Cube": CubeEvaluator,
}


def array_to_vtarray(arr, element_type):
    """
    Converts array @arr into a Vt-typed array, where each individual element of type @element_type.

    Args:
        arr (n-array): An array of values. Can be, e.g., a list, or numpy array
        element_type (type): Per-element type to convert the elements from @arr into.
            Valid options are keys of GF_TO_VT_MAPPING

    Returns:
        Vt.Array: Vt-typed array, of specified type corresponding to @element_type
    """
    # Make sure array type is valid
    assert_valid_key(key=element_type, valid_keys=GF_TO_VT_MAPPING, name="array element type")

    # Construct list of values
    arr_list = []

    # Check first to see if elements are vectors or not. If this is an iterable value that is not a string,
    # then this is a vector and we have to map it to the correct type via *
    is_vec_element = (isinstance(arr[0], Iterable)) and (not isinstance(arr[0], str))

    # Loop over array and set values
    for ele in arr:
        arr_list.append(element_type(*ele) if is_vec_element else ele)

    return GF_TO_VT_MAPPING[element_type](arr_list)


def get_prim_nested_children(prim):
    """
    Grabs all nested prims starting from root @prim via depth-first-search

    Args:
        prim (Usd.Prim): root prim from which to search for nested children prims

    Returns:
        list of Usd.Prim: nested prims
    """
    prims = []
    for child in get_prim_children(prim):
        prims.append(child)
        prims += get_prim_nested_children(prim=child)

    return prims


def get_camera_params(viewport):
    """
    Get active camera intrinsic and extrinsic parameters.

    Returns:
        dict: Keyword-mapped values of the active camera's parameters:

            pose (numpy.ndarray): camera position in world coordinates,
            fov (float): horizontal field of view in radians
            focal_length (float)
            horizontal_aperture (float)
            view_projection_matrix (numpy.ndarray(dtype=float64, shape=(4, 4)))
            resolution (dict): resolution as a dict with 'width' and 'height'.
            clipping_range (tuple(float, float)): Near and Far clipping values.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(viewport.get_active_camera())
    prim_tf = omni.usd.get_world_transform_matrix(prim)
    view_params = helpers.get_view_params(viewport)
    fov = 2 * math.atan(view_params["horizontal_aperture"] / (2 * view_params["focal_length"]))
    view_proj_mat = helpers.get_view_proj_mat(view_params)

    return {
        "pose": np.array(prim_tf),
        "fov": fov,
        "focal_length": view_params["focal_length"],
        "horizontal_aperture": view_params["horizontal_aperture"],
        "view_projection_matrix": view_proj_mat,
        "resolution": {"width": view_params["width"], "height": view_params["height"]},
        "clipping_range": view_params["clipping_range"],
    }


def get_semantic_objects_pose():
    """
    Get pose of all objects with a semantic label.
    """
    stage = omni.usd.get_context().get_stage()
    mappings = helpers.get_instance_mappings()
    pose = []
    for m in mappings:
        prim_path = m[1]
        prim = stage.GetPrimAtPath(prim_path)
        prim_tf = omni.usd.get_world_transform_matrix(prim)
        pose.append((str(prim_path), m[2], str(m[3]), np.array(prim_tf)))
    return pose


def create_joint(prim_path, joint_type, body0=None, body1=None, enabled=True,
                 joint_frame_in_parent_frame_pos=None, joint_frame_in_parent_frame_quat=None,
                 joint_frame_in_child_frame_pos=None, joint_frame_in_child_frame_quat=None,
                 break_force=None, break_torque=None):
    """
    Creates a joint between @body0 and @body1 of specified type @joint_type

    Args:
        prim_path (str): absolute path to where the joint will be created
        joint_type (str): type of joint to create. Valid options are:
            "FixedJoint", "Joint", "PrismaticJoint", "RevoluteJoint", "SphericalJoint"
                        (equivalently, one of JointType)
        body0 (str or None): absolute path to the first body's prim. At least @body0 or @body1 must be specified.
        body1 (str or None): absolute path to the second body's prim. At least @body0 or @body1 must be specified.
        enabled (bool): whether to enable this joint or not.
        joint_frame_in_parent_frame_pos (np.ndarray or None): relative position of the joint frame to the parent frame (body0).
        joint_frame_in_parent_frame_quat (np.ndarray or None): relative orientation of the joint frame to the parent frame (body0).
        joint_frame_in_child_frame_pos (np.ndarray or None): relative position of the joint frame to the child frame (body1).
        joint_frame_in_child_frame_quat (np.ndarray or None): relative orientation of the joint frame to the child frame (body1).
        break_force (float or None): break force for linear dofs, unit is Newton.
        break_torque (float or None): break torque for angular dofs, unit is Newton-meter.

    Returns:
        Usd.Prim: Created joint prim
    """
    # Make sure we have valid joint_type
    assert JointType.is_valid(joint_type=joint_type), \
        f"Invalid joint specified for creation: {joint_type}"

    # Make sure at least body0 or body1 is specified
    assert body0 is not None or body1 is not None, \
        f"At least either body0 or body1 must be specified when creating a joint!"

    # Create the joint
    joint = UsdPhysics.__dict__[joint_type].Define(og.sim.stage, prim_path)

    # Possibly add body0, body1 targets
    if body0 is not None:
        assert is_prim_path_valid(body0), f"Invalid body0 path specified: {body0}"
        joint.GetBody0Rel().SetTargets([Sdf.Path(body0)])
    if body1 is not None:
        assert is_prim_path_valid(body1), f"Invalid body1 path specified: {body1}"
        joint.GetBody1Rel().SetTargets([Sdf.Path(body1)])

    # Get the prim pointed to at this path
    joint_prim = get_prim_at_path(prim_path)

    # Apply joint API interface
    PhysxSchema.PhysxJointAPI.Apply(joint_prim)

    # We need to step rendering once to auto-fill the local pose before overwriting it.
    # Note that for some reason, if multi_gpu is used, this line will crash if create_joint is called during on_contact
    # callback, e.g. when an attachment joint is being created due to contacts.
    og.sim.render()

    if joint_frame_in_parent_frame_pos is not None:
        joint_prim.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*joint_frame_in_parent_frame_pos))
    if joint_frame_in_parent_frame_quat is not None:
        joint_prim.GetAttribute("physics:localRot0").Set(Gf.Quatf(*joint_frame_in_parent_frame_quat[[3, 0, 1, 2]]))
    if joint_frame_in_child_frame_pos is not None:
        joint_prim.GetAttribute("physics:localPos1").Set(Gf.Vec3f(*joint_frame_in_child_frame_pos))
    if joint_frame_in_child_frame_quat is not None:
        joint_prim.GetAttribute("physics:localRot1").Set(Gf.Quatf(*joint_frame_in_child_frame_quat[[3, 0, 1, 2]]))

    if break_force is not None:
        joint_prim.GetAttribute("physics:breakForce").Set(break_force)
    if break_torque is not None:
        joint_prim.GetAttribute("physics:breakTorque").Set(break_torque)

    # Possibly (un-/)enable this joint
    joint_prim.GetAttribute("physics:jointEnabled").Set(enabled)

    # We update the simulation now without stepping physics if sim is playing so we can bypass the snapping warning from PhysicsUSD
    if og.sim.is_playing():
        with suppress_omni_log(channels=["omni.physx.plugin"]):
            og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)

    # Return this joint
    return joint_prim


class RigidContactAPI:
    """
    Class containing class methods to aggregate rigid body contacts across all rigid bodies in the simulator
    """
    # Dictionary mapping rigid body prim path to corresponding index in the contact view matrix
    _PATH_TO_IDX = None

    # Contact view for generating contact matrices at each timestep
    _CONTACT_VIEW = None

    # Current aggregated contacts over all rigid bodies at the current timestep. Shape: (N, N, 3)
    _CONTACT_MATRIX = None

    # Current cache, mapping 2-tuple (prim_paths_a, prim_paths_b) to contact values
    _CONTACT_CACHE = None

    @classmethod
    def initialize_view(cls):
        """
        Initializes the rigid contact view. Note: Can only be done when sim is playing!
        """
        assert og.sim.is_playing(), "Cannot create rigid contact view while sim is not playing!"

        # Compile deterministic mapping from rigid body path to idx
        # Note that omni's ordering is based on the top-down object ordering path on the USD stage, which coincidentally
        # matches the same ordering we store objects in our registry. So the mapping we generate from our registry
        # mapping aligns with omni's ordering!
        i = 0
        cls._PATH_TO_IDX = dict()
        for obj in og.sim.scene.objects:
            if obj.prim_type == PrimType.RIGID:
                for link in obj.links.values():
                    if not link.kinematic_only:
                        cls._PATH_TO_IDX[link.prim_path] = i
                        i += 1

        # Generate rigid body view, making sure to update the simulation first (without physics) so that the physx
        # backend is synchronized with any newly added objects
        # We also suppress the omni tensor plugin from giving warnings we expect
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            cls._CONTACT_VIEW = og.sim.physics_sim_view.create_rigid_contact_view(
                pattern="/World/*/*",
                filter_patterns=list(cls._PATH_TO_IDX.keys()),
            )

        # Sanity check generated view -- this should generate square matrices of shape (N, N, 3)
        n_bodies = len(cls._PATH_TO_IDX)
        # from IPython import embed; embed()
        assert cls._CONTACT_VIEW.sensor_count == n_bodies and cls._CONTACT_VIEW.filter_count == n_bodies, \
            f"Got unexpected contact view shape. Expected: ({n_bodies}, {n_bodies}); " \
            f"got: ({cls._CONTACT_VIEW.sensor_count}, {cls._CONTACT_VIEW.filter_count})"

    @classmethod
    def get_body_idx(cls, prim_path):
        """
        Returns:
            int: idx assigned to the rigid body defined by @prim_path
        """
        return cls._PATH_TO_IDX[prim_path]

    @classmethod
    def get_all_impulses(cls):
        """
        Grab all impulses at the current timestep

        Returns:
            n-array: (N, N, 3) impulse array defining current impulses between all N contact-sensor enabled rigid bodies
                in the simulator
        """
        # Generate the contact matrix if it doesn't already exist
        if cls._CONTACT_MATRIX is None:
            cls._CONTACT_MATRIX = cls._CONTACT_VIEW.get_contact_force_matrix(dt=1.0)

        return cls._CONTACT_MATRIX

    @classmethod
    def get_impulses(cls, prim_paths_a, prim_paths_b):
        """
        Grabs the matrix representing all impulse forces between rigid prims from @prim_paths_a and
        rigid prims from @prim_paths_b

        Args:
            prim_paths_a (list of str): Rigid body prim path(s) with which to grab contact impulses against
                any of the rigid body prim path(s) defined by @prim_paths_b
            prim_paths_b (list of str): Rigid body prim path(s) with which to grab contact impulses against
                any of the rigid body prim path(s) defined by @prim_paths_a

        Returns:
            n-array: (N, M, 3) impulse array defining current impulses between N bodies from @prim_paths_a and M bodies
                from @prim_paths_b
        """
        # Compute subset of matrix and return
        idxs_a = [cls._PATH_TO_IDX[path] for path in prim_paths_a]
        idxs_b = [cls._PATH_TO_IDX[path] for path in prim_paths_b]
        return cls.get_all_impulses()[idxs_a][:, idxs_b]

    @classmethod
    def in_contact(cls, prim_paths_a, prim_paths_b):
        """
        Check if any rigid prim from @prim_paths_a is in contact with any rigid prim from @prim_paths_b

        Args:
            prim_paths_a (list of str): Rigid body prim path(s) with which to check contact against any of the rigid
                body prim path(s) defined by @prim_paths_b
            prim_paths_b (list of str): Rigid body prim path(s) with which to check contact against any of the rigid
                body prim path(s) defined by @prim_paths_a

        Returns:
            bool: Whether any body from @prim_paths_a is in contact with any body from @prim_paths_b
        """
        # Check if the contact tuple already exists in the cache; if so, return the value
        key = (tuple(prim_paths_a), tuple(prim_paths_b))
        if key not in cls._CONTACT_CACHE:
            # In contact if any of the matrix values representing the interaction between the two groups is non-zero
            cls._CONTACT_CACHE[key] = np.any(cls.get_impulses(prim_paths_a=prim_paths_a, prim_paths_b=prim_paths_b))
        return cls._CONTACT_CACHE[key]

    @classmethod
    def clear(cls):
        """
        Clears the internal contact matrix and cache
        """
        cls._CONTACT_MATRIX = None
        cls._CONTACT_CACHE = dict()


class CollisionAPI:
    """
    Class containing class methods to facilitate collision handling, e.g. collision groups
    """
    ACTIVE_COLLISION_GROUPS = {}

    @classmethod
    def add_to_collision_group(cls, col_group, prim_path, create_if_not_exist=False):
        """
        Adds the prim and all nested prims specified by @prim_path to the global collision group @col_group. If @col_group
        does not exist, then it will either be created if @create_if_not_exist is True, otherwise will raise an Error.
        Args:
            col_group (str): Name of the collision group to assign the prim at @prim_path to
            prim_path (str): Prim (and all nested prims) to assign to this @col_group
            create_if_not_exist (bool): True if @col_group should be created if it does not already exist, otherwise an
                error will be raised
        """
        # TODO: This slows things down and / or crashes the sim with large number of objects. Skipping this for now, look into this later
        pass
        # # Check if collision group exists or not
        # if col_group not in cls.ACTIVE_COLLISION_GROUPS:
        #     # Raise error if we don't explicitly want to create a new group
        #     if not create_if_not_exist:
        #         raise ValueError(f"Collision group {col_group} not found in current registry, and create_if_not_exist"
        #                          f"was set to False!")
        #     # Otherwise, create the new group
        #     col_group_name = f"/World/collisionGroup_{col_group}"
        #     group = UsdPhysics.CollisionGroup.Define(get_current_stage(), col_group_name)
        #     group.GetFilteredGroupsRel().AddTarget(col_group_name)  # Make sure that we can collide within our own group
        #     cls.ACTIVE_COLLISION_GROUPS[col_group] = group
        #
        # # Add this prim to the collision group
        # cls.ACTIVE_COLLISION_GROUPS[col_group].GetCollidersCollectionAPI().GetIncludesRel().AddTarget(prim_path)

    @classmethod
    def clear(cls):
        """
        Clears the internal state of this CollisionAPI
        """
        cls.ACTIVE_COLLISION_GROUPS = {}


class BoundingBoxAPI:
    """
    Class containing class methods to facilitate bounding box handling
    """
    # Non-flatcache-compatible cache -- this is a direct omni API-based object
    CACHE_NON_FLATCACHE = None

    # Flatcache-compatible cache -- this is a dictionary mapping prim paths to corresponding AABBs
    CACHE_FLATCACHE = dict()

    @classmethod
    def compute_aabb(cls, prim):
        """
        Computes the AABB (world-frame oriented) for @prim.

        NOTE: If @prim is an EntityPrim (i.e.: owns multiple links), then the computed bounding box will be
        the subsequent aggregate over all the links.

        Args:
            prim (XFormPrim): Prim to calculate AABB for

        Returns:
            2-tuple:
                - 3-array: start (x,y,z) corner of world-coordinate frame aligned bounding box
                - 3-array: end (x,y,z) corner of world-coordinate frame aligned bounding box
        """
        # Use the correct API to calculate AABB based on whether flatcache is enabled or not
        return cls._compute_flatcache_aabb(prim=prim) if gm.ENABLE_FLATCACHE else \
            cls._compute_non_flatcache_aabb(prim_path=prim.prim_path)

    @classmethod
    def _compute_flatcache_aabb(cls, prim):
        """
        Computes the AABB (world-frame oriented) for @prim. This an API compatible with flatcache, which manually
        updates the @prim's transforms on the USD stage before computing its AABB

        Args:
            prim (XFormPrim): Prim to calculate AABB for

        Returns:
            2-tuple:
                - 3-array: start (x,y,z) corner of world-coordinate frame aligned bounding box
                - 3-array: end (x,y,z) corner of world-coordinate frame aligned bounding box
        """
        # Run imports here to avoid circular imports
        from omnigibson.prims import EntityPrim, RigidPrim, XFormPrim

        # Simply grab the AABB if it's already been cached
        if prim in cls.CACHE_FLATCACHE:
            return cls.CACHE_FLATCACHE[prim]

        # Next, process the AABB depending on the type of prim it is
        if isinstance(prim, EntityPrim):
            obj = prim
        elif isinstance(prim, RigidPrim):
            # Find the obj owning this link
            obj = og.sim.scene.object_registry("prim_path", "/".join(prim.prim_path.split("/")[:-1]))
        elif isinstance(prim, XFormPrim):
            # See if this XForm belongs to any object
            obj = og.sim.scene.object_registry("prim_path", "/".join(prim.prim_path.split("/")[:2]), None)
        else:
            raise ValueError(f"Inputted prim must be an instance of EntityPrim, RigidPrim, or XFormPrim "
                             f"in order to calculate AABB!")

        # Update tfs for the object that owns this prim
        if obj is not None:
            FlatcacheAPI.sync_raw_object_transforms_in_usd(prim=obj)

        # Compute the AABB and cache it internally
        val = cls._compute_non_flatcache_aabb(prim_path=prim.prim_path)
        cls.CACHE_FLATCACHE[prim] = val

        return val

    @classmethod
    def _compute_non_flatcache_aabb(cls, prim_path):
        """
        Computes the AABB (world-frame oriented) for the prim specified at @prim_path using the underlying omniverse
        API.

        NOTE: This is NOT compatible with flatcache and will result in incorrect values if flatcache is enabled!! See:
        https://docs.omniverse.nvidia.com/app_code/prod_extensions/ext_physics.html#physx-short-flatcache-also-known-as-fabric-rename-in-next-release

        Args:
            prim_path (str): Path to the prim to calculate AABB for

        Returns:
            2-tuple:
                - 3-array: start (x,y,z) corner of world-coordinate frame aligned bounding box
                - 3-array: end (x,y,z) corner of world-coordinate frame aligned bounding box
        """
        # Create cache if it doesn't already exist
        if cls.CACHE_NON_FLATCACHE is None:
            og.sim.psi.fetch_results()
            cls.CACHE_NON_FLATCACHE = create_bbox_cache(use_extents_hint=False)

        # Grab aabb
        aabb = compute_aabb(bbox_cache=cls.CACHE_NON_FLATCACHE, prim_path=prim_path)

        # Sanity check values
        if np.any(aabb[3:] < aabb[:3]):
            raise ValueError(f"Got invalid aabb values: low={aabb[:3]}, high={aabb[3:]}")

        return aabb[:3], aabb[3:]

    @classmethod
    def compute_center_extent(cls, prim):
        """
        Computes the AABB (world-frame oriented) for @prim, and convert it into the center and extent values

        Args:
            prim (XFormPrim): Prim to calculate AABB for

        Returns:
            2-tuple:
                - 3-array: center position (x,y,z) of world-coordinate frame aligned bounding box
                - 3-array: end-to-end extent size (x,y,z) of world-coordinate frame aligned bounding box
        """
        low, high = cls.compute_aabb(prim=prim)

        return (low + high) / 2.0, high - low

    @classmethod
    def clear(cls):
        """
        Clears the internal state of this BoundingBoxAPI. This should occur at least once per sim step.
        """
        cls.CACHE_NON_FLATCACHE = None
        cls.CACHE_FLATCACHE = dict()

    @classmethod
    def aabb_contains_point(cls, point, container):
        """
        Returns true if the point is contained in the container AABB

        Args:
            point (tuple): (x,y,z) position in world-coordinates
            container (tuple):
                - 3-array: start (x,y,z) corner of world-coordinate frame aligned bounding box
                - 3-array: end (x,y,z) corner of world-coordinate frame aligned bounding box

        Returns:
            bool: True if AABB contains @point, otherwise False
        """
        lower, upper = container
        return np.less_equal(lower, point).all() and np.less_equal(point, upper).all()


class FlatcacheAPI:
    """
    Monolithic class for leveraging functionality meant to be used EXCLUSIVELY with flatcache.
    """
    # Modified prims since transition from sim being stopped to sim being played occurred
    # This should get cleared every time og.sim.stop() gets called
    MODIFIED_PRIMS = set()

    @classmethod
    def sync_raw_object_transforms_in_usd(cls, prim):
        """
        Manually synchronizes the per-link local raw transforms per-joint raw states from entity prim @prim using
        dynamic control interface as the ground truth.

        NOTE: This slightly abuses the dynamic control - usd integration, and should ONLY be used if flatcache
        is active, since the USD is not R/W at runtime and so we can write directly to child link poses on the USD
        without breaking the simulation!

        Args:
            prim (EntityPrim): prim whose owned links and joints should have their raw local states updated to match the
                "true" values found from the dynamic control interface
        """
        # Make sure flatcache is enabled -- this should NEVER be called otherwise!!
        assert gm.ENABLE_FLATCACHE, "Syncing raw object transforms should only occur if flatcache is being used!"

        # We're somewhat abusing low-level dynamic control - physx - usd integration, but we (supposedly) know
        # what we're doing so we suppress logging so we don't see any error messages :D
        with suppress_omni_log(["omni.physx.plugin"]):
            # Import here to avoid circular imports
            from omnigibson.prims.xform_prim import XFormPrim

            # 1. For every link, update its xformOp properties based on the delta_tf between object frame and link frame
            obj_pos, obj_quat = XFormPrim.get_local_pose(prim)
            for link in prim.links.values():
                rel_pos, rel_quat = T.relative_pose_transform(*link.get_position_orientation(), obj_pos, obj_quat)
                XFormPrim.set_local_pose(link, rel_pos, rel_quat)
            # 2. For every joint, update its linear / angular joint state
            if prim.n_joints > 0:
                joints_pos = prim.get_joint_positions()
                for joint, joint_pos in zip(prim.joints.values(), joints_pos):
                    state_name = "linear" if joint.joint_type == JointType.JOINT_PRISMATIC else "angular"
                    joint_pos = joint_pos if joint.joint_type == JointType.JOINT_PRISMATIC else joint_pos * 180.0 / np.pi
                    joint.set_attribute(f"state:{state_name}:physics:position", float(joint_pos))

            # Update the simulation without taking any time
            # This is needed because physx complains that we're manually writing to child links' poses, and will
            # subsequently not respect any additional writes to the object pose before an additional step is taken.
            # So we take a "zero" length step so that any additional writes to the object's pose at the current
            # timestep are respected
            og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)

        # Add this prim to the set of modified prims
        cls.MODIFIED_PRIMS.add(prim)

    @classmethod
    def reset_raw_object_transforms_in_usd(cls, prim):
        """
        Manually resets the per-link local raw transforms and per-joint raw states from entity prim @prim to be zero.

        NOTE: This slightly abuses the dynamic control - usd integration, and should ONLY be used if flatcache
        is active, since the USD is not R/W at runtime and so we can write directly to child link poses on the USD
        without breaking the simulation!

        Args:
            prim (EntityPrim): prim whose owned links and joints should have their local values reset to be zero
        """
        # Make sure flatcache is enabled -- this should NEVER be called otherwise!!
        assert gm.ENABLE_FLATCACHE, "Resetting raw object transforms should only occur if flatcache is being used!"

        # We're somewhat abusing low-level dynamic control - physx - usd integration, but we (supposedly) know
        # what we're doing so we suppress logging so we don't see any error messages :D
        with suppress_omni_log(["omni.physx.plugin"]):
            # Import here to avoid circular imports
            from omnigibson.prims.xform_prim import XFormPrim

            # 1. For every link, update its xformOp properties to be 0
            for link in prim.links.values():
                XFormPrim.set_local_pose(link, np.zeros(3), np.array([0, 0, 0, 1.0]))
            # 2. For every joint, update its linear / angular joint state to be 0
            if prim.n_joints > 0:
                for joint in prim.joints.values():
                    state_name = "linear" if joint.joint_type == JointType.JOINT_PRISMATIC else "angular"
                    joint.set_attribute(f"state:{state_name}:physics:position", 0.0)

            # Update the simulation without taking any time
            # This is needed because physx complains that we're manually writing to child links' poses, and will
            # subsequently not respect any additional writes to the object pose before an additional step is taken.
            # So we take a "zero" length step so that any additional writes to the object's pose at the current
            # timestep are respected
            og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)

    @classmethod
    def reset(cls):
        """
        Resets the internal state of this FlatcacheAPI.This should only occur when the simulator is stopped
        """
        # For any prim transforms that were manually updated, we need to restore their original transforms
        for prim in cls.MODIFIED_PRIMS:
            cls.reset_raw_object_transforms_in_usd(prim)
        cls.MODIFIED_PRIMS = set()


def clear():
    """
    Clear state tied to singleton classes
    """
    CollisionAPI.clear()
    BoundingBoxAPI.clear()


def create_mesh_prim_with_default_xform(primitive_type, prim_path, u_patches=None, v_patches=None):
    """
    Creates a mesh prim of the specified @primitive_type at the specified @prim_path

    Args:
        primitive_type (str): Primitive mesh type, should be one of PRIMITIVE_MESH_TYPES to be valid
        prim_path (str): Destination prim path to store the mesh prim
        u_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            u-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
        v_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            v-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
            Both u_patches and v_patches need to be specified for them to be effective.
    """

    assert primitive_type in PRIMITIVE_MESH_TYPES, "Invalid primitive mesh type: {primitive_type}"
    evaluator = MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING[primitive_type]
    u_backup = carb.settings.get_settings().get(evaluator.SETTING_U_SCALE)
    v_backup = carb.settings.get_settings().get(evaluator.SETTING_V_SCALE)
    hs_backup = carb.settings.get_settings().get(evaluator.SETTING_OBJECT_HALF_SCALE)
    carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, 1)
    carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, 1)

    # Default half_scale (i.e. half-extent, half_height, radius) is 1.
    # TODO (eric): change it to 0.5 once the mesh generator API accepts floating-number HALF_SCALE
    #  (currently it only accepts integer-number and floors 0.5 into 0).
    carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, 1)
    if u_patches is not None and v_patches is not None:
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type=primitive_type,
            prim_path=prim_path,
            u_patches=u_patches,
            v_patches=v_patches,
        )
    else:
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type=primitive_type,
            prim_path=prim_path,
        )

    carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, u_backup)
    carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, v_backup)
    carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, hs_backup)


def mesh_prim_to_trimesh_mesh(mesh_prim, include_normals=True, include_texcoord=True):
    """
    Generates trimesh mesh from @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to convert into trimesh mesh
        include_normals (bool): Whether to include the normals in the resulting trimesh or not
        include_texcoord (bool): Whether to include the corresponding 2D-texture coordinates in the resulting
            trimesh or not

    Returns:
        trimesh.Trimesh: Generated trimesh mesh
    """
    face_vertex_counts = np.array(mesh_prim.GetAttribute("faceVertexCounts").Get())
    vertices = np.array(mesh_prim.GetAttribute("points").Get())
    face_indices = np.array(mesh_prim.GetAttribute("faceVertexIndices").Get())

    faces = []
    i = 0
    for count in face_vertex_counts:
        for j in range(count - 2):
            faces.append([face_indices[i], face_indices[i + j + 1], face_indices[i + j + 2]])
        i += count

    kwargs = dict(vertices=vertices, faces=faces)

    if include_normals:
        kwargs["vertex_normals"] = np.array(mesh_prim.GetAttribute("normals").Get())

    if include_texcoord:
        kwargs["visual"] = trimesh.visual.TextureVisuals(uv=np.array(mesh_prim.GetAttribute("primvars:st").Get()))

    return trimesh.Trimesh(**kwargs)


def sample_mesh_keypoints(mesh_prim, n_keypoints, n_keyfaces, deterministic=True):
    """
    Samples keypoints and keyfaces for mesh @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to be sampled from
        n_keypoints (int): number of (unique) keypoints to randomly sample from @mesh_prim
        n_keyfaces (int): number of (unique) keyfaces to randomly sample from @mesh_prim
        deterministic (bool): Whether to deterministically sample or not (ie: whether to set random seed or not)

    Returns:
        2-tuple:
            - n-array: (n,) 1D int array representing the randomly sampled point idxs from @mesh_prim.
                Note that since this is without replacement, the total length of the array may be less than
                @n_keypoints
            - None or n-array: 1D int array representing the randomly sampled face idxs from @mesh_prim.
                Note that since this is without replacement, the total length of the array may be less than
                @n_keyfaces
    """
    # Set seed if deterministic
    if deterministic:
        np.random.seed(0)

    # Generate trimesh mesh from which to aggregate points
    tm = mesh_prim_to_trimesh_mesh(mesh_prim=mesh_prim, include_normals=False, include_texcoord=False)
    n_unique_vertices, n_unique_faces = len(tm.vertices), len(tm.faces)
    faces_flat = tm.faces.flatten()
    n_vertices = len(faces_flat)

    # Sample vertices
    unique_vertices = np.unique(faces_flat, return_index=True)[1]
    assert len(unique_vertices) == n_unique_vertices
    keypoint_idx = np.random.choice(unique_vertices, size=n_keypoints, replace=False) if \
        n_unique_vertices > n_keypoints else unique_vertices

    # Sample faces
    keyface_idx = np.random.choice(n_unique_faces, size=n_keyfaces, replace=False) if \
        n_unique_faces > n_keyfaces else np.arange(n_unique_faces)

    return keypoint_idx, keyface_idx


def get_mesh_volume_and_com(mesh_prim):
    """
    Computes the volume and center of mass for @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to compute volume and center of mass for

    Returns:
        Tuple[float, np.array]: Tuple containing the volume and center of mass in the mesh frame of @mesh_prim
    """
    mesh_type = mesh_prim.GetPrimTypeInfo().GetTypeName()
    assert mesh_type in GEOM_TYPES, f"Invalid mesh type: {mesh_type}"
    # Default volume and com
    volume = 0.0
    com = np.zeros(3)
    if mesh_type == "Mesh":
        # We construct a trimesh object from this mesh in order to infer its volume
        trimesh_mesh = mesh_prim_to_trimesh_mesh(mesh_prim, include_normals=False, include_texcoord=False)
        if trimesh_mesh.is_volume:
            volume = trimesh_mesh.volume
            com = trimesh_mesh.center_mass
        elif trimesh_mesh.vertices.shape[0] >= 4:  # At least 4 vertices are needed for computing the convex hull
            # If the mesh is not a volume, we compute its convex hull and use that instead
            trimesh_mesh_convex = trimesh_mesh.convex_hull
            volume = trimesh_mesh_convex.volume
            com = trimesh_mesh_convex.center_mass
        else:
            # Use the default volume and com
            pass
    elif mesh_type == "Sphere":
        volume = 4 / 3 * np.pi * (mesh_prim.GetAttribute("radius").Get() ** 3)
    elif mesh_type == "Cube":
        volume = mesh_prim.GetAttribute("size").Get() ** 3
    elif mesh_type == "Cone":
        volume = np.pi * (mesh_prim.GetAttribute("radius").Get() ** 2) * mesh_prim.GetAttribute("height").Get() / 3
        com = np.array([0, 0, mesh_prim.GetAttribute("height").Get() / 4])
    elif mesh_type == "Cylinder":
        volume = np.pi * (mesh_prim.GetAttribute("radius").Get() ** 2) * mesh_prim.GetAttribute("height").Get()
    else:
        raise ValueError(f"Cannot compute volume for mesh of type: {mesh_type}")

    return volume, com


def create_primitive_mesh(prim_path, primitive_type, extents=1.0, u_patches=None, v_patches=None):
    """
    Helper function that generates a UsdGeom.Mesh prim at specified @prim_path of type @primitive_type.

    NOTE: Generated mesh prim will, by default, have extents equaling [1, 1, 1]

    Args:
        prim_path (str): Where the loaded mesh should exist on the stage
        primitive_type (str): Type of primitive mesh to create. Should be one of:
            {"Cone", "Cube", "Cylinder", "Disk", "Plane", "Sphere", "Torus"}
        extents (float or 3-array): Specifies the extents of the generated mesh. Default is 1.0, i.e.:
            generated mesh will be in be contained in a [1,1,1] sized bounding box
        u_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            u-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
        v_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            v-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
            Both u_patches and v_patches need to be specified for them to be effective.

    Returns:
        UsdGeom.Mesh: Generated primitive mesh as a prim on the active stage
    """
    assert_valid_key(key=primitive_type, valid_keys=PRIMITIVE_MESH_TYPES, name="primitive mesh type")
    create_mesh_prim_with_default_xform(primitive_type, prim_path, u_patches=u_patches, v_patches=v_patches)
    mesh = UsdGeom.Mesh.Define(og.sim.stage, prim_path)

    # Modify the points and normals attributes so that total extents is the desired
    # This means multiplying omni's default by extents * 50.0, as the native mesh generated has extents [-0.01, 0.01]
    # -- i.e.: 2cm-wide mesh
    extents = np.ones(3) * extents if isinstance(extents, float) else np.array(extents)
    for attr in (mesh.GetPointsAttr(), mesh.GetNormalsAttr()):
        vals = np.array(attr.Get()).astype(np.float64)
        attr.Set(Vt.Vec3fArray([Gf.Vec3f(*(val * extents * 50.0)) for val in vals]))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*(-extents / 2.0)), Gf.Vec3f(*(extents / 2.0))]))

    return mesh


def add_asset_to_stage(asset_path, prim_path):
    """
    Adds asset file (either USD or OBJ) at @asset_path at the location @prim_path

    Args:
        asset_path (str): Absolute or relative path to the asset file to load
        prim_path (str): Where loaded asset should exist on the stage

    Returns:
        Usd.Prim: Loaded prim as a USD prim
    """
    # Make sure this is actually a supported asset type
    assert asset_path[-4:].lower() in {".usd", ".obj"}, f"Cannot load a non-USD or non-OBJ file as a USD prim!"
    asset_type = asset_path[-3:]

    # Make sure the path exists
    assert os.path.exists(asset_path), f"Cannot load {asset_type.upper()} file {asset_path} because it does not exist!"

    # Add reference to stage and grab prim
    add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
    prim = get_prim_at_path(prim_path)

    # Make sure prim was loaded correctly
    assert prim, f"Failed to load {asset_type.upper()} object from path: {asset_path}"

    return prim


def get_world_prim():
    """
    Returns:
        Usd.Prim: Active world prim in the current stage
    """
    return get_prim_at_path("/World")
