import collections
import math
import os
from collections.abc import Iterable

import numpy as np
import trimesh

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.utils.constants import PRIMITIVE_MESH_TYPES, JointType, PrimType
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.ui_utils import create_module_logger, suppress_omni_log

# Create module logger
log = create_module_logger(module_name=__name__)


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
    GF_TO_VT_MAPPING = {
        lazy.pxr.Gf.Vec3d: lazy.pxr.Vt.Vec3dArray,
        lazy.pxr.Gf.Vec3f: lazy.pxr.Vt.Vec3fArray,
        lazy.pxr.Gf.Vec3h: lazy.pxr.Vt.Vec3hArray,
        lazy.pxr.Gf.Quatd: lazy.pxr.Vt.QuatdArray,
        lazy.pxr.Gf.Quatf: lazy.pxr.Vt.QuatfArray,
        lazy.pxr.Gf.Quath: lazy.pxr.Vt.QuathArray,
        int: lazy.pxr.Vt.IntArray,
        float: lazy.pxr.Vt.FloatArray,
        bool: lazy.pxr.Vt.BoolArray,
        str: lazy.pxr.Vt.StringArray,
        chr: lazy.pxr.Vt.CharArray,
    }

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
    for child in lazy.omni.isaac.core.utils.prims.get_prim_children(prim):
        prims.append(child)
        prims += get_prim_nested_children(prim=child)

    return prims


def create_joint(
    prim_path,
    joint_type,
    body0=None,
    body1=None,
    enabled=True,
    joint_frame_in_parent_frame_pos=None,
    joint_frame_in_parent_frame_quat=None,
    joint_frame_in_child_frame_pos=None,
    joint_frame_in_child_frame_quat=None,
    break_force=None,
    break_torque=None,
):
    """
    Creates a joint between @body0 and @body1 of specified type @joint_type

    Args:
        prim_path (str): absolute path to where the joint will be created
        joint_type (str or JointType): type of joint to create. Valid options are:
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
    assert JointType.is_valid(joint_type=joint_type), f"Invalid joint specified for creation: {joint_type}"

    # Make sure at least body0 or body1 is specified
    assert (
        body0 is not None or body1 is not None
    ), f"At least either body0 or body1 must be specified when creating a joint!"

    # Create the joint
    joint = getattr(lazy.pxr.UsdPhysics, joint_type).Define(og.sim.stage, prim_path)

    # Possibly add body0, body1 targets
    if body0 is not None:
        assert lazy.omni.isaac.core.utils.prims.is_prim_path_valid(body0), f"Invalid body0 path specified: {body0}"
        joint.GetBody0Rel().SetTargets([lazy.pxr.Sdf.Path(body0)])
    if body1 is not None:
        assert lazy.omni.isaac.core.utils.prims.is_prim_path_valid(body1), f"Invalid body1 path specified: {body1}"
        joint.GetBody1Rel().SetTargets([lazy.pxr.Sdf.Path(body1)])

    # Get the prim pointed to at this path
    joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)

    # Apply joint API interface
    lazy.pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)

    # We need to step rendering once to auto-fill the local pose before overwriting it.
    # Note that for some reason, if multi_gpu is used, this line will crash if create_joint is called during on_contact
    # callback, e.g. when an attachment joint is being created due to contacts.
    og.sim.render()

    if joint_frame_in_parent_frame_pos is not None:
        joint_prim.GetAttribute("physics:localPos0").Set(lazy.pxr.Gf.Vec3f(*joint_frame_in_parent_frame_pos))
    if joint_frame_in_parent_frame_quat is not None:
        joint_prim.GetAttribute("physics:localRot0").Set(
            lazy.pxr.Gf.Quatf(*joint_frame_in_parent_frame_quat[[3, 0, 1, 2]])
        )
    if joint_frame_in_child_frame_pos is not None:
        joint_prim.GetAttribute("physics:localPos1").Set(lazy.pxr.Gf.Vec3f(*joint_frame_in_child_frame_pos))
    if joint_frame_in_child_frame_quat is not None:
        joint_prim.GetAttribute("physics:localRot1").Set(
            lazy.pxr.Gf.Quatf(*joint_frame_in_child_frame_quat[[3, 0, 1, 2]])
        )

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


class RigidContactAPIImpl:
    """
    Class containing class methods to aggregate rigid body contacts across all rigid bodies in the simulator
    """

    def __init__(self):
        self._PATH_TO_SCENE_IDX = dict()

        # Dictionary mapping rigid body prim path to corresponding index in the contact view matrix
        self._PATH_TO_ROW_IDX = dict()
        self._PATH_TO_COL_IDX = dict()

        # Numpy array of rigid body prim paths where its array index directly corresponds to the corresponding
        # index in the contact view matrix
        self._ROW_IDX_TO_PATH = dict()
        self._COL_IDX_TO_PATH = dict()

        # Contact view for generating contact matrices at each timestep
        self._CONTACT_VIEW = dict()

        # Current aggregated contacts over all rigid bodies at the current timestep. Shape: (N, N, 3)
        self._CONTACT_MATRIX = dict()

        # Current contact data cache containing forces, points, normals, separations, contact_counts, start_indices
        self._CONTACT_DATA = dict()

        # Current cache, mapping 2-tuple (prim_paths_a, prim_paths_b) to contact values
        self._CONTACT_CACHE = None

    @classmethod
    def get_row_filter(cls):
        return "/World/*/*"

    @classmethod
    def get_column_filters(cls):
        filters = dict()
        for scene_idx, scene in enumerate(og.sim.scenes):
            filters[scene_idx] = []
            for obj in scene.objects:
                if obj.prim_type == PrimType.RIGID:
                    for link in obj.links.values():
                        if not link.kinematic_only:
                            filters[scene_idx].append(link.prim_path)

        return filters

    @classmethod
    def get_max_contact_data_count(cls):
        return 0

    def initialize_view(self):
        """
        Initializes the rigid contact view. Note: Can only be done when sim is playing!
        """
        assert og.sim.is_playing(), "Cannot create rigid contact view while sim is not playing!"

        # Compile deterministic mapping from rigid body path to idx
        # Note that omni's ordering is based on the top-down object ordering path on the USD stage, which coincidentally
        # matches the same ordering we store objects in our registry. So the mapping we generate from our registry
        # mapping aligns with omni's ordering!
        column_filters = self.get_column_filters()
        for scene_idx, filters in column_filters.items():
            self._PATH_TO_COL_IDX[scene_idx] = dict()
            for i, link_path in enumerate(filters):
                self._PATH_TO_COL_IDX[scene_idx][link_path] = i
                self._PATH_TO_SCENE_IDX[link_path] = scene_idx

        # If there are no valid objects, clear the view and terminate early
        if len(column_filters) == 0:
            self._CONTACT_VIEW = dict()
            return

        # Generate rigid body view, making sure to update the simulation first (without physics) so that the physx
        # backend is synchronized with any newly added objects
        # We also suppress the omni tensor plugin from giving warnings we expect
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            for scene_idx, _ in enumerate(og.sim.scenes):
                self._CONTACT_VIEW[scene_idx] = og.sim.physics_sim_view.create_rigid_contact_view(
                    pattern=self.get_row_filter(),
                    filter_patterns=column_filters[scene_idx],
                    max_contact_data_count=self.get_max_contact_data_count(),
                )

        # Create deterministic mapping from path to row index
        for scene_idx, _ in enumerate(og.sim.scenes):
            self._PATH_TO_ROW_IDX[scene_idx] = {
                path: i for i, path in enumerate(self._CONTACT_VIEW[scene_idx].sensor_paths)
            }

        # Store the reverse mappings as well. This can just be a numpy array since the mapping uses integer indices
        for scene_idx, _ in enumerate(og.sim.scenes):
            self._ROW_IDX_TO_PATH[scene_idx] = np.array(list(self._PATH_TO_ROW_IDX[scene_idx].keys()))
            self._COL_IDX_TO_PATH[scene_idx] = np.array(list(self._PATH_TO_COL_IDX[scene_idx].keys()))

        # Sanity check generated view -- this should generate square matrices of shape (N, N, 3)
        # n_bodies = len(cls._PATH_TO_COL_IDX)
        # assert cls._CONTACT_VIEW.filter_count == n_bodies, \
        #     f"Got unexpected contact view shape. Expected: (N, {n_bodies}); " \
        #     f"got: (N, {cls._CONTACT_VIEW.filter_count})"

    def get_scene_idx(self, prim_path):
        """
        Returns:
            int: scene idx for the rigid body defined by @prim_path
        """
        return self._PATH_TO_SCENE_IDX[prim_path]

    def get_body_row_idx(self, prim_path):
        """
        Returns:
            int: row idx assigned to the rigid body defined by @prim_path
        """
        scene_idx = self._PATH_TO_SCENE_IDX[prim_path]
        return scene_idx, self._PATH_TO_ROW_IDX[scene_idx][prim_path]

    def get_body_col_idx(self, prim_path):
        """
        Returns:
            int: col idx assigned to the rigid body defined by @prim_path
        """
        scene_idx = self._PATH_TO_SCENE_IDX[prim_path]
        return scene_idx, self._PATH_TO_COL_IDX[scene_idx][prim_path]

    def get_row_idx_prim_path(self, scene_idx, idx):
        """
        Returns:
            str: @prim_path corresponding to the row idx @idx in the contact matrix
        """
        return self._ROW_IDX_TO_PATH[scene_idx][idx]

    def get_col_idx_prim_path(self, scene_idx, idx):
        """
        Returns:
            str: @prim_path corresponding to the column idx @idx in the contact matrix
        """
        return self._COL_IDX_TO_PATH[scene_idx][idx]

    def get_all_impulses(self, scene_idx):
        """
        Grab all impulses at the current timestep

        Returns:
            n-array: (N, M, 3) impulse array defining current impulses between all N contact-sensor enabled rigid bodies
                in the simulator and M tracked rigid bodies
        """
        # Generate the contact matrix if it doesn't already exist
        if scene_idx not in self._CONTACT_MATRIX:
            self._CONTACT_MATRIX[scene_idx] = self._CONTACT_VIEW[scene_idx].get_contact_force_matrix(dt=1.0)

        return self._CONTACT_MATRIX[scene_idx]

    def get_impulses(self, prim_paths_a, prim_paths_b):
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
        scene_idx = self._PATH_TO_SCENE_IDX[prim_paths_a[0]]
        idxs_a = [self._PATH_TO_ROW_IDX[scene_idx][path] for path in prim_paths_a]
        idxs_b = [self._PATH_TO_COL_IDX[scene_idx][path] for path in prim_paths_b]
        return self.get_all_impulses(scene_idx)[idxs_a][:, idxs_b]

    def get_contact_data(self, prim_path):
        scene_idx, row_idx = self.get_body_row_idx(prim_path)
        # First check if the object has any contacts
        impulses = self.get_all_impulses(scene_idx)
        if not np.any(impulses[row_idx] > 0):
            return []

        # Get the contact targets' prim paths
        col_idxs = np.nonzero(impulses[row_idx] > 0)[0]
        col_paths = [self.get_col_idx_prim_path(scene_idx, idx) for idx in col_idxs]

        # Get the contact data
        if scene_idx not in self._CONTACT_DATA:
            self._CONTACT_DATA[scene_idx] = self._CONTACT_VIEW[scene_idx].get_contact_data(dt=1.0)

        # Get the contact data for this prim
        forces, points, normals, separations, contact_counts, start_indices = self._CONTACT_DATA[scene_idx]
        start_idx = start_indices[row_idx]
        contact_count = contact_counts[row_idx]
        end_idx = start_idx + contact_count

        # Assert that one of two things is true: either the prim count and contact count are equal,
        # in which case we can zip them together, or the prim count is 1, in which case we can just
        # repeat the single prim data for all contacts. Otherwise, it is not clear which contacts are
        # happening between which two objects, so we return no contacts while printing an error.
        if len(col_paths) == contact_count:
            return list(
                zip(
                    col_paths,
                    forces[start_idx:end_idx],
                    points[start_idx:end_idx],
                    normals[start_idx:end_idx],
                    separations[start_idx:end_idx],
                )
            )
        elif len(col_paths) == 1:
            return [
                (col_paths[0], force, point, normal, separation)
                for force, point, normal, separation in zip(
                    forces[start_idx:end_idx],
                    points[start_idx:end_idx],
                    normals[start_idx:end_idx],
                    separations[start_idx:end_idx],
                )
            ]

        log.warning(
            f"Could not disambiguate which contacts are happening with which object for prim {prim_path}! Returning no contacts."
        )
        return []

    def get_contact_data_from_columns(self, scene_idx, col_paths):
        # First, find all of the rows that the prim is in contact with
        impulses = self.get_all_impulses(scene_idx)
        col_idx = [self.get_body_col_idx(prim_path) for prim_path in col_paths]
        if not np.any(impulses[:, col_idx] > 0):
            return []

        # Get the contact targets' prim paths
        row_idxs = np.nonzero(np.any(impulses[:, col_idx] > 0, axis=1))[0]
        row_paths = [self.get_row_idx_prim_path(scene_idx, idx) for idx in row_idxs]

        # Accumulate contacts for each row
        return [
            # TODO: Is it true that only the normal needs to be negated?
            (row_path, col_path, force, point, -normal, separation)
            for row_path in row_paths
            for col_path, force, point, normal, separation in self.get_contact_data(row_path)
            if col_path in col_paths
        ]

    def in_contact(self, prim_paths_a, prim_paths_b):
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
        if key not in self._CONTACT_CACHE:
            # In contact if any of the matrix values representing the interaction between the two groups is non-zero
            self._CONTACT_CACHE[key] = np.any(self.get_impulses(prim_paths_a=prim_paths_a, prim_paths_b=prim_paths_b))
        return self._CONTACT_CACHE[key]

    def clear(self):
        """
        Clears the internal contact matrix and cache
        """
        self._CONTACT_MATRIX = dict()
        self._CONTACT_DATA = dict()
        self._CONTACT_CACHE = dict()


# Instantiate the RigidContactAPI
RigidContactAPI = RigidContactAPIImpl()


class GripperRigidContactAPIImpl(RigidContactAPIImpl):
    @classmethod
    def get_column_filters(cls):
        from omnigibson.robots.manipulation_robot import ManipulationRobot

        filters = dict()
        for scene_idx, scene in enumerate(og.sim.scenes):
            filters[scene_idx] = []
            for robot in scene.robots:
                if isinstance(robot, ManipulationRobot):
                    filters[scene_idx].extend(link.prim_path for links in robot.finger_links.values() for link in links)

        return filters

    @classmethod
    def get_max_contact_data_count(cls):
        # 2x per finger link, to be safe.
        return len(cls.get_column_filters()[0]) * 2


# Instantiate the GripperRigidContactAPI
GripperRigidContactAPI = GripperRigidContactAPIImpl()


class CollisionAPI:
    """
    Class containing class methods to facilitate collision handling, e.g. collision groups
    """

    ACTIVE_COLLISION_GROUPS = dict()

    @classmethod
    def create_collision_group(cls, col_group, filter_self_collisions=False):
        """
        Creates a new collision group with name @col_group

        Args:
            col_group (str): Name of the collision group to create
            filter_self_collisions (bool): Whether to ignore self-collisions within the group. Default is False
        """
        # Can only be done when sim is stopped
        assert og.sim is None or og.sim.is_stopped(), "Cannot create a collision group unless og.sim is stopped!"

        # Make sure the group doesn't already exist
        assert (
            col_group not in cls.ACTIVE_COLLISION_GROUPS
        ), f"Cannot create collision group {col_group} because it already exists!"

        # Create the group
        col_group_prim_path = f"/World/collision_groups/{col_group}"
        group = lazy.pxr.UsdPhysics.CollisionGroup.Define(og.sim.stage, col_group_prim_path)
        if filter_self_collisions:
            # Do not collide with self
            group.GetFilteredGroupsRel().AddTarget(col_group_prim_path)
        cls.ACTIVE_COLLISION_GROUPS[col_group] = group

    @classmethod
    def add_to_collision_group(cls, col_group, prim_path):
        """
        Adds the prim and all nested prims specified by @prim_path to the global collision group @col_group. If @col_group
        does not exist, then it will either be created if @create_if_not_exist is True, otherwise will raise an Error.
        Args:
            col_group (str): Name of the collision group to assign the prim at @prim_path to
            prim_path (str): Prim (and all nested prims) to assign to this @col_group
        """
        # Make sure collision group exists
        assert (
            col_group in cls.ACTIVE_COLLISION_GROUPS
        ), f"Cannot add to collision group {col_group} because it does not exist!"

        # Add this prim to the collision group
        cls.ACTIVE_COLLISION_GROUPS[col_group].GetCollidersCollectionAPI().GetIncludesRel().AddTarget(prim_path)

    @classmethod
    def add_group_filter(cls, col_group, filter_group):
        """
        Adds a new group filter for group @col_group, filtering all collision with group @filter_group
        Args:
            col_group (str): Name of the collision group which will have a new filter group added
            filter_group (str): Name of the group that should be filtered
        """
        # Make sure the group doesn't already exist
        for group_name in (col_group, filter_group):
            assert group_name in cls.ACTIVE_COLLISION_GROUPS, (
                f"Cannot add group filter {filter_group} to collision group {col_group} because at least one group "
                f"does not exist!"
            )

        # Grab the group, and add the filter
        filter_group_prim_path = f"/World/collision_groups/{filter_group}"
        group = cls.ACTIVE_COLLISION_GROUPS[col_group]
        group.GetFilteredGroupsRel().AddTarget(filter_group_prim_path)

    @classmethod
    def clear(cls):
        """
        Clears the internal state of this CollisionAPI
        """
        cls.ACTIVE_COLLISION_GROUPS = {}


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
                    joint_pos = (
                        joint_pos if joint.joint_type == JointType.JOINT_PRISMATIC else joint_pos * 180.0 / np.pi
                    )
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


class PoseAPI:
    """
    This is a singleton class for getting world poses.
    Whenever we directly set the pose of a prim, we should call PoseAPI.invalidate().
    After that, if we need to access the pose of a prim without stepping physics,
    this class will refresh the poses by syncing across USD-fabric-PhysX depending on the flatcache setting.
    """

    VALID = False

    @classmethod
    def invalidate(cls):
        cls.VALID = False

    @classmethod
    def mark_valid(cls):
        cls.VALID = True

    @classmethod
    def _refresh(cls):
        if og.sim is not None and not cls.VALID:
            # when flatcache is on
            if og.sim._physx_fabric_interface:
                # no time step is taken here
                og.sim._physx_fabric_interface.update(og.sim.get_physics_dt(), og.sim.current_time)
            # when flatcache is off
            else:
                # no time step is taken here
                og.sim.psi.fetch_results()
            cls.mark_valid()

    @classmethod
    def get_world_pose(cls, prim_path):
        cls._refresh()
        position, orientation = lazy.omni.isaac.core.utils.xforms.get_world_pose(prim_path)
        return np.array(position), np.array(orientation)[[1, 2, 3, 0]]

    @classmethod
    def get_world_pose_with_scale(cls, prim_path):
        """
        This is used when information about the prim's global scale is needed,
        e.g. when converting points in the prim frame to the world frame.
        """
        cls._refresh()
        return np.array(lazy.omni.isaac.core.utils.xforms._get_world_pose_transform_w_scale(prim_path)).T


class ControllableObjectViewAPI:
    """
    A centralized view that allows for reading and writing to an ArticulationView that covers all
    controllable objects in the scene. This is used to avoid the overhead of reading from many views
    for each robot in each physics step, a source of significant overhead.
    """

    # The unified ArticulationView used to access all of the controllable objects in the scene.
    _VIEW = None

    # Cache for all of the view functions' return values within the same simulation step.
    # Keyed by function name without get_, the value is the return value of the function.
    _READ_CACHE = {}

    # Cache for all of the view functions' write values within the same simulation step.
    # Keyed by the function name without set_, the value is the set of indices that need to be updated.
    _WRITE_IDX_CACHE = collections.defaultdict(set)

    # Mapping from prim path to index in the view.
    _IDX = {}

    # Mapping from prim idx to a dict that maps link name to link index in the view.
    _LINK_IDX = {}

    @classmethod
    def clear(cls):
        cls._READ_CACHE = {}
        cls._WRITE_IDX_CACHE = collections.defaultdict(set)

    @classmethod
    def flush_control(cls):
        if "dof_position_targets" in cls._WRITE_IDX_CACHE:
            pos_indices = np.array(sorted(cls._WRITE_IDX_CACHE["dof_position_targets"]))
            pos_targets = cls._READ_CACHE["dof_position_targets"]
            cls._VIEW.set_dof_position_targets(pos_targets, np.array(pos_indices))

        if "dof_velocity_targets" in cls._WRITE_IDX_CACHE:
            vel_indices = np.array(sorted(cls._WRITE_IDX_CACHE["dof_velocity_targets"]))
            vel_targets = cls._READ_CACHE["dof_velocity_targets"]
            cls._VIEW.set_dof_velocity_targets(vel_targets, np.array(vel_indices))

        if "dof_actuation_forces" in cls._WRITE_IDX_CACHE:
            eff_indices = np.array(sorted(cls._WRITE_IDX_CACHE["dof_actuation_forces"]))
            eff_targets = cls._READ_CACHE["dof_actuation_forces"]
            cls._VIEW.set_dof_actuation_forces(eff_targets, np.array(eff_indices))

    @classmethod
    def initialize_view(cls):
        # First, get all of the controllable objects in the scene (avoiding circular import)
        from omnigibson.objects.controllable_object import ControllableObject

        controllable_objects = [
            obj for scene in og.sim.scenes for obj in scene.objects if isinstance(obj, ControllableObject)
        ]

        # This only works if the root link is called base_link for every controllable object, so assert that
        assert all(
            co.root_link.prim_path.endswith("/base_link") for co in controllable_objects
        ), "Controllable objects must have a link named base_link as the root link."

        # Get their corresponding prim paths
        expected_regular_prim_paths = {obj.articulation_root_path for obj in controllable_objects}
        expected_dummy_prim_paths = {
            obj._dummy.articulation_root_path
            for obj in controllable_objects
            if hasattr(obj, "_dummy") and obj._dummy is not None
        }
        expected_prim_paths = expected_regular_prim_paths | expected_dummy_prim_paths

        # Make sure we have at least one controllable object
        if len(expected_prim_paths) == 0:
            return

        # Create the actual articulation view. Note that even though we search for base_link here,
        # the returned things will not necessarily be the base_link prim paths, but the appropriate
        # articulation root path for every object (base_link for non-fixed, parent for fixed objects)
        cls._VIEW = og.sim.physics_sim_view.create_articulation_view("/World/controllable_*/base_link")
        view_prim_paths = cls._VIEW.prim_paths
        assert (
            set(view_prim_paths) == expected_prim_paths
        ), f"ControllableObjectViewAPI expected prim paths {expected_prim_paths} but got {view_prim_paths}"

        # Create the mapping from prim path to index
        cls._IDX = {prim_path: i for i, prim_path in enumerate(view_prim_paths)}
        cls._LINK_IDX = [
            {link_path.split("/")[-1]: j for j, link_path in enumerate(articulation_link_paths)}
            for articulation_link_paths in cls._VIEW.link_paths
        ]

    @classmethod
    def set_joint_position_targets(cls, prim_path, positions, indices):
        assert len(indices) == len(positions), "Indices and values must have the same length"
        idx = cls._IDX[prim_path]

        # Load the current targets.
        if "dof_position_targets" not in cls._READ_CACHE:
            cls._READ_CACHE["dof_position_targets"] = cls._VIEW.get_dof_position_targets()

        # Update the target
        cls._READ_CACHE["dof_position_targets"][idx][indices] = positions

        # Add this index to the write cache
        cls._WRITE_IDX_CACHE["dof_position_targets"].add(idx)

    @classmethod
    def set_joint_velocity_targets(cls, prim_path, velocities, indices):
        assert len(indices) == len(velocities), "Indices and values must have the same length"
        idx = cls._IDX[prim_path]

        # Load the current targets.
        if "dof_velocity_targets" not in cls._READ_CACHE:
            cls._READ_CACHE["dof_velocity_targets"] = cls._VIEW.get_dof_velocity_targets()

        # Update the target
        cls._READ_CACHE["dof_velocity_targets"][idx][indices] = velocities

        # Add this index to the write cache
        cls._WRITE_IDX_CACHE["dof_velocity_targets"].add(idx)

    @classmethod
    def set_joint_efforts(cls, prim_path, efforts, indices):
        assert len(indices) == len(efforts), "Indices and values must have the same length"
        idx = cls._IDX[prim_path]

        # Load the current targets.
        if "dof_actuation_forces" not in cls._READ_CACHE:
            cls._READ_CACHE["dof_actuation_forces"] = cls._VIEW.get_dof_actuation_forces()

        # Update the target
        cls._READ_CACHE["dof_actuation_forces"][idx][indices] = efforts

        # Add this index to the write cache
        cls._WRITE_IDX_CACHE["dof_actuation_forces"].add(idx)

    @classmethod
    def get_position_orientation(cls, prim_path):
        if "root_transforms" not in cls._READ_CACHE:
            cls._READ_CACHE["root_transforms"] = cls._VIEW.get_root_transforms()

        idx = cls._IDX[prim_path]
        pose = cls._READ_CACHE["root_transforms"][idx]
        return pose[:3], pose[3:]

    @classmethod
    def get_linear_velocity(cls, prim_path):
        if "root_velocities" not in cls._READ_CACHE:
            cls._READ_CACHE["root_velocities"] = cls._VIEW.get_root_velocities()

        idx = cls._IDX[prim_path]
        return cls._READ_CACHE["root_velocities"][idx][:3]

    @classmethod
    def get_angular_velocity(cls, prim_path):
        if "root_velocities" not in cls._READ_CACHE:
            cls._READ_CACHE["root_velocities"] = cls._VIEW.get_root_velocities()

        idx = cls._IDX[prim_path]
        return cls._READ_CACHE["root_velocities"][idx][3:]

    @classmethod
    def get_relative_linear_velocity(cls, prim_path):
        orn = cls.get_position_orientation(prim_path)[1]
        linvel = cls.get_linear_velocity(prim_path)
        return T.quat2mat(orn).T @ linvel

    @classmethod
    def get_relative_angular_velocity(cls, prim_path):
        orn = cls.get_position_orientation(prim_path)[1]
        angvel = cls.get_angular_velocity(prim_path)
        return T.mat2euler(T.quat2mat(orn).T @ T.euler2mat(angvel))

    @classmethod
    def get_joint_positions(cls, prim_path):
        if "dof_positions" not in cls._READ_CACHE:
            cls._READ_CACHE["dof_positions"] = cls._VIEW.get_dof_positions()

        idx = cls._IDX[prim_path]
        return cls._READ_CACHE["dof_positions"][idx]

    @classmethod
    def get_joint_velocities(cls, prim_path):
        if "dof_velocities" not in cls._READ_CACHE:
            cls._READ_CACHE["dof_velocities"] = cls._VIEW.get_dof_velocities()

        idx = cls._IDX[prim_path]
        return cls._READ_CACHE["dof_velocities"][idx]

    @classmethod
    def get_joint_efforts(cls, prim_path):
        if "dof_projected_joint_forces" not in cls._READ_CACHE:
            cls._READ_CACHE["dof_projected_joint_forces"] = cls._VIEW.get_dof_projected_joint_forces()

        idx = cls._IDX[prim_path]
        return cls._READ_CACHE["dof_projected_joint_forces"][idx]

    @classmethod
    def get_mass_matrix(cls, prim_path):
        if "mass_matrices" not in cls._READ_CACHE:
            cls._READ_CACHE["mass_matrices"] = cls._VIEW.get_mass_matrices()

        idx = cls._IDX[prim_path]
        # TODO: Maybe do the shape correction here. physics_view.mass_matrix_shape has it.
        return cls._READ_CACHE["mass_matrices"][idx]

    @classmethod
    def get_generalized_gravity_forces(cls, prim_path):
        if "generalized_gravity_forces" not in cls._READ_CACHE:
            cls._READ_CACHE["generalized_gravity_forces"] = cls._VIEW.get_generalized_gravity_forces()

        idx = cls._IDX[prim_path]
        return cls._READ_CACHE["generalized_gravity_forces"][idx]

    @classmethod
    def get_coriolis_and_centrifugal_forces(cls, prim_path):
        if "coriolis_and_centrifugal_forces" not in cls._READ_CACHE:
            cls._READ_CACHE["coriolis_and_centrifugal_forces"] = cls._VIEW.get_coriolis_and_centrifugal_forces()

        idx = cls._IDX[prim_path]
        return cls._READ_CACHE["coriolis_and_centrifugal_forces"][idx]

    @classmethod
    def get_link_relative_position_orientation(cls, prim_path, link_name):
        if "link_transforms" not in cls._READ_CACHE:
            cls._READ_CACHE["link_transforms"] = cls._VIEW.get_link_transforms()

        idx = cls._IDX[prim_path]
        link_idx = cls._LINK_IDX[idx][link_name]
        pose = cls._READ_CACHE["link_transforms"][idx][link_idx]
        pos, orn = pose[:3], pose[3:]

        # Get the root world transform too
        world_pos, world_orn = cls.get_position_orientation(prim_path)

        # Compute the relative position and orientation
        return T.relative_pose_transform(pos, orn, world_pos, world_orn)

    @classmethod
    def get_link_relative_linear_velocity(cls, prim_path, link_name):
        if "link_velocities" not in cls._READ_CACHE:
            cls._READ_CACHE["link_velocities"] = cls._VIEW.get_link_velocities()

        idx = cls._IDX[prim_path]
        link_idx = cls._LINK_IDX[idx][link_name]
        vel = cls._READ_CACHE["link_velocities"][idx][link_idx]
        linvel = vel[:3]

        # Get the root world transform too
        _, world_orn = cls.get_position_orientation(prim_path)

        # Compute the relative position and orientation
        return T.quat2mat(world_orn).T @ linvel

    @classmethod
    def get_link_relative_angular_velocity(cls, prim_path, link_name):
        if "link_velocities" not in cls._READ_CACHE:
            cls._READ_CACHE["link_velocities"] = cls._VIEW.get_link_velocities()

        idx = cls._IDX[prim_path]
        link_idx = cls._LINK_IDX[idx][link_name]
        vel = cls._READ_CACHE["link_velocities"][idx][link_idx]
        angvel = vel[3:]

        # Get the root world transform too
        _, world_orn = cls.get_position_orientation(prim_path)

        # Compute the relative position and orientation
        return T.mat2euler(T.quat2mat(world_orn).T @ T.euler2mat(angvel))


def clear():
    """
    Clear state tied to singleton classes
    """
    PoseAPI.invalidate()
    CollisionAPI.clear()
    ControllableObjectViewAPI.clear()


def create_mesh_prim_with_default_xform(primitive_type, prim_path, u_patches=None, v_patches=None, stage=None):
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
        stage (None or Usd.Stage): If specified, stage on which the primitive mesh should be generated. If None, will
            use og.sim.stage
    """
    MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING = {
        "Sphere": lazy.omni.kit.primitive.mesh.evaluators.sphere.SphereEvaluator,
        "Disk": lazy.omni.kit.primitive.mesh.evaluators.disk.DiskEvaluator,
        "Plane": lazy.omni.kit.primitive.mesh.evaluators.plane.PlaneEvaluator,
        "Cylinder": lazy.omni.kit.primitive.mesh.evaluators.cylinder.CylinderEvaluator,
        "Torus": lazy.omni.kit.primitive.mesh.evaluators.torus.TorusEvaluator,
        "Cone": lazy.omni.kit.primitive.mesh.evaluators.cone.ConeEvaluator,
        "Cube": lazy.omni.kit.primitive.mesh.evaluators.cube.CubeEvaluator,
    }

    assert primitive_type in PRIMITIVE_MESH_TYPES, "Invalid primitive mesh type: {primitive_type}"
    evaluator = MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING[primitive_type]
    u_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_U_SCALE)
    v_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_V_SCALE)
    hs_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_OBJECT_HALF_SCALE)
    lazy.carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, 1)
    lazy.carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, 1)
    stage = og.sim.stage if stage is None else stage

    # Default half_scale (i.e. half-extent, half_height, radius) is 1.
    # TODO (eric): change it to 0.5 once the mesh generator API accepts floating-number HALF_SCALE
    #  (currently it only accepts integer-number and floors 0.5 into 0).
    lazy.carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, 1)
    kwargs = dict(prim_type=primitive_type, prim_path=prim_path, stage=stage)
    if u_patches is not None and v_patches is not None:
        kwargs["u_patches"] = u_patches
        kwargs["v_patches"] = v_patches

    # Import now to avoid too-eager load of Omni classes due to inheritance
    from omnigibson.utils.deprecated_utils import CreateMeshPrimWithDefaultXformCommand

    CreateMeshPrimWithDefaultXformCommand(**kwargs).do()

    lazy.carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, u_backup)
    lazy.carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, v_backup)
    lazy.carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, hs_backup)


def mesh_prim_mesh_to_trimesh_mesh(mesh_prim, include_normals=True, include_texcoord=True):
    """
    Generates trimesh mesh from @mesh_prim if mesh_type is "Mesh"

    Args:
        mesh_prim (Usd.Prim): Mesh prim to convert into trimesh mesh
        include_normals (bool): Whether to include the normals in the resulting trimesh or not
        include_texcoord (bool): Whether to include the corresponding 2D-texture coordinates in the resulting
            trimesh or not

    Returns:
        trimesh.Trimesh: Generated trimesh mesh
    """
    mesh_type = mesh_prim.GetPrimTypeInfo().GetTypeName()
    assert mesh_type == "Mesh", f"Expected mesh prim to have type Mesh, got {mesh_type}"
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
        raw_texture = mesh_prim.GetAttribute("primvars:st").Get()
        if raw_texture is not None:
            kwargs["visual"] = trimesh.visual.TextureVisuals(uv=np.array(raw_texture))

    return trimesh.Trimesh(**kwargs)


def mesh_prim_shape_to_trimesh_mesh(mesh_prim):
    """
    Generates trimesh mesh from @mesh_prim if mesh_type is "Sphere", "Cube", "Cone" or "Cylinder"

    Args:
        mesh_prim (Usd.Prim): Mesh prim to convert into trimesh mesh

    Returns:
        trimesh.Trimesh: Generated trimesh mesh
    """
    mesh_type = mesh_prim.GetPrimTypeInfo().GetTypeName()
    if mesh_type == "Sphere":
        radius = mesh_prim.GetAttribute("radius").Get()
        trimesh_mesh = trimesh.creation.icosphere(subdivision=3, radius=radius)
    elif mesh_type == "Cube":
        extent = mesh_prim.GetAttribute("size").Get()
        trimesh_mesh = trimesh.creation.box([extent] * 3)
    elif mesh_type == "Cone":
        radius = mesh_prim.GetAttribute("radius").Get()
        height = mesh_prim.GetAttribute("height").Get()
        trimesh_mesh = trimesh.creation.cone(radius=radius, height=height)
        # Trimesh cones are centered at the base. We'll move them down by half the height.
        transform = trimesh.transformations.translation_matrix([0, 0, -height / 2])
        trimesh_mesh.apply_transform(transform)
    elif mesh_type == "Cylinder":
        radius = mesh_prim.GetAttribute("radius").Get()
        height = mesh_prim.GetAttribute("height").Get()
        trimesh_mesh = trimesh.creation.cylinder(radius=radius, height=height)
    else:
        raise ValueError(f"Expected mesh prim to have type Sphere, Cube, Cone or Cylinder, got {mesh_type}")

    return trimesh_mesh


def mesh_prim_to_trimesh_mesh(mesh_prim, include_normals=True, include_texcoord=True, world_frame=False):
    """
    Generates trimesh mesh from @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to convert into trimesh mesh
        include_normals (bool): Whether to include the normals in the resulting trimesh or not
        include_texcoord (bool): Whether to include the corresponding 2D-texture coordinates in the resulting
            trimesh or not
        world_frame (bool): Whether to convert the mesh to the world frame or not

    Returns:
        trimesh.Trimesh: Generated trimesh mesh
    """
    mesh_type = mesh_prim.GetTypeName()
    if mesh_type == "Mesh":
        trimesh_mesh = mesh_prim_mesh_to_trimesh_mesh(mesh_prim, include_normals, include_texcoord)
    else:
        trimesh_mesh = mesh_prim_shape_to_trimesh_mesh(mesh_prim)

    if world_frame:
        trimesh_mesh.apply_transform(PoseAPI.get_world_pose_with_scale(mesh_prim.GetPath().pathString))

    return trimesh_mesh


def sample_mesh_keypoints(mesh_prim, n_keypoints, n_keyfaces, seed=None):
    """
    Samples keypoints and keyfaces for mesh @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to be sampled from
        n_keypoints (int): number of (unique) keypoints to randomly sample from @mesh_prim
        n_keyfaces (int): number of (unique) keyfaces to randomly sample from @mesh_prim
        seed (None or int): If set, sets the random seed for deterministic results

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
    if seed is not None:
        np.random.seed(seed)

    # Generate trimesh mesh from which to aggregate points
    tm = mesh_prim_mesh_to_trimesh_mesh(mesh_prim=mesh_prim, include_normals=False, include_texcoord=False)
    n_unique_vertices, n_unique_faces = len(tm.vertices), len(tm.faces)
    faces_flat = tm.faces.flatten()
    n_vertices = len(faces_flat)

    # Sample vertices
    unique_vertices = np.unique(faces_flat)
    assert len(unique_vertices) == n_unique_vertices
    keypoint_idx = (
        np.random.choice(unique_vertices, size=n_keypoints, replace=False)
        if n_unique_vertices > n_keypoints
        else unique_vertices
    )

    # Sample faces
    keyface_idx = (
        np.random.choice(n_unique_faces, size=n_keyfaces, replace=False)
        if n_unique_faces > n_keyfaces
        else np.arange(n_unique_faces)
    )

    return keypoint_idx, keyface_idx


def get_mesh_volume_and_com(mesh_prim, world_frame=False):
    """
    Computes the volume and center of mass for @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to compute volume and center of mass for
        world_frame (bool): Whether to return the volume and CoM in the world frame

    Returns:
        Tuple[float, np.array]: Tuple containing the (volume, center_of_mass) in the mesh frame or the world frame
    """

    trimesh_mesh = mesh_prim_to_trimesh_mesh(
        mesh_prim, include_normals=False, include_texcoord=False, world_frame=world_frame
    )
    if trimesh_mesh.is_volume:
        volume = trimesh_mesh.volume
        com = trimesh_mesh.center_mass
    else:
        # If the mesh is not a volume, we compute its convex hull and use that instead
        try:
            trimesh_mesh_convex = trimesh_mesh.convex_hull
            volume = trimesh_mesh_convex.volume
            com = trimesh_mesh_convex.center_mass
        except:
            # if convex hull computation fails, it usually means the mesh is degenerated: use trivial values.
            volume = 0.0
            com = np.zeros(3)

    return volume, com


def check_extent_radius_ratio(mesh_prim):
    """
    Checks if the min extent in world frame and the extent radius ratio in local frame of @mesh_prim is within the
    acceptable range for PhysX GPU acceleration (not too thin, and not too oblong)

    Ref: https://github.com/NVIDIA-Omniverse/PhysX/blob/561a0df858d7e48879cdf7eeb54cfe208f660f18/physx/source/geomutils/src/convex/GuConvexMeshData.h#L183-L190

    Args:
        mesh_prim (Usd.Prim): Mesh prim to check

    Returns:
        bool: True if the min extent (world) and the extent radius ratio (local frame) is acceptable, False otherwise
    """
    mesh_type = mesh_prim.GetPrimTypeInfo().GetTypeName()
    # Non-mesh prims are always considered to be within the acceptable range
    if mesh_type != "Mesh":
        return True

    trimesh_mesh_world = mesh_prim_to_trimesh_mesh(
        mesh_prim, include_normals=False, include_texcoord=False, world_frame=True
    )
    min_extent = trimesh_mesh_world.extents.min()
    # If the mesh is too flat in the world frame, omniverse cannot create convex mesh for it
    if min_extent < 1e-5:
        return False

    trimesh_mesh = mesh_prim_to_trimesh_mesh(
        mesh_prim, include_normals=False, include_texcoord=False, world_frame=False
    )
    if not trimesh_mesh.is_volume:
        trimesh_mesh = trimesh_mesh.convex_hull

    max_radius = trimesh_mesh.extents.max() / 2.0
    min_radius = trimesh.proximity.closest_point(trimesh_mesh, np.array([trimesh_mesh.center_mass]))[1][0]
    ratio = max_radius / min_radius

    # PhysX requires ratio to be < 100.0. We use 95.0 to be safe.
    return ratio < 95.0


def create_primitive_mesh(prim_path, primitive_type, extents=1.0, u_patches=None, v_patches=None, stage=None):
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
        stage (None or Usd.Stage): If specified, stage on which the primitive mesh should be generated. If None, will
            use og.sim.stage

    Returns:
        UsdGeom.Mesh: Generated primitive mesh as a prim on the active stage
    """
    assert_valid_key(key=primitive_type, valid_keys=PRIMITIVE_MESH_TYPES, name="primitive mesh type")
    create_mesh_prim_with_default_xform(
        primitive_type, prim_path, u_patches=u_patches, v_patches=v_patches, stage=stage
    )
    mesh = lazy.pxr.UsdGeom.Mesh.Define(og.sim.stage if stage is None else stage, prim_path)

    # Modify the points and normals attributes so that total extents is the desired
    # This means multiplying omni's default by extents * 50.0, as the native mesh generated has extents [-0.01, 0.01]
    # -- i.e.: 2cm-wide mesh
    extents = np.ones(3) * extents if isinstance(extents, float) else np.array(extents)
    for attr in (mesh.GetPointsAttr(), mesh.GetNormalsAttr()):
        vals = np.array(attr.Get()).astype(np.float64)
        attr.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*(val * extents * 50.0)) for val in vals]))
    mesh.GetExtentAttr().Set(
        lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*(-extents / 2.0)), lazy.pxr.Gf.Vec3f(*(extents / 2.0))])
    )

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
    lazy.omni.isaac.core.utils.stage.add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
    prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)

    # Make sure prim was loaded correctly
    assert prim, f"Failed to load {asset_type.upper()} object from path: {asset_path}"

    return prim


def get_world_prim():
    """
    Returns:
        Usd.Prim: Active world prim in the current stage
    """
    return lazy.omni.isaac.core.utils.prims.get_prim_at_path("/World")
