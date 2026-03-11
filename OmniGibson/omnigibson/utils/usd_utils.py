import collections
import itertools
import os
import re
from collections.abc import Iterable
from typing import Tuple

import numpy as np
import torch as th
import trimesh
from numba import jit, prange

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
import omnigibson.utils.transform_utils as TT
import omnigibson.utils.transform_utils_np as NT
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.backend_utils import add_compute_function
from omnigibson.macros import gm
from omnigibson.utils.constants import PRIMITIVE_MESH_TYPES, JointType, PrimType
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.python_utils import assert_valid_key, torch_compile
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
    for child in lazy.isaacsim.core.utils.prims.get_prim_children(prim):
        prims.append(child)
        prims += get_prim_nested_children(prim=child)

    return prims


def create_joint(
    prim_path,
    joint_type,
    body0=None,
    body1=None,
    enabled=True,
    exclude_from_articulation=False,
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
        exclude_from_articulation (bool): whether to exclude this joint from the articulation or not.
        joint_frame_in_parent_frame_pos (th.tensor or None): relative position of the joint frame to the parent frame (body0).
        joint_frame_in_parent_frame_quat (th.tensor or None): relative orientation of the joint frame to the parent frame (body0).
        joint_frame_in_child_frame_pos (th.tensor or None): relative position of the joint frame to the child frame (body1).
        joint_frame_in_child_frame_quat (th.tensor or None): relative orientation of the joint frame to the child frame (body1).
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
    ), "At least either body0 or body1 must be specified when creating a joint!"

    # Create the joint
    joint = getattr(lazy.pxr.UsdPhysics, joint_type).Define(og.sim.stage, prim_path)

    # Possibly add body0, body1 targets
    if body0 is not None:
        assert lazy.isaacsim.core.utils.prims.is_prim_path_valid(body0), f"Invalid body0 path specified: {body0}"
        joint.GetBody0Rel().SetTargets([lazy.pxr.Sdf.Path(body0)])
    if body1 is not None:
        assert lazy.isaacsim.core.utils.prims.is_prim_path_valid(body1), f"Invalid body1 path specified: {body1}"
        joint.GetBody1Rel().SetTargets([lazy.pxr.Sdf.Path(body1)])

    # Get the prim pointed to at this path
    joint_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)

    # Apply joint API interface
    lazy.pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)

    # We need to step rendering once to auto-fill the local pose before overwriting it.
    # Note that for some reason, if multi_gpu is used, this line will crash if create_joint is called during on_contact
    # callback, e.g. when an attachment joint is being created due to contacts.
    og.sim.render()

    if joint_frame_in_parent_frame_pos is not None:
        joint_prim.GetAttribute("physics:localPos0").Set(lazy.pxr.Gf.Vec3f(*joint_frame_in_parent_frame_pos.tolist()))
    if joint_frame_in_parent_frame_quat is not None:
        joint_prim.GetAttribute("physics:localRot0").Set(
            lazy.pxr.Gf.Quatf(*joint_frame_in_parent_frame_quat[[3, 0, 1, 2]].tolist())
        )
    if joint_frame_in_child_frame_pos is not None:
        joint_prim.GetAttribute("physics:localPos1").Set(lazy.pxr.Gf.Vec3f(*joint_frame_in_child_frame_pos.tolist()))
    if joint_frame_in_child_frame_quat is not None:
        joint_prim.GetAttribute("physics:localRot1").Set(
            lazy.pxr.Gf.Quatf(*joint_frame_in_child_frame_quat[[3, 0, 1, 2]].tolist())
        )

    if break_force is not None:
        joint_prim.GetAttribute("physics:breakForce").Set(break_force)
    if break_torque is not None:
        joint_prim.GetAttribute("physics:breakTorque").Set(break_torque)

    # Possibly (un-/)enable this joint
    joint_prim.GetAttribute("physics:jointEnabled").Set(enabled)

    # Possibly exclude this joint from the articulation
    joint_prim.GetAttribute("physics:excludeFromArticulation").Set(exclude_from_articulation)

    # We update the simulation now without stepping physics if sim is playing so we can bypass the snapping warning from PhysicsUSD
    if og.sim.is_playing():
        with suppress_omni_log(channels=["omni.physx.plugin"]):
            og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)

    # Return this joint
    return joint_prim


class RigidContactAPIImpl:
    """
    Class containing class methods to aggregate rigid body contacts across all rigid bodies in the simulator

    NOTE: The RigidContactAPI only works when the contacting objects are awake. If the objects could be asleep, use ContactBodies instead.
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
    def get_row_filters(cls):
        return [f"/World/scene_{i}/*/*" for i in range(len(og.sim.scenes))]

    @classmethod
    def get_column_filters(cls):
        filters = dict()
        for scene_idx, scene in enumerate(og.sim.scenes):
            filters[scene_idx] = []
            for obj in scene.objects:
                if obj.prim_type == PrimType.RIGID:
                    for link in obj.links.values():
                        from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim

                        if isinstance(link, RigidDynamicPrim):
                            filters[scene_idx].append(link.prim_path)

        return filters

    @classmethod
    def get_max_contact_data_count(cls):
        return 256

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

        # Get the row filters too
        row_filters = self.get_row_filters()

        # Generate rigid body view, making sure to update the simulation first (without physics) so that the physx
        # backend is synchronized with any newly added objects
        # We also suppress the omni tensor plugin from giving warnings we expect
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            for scene_idx, _ in enumerate(og.sim.scenes):
                # TODO: How to make this work with the floor plane?
                # If there are no collidable objects in the scene, skip.
                if len(column_filters[scene_idx]) == 0:
                    continue

                self._CONTACT_VIEW[scene_idx] = og.sim.physics_sim_view.create_rigid_contact_view(
                    pattern=row_filters[scene_idx],
                    filter_patterns=column_filters[scene_idx],
                    max_contact_data_count=self.get_max_contact_data_count(),
                )

                self._PATH_TO_ROW_IDX[scene_idx] = {
                    path: i for i, path in enumerate(self._CONTACT_VIEW[scene_idx].sensor_paths)
                }

                self._ROW_IDX_TO_PATH[scene_idx] = list(self._PATH_TO_ROW_IDX[scene_idx].keys())
                self._COL_IDX_TO_PATH[scene_idx] = list(self._PATH_TO_COL_IDX[scene_idx].keys())

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

    def get_contact_pairs(self, scene_idx, row_prim_paths=None, column_prim_paths=None):
        """Get pairs of prim paths that are in contact."""
        impulses = th.norm(self.get_all_impulses(scene_idx), dim=-1)
        assert impulses.ndim == 2, f"Impulse matrix should be 2D, found shape {impulses.shape}"
        interesting_col_paths = [
            p for p in self._PATH_TO_COL_IDX[scene_idx].keys() if column_prim_paths is None or p in column_prim_paths
        ]
        interesting_columns = [list(self.get_body_col_idx(pp))[1] for pp in interesting_col_paths]

        # Get the interesting-columns from the impulse matrix
        interesting_impulse_columns = impulses[:, interesting_columns]
        assert (
            interesting_impulse_columns.ndim == 2
        ), f"Impulse matrix should be 2D, found shape {interesting_impulse_columns.shape}"
        interesting_row_idxes = th.nonzero(th.any(interesting_impulse_columns > 0, dim=1)).flatten()
        interesting_row_paths = [self.get_row_idx_prim_path(scene_idx, i) for i in interesting_row_idxes]

        # Filter out idxes by whether or not the row path is in the row prim paths list
        if row_prim_paths is not None:
            interesting_row_idxes, interesting_row_paths = zip(
                *[(i, p) for i, p in zip(interesting_row_idxes, interesting_row_paths) if p in row_prim_paths]
            )
            interesting_row_idxes = th.tensor(list(interesting_row_idxes))

        # Get the full interesting section of the impulse matrix
        interesting_impulses = interesting_impulse_columns[interesting_row_idxes]
        assert interesting_impulses.ndim == 2, f"Impulse matrix should be 2D, found shape {interesting_impulses.shape}"

        # Early return if not in contact.
        if not th.any(interesting_impulses > 0):
            return set()

        # Get all of the (row, col) pairs where the impulse is greater than 0
        return {
            (interesting_row_paths[row], interesting_col_paths[col])
            for row, col in zip(*th.nonzero(interesting_impulses > 0, as_tuple=True))
        }

    def get_contact_data(self, scene_idx, row_prim_paths=None, column_prim_paths=None):
        # First check if the object has any contacts
        impulses = th.norm(self.get_all_impulses(scene_idx), dim=-1)
        assert impulses.ndim == 2, f"Impulse matrix should be 2D, found shape {impulses.shape}"

        row_idx = (
            list(range(impulses.shape[0]))
            if row_prim_paths is None
            else [self.get_body_row_idx(path)[1] for path in row_prim_paths]
        )
        col_idx = (
            list(range(impulses.shape[1]))
            if column_prim_paths is None
            else [self.get_body_col_idx(path)[1] for path in column_prim_paths]
        )
        relevant_impulses = impulses[row_idx][:, col_idx]

        # Early return if not in contact.
        if not th.any(relevant_impulses > 0):
            return []

        # Get the contact data
        if scene_idx not in self._CONTACT_DATA:
            self._CONTACT_DATA[scene_idx] = self._CONTACT_VIEW[scene_idx].get_contact_data(dt=1.0)

        # Get the contact data for this prim
        forces, points, normals, separations, contact_counts, start_indices = self._CONTACT_DATA[scene_idx]

        # Assert that one of two things is true: either the prim count and contact count are equal,
        # in which case we can zip them together, or the prim count is 1, in which case we can just
        # repeat the single prim data for all contacts. Otherwise, it is not clear which contacts are
        # happening between which two objects, so we return no contacts while printing an error.
        contacts = []
        for row in row_idx:
            row_prim_path = self.get_row_idx_prim_path(scene_idx, row)
            for col in col_idx:
                if contact_counts[row, col] == 0:
                    continue
                col_prim_path = self.get_col_idx_prim_path(scene_idx, col)
                start_idx = start_indices[row, col]
                end_idx = start_idx + contact_counts[row, col]
                if start_idx >= len(forces):
                    log.warning(
                        f"Contact data for prim {row_prim_path} and "
                        f"{col_prim_path} is missing because there are too many contacts!"
                    )
                    continue
                if end_idx > len(forces):
                    log.warning(
                        f"Contact data for prim {row_prim_path} and "
                        f"{col_prim_path} is partial because there are too many contacts!"
                    )
                    continue
                contacts.extend(
                    zip(
                        itertools.repeat(row_prim_path),
                        itertools.repeat(col_prim_path),
                        forces[start_idx:end_idx],
                        points[start_idx:end_idx],
                        normals[start_idx:end_idx],
                        separations[start_idx:end_idx],
                    )
                )

        return contacts

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
            self._CONTACT_CACHE[key] = th.any(self.get_impulses(prim_paths_a=prim_paths_a, prim_paths_b=prim_paths_b))
        return self._CONTACT_CACHE[key].item()

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
        filters = dict()
        for scene_idx, scene in enumerate(og.sim.scenes):
            filters[scene_idx] = []
            for robot in scene.robots:
                if robot.is_manipulation:
                    filters[scene_idx].extend(link.prim_path for links in robot.finger_links.values() for link in links)

        return filters

    @classmethod
    def get_max_contact_data_count(cls):
        return len(cls.get_column_filters()[0]) * 128


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
        # Remove all the collision group prims
        for col_group_prim in cls.ACTIVE_COLLISION_GROUPS.values():
            delete_or_deactivate_prim(col_group_prim.GetPath().pathString)

        # Remove the collision groups tree
        delete_or_deactivate_prim("/World/collision_groups")

        # Clear the dictionary
        cls.ACTIVE_COLLISION_GROUPS = {}


def setup_collision_apis(prim):
    """
    Apply collision-related physics APIs to a USD prim. This should be called for prims
    that are identified as collision meshes (e.g. those appearing under a "collisions" scope prim).

    This applies the CollisionAPI, PhysxCollisionAPI, and (for meshes) MeshCollisionAPI to the prim,
    sets a default convex hull collision approximation for mesh types, and enables/disables collisions
    based on the global VISUAL_ONLY setting.

    Note: This does NOT set the prim's purpose. The caller should set purpose as appropriate
    (e.g. "guide" for collision-only meshes, "default" for collision+visual meshes).

    Args:
        prim: The USD prim to set up collision APIs on.

    Returns:
        tuple: (collision_api, physx_collision_api, mesh_collision_api) where mesh_collision_api
            may be None for non-mesh prims.
    """
    # Create / get CollisionAPI reference
    collision_api = (
        lazy.pxr.UsdPhysics.CollisionAPI(prim)
        if prim.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI)
        else lazy.pxr.UsdPhysics.CollisionAPI.Apply(prim)
    )
    physx_collision_api = (
        lazy.pxr.PhysxSchema.PhysxCollisionAPI(prim)
        if prim.HasAPI(lazy.pxr.PhysxSchema.PhysxCollisionAPI)
        else lazy.pxr.PhysxSchema.PhysxCollisionAPI.Apply(prim)
    )

    # Optionally add mesh collision API if this is a mesh
    mesh_collision_api = None
    if prim.GetPrimTypeInfo().GetTypeName() == "Mesh":
        mesh_collision_api = (
            lazy.pxr.UsdPhysics.MeshCollisionAPI(prim)
            if prim.HasAPI(lazy.pxr.UsdPhysics.MeshCollisionAPI)
            else lazy.pxr.UsdPhysics.MeshCollisionAPI.Apply(prim)
        )
        # Set the approximation to be convex hull by default
        apply_collision_approximation(prim, mesh_collision_api, "convexHull")

    # Set collision enabled based on global setting
    collision_api.GetCollisionEnabledAttr().Set(not gm.VISUAL_ONLY)

    return collision_api, physx_collision_api, mesh_collision_api


def apply_collision_approximation(prim, mesh_collision_api, approximation_type):
    """
    Apply a collision approximation type to a single collision mesh prim.

    Args:
        prim: The USD prim to apply the collision approximation to.
        mesh_collision_api: The UsdPhysics.MeshCollisionAPI for this prim.
        approximation_type (str): Approximation type to use. One of:
            {"none", "convexHull", "convexDecomposition", "meshSimplification", "sdf",
             "boundingSphere", "boundingCube"}
    """
    assert mesh_collision_api is not None, "collision_approximation only applicable for meshes!"
    assert_valid_key(
        key=approximation_type,
        valid_keys={
            "none",
            "convexHull",
            "convexDecomposition",
            "meshSimplification",
            "sdf",
            "boundingSphere",
            "boundingCube",
        },
        name="collision approximation type",
    )

    # Make sure to add the appropriate API if we're setting certain values
    if approximation_type == "convexHull" and not prim.HasAPI(lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI):
        lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
    elif approximation_type == "convexDecomposition" and not prim.HasAPI(
        lazy.pxr.PhysxSchema.PhysxConvexDecompositionCollisionAPI
    ):
        lazy.pxr.PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    elif approximation_type == "meshSimplification" and not prim.HasAPI(
        lazy.pxr.PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI
    ):
        lazy.pxr.PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI.Apply(prim)
    elif approximation_type == "sdf" and not prim.HasAPI(lazy.pxr.PhysxSchema.PhysxSDFMeshCollisionAPI):
        lazy.pxr.PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    elif approximation_type == "none" and not prim.HasAPI(lazy.pxr.PhysxSchema.PhysxTriangleMeshCollisionAPI):
        lazy.pxr.PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim)

    if approximation_type == "convexHull":
        pch_api = lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI(prim)
        # Also make sure the maximum vertex count is 60 (max number compatible with GPU)
        # https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html#collision-settings
        if pch_api.GetHullVertexLimitAttr().Get() is None:
            pch_api.CreateHullVertexLimitAttr()
        pch_api.GetHullVertexLimitAttr().Set(60)

    mesh_collision_api.GetApproximationAttr().Set(approximation_type)


class PoseAPI:
    """
    This is a singleton class for getting world poses.
    Whenever we directly set the pose of a prim, we should call PoseAPI.invalidate().
    After that, if we need to access the pose of a prim without stepping physics,
    this class will refresh the poses by syncing across USD-fabric-PhysX depending on the flatcache setting.
    """

    VALID = False

    # Dictionary mapping prim path to fabric prim
    PRIMS = dict()

    @classmethod
    def clear(cls):
        cls.PRIMS = dict()

    @classmethod
    def invalidate(cls):
        cls.VALID = False

    @classmethod
    def mark_valid(cls):
        cls.VALID = True

    @classmethod
    def _refresh(cls):
        if og.sim is not None and not cls.VALID:
            # Check that no reads from PoseAPI are happening during a physics step, this is quite slow!
            assert not og.sim.currently_stepping, "Cannot refresh poses during a physics step!"

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
        """
        Gets pose of the prim object with respect to the world frame
        Args:
            Prim_path: the path of the prim object
        Returns:
            2-tuple:
                - torch.Tensor: (x,y,z) position in the world frame
                - torch.Tensor: (x,y,z,w) quaternion orientation in the world frame
        """
        # Check that no reads from PoseAPI are happening during a physics step.
        assert (
            not og.sim.currently_stepping
        ), "Do not read poses from PoseAPI during a physics step, this is quite slow!"

        # Add to stored prims if not already existing
        if prim_path not in cls.PRIMS:
            cls.PRIMS[prim_path] = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path=prim_path, fabric=True)

        cls._refresh()

        # Avoid premature imports
        from omnigibson.utils.deprecated_utils import get_world_pose

        position, orientation = get_world_pose(cls.PRIMS[prim_path])
        return th.tensor(position, dtype=th.float32), th.tensor(orientation, dtype=th.float32)

    @classmethod
    def get_world_pose_with_scale(cls, prim_path):
        """
        This is used when information about the prim's global scale is needed,
        e.g. when converting points in the prim frame to the world frame.
        """
        # Add to stored prims if not already existing
        if prim_path not in cls.PRIMS:
            cls.PRIMS[prim_path] = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path=prim_path, fabric=True)

        cls._refresh()
        # Avoid premature imports
        from omnigibson.utils.deprecated_utils import _get_world_pose_transform_w_scale

        return th.tensor(_get_world_pose_transform_w_scale(cls.PRIMS[prim_path]), dtype=th.float32).T

    @classmethod
    def convert_world_pose_to_local(cls, prim, position, orientation):
        """Converts a world pose to a local pose under a prim's parent."""
        world_transform = T.pose2mat((position, orientation))
        parent_path = str(lazy.isaacsim.core.utils.prims.get_prim_parent(prim).GetPath())
        parent_world_transform = cls.get_world_pose_with_scale(parent_path)

        local_transform = th.linalg.inv_ex(parent_world_transform).inverse @ world_transform
        local_transform[:3, :3] /= th.linalg.norm(local_transform[:3, :3], dim=0)  # unscale local transform's rotation

        # Check that the local transform consists only of a position, scale and rotation
        product = local_transform[:3, :3] @ local_transform[:3, :3].T
        assert th.allclose(
            product, th.diag(th.diag(product)), atol=1e-3
        ), f"{prim.GetPath()} local transform is not orthogonal."

        # Return the local pose
        return T.mat2pose(local_transform)


class BatchControlViewAPIImpl:
    """
    A centralized view that allows for reading and writing to an ArticulationView that covers multiple
    controllable objects in the scene. This is used to avoid the overhead of reading from many views
    for each robot in each physics step, a source of significant overhead.
    """

    def __init__(self, pattern):
        # The prim path pattern that will be passed into the view
        self._pattern = pattern

        # The unified ArticulationView used to access all of the controllable objects in the scene.
        self._view = None

        # Cache for all of the view functions' return values within the same simulation step.
        # Keyed by function name without get_, the value is the return value of the function.
        self._read_cache = {}

        # Cache for all of the view functions' write values within the same simulation step.
        # Keyed by the function name without set_, the value is the set of indices that need to be updated.
        self._write_idx_cache = collections.defaultdict(set)

        # Mapping from prim path to index in the view.
        self._idx = {}

        # Mapping from prim idx to a dict that maps link name to link index in the view.
        self._link_idx = {}

        # Mapping from prim path to base footprint link name if one exists, None if the root is the base link.
        self._base_footprint_link_names = {}

        # Prior link transforms / dof positions for estimating velocities since Omni gives inaccurate values
        self._last_state = None

    def post_physics_step(self):
        # Should be called every sim physics step, right after a new physics step occurs
        # The current poses (if it exists) are now the former poses from the previous timestep
        # These values are needed to compute velocity estimates
        if (
            "root_transforms" in self._read_cache
            and "link_transforms" in self._read_cache
            and "dof_positions" in self._read_cache
        ):
            self._last_state = {
                "root_transforms": cb.copy(self._read_cache["root_transforms"]),
                "link_transforms": cb.copy(self._read_cache["link_transforms"]),
                "dof_positions": cb.copy(self._read_cache["dof_positions"]),
            }
        else:
            # We don't have enough info to populate the history, so simply clear it instead
            self._last_state = None

        # Clear the internal data since everything is outdated
        self.clear(keep_last_pose=True)

    def clear(self, keep_last_pose=False):
        self._read_cache = {}
        self._write_idx_cache = collections.defaultdict(set)

        # Clear our last timestep's cached values by default
        if not keep_last_pose:
            self._last_state = None

        # Cache the (now current) transforms so that they're guaranteed to exist throughout the duration of this
        # timestep, and available for caching during the next timestep's post_physics_step() call
        if og.sim.is_playing():
            self._read_cache["root_transforms"] = cb.from_torch(self._view.get_root_transforms())
            self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())
            self._read_cache["dof_positions"] = cb.from_torch(self._view.get_dof_positions())

    def _set_dof_position_targets(self, data, indices, cast=True):
        # No casting results in better efficiency
        if cast:
            data = self._view._frontend.as_contiguous_float32(data)
            indices = self._view._frontend.as_contiguous_uint32(indices)
        data_desc = self._view._frontend.get_tensor_desc(data)
        indices_desc = self._view._frontend.get_tensor_desc(indices)

        if not self._view._backend.set_dof_position_targets(data_desc, indices_desc):
            raise Exception("Failed to set DOF positions in backend")

    def _set_dof_velocity_targets(self, data, indices, cast=True):
        # No casting results in better efficiency
        if cast:
            data = self._view._frontend.as_contiguous_float32(data)
            indices = self._view._frontend.as_contiguous_uint32(indices)
        data_desc = self._view._frontend.get_tensor_desc(data)
        indices_desc = self._view._frontend.get_tensor_desc(indices)

        if not self._view._backend.set_dof_velocity_targets(data_desc, indices_desc):
            raise Exception("Failed to set DOF velocities in backend")

    def _set_dof_actuation_forces(self, data, indices, cast=True):
        # No casting results in better efficiency
        if cast:
            data = self._view._frontend.as_contiguous_float32(data)
            indices = self._view._frontend.as_contiguous_uint32(indices)
        data_desc = self._view._frontend.get_tensor_desc(data)
        indices_desc = self._view._frontend.get_tensor_desc(indices)

        if not self._view._backend.set_dof_actuation_forces(data_desc, indices_desc):
            raise Exception("Failed to set DOF actuation forces in backend")

    def flush_control(self):
        if "dof_position_targets" in self._write_idx_cache:
            pos_indices = cb.int_array(sorted(self._write_idx_cache["dof_position_targets"]))
            pos_targets = self._read_cache["dof_position_targets"]
            self._set_dof_position_targets(cb.to_torch(pos_targets), cb.to_torch(pos_indices), cast=False)

        if "dof_velocity_targets" in self._write_idx_cache:
            vel_indices = cb.int_array(sorted(self._write_idx_cache["dof_velocity_targets"]))
            vel_targets = self._read_cache["dof_velocity_targets"]
            self._set_dof_velocity_targets(cb.to_torch(vel_targets), cb.to_torch(vel_indices), cast=False)

        if "dof_actuation_forces" in self._write_idx_cache:
            eff_indices = cb.int_array(sorted(self._write_idx_cache["dof_actuation_forces"]))
            eff_targets = self._read_cache["dof_actuation_forces"]
            self._set_dof_actuation_forces(cb.to_torch(eff_targets), cb.to_torch(eff_indices), cast=False)

    def initialize_view(self):
        # First, get all of the controllable objects in the scene (avoiding circular import)
        from omnigibson.robots import Robot

        controllable_objects = [obj for scene in og.sim.scenes for obj in scene.objects if isinstance(obj, Robot)]

        # Get their corresponding prim paths
        expected_prim_paths = {obj.articulation_root_path for obj in controllable_objects}

        # Apply the pattern to find the expected prim paths
        expected_prim_paths = {
            prim_path for prim_path in expected_prim_paths if re.fullmatch(self._pattern.replace("*", ".*"), prim_path)
        }

        # Make sure we have at least one controllable object
        if len(expected_prim_paths) == 0:
            return

        # Create the actual articulation view. Note that even though we search for base_link here,
        # the returned things will not necessarily be the base_link prim paths, but the appropriate
        # articulation root path for every object (base_link for non-fixed, parent for fixed objects)
        self._view = og.sim.physics_sim_view.create_articulation_view(self._pattern)
        view_prim_paths = self._view.prim_paths
        assert (
            set(view_prim_paths) == expected_prim_paths
        ), f"ControllableObjectViewAPI expected prim paths {expected_prim_paths} but got {view_prim_paths}"

        # Create the mapping from prim path to index
        self._idx = {prim_path: i for i, prim_path in enumerate(view_prim_paths)}
        self._link_idx = [
            {link_path.split("/")[-1]: j for j, link_path in enumerate(articulation_link_paths)}
            for articulation_link_paths in self._view.link_paths
        ]
        self._base_footprint_link_names = {
            obj.articulation_root_path: (
                obj.base_footprint_link_name if obj.base_footprint_link_name != obj.root_link_name else None
            )
            for obj in controllable_objects
            if obj.articulation_root_path in expected_prim_paths
        }

    def set_joint_position_targets(self, prim_path, positions, indices):
        assert len(indices) == len(positions), "Indices and values must have the same length"
        idx = self._idx[prim_path]

        # Load the current targets.
        if "dof_position_targets" not in self._read_cache:
            self._read_cache["dof_position_targets"] = cb.from_torch(self._view.get_dof_position_targets())

        # Update the target
        self._read_cache["dof_position_targets"][idx, indices] = positions

        # Add this index to the write cache
        self._write_idx_cache["dof_position_targets"].add(idx)

    def set_joint_velocity_targets(self, prim_path, velocities, indices):
        assert len(indices) == len(velocities), "Indices and values must have the same length"
        idx = self._idx[prim_path]

        # Load the current targets.
        if "dof_velocity_targets" not in self._read_cache:
            self._read_cache["dof_velocity_targets"] = cb.from_torch(self._view.get_dof_velocity_targets())

        # Update the target
        self._read_cache["dof_velocity_targets"][idx, indices] = velocities

        # Add this index to the write cache
        self._write_idx_cache["dof_velocity_targets"].add(idx)

    def set_joint_efforts(self, prim_path, efforts, indices):
        assert len(indices) == len(efforts), "Indices and values must have the same length"
        idx = self._idx[prim_path]

        # Load the current targets.
        if "dof_actuation_forces" not in self._read_cache:
            self._read_cache["dof_actuation_forces"] = cb.from_torch(self._view.get_dof_actuation_forces())

        # Update the target
        self._read_cache["dof_actuation_forces"][idx, indices] = efforts

        # Add this index to the write cache
        self._write_idx_cache["dof_actuation_forces"].add(idx)

    def get_member_view_indices(self, prim_paths):
        """Return view row index for each prim_path (in input order)."""
        return [self._idx[p] for p in prim_paths]

    def set_all_joint_position_targets(self, enabled_rows, controls, dof_idx):
        """
        Args:
            enabled_rows: list[int] — view row indices for enabled members (pre-filtered)
            controls: (N_enabled, len(dof_idx)) array — pre-stacked by controller
            dof_idx: DOF column indices (cb.arr_type)
        """
        if "dof_position_targets" not in self._read_cache:
            self._read_cache["dof_position_targets"] = cb.from_torch(self._view.get_dof_position_targets())
        targets = self._read_cache["dof_position_targets"]
        row_idx = cb.int_array(enabled_rows).reshape(-1, 1)
        targets[row_idx, dof_idx] = controls
        self._write_idx_cache["dof_position_targets"].update(enabled_rows)

    def set_all_joint_velocity_targets(self, enabled_rows, velocities, dof_idx):
        if "dof_velocity_targets" not in self._read_cache:
            self._read_cache["dof_velocity_targets"] = cb.from_torch(self._view.get_dof_velocity_targets())
        targets = self._read_cache["dof_velocity_targets"]
        row_idx = cb.int_array(enabled_rows).reshape(-1, 1)
        targets[row_idx, dof_idx] = velocities
        self._write_idx_cache["dof_velocity_targets"].update(enabled_rows)

    def set_all_joint_efforts(self, enabled_rows, efforts, dof_idx):
        if "dof_actuation_forces" not in self._read_cache:
            self._read_cache["dof_actuation_forces"] = cb.from_torch(self._view.get_dof_actuation_forces())
        targets = self._read_cache["dof_actuation_forces"]
        row_idx = cb.int_array(enabled_rows).reshape(-1, 1)
        targets[row_idx, dof_idx] = efforts
        self._write_idx_cache["dof_actuation_forces"].update(enabled_rows)

    def get_all_root_transform(self):
        if "root_transforms" not in self._read_cache:
            self._read_cache["root_transforms"] = cb.from_torch(self._view.get_root_transforms())
        pose = self._read_cache["root_transforms"]
        return pose[:, :3], pose[:, 3:]

    def get_root_transform(self, prim_path):
        idx = self._idx[prim_path]
        pos, quat = self.get_all_root_transform()
        return pos[idx], quat[idx]

    def get_all_position_orientation(self):
        # Here we want to return the position of the base footprint link.
        # If the base footprint link is None, we return the position of the root link.

        # we assume that in a view, all base link_name is the same
        link_name = next(iter(self._base_footprint_link_names.values()))
        if link_name is None:
            return self.get_all_root_transform()
        else:
            return self.get_all_link_transform(link_name)

    def get_position_orientation(self, prim_path):
        # Here we want to return the position of the base footprint link. If the base footprint link is None,
        # we return the position of the root link.
        if self._base_footprint_link_names[prim_path] is not None:
            link_name = self._base_footprint_link_names[prim_path]
            return self.get_link_transform(prim_path, link_name)
        else:
            return self.get_root_transform(prim_path)

    def _get_all_velocities(self, estimate=False):
        link_name = next(iter(self._base_footprint_link_names.values()))
        if link_name is not None:
            return self._get_all_link_velocities(link_name, estimate=estimate)
        else:
            return self._get_all_root_velocities(estimate=estimate)

    def _get_all_relative_velocities(self, estimate=False):
        """Returns (N, n_links+1, 6) relative velocities for all robots; final slot [-1] is the base."""
        vel_str = "velocities_estimate" if estimate else "velocities"

        if f"all_relative_{vel_str}" not in self._read_cache:
            # Warm the (N, L, 6) link velocity cache and fetch it
            any_link_name = next(iter(self._link_idx[0]))
            self._get_all_link_velocities(any_link_name, estimate=estimate)
            link_vels = cb.to_torch(self._read_cache[f"link_{vel_str}"])  # (N, L, 6)

            # Get base velocities (N, 6): reuse link cache if a base footprint link is configured
            # (all robots in a view share the same base footprint link name)
            base_footprint_link_name = next(iter(self._base_footprint_link_names.values()))
            if base_footprint_link_name is not None:
                base_link_idx = self._link_idx[0][base_footprint_link_name]
                base_vels = link_vels[:, base_link_idx, :]  # (N, 6) — already in cache, no extra fetch
            else:
                # Warm root velocities cache and get (N, 6)
                self._get_all_root_velocities(estimate=estimate)
                base_vels = cb.to_torch(self._read_cache[f"root_{vel_str}"])  # (N, 6)

            # Build (N, L+1, 6): link vels followed by base vel (base at final index, matching _get_relative_velocities)
            all_vels = th.cat([link_vels, base_vels.unsqueeze(1)], dim=1)  # (N, L+1, 6)

            # Build block-diagonal rotation transform per robot: (N, 6, 6)
            all_quats = cb.to_torch(self.get_all_position_orientation()[1])  # (N, 4)
            ori_t_batch = TT.quat2mat(all_quats).transpose(-2, -1)  # (N, 3, 3)
            tf = th.zeros(all_vels.shape[0], 6, 6, dtype=all_vels.dtype)
            tf[:, :3, :3] = ori_t_batch
            tf[:, 3:, 3:] = ori_t_batch

            # Batched matmul: (N, 1, 6, 6) @ (N, L+1, 6, 1) → (N, L+1, 6)
            rel_vels = (tf.unsqueeze(1) @ all_vels.unsqueeze(-1)).squeeze(-1)
            self._read_cache[f"all_relative_{vel_str}"] = cb.from_torch(rel_vels)

        return self._read_cache[f"all_relative_{vel_str}"]

    def _get_relative_velocities(self, prim_path, estimate=False):
        idx = self._idx[prim_path]
        return self._get_all_relative_velocities(estimate=estimate)[idx]

    def get_linear_velocity(self, prim_path, estimate=False):
        return self._get_velocities(prim_path, estimate=estimate)[:3]

    def get_angular_velocity(self, prim_path, estimate=False):
        return self._get_velocities(prim_path, estimate=estimate)[3:]

    def _get_all_root_velocities(self, estimate=False):
        vel_str = "velocities_estimate" if estimate else "velocities"

        # Use estimated calculation if requested and we have prior history info
        if f"root_{vel_str}" not in self._read_cache:
            if estimate and self._last_state is not None:
                # Compute root velocities estimate as delta between prior timestep and current timestep
                vels = cb.zeros((self._last_state["root_transforms"].shape[0], 6))

                if "root_transforms" not in self._read_cache:
                    self._read_cache["root_transforms"] = cb.from_torch(self._view.get_root_transforms())

                vels[:, :3] = self._read_cache["root_transforms"][:, :3] - self._last_state["root_transforms"][:, :3]
                vels[:, 3:] = cb.T.quat2axisangle(
                    cb.T.quat_distance(
                        self._read_cache["root_transforms"][:, 3:], self._last_state["root_transforms"][:, 3:]
                    )
                )
                self._read_cache[f"root_{vel_str}"] = vels / og.sim.get_physics_dt()
            else:
                self._read_cache[f"root_{vel_str}"] = cb.from_torch(self._view.get_root_velocities())

        return self._read_cache[f"root_{vel_str}"]

    def get_relative_linear_velocity(self, prim_path, estimate=False):
        # base corresponds to final index
        return self._get_relative_velocities(prim_path, estimate=estimate)[-1, :3]

    def get_relative_angular_velocity(self, prim_path, estimate=False):
        # base corresponds to final index
        return self._get_relative_velocities(prim_path, estimate=estimate)[-1, 3:]

    def get_link_index(self, link_name):
        """Returns the integer body index for the named link in the articulation view's link_paths."""
        return self._link_idx[0][link_name]

    def get_all_link_relative_position_orientation(self, link_name):
        """Returns (N, 3) positions and (N, 4) quaternions for the given link across all robots."""
        cache_key = f"all_link_rel_pose_{link_name}"
        if cache_key not in self._read_cache:
            link_idx = self._link_idx[0][link_name]
            # _get_all_relative_poses returns (N, n_links, 7); slice the desired link: (N, 7)
            poses = self._get_all_relative_poses()[:, link_idx, :]
            self._read_cache[cache_key] = poses
        poses = self._read_cache[cache_key]
        return poses[:, :3], poses[:, 3:]

    def get_all_link_relative_linear_velocity(self, link_name, estimate=False):
        """Returns (N, 3) link linear velocities for all robots."""
        cache_key = f"all_link_rel_lin_vel{'_est' if estimate else ''}_{link_name}"
        if cache_key not in self._read_cache:
            link_idx = self._link_idx[0][link_name]
            self._read_cache[cache_key] = self._get_all_relative_velocities(estimate=estimate)[:, link_idx, :3]
        return self._read_cache[cache_key]

    def get_all_link_relative_angular_velocity(self, link_name, estimate=False):
        """Returns (N, 3) link angular velocities for all robots."""
        cache_key = f"all_link_rel_ang_vel{'_est' if estimate else ''}_{link_name}"
        if cache_key not in self._read_cache:
            link_idx = self._link_idx[0][link_name]
            self._read_cache[cache_key] = self._get_all_relative_velocities(estimate=estimate)[:, link_idx, 3:]
        return self._read_cache[cache_key]

    def get_all_relative_linear_velocity(self, estimate=False):
        """Returns (N, 3) base linear velocities for all robots in this view."""
        cache_key = f"all_relative_lin_vel{'_est' if estimate else ''}"
        if cache_key not in self._read_cache:
            # Base is appended at the final index in _get_all_relative_velocities
            self._read_cache[cache_key] = self._get_all_relative_velocities(estimate=estimate)[:, -1, :3]
        return self._read_cache[cache_key]

    def get_all_relative_angular_velocity(self, estimate=False):
        """Returns (N, 3) base angular velocities for all robots in this view."""
        cache_key = f"all_relative_ang_vel{'_est' if estimate else ''}"
        if cache_key not in self._read_cache:
            # Base is appended at the final index in _get_all_relative_velocities
            self._read_cache[cache_key] = self._get_all_relative_velocities(estimate=estimate)[:, -1, 3:]
        return self._read_cache[cache_key]

    def get_all_joint_positions(self):
        """Returns (N, n_dof) joint positions for all robots in this view."""
        if "dof_positions" not in self._read_cache:
            self._read_cache["dof_positions"] = cb.from_torch(self._view.get_dof_positions())
        return self._read_cache["dof_positions"]

    def get_joint_positions(self, prim_path):
        return self.get_all_joint_positions()[self._idx[prim_path]]

    def get_all_joint_velocities(self, estimate=False):
        """Returns (N, n_dof) joint velocities for all robots in this view."""
        vel_str = "velocities_estimate" if estimate else "velocities"
        if f"dof_{vel_str}" not in self._read_cache:
            if estimate and self._last_state is not None:
                if "dof_positions" not in self._read_cache:
                    self._read_cache["dof_positions"] = cb.from_torch(self._view.get_dof_positions())
                self._read_cache[f"dof_{vel_str}"] = (
                    self._read_cache["dof_positions"] - self._last_state["dof_positions"]
                ) / og.sim.get_physics_dt()
            else:
                self._read_cache[f"dof_{vel_str}"] = cb.from_torch(self._view.get_dof_velocities())
        return self._read_cache[f"dof_{vel_str}"]

    def get_joint_velocities(self, prim_path, estimate=False):
        return self.get_all_joint_velocities(estimate=estimate)[self._idx[prim_path]]

    def get_all_joint_efforts(self):
        """Returns (N, n_dof) joint efforts for all robots in this view."""
        if "dof_projected_joint_forces" not in self._read_cache:
            self._read_cache["dof_projected_joint_forces"] = cb.from_torch(self._view.get_dof_projected_joint_forces())
        return self._read_cache["dof_projected_joint_forces"]

    def get_joint_efforts(self, prim_path):
        return self.get_all_joint_efforts()[self._idx[prim_path]]

    def get_all_generalized_mass_matrices(self):
        """Returns (N, n_dof, n_dof) mass matrices for all robots in this view."""
        if "mass_matrices" not in self._read_cache:
            self._read_cache["mass_matrices"] = cb.from_torch(self._view.get_generalized_mass_matrices())
        return self._read_cache["mass_matrices"]

    def get_generalized_mass_matrices(self, prim_path):
        return self.get_all_generalized_mass_matrices()[self._idx[prim_path]]

    def get_all_gravity_compensation_forces(self):
        """Returns (N, n_dof) gravity compensation forces for all robots in this view."""
        if "generalized_gravity_forces" not in self._read_cache:
            self._read_cache["generalized_gravity_forces"] = cb.from_torch(self._view.get_gravity_compensation_forces())
        return self._read_cache["generalized_gravity_forces"]

    def get_gravity_compensation_forces(self, prim_path):
        return self.get_all_gravity_compensation_forces()[self._idx[prim_path]]

    def get_all_coriolis_and_centrifugal_compensation_forces(self):
        """Returns (N, n_dof) Coriolis/centrifugal forces for all robots in this view."""
        if "coriolis_and_centrifugal_forces" not in self._read_cache:
            self._read_cache["coriolis_and_centrifugal_forces"] = cb.from_torch(
                self._view.get_coriolis_and_centrifugal_compensation_forces()
            )
        return self._read_cache["coriolis_and_centrifugal_forces"]

    def get_coriolis_and_centrifugal_compensation_forces(self, prim_path):
        return self.get_all_coriolis_and_centrifugal_compensation_forces()[self._idx[prim_path]]

    def get_link_transform(self, prim_path, link_name):
        idx = self._idx[prim_path]
        pos, quat = self.get_all_link_transform(link_name)
        return pos[idx], quat[idx]

    def get_all_link_transform(self, link_name):
        if "link_transforms" not in self._read_cache:
            self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())

        # We assume that in a view, link_idx for the same link_name is the same across all members
        link_idx = self._link_idx[0][link_name]
        pose = self._read_cache["link_transforms"][:, link_idx]
        return pose[:, :3], pose[:, 3:]

    def _get_relative_poses(self, prim_path):
        idx = self._idx[prim_path]
        return self._get_all_relative_poses()[idx]

    def get_link_relative_position_orientation(self, prim_path, link_name):
        idx = self._idx[prim_path]
        pos, quat = self.get_all_link_relative_position_orientation(link_name)
        return pos[idx], quat[idx]

    def _get_all_link_velocities(self, link_name, estimate=False):
        """Returns (N, 6) velocities (linear + angular) for the given link across all robots."""
        vel_str = "velocities_estimate" if estimate else "velocities"

        # Build and cache the full (N, L, 6) tensor for all robots and all links
        if f"link_{vel_str}" not in self._read_cache:
            if estimate and self._last_state is not None:
                # Compute link velocities estimate as delta between prior timestep and current timestep
                N, L, _ = self._last_state["link_transforms"].shape
                vels = cb.zeros((N, L, 6))

                if "link_transforms" not in self._read_cache:
                    self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())

                vels[:, :, :3] = (
                    self._read_cache["link_transforms"][:, :, :3] - self._last_state["link_transforms"][:, :, :3]
                )
                vels[:, :, 3:] = cb.view(
                    cb.T.quat2axisangle(
                        cb.T.quat_distance(
                            cb.view(self._read_cache["link_transforms"][:, :, 3:], (-1, 4)),
                            cb.view(self._last_state["link_transforms"][:, :, 3:], (-1, 4)),
                        )
                    ),
                    (N, L, 3),
                )
                self._read_cache[f"link_{vel_str}"] = vels / og.sim.get_physics_dt()

            # Otherwise, directly grab velocities
            else:
                self._read_cache[f"link_{vel_str}"] = cb.from_torch(self._view.get_link_velocities())

        link_idx = self._link_idx[0][link_name]
        return self._read_cache[f"link_{vel_str}"][:, link_idx, :]  # (N, 6)

    def _get_link_velocities(self, prim_path, link_name, estimate=False):
        idx = self._idx[prim_path]
        return self._get_all_link_velocities(link_name, estimate=estimate)[idx]

    def get_link_linear_velocity(self, prim_path, link_name, estimate=False):
        return self._get_link_velocities(prim_path, link_name, estimate=estimate)[:3]

    def get_all_link_linear_velocity(self, link_name, estimate=False):
        return self._get_all_link_velocities(link_name, estimate=estimate)[:, :3]

    def get_link_relative_linear_velocity(self, prim_path, link_name, estimate=False):
        idx = self._idx[prim_path]
        link_idx = self._link_idx[idx][link_name]
        return self._get_relative_velocities(prim_path, estimate=estimate)[link_idx, :3]

    def get_all_link_angular_velocity(self, link_name, estimate=False):
        return self._get_all_link_velocities(link_name, estimate=estimate)[:, 3:]

    def get_link_relative_angular_velocity(self, prim_path, link_name, estimate=False):
        idx = self._idx[prim_path]
        link_idx = self._link_idx[idx][link_name]
        return self._get_relative_velocities(prim_path, estimate=estimate)[link_idx, 3:]

    def get_all_jacobian(self):
        if "jacobians" not in self._read_cache:
            self._read_cache["jacobians"] = cb.from_torch(self._view.get_jacobians())
        return self._read_cache["jacobians"]

    def get_jacobian(self, prim_path):
        idx = self._idx[prim_path]
        return self.get_all_jacobian()[idx]

    def _get_all_relative_poses(self):
        """Returns (N, n_links, 7) relative poses (pos + quat) for all robots in this view, batched."""
        if "relative_poses" not in self._read_cache:
            # All link world transforms: (N, n_links, 7)
            if "link_transforms" not in self._read_cache:
                self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())
            all_link_tfs = cb.to_torch(self._read_cache["link_transforms"])  # (N, n_links, 7)

            # All base poses
            all_pos, all_quat = self.get_all_position_orientation()  # (N, 3), (N, 4)
            all_pos = cb.to_torch(all_pos)
            all_quat = cb.to_torch(all_quat)

            N, n_links = all_link_tfs.shape[:2]

            # Build link homogeneous transform matrices: (N, n_links, 4, 4)
            tfs = th.zeros(N, n_links, 4, 4, dtype=th.float32)
            tfs[:, :, 3, 3] = 1.0
            tfs[:, :, :3, 3] = all_link_tfs[:, :, :3]
            # quat2mat doesn't handle rank-3 input; flatten the N*n_links batch dimension
            tfs[:, :, :3, :3] = TT.quat2mat(all_link_tfs[:, :, 3:].reshape(-1, 4)).reshape(N, n_links, 3, 3)

            # Build batched base pose inverses: (N, 4, 4)
            # For a rigid transform [R, t; 0, 1], the inverse is [R^T, -R^T t; 0, 1]
            base_rot_T = TT.quat2mat(all_quat).transpose(-2, -1)  # (N, 3, 3)
            base_tf_inv = th.zeros(N, 4, 4, dtype=th.float32)
            base_tf_inv[:, 3, 3] = 1.0
            base_tf_inv[:, :3, :3] = base_rot_T
            base_tf_inv[:, :3, 3] = -(base_rot_T @ all_pos.unsqueeze(-1)).squeeze(-1)

            # Batched matmul: (N, 1, 4, 4) @ (N, n_links, 4, 4) → (N, n_links, 4, 4)
            rel_tfs = base_tf_inv.unsqueeze(1) @ tfs

            # Convert back to (N, n_links, 7) pos + quat
            rel_poses = th.zeros(N, n_links, 7, dtype=th.float32)
            rel_poses[:, :, :3] = rel_tfs[:, :, :3, 3]
            rel_poses[:, :, 3:] = TT.mat2quat(rel_tfs[:, :, :3, :3].reshape(-1, 3, 3)).reshape(N, n_links, 4)

            self._read_cache["relative_poses"] = cb.from_torch(rel_poses)
        return self._read_cache["relative_poses"]

    def get_all_relative_jacobians(self):
        """Returns (N, n_links, 6, n_dof_total) relative jacobians for all robots in this view."""
        if "relative_jacobians" not in self._read_cache:
            # All raw jacobians: (N, n_links, 6, n_dof_total)
            all_jacobians = cb.to_torch(self.get_all_jacobian())
            # Base orientation quaternions for all robots: (N, 4)
            all_quats = cb.to_torch(self.get_all_position_orientation()[1])
            N = all_quats.shape[0]

            # Rotation matrices transposed per robot: (N, 3, 3)
            ori_t_batch = TT.quat2mat(all_quats).transpose(-2, -1)

            # Build block-diagonal transform tf = [[ori_t, 0], [0, ori_t]]: (N, 6, 6)
            tf = th.zeros(N, 6, 6, dtype=all_jacobians.dtype)
            tf[:, :3, :3] = ori_t_batch
            tf[:, 3:, 3:] = ori_t_batch

            # Batched matmul: (N, 1, 6, 6) @ (N, n_links, 6, n_dof_total) → (N, n_links, 6, n_dof_total)
            # Run in pytorch since it's order of magnitude faster than numpy!
            self._read_cache["relative_jacobians"] = cb.from_torch(tf.unsqueeze(1) @ all_jacobians)
        return self._read_cache["relative_jacobians"]

    def get_relative_jacobian(self, prim_path):
        idx = self._idx[prim_path]
        return self.get_all_relative_jacobians()[idx]


def get_robot_kinematic_tree_pattern(articulation_root_path: str) -> str:
    """
    Returns a glob pattern that matches all robots of the same type and fixedness as the
    given articulation root path.

    The pattern generalizes over scene index and robot instance name, preserving the
    robot-type component and any path suffix (e.g. base link name for floating-base robots).

    Examples:
        "/World/scene_0/controllable__fetch__robot0"
            -> "/World/scene_*/controllable__fetch__*"
        "/World/scene_0/controllable__fetch__robot0/base_link"
            -> "/World/scene_*/controllable__fetch__*/base_link"
    """
    scene_id, robot_name = articulation_root_path.split("/")[2:4]
    assert scene_id.startswith("scene_"), f"Prim path 2nd component {articulation_root_path} does not start with scene_"
    components = robot_name.split("__")
    assert len(components) == 3, (
        f"Robot prim path's 3rd component {robot_name} does not match "
        "expected format of prefix__robottype__robotname."
    )
    assert (
        components[0] == "controllable"
    ), f"Prim path {articulation_root_path} 3rd component does not start with 'controllable__'"
    return articulation_root_path.replace(f"/{scene_id}/", "/scene_*/").replace(
        f"/{robot_name}", f"/{components[0]}__{components[1]}__*"
    )


class ControllableObjectViewAPI:
    """
    An interface that creates BatchControlViewAPIImpl instances for each robot type in the scene.

    This is done to avoid the overhead of reading from many views for each robot in each physics step,
    providing major speed improvements in vector env use cases.

    This class is a singleton, and should be used to access the BatchControlViewAPIImpl instances.

    The pattern used to group the robots is based on the robot prim paths, which is assumed to be in the format
    /World/scene_*/controllable__robottype__robotname.

    The patterns used by the subviews are generated by replacing the robot name with a wildcard, so that all robots
    of the same type are grouped together. If there are fixed base robots, they will be grouped separately from
    non-fixed base robots even within the same robot type, by virtue of their different articulation root paths.
    """

    # Dictionary mapping from pattern to BatchControlViewAPIImpl
    _VIEWS_BY_PATTERN = {}

    @classmethod
    def post_physics_step(cls):
        for view in cls._VIEWS_BY_PATTERN.values():
            view.post_physics_step()

    @classmethod
    def clear(cls):
        for view in cls._VIEWS_BY_PATTERN.values():
            view.clear()

    @classmethod
    def clear_object(cls, prim_path):
        if get_robot_kinematic_tree_pattern(prim_path) in cls._VIEWS_BY_PATTERN:
            cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].clear()

    @classmethod
    def flush_control(cls):
        for view in cls._VIEWS_BY_PATTERN.values():
            view.flush_control()

    @classmethod
    def initialize_view(cls):
        cls._VIEWS_BY_PATTERN = {}

        # First, get all of the controllable objects in the scene (avoiding circular import)
        from omnigibson.robots import Robot

        controllable_objects = [obj for scene in og.sim.scenes for obj in scene.objects if isinstance(obj, Robot)]

        # Get their corresponding prim paths
        expected_prim_paths = {obj.articulation_root_path for obj in controllable_objects}

        # Group the prim paths by robot type
        patterns = {get_robot_kinematic_tree_pattern(prim_path) for prim_path in expected_prim_paths}

        # Create the view for each robot type / fixedness combo
        for pattern in patterns:
            if pattern not in cls._VIEWS_BY_PATTERN:
                cls._VIEWS_BY_PATTERN[pattern] = BatchControlViewAPIImpl(pattern)

        # Initialize the views
        for view in cls._VIEWS_BY_PATTERN.values():
            view.initialize_view()

        # Assert that the views' prim paths are disjoint
        all_prim_paths = []
        for view in cls._VIEWS_BY_PATTERN.values():
            all_prim_paths.extend(view._idx.keys())
        counts = collections.Counter(all_prim_paths)

        missing = set(expected_prim_paths) - set(all_prim_paths)
        assert len(missing) == 0, f"Prim paths {missing} are missing from the views!"

        more_than_once = {prim_path: count for prim_path, count in counts.items() if count > 1}
        assert len(more_than_once) == 0, f"Prim paths {more_than_once} are present in multiple views!"

    @classmethod
    def set_joint_position_targets(cls, prim_path, positions, indices):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].set_joint_position_targets(
            prim_path, positions, indices
        )

    @classmethod
    def set_joint_velocity_targets(cls, prim_path, velocities, indices):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].set_joint_velocity_targets(
            prim_path, velocities, indices
        )

    @classmethod
    def set_joint_efforts(cls, prim_path, efforts, indices):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].set_joint_efforts(
            prim_path, efforts, indices
        )

    @classmethod
    def get_member_view_indices(cls, routing_path, prim_paths):
        """Return view row indices for prim_paths (all in same view as routing_path)."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(routing_path)].get_member_view_indices(prim_paths)

    @classmethod
    def set_all_joint_position_targets(cls, routing_path, enabled_rows, controls, dof_idx):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(routing_path)].set_all_joint_position_targets(
            enabled_rows, controls, dof_idx
        )

    @classmethod
    def set_all_joint_velocity_targets(cls, routing_path, enabled_rows, velocities, dof_idx):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(routing_path)].set_all_joint_velocity_targets(
            enabled_rows, velocities, dof_idx
        )

    @classmethod
    def set_all_joint_efforts(cls, routing_path, enabled_rows, efforts, dof_idx):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(routing_path)].set_all_joint_efforts(
            enabled_rows, efforts, dof_idx
        )

    @classmethod
    def get_position_orientation(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_position_orientation(prim_path)

    @classmethod
    def get_root_position_orientation(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_root_transform(prim_path)

    @classmethod
    def get_linear_velocity(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_linear_velocity(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_angular_velocity(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_angular_velocity(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_all_joint_positions(cls, prim_path):
        """Returns (N, n_dof) joint positions for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_joint_positions()

    @classmethod
    def get_joint_positions(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_joint_positions(prim_path)

    @classmethod
    def get_all_joint_velocities(cls, prim_path, estimate=False):
        """Returns (N, n_dof) joint velocities for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_joint_velocities(
            estimate=estimate
        )

    @classmethod
    def get_joint_velocities(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_joint_velocities(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_all_joint_efforts(cls, prim_path):
        """Returns (N, n_dof) joint efforts for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_joint_efforts()

    @classmethod
    def get_joint_efforts(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_joint_efforts(prim_path)

    @classmethod
    def get_all_generalized_mass_matrices(cls, prim_path):
        """Returns (N, n_dof, n_dof) mass matrices for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_generalized_mass_matrices()

    @classmethod
    def get_all_gravity_compensation_forces(cls, prim_path):
        """Returns (N, n_dof) gravity compensation forces for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_gravity_compensation_forces()

    @classmethod
    def get_all_coriolis_and_centrifugal_compensation_forces(cls, prim_path):
        """Returns (N, n_dof) Coriolis/centrifugal forces for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[
            get_robot_kinematic_tree_pattern(prim_path)
        ].get_all_coriolis_and_centrifugal_compensation_forces()

    @classmethod
    def get_link_relative_position_orientation(cls, prim_path, link_name):
        return cls._VIEWS_BY_PATTERN[
            get_robot_kinematic_tree_pattern(prim_path)
        ].get_link_relative_position_orientation(prim_path, link_name)

    @classmethod
    def get_link_index(cls, prim_path, link_name):
        """Returns the integer body index for the named link in the articulation view's link_paths."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_link_index(link_name)

    @classmethod
    def get_all_relative_jacobians(cls, prim_path):
        """Returns (N, n_links, 6, n_dof_total) relative jacobians for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_relative_jacobians()

    @classmethod
    def get_all_link_relative_position_orientation(cls, prim_path, link_name):
        """Returns (N, 3) positions and (N, 4) quaternions for the given link across all robots."""
        return cls._VIEWS_BY_PATTERN[
            get_robot_kinematic_tree_pattern(prim_path)
        ].get_all_link_relative_position_orientation(link_name)

    @classmethod
    def get_all_link_relative_linear_velocity(cls, prim_path, link_name, estimate=False):
        """Returns (N, 3) link linear velocities for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_link_relative_linear_velocity(
            link_name, estimate=estimate
        )

    @classmethod
    def get_all_link_relative_angular_velocity(cls, prim_path, link_name, estimate=False):
        """Returns (N, 3) link angular velocities for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[
            get_robot_kinematic_tree_pattern(prim_path)
        ].get_all_link_relative_angular_velocity(link_name, estimate=estimate)

    @classmethod
    def get_all_relative_linear_velocity(cls, prim_path, estimate=False):
        """Returns (N, 3) base linear velocities for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_relative_linear_velocity(
            estimate=estimate
        )

    @classmethod
    def get_all_relative_angular_velocity(cls, prim_path, estimate=False):
        """Returns (N, 3) base angular velocities for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_relative_angular_velocity(
            estimate=estimate
        )


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
    face_vertex_counts = vtarray_to_torch(mesh_prim.GetAttribute("faceVertexCounts").Get(), dtype=th.int)
    vertices = vtarray_to_torch(mesh_prim.GetAttribute("points").Get())
    face_indices = vtarray_to_torch(mesh_prim.GetAttribute("faceVertexIndices").Get(), dtype=th.int)

    faces = []
    i = 0
    for count in face_vertex_counts:
        for j in range(count - 2):
            faces.append([face_indices[i], face_indices[i + j + 1], face_indices[i + j + 2]])
        i += count

    kwargs = dict(vertices=vertices, faces=faces)

    if include_normals:
        kwargs["vertex_normals"] = vtarray_to_torch(mesh_prim.GetAttribute("normals").Get())

    if include_texcoord:
        raw_texture = mesh_prim.GetAttribute("primvars:st").Get()
        if raw_texture is not None:
            kwargs["visual"] = trimesh.visual.TextureVisuals(uv=vtarray_to_torch(raw_texture))

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
        th.manual_seed(seed)

    # Generate trimesh mesh from which to aggregate points
    tm = mesh_prim_mesh_to_trimesh_mesh(mesh_prim=mesh_prim, include_normals=False, include_texcoord=False)
    n_unique_vertices, n_unique_faces = len(tm.vertices), len(tm.faces)
    faces_flat = th.tensor(tm.faces.flatten(), dtype=th.int32)

    # Sample vertices
    unique_vertices = th.unique(faces_flat)
    assert len(unique_vertices) == n_unique_vertices
    keypoint_idx = (
        th.randperm(len(unique_vertices))[:n_keypoints] if n_unique_vertices > n_keypoints else unique_vertices
    )

    # Sample faces
    keyface_idx = th.randperm(n_unique_faces)[:n_keyfaces] if n_unique_faces > n_keyfaces else th.arange(n_unique_faces)

    return keypoint_idx, keyface_idx


def get_mesh_volume_and_com(mesh_prim, world_frame=False):
    """
    Computes the volume and center of mass for @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to compute volume and center of mass for
        world_frame (bool): Whether to return the volume and CoM in the world frame

    Returns:
        Tuple[float, th.tensor]: Tuple containing the (volume, center_of_mass) in the mesh frame or the world frame
    """

    trimesh_mesh = mesh_prim_to_trimesh_mesh(
        mesh_prim, include_normals=False, include_texcoord=False, world_frame=world_frame
    )
    if trimesh_mesh.is_volume:
        volume = trimesh_mesh.volume
        com = th.tensor(trimesh_mesh.center_mass)
    else:
        # If the mesh is not a volume, we compute its convex hull and use that instead
        try:
            trimesh_mesh_convex = trimesh_mesh.convex_hull
            volume = trimesh_mesh_convex.volume
            com = th.tensor(trimesh_mesh_convex.center_mass)
        except:
            # if convex hull computation fails, it usually means the mesh is degenerated: use trivial values.
            volume = 0.0
            com = th.zeros(3)

    return volume, com.to(dtype=th.float32)


def check_extent_radius_ratio(geom_prim, com):
    """
    Checks if the min extent in world frame and the extent radius ratio in local frame of @geom_prim is within the
    acceptable range for PhysX GPU acceleration (not too thin, and not too oblong)

    Ref: https://github.com/NVIDIA-Omniverse/PhysX/blob/561a0df858d7e48879cdf7eeb54cfe208f660f18/physx/source/geomutils/src/convex/GuConvexMeshData.h#L183-L190

    Args:
        geom_prim (GeomPrim): Geom prim to check
        com (th.tensor): Center of mass of the mesh. Obtained from get_mesh_volume_and_com

    Returns:
        bool: True if the min extent (world) and the extent radius ratio (local frame) is acceptable, False otherwise
    """
    mesh_type = geom_prim.prim.GetPrimTypeInfo().GetTypeName()
    # Non-mesh prims are always considered to be within the acceptable range
    if mesh_type != "Mesh":
        return True

    extent = geom_prim.extent
    min_extent = extent.min()
    # If the mesh is too flat in the world frame, omniverse cannot create convex mesh for it
    if min_extent < 1e-5:
        return False

    max_radius = extent.max() / 2.0
    min_radius = th.min(th.norm(geom_prim.points - com, dim=-1), dim=0).values
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
    extents = th.ones(3) * extents if isinstance(extents, float) else th.tensor(extents)
    for attr in (mesh.GetPointsAttr(), mesh.GetNormalsAttr()):
        vals = th.tensor(attr.Get()).double()
        attr.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*(val * extents * 50.0).tolist()) for val in vals]))
    mesh.GetExtentAttr().Set(
        lazy.pxr.Vt.Vec3fArray(
            [lazy.pxr.Gf.Vec3f(*(-extents / 2.0).tolist()), lazy.pxr.Gf.Vec3f(*(extents / 2.0).tolist())]
        )
    )

    return triangularize_mesh(mesh)


def triangularize_mesh(mesh):
    """
    Triangulates the mesh @mesh, modification in-place
    """
    tm = mesh_prim_to_trimesh_mesh(mesh.GetPrim())

    face_vertex_counts = np.array([len(face) for face in tm.faces], dtype=int)
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(tm.faces.flatten())
    mesh.GetNormalsAttr().Set(lazy.pxr.Vt.Vec3fArray.FromNumpy(tm.vertex_normals[tm.faces.flatten()]))

    # Modify the UV mapping if it exists
    if isinstance(tm.visual, trimesh.visual.TextureVisuals):
        mesh.GetPrim().GetAttribute("primvars:st").Set(
            lazy.pxr.Vt.Vec2fArray.FromNumpy(tm.visual.uv[tm.faces.flatten()])
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
    asset_type = asset_path.split(".")[-1]
    assert asset_type in {"usd", "usda", "obj"}, "Cannot load a non-USD or non-OBJ file as a USD prim!"

    # Make sure the path exists
    assert os.path.exists(asset_path), f"Cannot load {asset_type.upper()} file {asset_path} because it does not exist!"

    # Add reference to stage and grab prim
    lazy.isaacsim.core.utils.stage.add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
    prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)

    # Make sure prim was loaded correctly
    assert prim, f"Failed to load {asset_type.upper()} object from path: {asset_path}"

    return prim


def get_world_prim():
    """
    Returns:
        Usd.Prim: Active world prim in the current stage
    """
    return lazy.isaacsim.core.utils.prims.get_prim_at_path("/World")


def scene_relative_prim_path_to_absolute(scene, relative_prim_path):
    """
    Converts a scene-relative prim path to an absolute prim path.

    Args:
        scene (Scene or None): Scene object that the prim is in. None if it's global.
        relative_prim_path (str): Relative prim path in the scene

    Returns:
        str: Absolute prim path in the stage
    """
    # Special case for OmniGraph prims
    if relative_prim_path.startswith("/OmniGraph"):
        return relative_prim_path

    # Make sure the relative path is actually relative
    assert not relative_prim_path.startswith("/World"), f"Expected relative prim path, got {relative_prim_path}"

    # When the scene is set to None, this prim is not in a scene but is global e.g. like the
    # viewer camera or one of the scene prims.
    if scene is None:
        return "/World" + relative_prim_path

    return scene.prim_path + relative_prim_path


def absolute_prim_path_to_scene_relative(scene, absolute_prim_path):
    """
    Converts an absolute prim path to a scene-relative prim path.

    Args:
        scene (Scene): Scene object that the prim is in. None if it's global.
        absolute_prim_path (str): Absolute prim path in the stage

    Returns:
        str: Relative prim path in the scene
    """
    # Special case for OmniGraph prims
    if absolute_prim_path.startswith("/OmniGraph"):
        return absolute_prim_path

    assert absolute_prim_path.startswith("/World"), f"Expected absolute prim path, got {absolute_prim_path}"

    # When the scene is set to None, this prim is not in a scene but is global e.g. like the
    # viewer camera or one of the scene prims.
    if scene is None:
        assert not absolute_prim_path.startswith(
            "/World/scene_"
        ), f"Expected global prim path, got {absolute_prim_path}"
        return absolute_prim_path[len("/World") :]

    return absolute_prim_path[len(scene.prim_path) :]


def deep_copy_prim(source_root_prim, dest_stage, dest_root_path):
    queue = [(source_root_prim, dest_root_path)]

    while queue:
        source_prim, dest_path = queue.pop(0)

        # Create a new prim in the destination stage with the same type as the source
        if source_prim.GetTypeName():
            dest_prim = dest_stage.DefinePrim(dest_path, source_prim.GetTypeName())
        else:
            dest_prim = dest_stage.OverridePrim(dest_path)

        # Copy attributes
        for attr in source_prim.GetAttributes():
            # Create a new attribute with the same specifications
            dest_attr = dest_prim.CreateAttribute(
                attr.GetName(), attr.GetTypeName(), attr.IsCustom(), attr.GetVariability()
            )

            # Check if the source attribute has a value
            if attr.HasValue():
                # Copy the value
                dest_attr.Set(attr.Get())

        # Copy relationships
        for rel in source_prim.GetRelationships():
            dest_rel = dest_prim.CreateRelationship(rel.GetName(), rel.IsCustom())
            targets = rel.GetTargets()
            updated_targets = [
                x.ReplacePrefix(source_root_prim.GetPath(), lazy.pxr.Sdf.Path(dest_root_path)) for x in targets
            ]
            if targets:
                dest_rel.SetTargets(updated_targets)

        # Copy child prims breadth-first
        for child in source_prim.GetAllChildren():
            new_dest_path = dest_path + "/" + child.GetName()
            queue.append((child, new_dest_path))


def delete_or_deactivate_prim(prim_path):
    """
    Attept to delete or deactivate the prim defined at @prim_path.

    Args:
        prim_path (str): Path defining which prim should be deleted or deactivated

    Returns:
        bool: Whether the operation was successful or not
    """
    if not lazy.isaacsim.core.utils.prims.is_prim_path_valid(prim_path):
        return False
    if lazy.isaacsim.core.utils.prims.is_prim_no_delete(prim_path):
        return False
    if lazy.isaacsim.core.utils.prims.get_prim_type_name(prim_path=prim_path) == "PhysicsScene":
        return False
    if prim_path == "/World":
        return False
    if prim_path == "/":
        return False
    # Don't remove any /Render prims as that can cause crashes
    if prim_path.startswith("/Render"):
        return False

    # If the prim is not ancestral, we can delete it.
    if not lazy.isaacsim.core.utils.prims.is_prim_ancestral(prim_path):
        lazy.omni.usd.commands.DeletePrimsCommand([prim_path], destructive=True).do()

    # Otherwise, we can only deactivate it, which essentially serves the same purpose.
    # All objects that are originally in the scene are ancestral because we add the pre-build scene to the stage.
    else:
        # Clear all default attributes before deactivating the prim to ensure clean reactivation.
        # Note: Prim deactivation preserves attribute values, so we must explicitly clear defaults
        # to prevent stale custom values from persisting when the prim is reactivated later.
        prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)
        for attr in prim.GetAttributes():
            assert attr.ClearDefault()
        lazy.omni.usd.commands.DeletePrimsCommand([prim_path], destructive=False).do()

    return True


def activate_prim_and_children(prim_path):
    """
    Recursively activates the prim at @prim_path and all of its children.

    Args:
        prim_path (str): Path to the prim to activate
    """
    current_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)
    current_prim.SetActive(True)
    # Use GetAllChildren to also find those that are inactive
    for child in current_prim.GetAllChildren():
        activate_prim_and_children(child.GetPath().pathString)


def get_sdf_value_type_name(val):
    """
    Determines the appropriate Sdf value type based on the input value.
    Args:
        val: The input value to determine the type for.
    Returns:
        lazy.pxr.Sdf.ValueTypeName: The corresponding Sdf value type.
    Raises:
        ValueError: If the input value type is not supported.
    """
    SDF_TYPE_MAPPING = {
        lazy.pxr.Gf.Vec3f: lazy.pxr.Sdf.ValueTypeNames.Float3,
        lazy.pxr.Gf.Vec2f: lazy.pxr.Sdf.ValueTypeNames.Float2,
        lazy.pxr.Sdf.AssetPath: lazy.pxr.Sdf.ValueTypeNames.Asset,
        bool: lazy.pxr.Sdf.ValueTypeNames.Bool,
        int: lazy.pxr.Sdf.ValueTypeNames.Int,
        float: lazy.pxr.Sdf.ValueTypeNames.Float,
        str: lazy.pxr.Sdf.ValueTypeNames.String,
    }
    for type_, usd_type in SDF_TYPE_MAPPING.items():
        if isinstance(val, type_):
            return usd_type
    raise ValueError(f"Unsupported input type: {type(val)}")


def replace_collision_blocks(old_usd_path: str, new_usd_path: str, output_usd_path: str):
    """
    Replace all collisions blocks in new_usd_path with those from old_usd_path.
    """

    def extract_collision_blocks(text):
        """
        Extract all top-level 'def [Mesh] "collisions"' blocks using brace matching.
        Returns a list of (start_idx, end_idx, block_text)
        """
        blocks = []
        lines = text.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("def") and '"collisions"' in line:
                start = i
                brace_count = 0
                # Find the opening brace
                while "{" not in lines[i]:
                    i += 1
                    if i >= len(lines):
                        break
                if i >= len(lines):
                    break
                brace_count += lines[i].count("{") - lines[i].count("}")
                i += 1
                # Count braces to find block end
                while brace_count > 0 and i < len(lines):
                    brace_count += lines[i].count("{") - lines[i].count("}")
                    i += 1
                end = i
                block_text = "".join(lines[start:end])
                blocks.append((start, end, block_text))
            else:
                i += 1
        return blocks

    # Load USDA files
    with open(old_usd_path, "r") as f:
        source_usda = f.read()
    with open(new_usd_path, "r") as f:
        target_usda = f.read()

    # Extract collision blocks
    source_collision_blocks = extract_collision_blocks(source_usda)
    target_blocks = extract_collision_blocks(target_usda)

    # Replace in target
    if len(target_blocks) != len(source_collision_blocks):
        print(f"Warning: Replacing {min(len(target_blocks), len(source_collision_blocks))} blocks due to mismatch.")

    new_lines = []
    last_idx = 0
    target_lines = target_usda.splitlines(keepends=True)
    for (start, end, _), (_, _, new_block) in zip(target_blocks, source_collision_blocks):
        new_lines.extend(target_lines[last_idx:start])
        new_lines.append(new_block)
        last_idx = end
    new_lines.extend(target_lines[last_idx:])

    new_usda_text = "".join(new_lines)

    # Save result
    with open(output_usd_path, "w") as f:
        f.write(new_usda_text)

    print(f"Finished replacing all {len(source_collision_blocks)} collision blocks.")


@torch_compile
def _compute_relative_poses_torch(
    idx: int,
    n_links: int,
    all_tfs: th.Tensor,
    base_pose: Tuple[th.Tensor, th.Tensor],
):
    tfs = th.zeros((n_links, 4, 4), dtype=th.float32)
    # base vel is the final -1 index
    link_tfs = all_tfs[idx, :]
    tfs[:, 3, 3] = 1.0
    tfs[:, :3, 3] = link_tfs[:, :3]
    tfs[:, :3, :3] = TT.quat2mat(link_tfs[:, 3:])
    base_tf_inv = th.zeros((1, 4, 4), dtype=th.float32)
    base_tf_inv[0, :, :] = TT.pose_inv(TT.pose2mat(base_pose))

    # (1, 4, 4) @ (n_links, 4, 4) -> (n_links, 4, 4)
    rel_tfs = base_tf_inv @ tfs

    # Re-convert to quat form
    rel_poses = th.zeros((n_links, 7), dtype=th.float32)
    rel_poses[:, :3] = rel_tfs[:, :3, 3]
    rel_poses[:, 3:] = TT.mat2quat(rel_tfs[:, :3, :3])

    return rel_poses


@jit(nopython=True)
def _compute_relative_poses_numpy(idx, n_links, all_tfs, base_pose):
    tfs = np.zeros((n_links, 4, 4), dtype=np.float32)
    # base vel is the final -1 index
    link_tfs = all_tfs[idx, :]
    tfs[:, 3, 3] = 1.0
    tfs[:, :3, 3] = link_tfs[:, :3]
    tfs[:, :3, :3] = NT._quat2mat(link_tfs[:, 3:])
    # base_tf_inv = np.zeros((1, 4, 4), dtype=np.float32)
    # base_tf_inv[0, :, :] = NT._pose_inv(NT.pose2mat(base_pose))
    base_tf_inv = NT._pose_inv(NT.pose2mat(base_pose))

    # (1, 4, 4) @ (n_links, 4, 4) -> (n_links, 4, 4)
    rel_tfs = np.zeros((n_links, 4, 4), dtype=np.float32)
    for i in prange(n_links):
        rel_tfs[i, :, :] = base_tf_inv @ tfs[i, :, :]
    # rel_tfs = base_tf_inv @ tfs

    # Re-convert to quat form
    rel_poses = np.zeros((n_links, 7), dtype=np.float32)
    rel_poses[:, :3] = rel_tfs[:, :3, 3]
    rel_poses[:, 3:] = NT.mat2quat_batch(rel_tfs[:, :3, :3].copy())

    return rel_poses


# Set these as part of the backend values
add_compute_function(
    name="compute_relative_poses", np_function=_compute_relative_poses_numpy, th_function=_compute_relative_poses_torch
)
