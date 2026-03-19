import collections
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
    Class containing class methods to aggregate rigid body contacts across all rigid bodies in the simulator.

    This API checks for contacts on every physics step, and then aggregates this into a boolean contact matrix
    on every non-physics step. Callers can then use this API to query either for contacts that are still ongoing,
    or contacts that occurred at any point since the last non-physics step (e.g. for checking for contact events).
    Contact information is cached per-physics-step and only updated for body pairs who have at least one side
    not asleep, which allows this API to bypass the limitations of the view (which returns contacts only for awake bodies).

    Since there is no direct tensorized way to check for object sleep state, this API approximates this by checking for
    the net contact force on an object (only reported for awake bodies) and also the position change since the last step.
    """

    def __init__(self):
        # Dictionary mapping rigid body prim path to corresponding row / col index in the contact view matrix
        self._PATH_TO_ROW_IDX = dict()
        self._PATH_TO_COL_IDX = dict()

        # Arrays of rigid body prim paths where each array index maps directly to the contact matrix row / col
        self._ROW_IDX_TO_PATH = dict()
        self._COL_IDX_TO_PATH = dict()

        # Contact view for generating contact matrices at each timestep
        self._CONTACT_VIEW = dict()

        # Rigid body view for batched body transform reads used by persistence logic
        self._RIGID_BODY_VIEW = dict()

        # Precomputed tensors mapping row/col indices to rigid body view indices
        self._CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS = dict()
        self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS = dict()
        self._CONTACT_MATRIX_COLS_HAS_RIGID_BODY = dict()

        # Contact matrix tracking contacts that occurred at any point during the last N physics steps
        # (between consecutive update_contact_cache calls). Shape: (R, C)
        self._CONTACT_MATRIX = dict()

        # Contact matrix tracking contacts at only the most recent physics step. Shape: (R, C)
        self._CURRENT_CONTACT_MATRIX = dict()

        # A matrix of indices for the contact matrix. This can be indexed the same way as the contact matrix
        # to obtain row and column indices to map back to prim paths. Shape: (R, C, 2)
        self._INDEX_MATRIX = dict()

        # Cached body transforms used for change detection. Shape: (N, 7) [pos(3), quat(4)]
        self._BODY_TRANSFORMS = dict()

        # Accumulated impulse matrices and transforms from individual physics steps,
        # collected between consecutive update_contact_cache calls.
        self._PENDING_IMPULSES = dict()
        self._PENDING_TRANSFORMS = dict()
        self._PENDING_NET_FORCES = dict()

        # Position / orientation tolerances for deciding whether a pair should be updated
        self._POS_EPS = 1e-6
        self._ORI_EPS = 1e-4

    @classmethod
    def get_body_filters(cls):
        filters = dict()
        for scene_idx, scene in enumerate(og.sim.scenes):
            filters[scene_idx] = []
            for obj in scene.objects:
                if obj.prim_type == PrimType.RIGID:
                    for link in obj.links.values():
                        from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
                        from omnigibson.prims.rigid_kinematic_prim import RigidKinematicPrim

                        if isinstance(link, (RigidDynamicPrim, RigidKinematicPrim)) and link.contact_reporting_enabled:
                            filters[scene_idx].append(link.prim_path)

        return filters

    @classmethod
    def get_max_contact_data_count(cls, n_bodies):
        return 256

    def initialize_view(self):
        """
        Initializes the rigid contact view. Note: Can only be done when sim is playing!
        """
        assert og.sim.is_playing(), "Cannot create rigid contact view while sim is not playing!"

        # Snapshot the old contact matrices and path mappings so we can carry over
        # cached contact state for pairs of bodies that already existed.
        prev_contact_matrix = dict(self._CONTACT_MATRIX)
        prev_current_contact_matrix = dict(self._CURRENT_CONTACT_MATRIX)
        prev_path_to_row_idx = dict(self._PATH_TO_ROW_IDX)
        prev_path_to_col_idx = dict(self._PATH_TO_COL_IDX)

        # Rebuild views from scratch to pick up any new/removed bodies.
        self.clear()

        body_filters = self.get_body_filters()

        # If there are no valid bodies, clear all views / mappings and terminate early
        if not any(len(filters) > 0 for filters in body_filters.values()):
            self.clear()
            return

        # Generate views, making sure to update simulation first so the physx backend is synchronized.
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            for scene_idx, _ in enumerate(og.sim.scenes):
                scene_body_filters = body_filters[scene_idx]
                if len(scene_body_filters) == 0:
                    continue

                # Rows correspond to dynamic rigid prims only, while columns correspond to all rigid prims.
                scene_dynamic_body_filters = []
                for obj in og.sim.scenes[scene_idx].objects:
                    if obj.prim_type == PrimType.RIGID:
                        for link in obj.links.values():
                            from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim

                            if isinstance(link, RigidDynamicPrim) and link.contact_reporting_enabled:
                                scene_dynamic_body_filters.append(link.prim_path)

                self._CONTACT_VIEW[scene_idx] = og.sim.physics_sim_view.create_rigid_contact_view(
                    pattern=f"/World/scene_{scene_idx}/*/*",
                    filter_patterns=scene_body_filters,
                    max_contact_data_count=self.get_max_contact_data_count(len(scene_body_filters)),
                )
                row_paths = list(self._CONTACT_VIEW[scene_idx].sensor_paths)
                col_paths = list(getattr(self._CONTACT_VIEW[scene_idx], "filter_patterns", scene_body_filters))

                if set(row_paths) != set(scene_dynamic_body_filters):
                    missing_rows = sorted(set(scene_dynamic_body_filters) - set(row_paths))
                    extra_rows = sorted(set(row_paths) - set(scene_dynamic_body_filters))
                    raise AssertionError(
                        "RigidContactAPI contact-view row mismatch. "
                        f"Expected {len(scene_dynamic_body_filters)} dynamic rows, got {len(row_paths)} rows. "
                        f"Missing rows ({len(missing_rows)}): {missing_rows}. "
                        f"Extra rows ({len(extra_rows)}): {extra_rows}."
                    )

                if set(col_paths) != set(scene_body_filters):
                    missing_cols = sorted(set(scene_body_filters) - set(col_paths))
                    extra_cols = sorted(set(col_paths) - set(scene_body_filters))
                    raise AssertionError(
                        "RigidContactAPI contact-view column mismatch. "
                        f"Expected {len(scene_body_filters)} rigid columns, got {len(col_paths)} columns. "
                        f"Missing columns ({len(missing_cols)}): {missing_cols}. "
                        f"Extra columns ({len(extra_cols)}): {extra_cols}."
                    )

                # Create the lookup tables
                self._ROW_IDX_TO_PATH[scene_idx] = row_paths
                self._COL_IDX_TO_PATH[scene_idx] = col_paths
                self._PATH_TO_ROW_IDX[scene_idx] = {path: i for i, path in enumerate(row_paths)}
                self._PATH_TO_COL_IDX[scene_idx] = {path: i for i, path in enumerate(col_paths)}

                # Create the rigid body view, and create some indexing tensors that allow for fast lookups
                # between the contact matrix rows and the rigid body view indices.
                self._RIGID_BODY_VIEW[scene_idx] = og.sim.physics_sim_view.create_rigid_body_view(
                    pattern=f"/World/scene_{scene_idx}/*/*"
                )
                path_to_view_idx = {path: i for i, path in enumerate(list(self._RIGID_BODY_VIEW[scene_idx].prim_paths))}
                self._CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS[scene_idx] = th.tensor(
                    [path_to_view_idx[path] for path in row_paths], dtype=th.long
                )

                # Some contact-matrix columns can correspond to kinematic-only links that do not appear
                # in the rigid-body view. We encode those as -1 and track a validity mask.
                col_to_rigid_rows = [path_to_view_idx.get(path, -1) for path in col_paths]
                self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS[scene_idx] = th.tensor(col_to_rigid_rows, dtype=th.long)
                self._CONTACT_MATRIX_COLS_HAS_RIGID_BODY[scene_idx] = (
                    self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS[scene_idx] >= 0
                )
                ii, jj = th.meshgrid(th.arange(len(row_paths)), th.arange(len(col_paths)), indexing="ij")
                self._INDEX_MATRIX[scene_idx] = th.stack([ii, jj], dim=-1)
                self._BODY_TRANSFORMS[scene_idx] = self._RIGID_BODY_VIEW[scene_idx].get_transforms().clone()

                # Build the new contact matrices. Start from current impulses (captures contacts
                # for newly added bodies), then overwrite with previously cached values for
                # every pair of bodies that already existed before the rebuild.
                initial_impulses = self._CONTACT_VIEW[scene_idx].get_contact_force_matrix(dt=1.0)
                initial_contacts = th.any(initial_impulses != 0, dim=-1)
                self._CONTACT_MATRIX[scene_idx] = initial_contacts.clone()
                self._CURRENT_CONTACT_MATRIX[scene_idx] = initial_contacts.clone()

                # Initialize pending accumulation lists for this scene
                # Note that existing data in these lists will be lost when the view is rebuilt.
                # TODO: Assert here that this is not happening during a physics step, and that these buffers are empty.
                # This TODO can be accomplished after the follow-up PR removes RigidContactAPI use in assisted grasping.
                self._PENDING_IMPULSES[scene_idx] = []
                self._PENDING_TRANSFORMS[scene_idx] = []
                self._PENDING_NET_FORCES[scene_idx] = []

                # Finally, remap data from the old matrices into the new ones. This lets us avoid losing our
                # cached data when new bodies are added or removed.
                old_matrix = prev_contact_matrix.get(scene_idx)
                old_current_matrix = prev_current_contact_matrix.get(scene_idx)
                old_row_map = prev_path_to_row_idx.get(scene_idx)
                old_col_map = prev_path_to_col_idx.get(scene_idx)
                if old_matrix is not None and old_row_map is not None and old_col_map is not None:
                    # Find rows and columns that existed in both the old and new views
                    surviving_row_paths = [p for p in row_paths if p in old_row_map]
                    surviving_col_paths = [p for p in col_paths if p in old_col_map]
                    if surviving_row_paths and surviving_col_paths:
                        old_row_idxs = th.tensor([old_row_map[p] for p in surviving_row_paths], dtype=th.long)
                        old_col_idxs = th.tensor([old_col_map[p] for p in surviving_col_paths], dtype=th.long)
                        new_row_idxs = th.tensor(
                            [self._PATH_TO_ROW_IDX[scene_idx][p] for p in surviving_row_paths], dtype=th.long
                        )
                        new_col_idxs = th.tensor(
                            [self._PATH_TO_COL_IDX[scene_idx][p] for p in surviving_col_paths], dtype=th.long
                        )
                        self._CONTACT_MATRIX[scene_idx][new_row_idxs[:, None], new_col_idxs[None, :]] = old_matrix[
                            old_row_idxs[:, None], old_col_idxs[None, :]
                        ]
                        self._CURRENT_CONTACT_MATRIX[scene_idx][new_row_idxs[:, None], new_col_idxs[None, :]] = (
                            old_current_matrix[old_row_idxs[:, None], old_col_idxs[None, :]]
                        )

    def add_contacts_from_physics_step(self):
        """
        Fetches contact impulse matrices and body transforms from the current physics step
        and appends them to pending lists. Should be called by the simulator after every
        individual physics step. The accumulated data is later processed in bulk by
        update_contact_cache.
        """
        assert og.sim.currently_stepping, "add_contacts_from_physics_step must be called during a physics step"

        for scene_idx in list(self._CONTACT_VIEW.keys()):
            try:
                # Get the contact impulse and net force matrices for this scene
                impulses = self._CONTACT_VIEW[scene_idx].get_contact_force_matrix(dt=og.sim.get_physics_dt())
                net_forces = self._CONTACT_VIEW[scene_idx].get_net_contact_forces(dt=og.sim.get_physics_dt())
            except Exception:
                log.warning(
                    "RigidContactAPI cannot compute contacts because the physics sim view is invalid. "
                    "This is expected if the physics sim view is not yet initialized, e.g. you are loading "
                    "a scene for the first time."
                )
                continue

            # Get the body transforms for this scene
            transforms = self._RIGID_BODY_VIEW[scene_idx].get_transforms()

            # Append the data to the pending lists. Note that we have to clone these matrices because
            # the view actually reuses the buffer.
            if scene_idx not in self._PENDING_IMPULSES:
                self._PENDING_IMPULSES[scene_idx] = []
                self._PENDING_TRANSFORMS[scene_idx] = []
                self._PENDING_NET_FORCES[scene_idx] = []
            self._PENDING_IMPULSES[scene_idx].append(impulses.clone())
            self._PENDING_TRANSFORMS[scene_idx].append(transforms.clone())
            self._PENDING_NET_FORCES[scene_idx].append(net_forces.clone())

    def update_contact_cache(self):
        """
        Processes all accumulated physics-step data (collected via add_contacts_from_physics_step)
        to update both the "recent" contact matrix (any contact in the last N steps) and the
        "current" contact matrix (contact at only the most recent step).

        Awakeness is evaluated per individual physics step by prepending the previously cached
        transforms and diffing consecutive frames.  Bodies that report any nonzero net contact
        force are also treated as awake.  Contact matrices are only updated from awake steps.
        """
        for scene_idx in list(self._CONTACT_VIEW.keys()):
            # Get the pending data for this scene
            pending_impulses = self._PENDING_IMPULSES[scene_idx]
            pending_transforms = self._PENDING_TRANSFORMS[scene_idx]
            pending_net_forces = self._PENDING_NET_FORCES[scene_idx]
            assert len(pending_impulses) == len(pending_transforms), "Number of impulses and transforms must match"
            assert len(pending_impulses) == len(pending_net_forces), "Number of impulses and net forces must match"
            N = len(pending_impulses)
            if N == 0:
                continue

            # Stack the pending data for fast operations: (N, R, C, 3) and (N, num_bodies, 7)
            all_impulses = th.stack(pending_impulses, dim=0)
            all_transforms = th.stack(pending_transforms, dim=0)
            all_net_forces = th.stack(pending_net_forces, dim=0)

            # Get the previous body transforms for the cache
            prev_transforms = self._BODY_TRANSFORMS[scene_idx]

            # Apply the position-based sleep state approximation. Here we compute the delta between
            # each physics step's transform with the previous physics step's transform (using the cached transform
            # for the first physics step).
            extended_transforms = th.cat([prev_transforms.unsqueeze(0), all_transforms], dim=0)  # (N+1, num_bodies, 7)
            pos_changed = th.any(
                th.abs(extended_transforms[1:, :, :3] - extended_transforms[:-1, :, :3]) > self._POS_EPS, dim=-1
            )  # (N, num_bodies)
            quat_dot = th.sum(
                extended_transforms[1:, :, 3:7] * extended_transforms[:-1, :, 3:7], dim=-1
            )  # (N, num_bodies)
            ori_changed = th.abs(quat_dot) < (1.0 - self._ORI_EPS)  # (N, num_bodies)
            per_step_awake = pos_changed | ori_changed  # (N, num_bodies)

            # Other than the position change, we also know that an object cannot be asleep if the net contact force is nonzero.
            row_to_rigid = self._CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS[scene_idx]
            net_force_awake = th.any(all_net_forces != 0, dim=-1)  # (N, R)
            per_step_awake[:, row_to_rigid] = per_step_awake[:, row_to_rigid] | net_force_awake

            # Now we compute the last physics step that the body was awake. We need to do this because we want to use the
            # data for all the indices where the object is not asleep.
            body_step_indices = th.arange(N, dtype=th.long).unsqueeze(1).expand_as(per_step_awake)
            last_awake_body_step = (
                th.where(per_step_awake, body_step_indices, th.tensor(-1, dtype=th.long)).max(dim=0).values
            )  # (num_bodies,)

            # For each step, compute the rows that are awake
            per_step_row_awake = per_step_awake[:, row_to_rigid]  # (N, R)

            # For each step, compute the columns that are awake
            col_to_rigid = self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS[scene_idx]
            valid_col_mask = self._CONTACT_MATRIX_COLS_HAS_RIGID_BODY[scene_idx]
            per_step_col_awake = th.zeros(N, len(col_to_rigid), dtype=th.bool)  # (N, C)
            per_step_col_awake[:, valid_col_mask] = per_step_awake[:, col_to_rigid[valid_col_mask]]

            # For each step, compute the pairs that are awake. This is an outer-OR of the row and column awake masks.
            per_step_awake_pairs = per_step_row_awake[:, :, None] | per_step_col_awake[:, None, :]  # (N, R, C)

            # What is the last step that the pair was awake?
            pair_step_indices = th.arange(N, dtype=th.long).reshape(N, 1, 1).expand_as(per_step_awake_pairs)
            last_awake_pair_step = (
                th.where(per_step_awake_pairs, pair_step_indices, th.tensor(-1, dtype=th.long)).max(dim=0).values
            )  # (R, C)
            pair_was_awake = last_awake_pair_step >= 0  # (R, C)

            # "Current" contact matrix: impulses from the last awake step per pair.
            # Pairs that were never awake retain their previous value.
            awake_rc = th.where(pair_was_awake)
            awake_pair_steps = last_awake_pair_step[pair_was_awake]
            last_awake_impulses = all_impulses[awake_pair_steps, awake_rc[0], awake_rc[1]]  # (num_awake, 3)
            self._CURRENT_CONTACT_MATRIX[scene_idx][pair_was_awake] = th.any(last_awake_impulses != 0, dim=-1)

            # "Recent" contact matrix: any contact across awake steps for awake pairs,
            # or the (now-updated) current contact value for non-awake pairs.
            any_contact_per_step = th.any(all_impulses != 0, dim=-1)  # (N, R, C)
            any_awake_contact = th.any(any_contact_per_step & per_step_awake_pairs, dim=0)  # (R, C)
            self._CONTACT_MATRIX[scene_idx][pair_was_awake] = any_awake_contact[pair_was_awake]
            self._CONTACT_MATRIX[scene_idx][~pair_was_awake] = self._CURRENT_CONTACT_MATRIX[scene_idx][~pair_was_awake]

            # Update body transforms from each body's last awake step
            awake_body_indices = th.where(last_awake_body_step >= 0)[0]
            self._BODY_TRANSFORMS[scene_idx][awake_body_indices] = all_transforms[
                last_awake_body_step[awake_body_indices], awake_body_indices
            ]

            # Clear pending data for this scene
            self._PENDING_IMPULSES[scene_idx] = []
            self._PENDING_TRANSFORMS[scene_idx] = []
            self._PENDING_NET_FORCES[scene_idx] = []

    def _get_prim_paths(self, objects_links_or_prim_paths):
        """
        Converts a set of objects, links, or prim paths to a list of prim paths for contact matrix lookups.

        Args:
            objects_links_or_prim_paths (set of EntityPrim, RigidPrim, str, or BaseObject): Objects, links, or prim paths to convert to prim paths.

        Returns:
            list[str]: List of prim paths.
        """
        # Avoid circular imports
        from omnigibson.prims.entity_prim import EntityPrim
        from omnigibson.prims.rigid_prim import RigidPrim

        outputs = []
        for inp in objects_links_or_prim_paths:
            if isinstance(inp, EntityPrim):
                outputs.extend([link.prim_path for link in inp.links.values()])
            elif isinstance(inp, RigidPrim):
                outputs.append(inp.prim_path)
            elif isinstance(inp, str):
                outputs.append(inp)
            else:
                raise ValueError(f"Input set must be a set of EntityPrim, RigidPrim, or str, found {type(inp)}")
        return outputs

    def get_contact_row_indices(self, scene_idx, objects_links_or_prim_paths):
        """
        Gets the row indices of the contact matrix for a given set of objects, links, or prim paths.
        This is the index of the rigid body in the contact matrix. This can be used by external callers to
        pre-cache the indices they care about for faster lookups later (e.g. avoiding a lookup on every call to is_in_contact).

        Args:
            scene_idx (int): Scene index to get the contact row indices for.
            objects_links_or_prim_paths (set of EntityPrim, RigidPrim, str, or BaseObject): Objects, links, or prim paths to get the contact row indices for.

        Returns:
            th.Tensor: Tensor of row indices.
        """
        # If the input is already a tensor just return it
        if isinstance(objects_links_or_prim_paths, th.Tensor):
            return objects_links_or_prim_paths

        # Otherwise, convert to prim paths
        prim_paths = self._get_prim_paths(objects_links_or_prim_paths)
        return th.tensor([self._PATH_TO_ROW_IDX[scene_idx][path] for path in prim_paths])

    def get_contact_col_indices(self, scene_idx, objects_links_or_prim_paths):
        """
        Gets the column indices of the contact matrix for a given set of objects, links, or prim paths.
        This is the index of the rigid body in the contact matrix. This can be used by external callers to
        pre-cache the indices they care about for faster lookups later (e.g. avoiding a lookup on every call to is_in_contact).

        Args:
            scene_idx (int): Scene index to get the contact column indices for.
            objects_links_or_prim_paths (set of EntityPrim, RigidPrim, str, or BaseObject): Objects, links, or prim paths to get the contact column indices for.

        Returns:
            th.Tensor: Tensor of column indices.
        """
        # If the input is already a tensor just return it
        if isinstance(objects_links_or_prim_paths, th.Tensor):
            return objects_links_or_prim_paths

        # Otherwise, convert to prim paths
        prim_paths = self._get_prim_paths(objects_links_or_prim_paths)
        return th.tensor([self._PATH_TO_COL_IDX[scene_idx][path] for path in prim_paths])

    def get_contact_pairs(self, scene_idx, query_set, with_set, current_only):
        """
        Get pairs of prim paths that are in contact.

        Args:
            scene_idx (int): Scene index to get the contact pairs for.
            query_set (set of RigidPrim, str, or BaseObject): Prims, prim paths, or objects for contact sensor objects to check. Must be specified.
            with_set (set of RigidPrim, str, or BaseObject): Prims, prim paths, or objects to filter the contact pairs by. Only these objects will be considered for contact. Can be None to check for contact with any object.
            current_only (bool): If True, only checks the most recent physics step. If False, checks whether contact occurred at any physics step since the last non-physics step.
                The True mode is recommended for use cases like Touching state etc. where a contact at the current position of the object is important.
                The False mode is recommended for use cases like transition rules etc. where a contact at any point during the last N physics steps is enough (e.g. as a trigger event).

        Returns:
            set of tuples: Set of tuples of (query_prim_path, filter_prim_path) pairs that are in contact.
        """
        if scene_idx not in self._CONTACT_MATRIX or scene_idx not in self._PATH_TO_COL_IDX:
            return set()
        contact_matrix = self._CURRENT_CONTACT_MATRIX[scene_idx] if current_only else self._CONTACT_MATRIX[scene_idx]
        assert contact_matrix.ndim == 2, f"Contact matrix should be 2D, found shape {contact_matrix.shape}"

        # Get the row indices corresponding to the sensor prim paths
        row_idxs = self.get_contact_row_indices(scene_idx, query_set)

        # Slice the contact matrix and the index matrix with the same indexing so that
        # nonzero positions in the submatrix can be mapped back to original row/col indices.
        idx_matrix = self._INDEX_MATRIX[scene_idx]
        if with_set is not None:
            col_idxs = self.get_contact_col_indices(scene_idx, with_set)
            in_contact = contact_matrix[row_idxs[:, None], col_idxs[None, :]]
            idx_matrix = idx_matrix[row_idxs[:, None], col_idxs[None, :]]
        else:
            in_contact = contact_matrix[row_idxs, :]
            idx_matrix = idx_matrix[row_idxs, :]

        # Early return if not in contact.
        if not th.any(in_contact).item():
            return set()

        original_indices = idx_matrix[in_contact].cpu().tolist()

        return {
            (self._ROW_IDX_TO_PATH[scene_idx][row], self._COL_IDX_TO_PATH[scene_idx][col])
            for row, col in original_indices
        }

    def is_in_contact(self, scene_idx, query_set, with_set, ignore_set, current_only):
        """
        Check if any of the prims in @query_set are in contact with any of the prims in @with_set, or not in contact with any of the prims in @ignore_set.

        Args:
            scene_idx (int): Scene index to check for contact.
            query_set (set of RigidPrim, str, or BaseObject): Prims, prim paths, or objects to check for contact.
            with_set (set of RigidPrim, str, or BaseObject): Prims, prim paths, or objects to check for contact with. Can be None to check for contact with any object.
            ignore_set (set of RigidPrim, str, or BaseObject): Prims, prim paths, or objects to ignore contact with. Can be None to not ignore any objects.
            current_only (bool): If True, only checks the most recent physics step. If False, checks whether contact occurred at any physics step since the last non-physics step.
                The True mode is recommended for use cases like Touching state etc. where a contact at the current position of the object is important.
                The False mode is recommended for use cases like transition rules etc. where a contact at any point during the last N physics steps is enough (e.g. as a trigger event).

        Returns:
            bool: True if any of the prims in @query_set are in contact with any of the prims in @with_set, or not in contact with any of the prims in @ignore_set, else False.
        """
        if with_set is not None and ignore_set is not None:
            raise ValueError("At most one of with_set or ignore_set may be specified.")

        if scene_idx not in self._CONTACT_MATRIX or scene_idx not in self._PATH_TO_COL_IDX:
            return False

        contact_matrix = self._CURRENT_CONTACT_MATRIX[scene_idx] if current_only else self._CONTACT_MATRIX[scene_idx]
        rows = self.get_contact_row_indices(scene_idx, query_set)
        if with_set is not None:
            cols = self.get_contact_col_indices(scene_idx, with_set)
            return th.any(contact_matrix[rows, :][:, cols]).item()
        elif ignore_set is not None:
            ignore_mask = th.ones(contact_matrix.shape[1], dtype=th.bool)
            ignore_mask[self.get_contact_col_indices(scene_idx, ignore_set)] = False
            return th.any(contact_matrix[rows, :][:, ignore_mask]).item()

        # Base case, return any collisions with any other prim
        return th.any(contact_matrix[rows]).item()

    def get_contact_row_mask(self, scene_idx, objects_links_or_prim_paths):
        """
        Gets a boolean mask over contact matrix rows for a given set of objects, links, or prim paths.
        Useful for building batch query masks for :meth:`is_in_contact_batch`.

        Args:
            scene_idx (int): Scene index.
            objects_links_or_prim_paths: Objects, links, or prim paths (or a pre-cached index tensor).

        Returns:
            th.Tensor: (R,) boolean tensor where R is the number of contact matrix rows.
        """
        R = self._CONTACT_MATRIX[scene_idx].shape[0]
        idxs = self.get_contact_row_indices(scene_idx, objects_links_or_prim_paths)
        mask = th.zeros(R, dtype=th.bool)
        mask[idxs] = True
        return mask

    def get_contact_col_mask(self, scene_idx, objects_links_or_prim_paths):
        """
        Gets a boolean mask over contact matrix columns for a given set of objects, links, or prim paths.
        Useful for building batch with/ignore masks for :meth:`is_in_contact_batch`.

        Args:
            scene_idx (int): Scene index.
            objects_links_or_prim_paths: Objects, links, or prim paths (or a pre-cached index tensor).

        Returns:
            th.Tensor: (C,) boolean tensor where C is the number of contact matrix columns.
        """
        C = self._CONTACT_MATRIX[scene_idx].shape[1]
        idxs = self.get_contact_col_indices(scene_idx, objects_links_or_prim_paths)
        mask = th.zeros(C, dtype=th.bool)
        mask[idxs] = True
        return mask

    def is_in_contact_batch(self, scene_idx, query_masks, with_masks, ignore_masks, current_only):
        """
        Batch contact check for N queries, fully tensorized.

        Each row ``i`` of the input masks defines one independent contact query. The method
        returns an ``(N,)`` boolean tensor where entry ``i`` is ``True`` iff any row selected
        by ``query_masks[i]`` is in contact with any column selected by ``with_masks[i]``
        (or any column *not* in ``ignore_masks[i]``).

        Provide either ``with_masks`` for all N queries or ``ignore_masks`` for all N queries,
        but not both (and not a per-query mix).

        Use :meth:`get_contact_row_mask` / :meth:`get_contact_col_mask` to build individual
        masks, then ``torch.stack`` them into the ``(N, R)`` / ``(N, C)`` tensors expected here.

        Args:
            scene_idx (int): Scene index to check for contact.
            query_masks (th.Tensor): ``(N, R)`` boolean tensor. ``query_masks[i, j]`` is True
                if contact-matrix row ``j`` belongs to query set ``i``.
            with_masks (th.Tensor or None): ``(N, C)`` boolean tensor. ``with_masks[i, j]`` is
                True if contact-matrix column ``j`` belongs to the with-set for query ``i``. Can be None to check for contact with any object.
            ignore_masks (th.Tensor or None): ``(N, C)`` boolean tensor. ``ignore_masks[i, j]``
                is True if contact-matrix column ``j`` should be *ignored* for query ``i``. Can be None to not ignore any objects.
            current_only (bool): If True, only checks the most recent physics step. If False, checks whether contact occurred at any physics step since the last non-physics step.
                The True mode is recommended for use cases like Touching state etc. where a contact at the current position of the object is important.
                The False mode is recommended for use cases like transition rules etc. where a contact at any point during the last N physics steps is enough (e.g. as a trigger event).

        Returns:
            th.Tensor: ``(N,)`` boolean tensor of contact results.
        """
        assert with_masks is None or ignore_masks is None, "Provide either with_masks or ignore_masks, not both."

        if scene_idx not in self._CONTACT_MATRIX or scene_idx not in self._PATH_TO_COL_IDX:
            return th.zeros(query_masks.shape[0], dtype=th.bool)

        contact_matrix = self._CURRENT_CONTACT_MATRIX[scene_idx] if current_only else self._CONTACT_MATRIX[scene_idx]

        # query_contacts[i, c] = True iff any row in query set i is in contact with column c.
        # We use float matmul for speed: (N, R) @ (R, C) -> (N, C), then threshold.
        query_contacts = (query_masks.float() @ contact_matrix.float()) > 0

        if with_masks is not None:
            return (query_contacts & with_masks).any(dim=1)
        elif ignore_masks is not None:
            return (query_contacts & ~ignore_masks).any(dim=1)

        return query_contacts.any(dim=1)

    def clear(self):
        """
        Clears internal contact views, mappings, and caches.
        """
        self._PATH_TO_ROW_IDX = dict()
        self._PATH_TO_COL_IDX = dict()
        self._ROW_IDX_TO_PATH = dict()
        self._COL_IDX_TO_PATH = dict()
        self._CONTACT_VIEW = dict()
        self._RIGID_BODY_VIEW = dict()
        self._CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS = dict()
        self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS = dict()
        self._CONTACT_MATRIX_COLS_HAS_RIGID_BODY = dict()
        self._CONTACT_MATRIX = dict()
        self._CURRENT_CONTACT_MATRIX = dict()
        self._INDEX_MATRIX = dict()
        self._BODY_TRANSFORMS = dict()
        self._PENDING_IMPULSES = dict()
        self._PENDING_TRANSFORMS = dict()
        self._PENDING_NET_FORCES = dict()


# Instantiate the RigidContactAPI
RigidContactAPI = RigidContactAPIImpl()


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

    def get_root_transform(self, prim_path):
        if "root_transforms" not in self._read_cache:
            self._read_cache["root_transforms"] = cb.from_torch(self._view.get_root_transforms())

        idx = self._idx[prim_path]
        pose = self._read_cache["root_transforms"][idx]
        return pose[:3], pose[3:]

    def get_position_orientation(self, prim_path):
        # Here we want to return the position of the base footprint link. If the base footprint link is None,
        # we return the position of the root link.
        if self._base_footprint_link_names[prim_path] is not None:
            link_name = self._base_footprint_link_names[prim_path]
            return self.get_link_transform(prim_path, link_name)
        else:
            return self.get_root_transform(prim_path)

    def _get_velocities(self, prim_path, estimate=False):
        if self._base_footprint_link_names[prim_path] is not None:
            link_name = self._base_footprint_link_names[prim_path]
            return self._get_link_velocities(prim_path, link_name, estimate=estimate)
        else:
            return self._get_root_velocities(prim_path, estimate=estimate)

    def _get_relative_velocities(self, prim_path, estimate=False):
        vel_str = "velocities_estimate" if estimate else "velocities"

        if f"relative_{vel_str}" not in self._read_cache:
            self._read_cache[f"relative_{vel_str}"] = {}

        if prim_path not in self._read_cache[f"relative_{vel_str}"]:
            # Compute all tfs at once, including base as well as all links
            idx = self._idx[prim_path]
            if f"link_{vel_str}" not in self._read_cache:
                # Force the internal cache to update
                self._get_link_velocities(
                    prim_path=prim_path, link_name=next(iter(self._link_idx[idx].keys())), estimate=estimate
                )

            vels = cb.zeros((len(self._link_idx[idx]) + 1, 6, 1))
            # base vel is the final -1 index
            vels[:-1, :, 0] = self._read_cache[f"link_{vel_str}"][idx, :]
            vels[-1, :, 0] = self._get_velocities(prim_path=prim_path, estimate=estimate)

            tf = cb.zeros((1, 6, 6))
            orn_t = cb.T.quat2mat(self.get_position_orientation(prim_path)[1]).T
            tf[0, :3, :3] = orn_t
            tf[0, 3:, 3:] = orn_t
            # x.T --> transpose (inverse) orientation
            # (1, 6, 6) @ (n_links, 6, 1) -> (n_links, 6, 1) -> (n_links, 6)
            self._read_cache[f"relative_{vel_str}"][prim_path] = cb.squeeze(tf @ vels, dim=-1)

        return self._read_cache[f"relative_{vel_str}"][prim_path]

    def get_linear_velocity(self, prim_path, estimate=False):
        return self._get_velocities(prim_path, estimate=estimate)[:3]

    def get_angular_velocity(self, prim_path, estimate=False):
        return self._get_velocities(prim_path, estimate=estimate)[3:]

    def _get_root_velocities(self, prim_path, estimate=False):
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

        idx = self._idx[prim_path]
        return self._read_cache[f"root_{vel_str}"][idx]

    def get_relative_linear_velocity(self, prim_path, estimate=False):
        # base corresponds to final index
        return self._get_relative_velocities(prim_path, estimate=estimate)[-1, :3]

    def get_relative_angular_velocity(self, prim_path, estimate=False):
        # base corresponds to final index
        return self._get_relative_velocities(prim_path, estimate=estimate)[-1, 3:]

    def get_joint_positions(self, prim_path):
        if "dof_positions" not in self._read_cache:
            self._read_cache["dof_positions"] = cb.from_torch(self._view.get_dof_positions())

        idx = self._idx[prim_path]
        return self._read_cache["dof_positions"][idx]

    def get_joint_velocities(self, prim_path, estimate=False):
        vel_str = "velocities_estimate" if estimate else "velocities"

        # Use estimated calculation if requested and we have prior history info
        if f"dof_{vel_str}" not in self._read_cache:
            if estimate and self._last_state is not None:
                if "dof_positions" not in self._read_cache:
                    self._read_cache["dof_positions"] = cb.from_torch(self._view.get_dof_positions())
                self._read_cache[f"dof_{vel_str}"] = (
                    self._read_cache["dof_positions"] - self._last_state["dof_positions"]
                ) / og.sim.get_physics_dt()
            else:
                self._read_cache[f"dof_{vel_str}"] = cb.from_torch(self._view.get_dof_velocities())

        idx = self._idx[prim_path]
        return self._read_cache[f"dof_{vel_str}"][idx]

    def get_joint_efforts(self, prim_path):
        if "dof_projected_joint_forces" not in self._read_cache:
            self._read_cache["dof_projected_joint_forces"] = cb.from_torch(self._view.get_dof_projected_joint_forces())

        idx = self._idx[prim_path]
        return self._read_cache["dof_projected_joint_forces"][idx]

    def get_generalized_mass_matrices(self, prim_path):
        if "mass_matrices" not in self._read_cache:
            self._read_cache["mass_matrices"] = cb.from_torch(self._view.get_generalized_mass_matrices())

        idx = self._idx[prim_path]
        return self._read_cache["mass_matrices"][idx]

    def get_gravity_compensation_forces(self, prim_path):
        if "generalized_gravity_forces" not in self._read_cache:
            self._read_cache["generalized_gravity_forces"] = cb.from_torch(self._view.get_gravity_compensation_forces())

        idx = self._idx[prim_path]
        return self._read_cache["generalized_gravity_forces"][idx]

    def get_coriolis_and_centrifugal_compensation_forces(self, prim_path):
        if "coriolis_and_centrifugal_forces" not in self._read_cache:
            self._read_cache["coriolis_and_centrifugal_forces"] = cb.from_torch(
                self._view.get_coriolis_and_centrifugal_compensation_forces()
            )

        idx = self._idx[prim_path]
        return self._read_cache["coriolis_and_centrifugal_forces"][idx]

    def get_link_transform(self, prim_path, link_name):
        if "link_transforms" not in self._read_cache:
            self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())

        idx = self._idx[prim_path]
        link_idx = self._link_idx[idx][link_name]
        pose = self._read_cache["link_transforms"][idx, link_idx]
        return pose[:3], pose[3:]

    def _get_relative_poses(self, prim_path):
        if "relative_poses" not in self._read_cache:
            self._read_cache["relative_poses"] = {}

        if prim_path not in self._read_cache["relative_poses"]:
            # Compute all tfs at once, including base as well as all links
            if "link_transforms" not in self._read_cache:
                self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())

            idx = self._idx[prim_path]
            self._read_cache["relative_poses"][prim_path] = cb.get_custom_method("compute_relative_poses")(
                idx,
                len(self._link_idx[idx]),
                self._read_cache["link_transforms"],
                self.get_position_orientation(prim_path=prim_path),
            )

        return self._read_cache["relative_poses"][prim_path]

    def get_link_relative_position_orientation(self, prim_path, link_name):
        idx = self._idx[prim_path]
        link_idx = self._link_idx[idx][link_name]
        rel_pose = self._get_relative_poses(prim_path)[link_idx]
        return rel_pose[:3], rel_pose[3:]

    def _get_link_velocities(self, prim_path, link_name, estimate=False):
        vel_str = "velocities_estimate" if estimate else "velocities"

        # Use estimated calculation if requested and we have prior history info
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

        idx = self._idx[prim_path]
        link_idx = self._link_idx[idx][link_name]
        vel = self._read_cache[f"link_{vel_str}"][idx, link_idx]

        return vel

    def get_link_linear_velocity(self, prim_path, link_name, estimate=False):
        return self._get_link_velocities(prim_path, link_name, estimate=estimate)[:3]

    def get_link_relative_linear_velocity(self, prim_path, link_name, estimate=False):
        idx = self._idx[prim_path]
        link_idx = self._link_idx[idx][link_name]
        return self._get_relative_velocities(prim_path, estimate=estimate)[link_idx, :3]

    def get_link_angular_velocity(self, prim_path, link_name, estimate=False):
        return self._get_link_velocities(prim_path, link_name, estimate=estimate)[3:]

    def get_link_relative_angular_velocity(self, prim_path, link_name, estimate=False):
        idx = self._idx[prim_path]
        link_idx = self._link_idx[idx][link_name]
        return self._get_relative_velocities(prim_path, estimate=estimate)[link_idx, 3:]

    def get_jacobian(self, prim_path):
        if "jacobians" not in self._read_cache:
            self._read_cache["jacobians"] = cb.from_torch(self._view.get_jacobians())

        idx = self._idx[prim_path]
        return self._read_cache["jacobians"][idx]

    def get_relative_jacobian(self, prim_path):
        if "relative_jacobians" not in self._read_cache:
            self._read_cache["relative_jacobians"] = {}

        if prim_path not in self._read_cache["relative_jacobians"]:
            jacobian = self.get_jacobian(prim_path)
            ori_t = cb.T.quat2mat(self.get_position_orientation(prim_path)[1]).T
            tf = cb.zeros((1, 6, 6))
            tf[:, :3, :3] = ori_t
            tf[:, 3:, 3:] = ori_t
            # Run this explicitly in pytorch since it's order of magnitude faster than numpy!
            # e.g.: 2e-4 vs. 3e-5 on R1
            rel_jac = cb.from_torch(cb.to_torch(tf) @ cb.to_torch(jacobian))
            self._read_cache["relative_jacobians"][prim_path] = rel_jac

        return self._read_cache["relative_jacobians"][prim_path]


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
        if cls._get_pattern_from_prim_path(prim_path) in cls._VIEWS_BY_PATTERN:
            cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].clear()

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
        patterns = {cls._get_pattern_from_prim_path(prim_path) for prim_path in expected_prim_paths}

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
    def _get_pattern_from_prim_path(cls, prim_path):
        """
        Returns which of the regexes this prim path should be found in.

        Note that since the prim path will be an articulation root path, this works for both fixed base and
        non-fixed base robots. Fixed and non-fixed versions of the same robot will be mapped to different
        patterns since they have different articulation root paths (object prim vs base link prim).
        """
        scene_id, robot_name = prim_path.split("/")[2:4]
        assert scene_id.startswith("scene_"), f"Prim path 2nd component {prim_path} does not start with scene_"
        components = robot_name.split("__")
        assert (
            len(components) == 3
        ), f"Robot prim path's 3rd component {robot_name} does not match expected format of prefix__robottype__robotname."
        assert (
            components[0] == "controllable"
        ), f"Prim path {prim_path} 3rd component does not start with prefix {cls._prefix}__"
        robot_name_pattern = prim_path.replace(f"/{scene_id}/", "/scene_*/").replace(
            f"/{robot_name}", f"/{components[0]}__{components[1]}__*"
        )
        return robot_name_pattern

    @classmethod
    def set_joint_position_targets(cls, prim_path, positions, indices):
        cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].set_joint_position_targets(
            prim_path, positions, indices
        )

    @classmethod
    def set_joint_velocity_targets(cls, prim_path, velocities, indices):
        cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].set_joint_velocity_targets(
            prim_path, velocities, indices
        )

    @classmethod
    def set_joint_efforts(cls, prim_path, efforts, indices):
        cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].set_joint_efforts(prim_path, efforts, indices)

    @classmethod
    def get_position_orientation(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_position_orientation(prim_path)

    @classmethod
    def get_root_position_orientation(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_root_transform(prim_path)

    @classmethod
    def get_linear_velocity(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_linear_velocity(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_angular_velocity(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_angular_velocity(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_relative_linear_velocity(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_relative_linear_velocity(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_relative_angular_velocity(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_relative_angular_velocity(
            prim_path,
            estimate=estimate,
        )

    @classmethod
    def get_joint_positions(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_joint_positions(prim_path)

    @classmethod
    def get_joint_velocities(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_joint_velocities(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_joint_efforts(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_joint_efforts(prim_path)

    @classmethod
    def get_generalized_mass_matrices(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_generalized_mass_matrices(
            prim_path
        )

    @classmethod
    def get_gravity_compensation_forces(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_gravity_compensation_forces(
            prim_path
        )

    @classmethod
    def get_coriolis_and_centrifugal_compensation_forces(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[
            cls._get_pattern_from_prim_path(prim_path)
        ].get_coriolis_and_centrifugal_compensation_forces(prim_path)

    @classmethod
    def get_link_position_orientation(cls, prim_path, link_name):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_link_transform(
            prim_path, link_name
        )

    @classmethod
    def get_link_relative_position_orientation(cls, prim_path, link_name):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_link_relative_position_orientation(
            prim_path, link_name
        )

    @classmethod
    def get_link_relative_linear_velocity(cls, prim_path, link_name, estimate=False):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_link_relative_linear_velocity(
            prim_path,
            link_name,
            estimate=estimate,
        )

    @classmethod
    def get_link_relative_angular_velocity(cls, prim_path, link_name, estimate=False):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_link_relative_angular_velocity(
            prim_path,
            link_name,
            estimate=estimate,
        )

    @classmethod
    def get_jacobian(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_jacobian(prim_path)

    @classmethod
    def get_relative_jacobian(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[cls._get_pattern_from_prim_path(prim_path)].get_relative_jacobian(prim_path)


def clear():
    """
    Clear state tied to singleton classes
    """
    PoseAPI.invalidate()
    CollisionAPI.clear()
    RigidContactAPI.clear()
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


def create_usd_stage(usd_path):
    stage = lazy.pxr.Usd.Stage.CreateNew(usd_path)
    lazy.pxr.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    lazy.pxr.UsdGeom.SetStageUpAxis(stage, "Z")
    return stage


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
