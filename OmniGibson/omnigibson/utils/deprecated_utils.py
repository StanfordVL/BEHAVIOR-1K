"""
A set of utility functions slated to be deprecated once Omniverse bugs are fixed
"""

import math
from typing import List, Optional, Tuple, Union

import carb
import numpy as np
import omni
import omni.timeline
import torch
import usdrt
import warp as wp
from isaacsim.core.prims import Articulation as _ArticulationView
from isaacsim.core.prims import RigidPrim as _RigidPrimView
from isaacsim.core.prims import XFormPrim as _XFormPrimView
from isaacsim.core.utils.prims import get_prim_at_path
from omni.kit.primitive.mesh.command import CreateMeshPrimWithDefaultXformCommand as CMPWDXC
from omni.kit.primitive.mesh.command import _get_all_evaluators
from omni.replicator.core import random_colours
from PIL import Image, ImageDraw
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

DEG2RAD = math.pi / 180.0


class CreateMeshPrimWithDefaultXformCommand(CMPWDXC):
    def __init__(self, prim_type: str, **kwargs):
        """
        Creates primitive.

        Args:
            prim_type (str): It supports Plane/Sphere/Cone/Cylinder/Disk/Torus/Cube.

        kwargs:
            object_origin (Gf.Vec3f): Position of mesh center in stage units.

            u_patches (int): The number of patches to tessellate U direction.

            v_patches (int): The number of patches to tessellate V direction.

            w_patches (int): The number of patches to tessellate W direction.
                             It only works for Cone/Cylinder/Cube.

            half_scale (float): Half size of mesh in centimeters. Default is None, which means it's controlled by settings.

            u_verts_scale (int): Tessellation Level of U. It's a multiplier of u_patches.

            v_verts_scale (int): Tessellation Level of V. It's a multiplier of v_patches.

            w_verts_scale (int): Tessellation Level of W. It's a multiplier of w_patches.
                                 It only works for Cone/Cylinder/Cube.
                                 For Cone/Cylinder, it's to tessellate the caps.
                                 For Cube, it's to tessellate along z-axis.

            above_ground (bool): It will offset the center of mesh above the ground plane if it's True,
                False otherwise. It's False by default. This param only works when param object_origin is not given.
                Otherwise, it will be ignored.

            stage (Usd.Stage): If specified, stage to create prim on
        """

        self._prim_type = prim_type[0:1].upper() + prim_type[1:].lower()
        self._usd_context = omni.usd.get_context()
        self._selection = self._usd_context.get_selection()
        self._stage = kwargs.get("stage", self._usd_context.get_stage())
        self._settings = carb.settings.get_settings()
        self._default_path = kwargs.get("prim_path", None)
        self._select_new_prim = kwargs.get("select_new_prim", True)
        self._prepend_default_prim = kwargs.get("prepend_default_prim", True)
        self._above_round = kwargs.get("above_ground", False)

        self._attributes = {**kwargs}
        # Supported mesh types should have an associated evaluator class
        self._evaluator_class = _get_all_evaluators()[prim_type]
        assert isinstance(self._evaluator_class, type)


class ArticulationView(_ArticulationView):
    """ArticulationView with some additional functionality implemented."""

    def set_joint_limits(
        self,
        values: Union[np.ndarray, torch.Tensor],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Sets joint limits for articulation joints in the view.

        Args:
            values (Union[np.ndarray, torch.Tensor, wp.array]): joint limits for articulations in the view. shape (M, K, 2).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indicies to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, "cpu")
            new_values = self._physics_view.get_dof_limits()
            values = self._backend_utils.move_data(values, device="cpu")
            new_values = self._backend_utils.assign(
                values,
                new_values,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_limits(new_values, indices)
        else:
            indices = self._backend_utils.to_list(
                self._backend_utils.resolve_indices(indices, self.count, self._device)
            )
            dof_types = self._backend_utils.to_list(self.get_dof_types())
            joint_indices = self._backend_utils.to_list(
                self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            )
            values = self._backend_utils.to_list(values)
            articulation_read_idx = 0
            for i in indices:
                dof_read_idx = 0
                for dof_index in joint_indices:
                    dof_val = values[articulation_read_idx][dof_read_idx]
                    if dof_types[dof_index] == omni.physics.tensors.DofType.Rotation:
                        dof_val /= DEG2RAD
                    prim = get_prim_at_path(self._dof_paths[i][dof_index])
                    prim.GetAttribute("physics:lowerLimit").Set(dof_val[0])
                    prim.GetAttribute("physics:upperLimit").Set(dof_val[1])
                    dof_read_idx += 1
                articulation_read_idx += 1
        return

    def get_joint_limits(
        self,
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        clone: bool = True,
    ) -> Union[np.ndarray, torch.Tensor, wp.array]:
        """Gets joint limits for articulation in the view.

        Args:
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indicies to specify which joints
                                                                                 to query. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
            clone (Optional[bool]): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: joint limits for articulations in the view. shape (M, K).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            values = self._backend_utils.move_data(self._physics_view.get_dof_limits(), self._device)
            if clone:
                values = self._backend_utils.clone_tensor(values, device=self._device)
            result = values[
                self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices
            ]
            return result
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            dof_types = self._backend_utils.to_list(self.get_dof_types())
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            values = np.zeros(shape=(indices.shape[0], joint_indices.shape[0], 2), dtype="float32")
            articulation_write_idx = 0
            indices = self._backend_utils.to_list(indices)
            joint_indices = self._backend_utils.to_list(joint_indices)
            for i in indices:
                dof_write_idx = 0
                for dof_index in joint_indices:
                    prim = get_prim_at_path(self._dof_paths[i][dof_index])
                    values[articulation_write_idx][dof_write_idx][0] = prim.GetAttribute("physics:lowerLimit").Get()
                    values[articulation_write_idx][dof_write_idx][1] = prim.GetAttribute("physics:upperLimit").Get()
                    if dof_types[dof_index] == omni.physics.tensors.DofType.Rotation:
                        values[articulation_write_idx][dof_write_idx] = (
                            values[articulation_write_idx][dof_write_idx] * DEG2RAD
                        )
                    dof_write_idx += 1
                articulation_write_idx += 1
            values = self._backend_utils.convert(values, dtype="float32", device=self._device, indexed=True)
            return values

    def get_joint_position_targets(
        self,
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        clone: bool = True,
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Get the joint position targets of articulations in the view

        Args:
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indices to specify which joints
                                                                                 to query. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: joint positions of articulations in the view.
            Shape is (M, K).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            current_joint_positions = self._physics_view.get_dof_position_targets()
            if clone:
                current_joint_positions = self._backend_utils.clone_tensor(current_joint_positions, device=self._device)
            result = current_joint_positions[
                self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices
            ]
            return result
        else:
            carb.log_warn("Physics Simulation View is not created yet in order to use get_joint_position_targets")
            return None

    def get_joint_velocity_targets(
        self,
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        clone: bool = True,
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Get the joint velocity targets of articulations in the view

        Args:
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indices to specify which joints
                                                                                 to query. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: joint velocities of articulations in the view.
            Shape is (M, K).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            current_joint_velocities = self._physics_view.get_dof_velocity_targets()
            if clone:
                current_joint_velocities = self._backend_utils.clone_tensor(
                    current_joint_velocities, device=self._device
                )
            result = current_joint_velocities[
                self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices
            ]
            return result
        else:
            carb.log_warn("Physics Simulation View is not created yet in order to use get_joint_velocity_targets")
            return None

    def set_max_velocities(
        self,
        values: Union[np.ndarray, torch.Tensor, wp.array],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Sets maximum velocities for articulation in the view.

        Args:
            values (Union[np.ndarray, torch.Tensor, wp.array]): maximum velocities for articulations in the view. shape (M, K).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indicies to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, "cpu")
            new_values = self._physics_view.get_dof_max_velocities()
            new_values = self._backend_utils.assign(
                self._backend_utils.move_data(values, device="cpu"),
                new_values,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_max_velocities(new_values, indices)
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            articulation_read_idx = 0
            indices = self._backend_utils.to_list(indices)
            joint_indices = self._backend_utils.to_list(joint_indices)
            values = self._backend_utils.to_list(values)
            for i in indices:
                dof_read_idx = 0
                for dof_index in joint_indices:
                    prim = PhysxSchema.PhysxJointAPI(get_prim_at_path(self._dof_paths[i][dof_index]))
                    if not prim.GetMaxJointVelocityAttr():
                        prim.CreateMaxJointVelocityAttr().Set(values[articulation_read_idx][dof_read_idx])
                    else:
                        prim.GetMaxJointVelocityAttr().Set(values[articulation_read_idx][dof_read_idx])
                    dof_read_idx += 1
                articulation_read_idx += 1
        return

    def get_max_velocities(
        self,
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        clone: bool = True,
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Gets maximum velocities for articulation in the view.

        Args:
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indicies to specify which joints
                                                                                 to query. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
            clone (Optional[bool]): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: maximum velocities for articulations in the view. shape (M, K).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, "cpu")
            max_velocities = self._physics_view.get_dof_max_velocities()
            if clone:
                max_velocities = self._backend_utils.clone_tensor(max_velocities, device="cpu")
            result = self._backend_utils.move_data(
                max_velocities[
                    self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices
                ],
                device=self._device,
            )
            return result
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            max_velocities = np.zeros(shape=(indices.shape[0], joint_indices.shape[0]), dtype="float32")
            indices = self._backend_utils.to_list(indices)
            joint_indices = self._backend_utils.to_list(joint_indices)
            articulation_write_idx = 0
            for i in indices:
                dof_write_idx = 0
                for dof_index in joint_indices:
                    prim = PhysxSchema.PhysxJointAPI(get_prim_at_path(self._dof_paths[i][dof_index]))
                    max_velocities[articulation_write_idx][dof_write_idx] = prim.GetMaxJointVelocityAttr().Get()
                    dof_write_idx += 1
                articulation_write_idx += 1
            max_velocities = self._backend_utils.convert(
                max_velocities, dtype="float32", device=self._device, indexed=True
            )
            return max_velocities

    def set_joint_positions(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor, wp.array]],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the joint positions of articulations in the view

        .. warning::

            This method will immediately set (teleport) the affected joints to the indicated value.
            Use the ``set_joint_position_targets`` or the ``apply_action`` methods to control the articulation joints.

        Args:
            positions (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): joint positions of articulations in the view to be set to in the next frame.
                                                                    shape is (M, K).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indices to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).

        .. hint::

            This method belongs to the methods used to set the articulation kinematic states:

            ``set_velocities`` (``set_linear_velocities``, ``set_angular_velocities``),
            ``set_joint_positions``, ``set_joint_velocities``, ``set_joint_efforts``

        Example:

        .. code-block:: python

            >>> # set all the articulation joints.
            >>> # Since there are 5 envs, the joint positions are repeated 5 times
            >>> positions = np.tile(np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0.04, 0.04]), (num_envs, 1))
            >>> prims.set_joint_positions(positions)
            >>>
            >>> # set only the fingers in closed position: panda_finger_joint1 (7) and panda_finger_joint2 (8) to 0.0
            >>> # for the first, middle and last of the 5 envs
            >>> positions = np.tile(np.array([0.0, 0.0]), (3, 1))
            >>> prims.set_joint_positions(positions, indices=np.array([0, 2, 4]), joint_indices=np.array([7, 8]))
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            new_dof_pos = self._physics_view.get_dof_positions()
            new_dof_pos = self._backend_utils.assign(
                self._backend_utils.move_data(positions, device=self._device),
                new_dof_pos,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_positions(new_dof_pos, indices)

            # THIS IS THE FIX:
            #   FIX V1: COMMENT OUT THE BELOW LINE AND SET TARGETS INSTEAD
            #   FIX V2: DECOUPLE INSTANTANEOUS POSITION FROM TARGETS
            # self._physics_view.set_dof_position_targets(new_dof_pos, indices)
            # self.set_joint_position_targets(positions, indices, joint_indices)
        else:
            carb.log_warn("Physics Simulation View is not created yet in order to use set_joint_positions")

    def set_joint_velocities(
        self,
        velocities: Optional[Union[np.ndarray, torch.Tensor, wp.array]],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the joint velocities of articulations in the view

        .. warning::

            This method will immediately set the affected joints to the indicated value.
            Use the ``set_joint_velocity_targets`` or the ``apply_action`` methods to control the articulation joints.

        Args:
            velocities (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): joint velocities of articulations in the view to be set to in the next frame.
                                                                    shape is (M, K).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indices to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).

        .. hint::

            This method belongs to the methods used to set the articulation kinematic states:

            ``set_velocities`` (``set_linear_velocities``, ``set_angular_velocities``),
            ``set_joint_positions``, ``set_joint_velocities``, ``set_joint_efforts``

        Example:

        .. code-block:: python

            >>> # set the velocities for all the articulation joints to the indicated values.
            >>> # Since there are 5 envs, the joint velocities are repeated 5 times
            >>> velocities = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), (num_envs, 1))
            >>> prims.set_joint_velocities(velocities)
            >>>
            >>> # set the fingers velocities: panda_finger_joint1 (7) and panda_finger_joint2 (8) to -0.1
            >>> # for the first, middle and last of the 5 envs
            >>> velocities = np.tile(np.array([-0.1, -0.1]), (3, 1))
            >>> prims.set_joint_velocities(velocities, indices=np.array([0, 2, 4]), joint_indices=np.array([7, 8]))
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            new_dof_vel = self._physics_view.get_dof_velocities()
            new_dof_vel = self._backend_utils.assign(
                self._backend_utils.move_data(velocities, device=self._device),
                new_dof_vel,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_velocities(new_dof_vel, indices)

            # THIS IS THE FIX:
            #   FIX V1: COMMENT OUT THE BELOW LINE AND SET TARGETS INSTEAD
            #   FIX V2: DECOUPLE INSTANTANEOUS VELOCITY FROM TARGETS
            # self._physics_view.set_dof_velocity_targets(new_dof_vel, indices)
            # self.set_joint_velocity_targets(velocities, indices, joint_indices)
        else:
            carb.log_warn("Physics Simulation View is not created yet in order to use set_joint_velocities")
        return

    def set_joint_efforts(
        self,
        efforts: Optional[Union[np.ndarray, torch.Tensor, wp.array]],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the joint efforts of articulations in the view

        .. note::

            This method can be used for effort control. For this purpose, there must be no joint drive
            or the stiffness and damping must be set to zero.

        Args:
            efforts (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): efforts of articulations in the view to be set to in the next frame.
                                                                    shape is (M, K).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indices to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).

        .. hint::

            This method belongs to the methods used to set the articulation kinematic states:

            ``set_velocities`` (``set_linear_velocities``, ``set_angular_velocities``),
            ``set_joint_positions``, ``set_joint_velocities``, ``set_joint_efforts``

        Example:

        .. code-block:: python

            >>> # set the efforts for all the articulation joints to the indicated values.
            >>> # Since there are 5 envs, the joint efforts are repeated 5 times
            >>> efforts = np.tile(np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]), (num_envs, 1))
            >>> prims.set_joint_efforts(efforts)
            >>>
            >>> # set the fingers efforts: panda_finger_joint1 (7) and panda_finger_joint2 (8) to 10
            >>> # for the first, middle and last of the 5 envs
            >>> efforts = np.tile(np.array([10, 10]), (3, 1))
            >>> prims.set_joint_efforts(efforts, indices=np.array([0, 2, 4]), joint_indices=np.array([7, 8]))
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return

        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)

            # THIS IS THE FIX: COMMENT OUT THE BELOW LINE AND USE ACTUATION FORCES INSTEAD
            # new_dof_efforts = self._backend_utils.create_zeros_tensor(
            #     shape=[self.count, self.num_dof], dtype="float32", device=self._device
            # )
            new_dof_efforts = self._physics_view.get_dof_actuation_forces()
            new_dof_efforts = self._backend_utils.assign(
                self._backend_utils.move_data(efforts, device=self._device),
                new_dof_efforts,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_actuation_forces(new_dof_efforts, indices)
        else:
            carb.log_warn("Physics Simulation View is not created yet in order to use set_joint_efforts")
        return

    def _invalidate_physics_handle_callback(self, event):
        # Overwrite super method, add additional de-initialization
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._physics_view = None
            self._invalidate_physics_handle_event = None
            self._is_initialized = False

    @property
    def initialized(self):
        # THIS IS THE FIX: another crazy bug from isaac, specifically isaac 4.5
        return self._is_initialized

    def _on_prim_deletion(self, prim_path):
        _XFormPrimView._on_prim_deletion(self, prim_path)
        self._physics_view = None


class RigidPrimView(_RigidPrimView):
    def get_linear_velocities(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None, clone: bool = True
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Get the linear velocities of prims in the view.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view)
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: linear velocities of the prims in the view. shape is (M, 3).

        Example:

        .. code-block:: python

            >>> # get all rigid prim linear velocities. Returned shape is (5, 3) for the example: 5 envs, linear (3)
            >>> prims.get_linear_velocities()
            [[0. 0. 0.]
             [0. 0. 0.]
             [0. 0. 0.]
             [0. 0. 0.]
             [0. 0. 0.]]
            >>>
            >>> # get only the rigid prim linear velocities for the first, middle and last of the 5 envs.
            >>> # Returned shape is (3, 3) for the example: 3 envs selected, linear (3)
            >>> prims.get_linear_velocities(indices=np.array([0, 2, 4]))
            [[0. 0. 0.]
             [0. 0. 0.]
             [0. 0. 0.]]
        """
        if not self._is_valid:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)

        if self.is_physics_handle_valid():
            linear_velocities = self._physics_view.get_velocities()
            if clone:
                # THIS LINE WAS NAMED INCORRECTLY!!!
                linear_velocities = self._backend_utils.clone_tensor(linear_velocities, device=self._device)
            return linear_velocities[indices, 0:3]
        else:
            linear_velocities = np.zeros(shape=(indices.shape[0], 3), dtype=np.float32)
            write_idx = 0
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                if self._rigid_body_apis[i] is None:
                    if self._prims[i].HasAPI(UsdPhysics.RigidBodyAPI):
                        rigid_api = UsdPhysics.RigidBodyAPI(self._prims[i])
                    else:
                        rigid_api = UsdPhysics.RigidBodyAPI.Apply(self._prims[i])
                    self._rigid_body_apis[i] = rigid_api
                linear_velocities[write_idx] = np.array(
                    self._rigid_body_apis[i].GetVelocityAttr().Get(), dtype=np.float32
                )
                write_idx += 1
            linear_velocities = self._backend_utils.convert(
                linear_velocities, dtype="float32", device=self._device, indexed=True
            )
            return linear_velocities

    def get_angular_velocities(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None, clone: bool = True
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Get the angular velocities of prims in the view.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view)
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: angular velocities of the prims in the view. shape is (M, 3).

        Example:

        .. code-block:: python

            >>> # get all rigid prim angular velocities. Returned shape is (5, 3) for the example: 5 envs, angular (3)
            >>> prims.get_angular_velocities()
            [[0. 0. 0.]
             [0. 0. 0.]
             [0. 0. 0.]
             [0. 0. 0.]
             [0. 0. 0.]]
            >>>
            >>> # get only the rigid prim angular velocities for the first, middle and last of the 5 envs
            >>> # Returned shape is (5, 3) for the example: 3 envs selected, angular (3)
            >>> prims.get_angular_velocities(indices=np.array([0, 2, 4]))
            [[0. 0. 0.]
             [0. 0. 0.]
             [0. 0. 0.]]
        """
        if not self._is_valid:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        if self.is_physics_handle_valid():
            angular_velocities = self._physics_view.get_velocities()
            if clone:
                # THIS LINE WAS NAMED INCORRECTLY!!
                angular_velocities = self._backend_utils.clone_tensor(angular_velocities, device=self._device)
            return angular_velocities[indices, 3:6]
        else:
            angular_velocities = np.zeros(shape=(indices.shape[0], 3), dtype=np.float32)
            write_idx = 0
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                if self._rigid_body_apis[i] is None:
                    if self._prims[i].HasAPI(UsdPhysics.RigidBodyAPI):
                        rigid_api = UsdPhysics.RigidBodyAPI(self._prims[i])
                    else:
                        rigid_api = UsdPhysics.RigidBodyAPI.Apply(self._prims[i])
                    self._rigid_body_apis[i] = rigid_api
                angular_velocities[write_idx] = np.array(
                    self._rigid_body_apis[i].GetAngularVelocityAttr().Get(), dtype="float32"
                )
                write_idx += 1
            angular_velocities = self._backend_utils.convert(
                angular_velocities, dtype="float32", device=self._device, indexed=True
            )
            return angular_velocities

    def get_world_poses(
        self,
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
        clone: bool = True,
        usd: bool = True,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]
    ]:
        """Get the poses of the prims in the view with respect to the world's frame.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.
            usd (bool, optional): True to query from usd. Otherwise False to query from Fabric data. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]]:
            first index is positions in the world frame of the prims. shape is (M, 3).
            second index is quaternion orientations in the world frame of the prims.
            quaternion is scalar-first (w, x, y, z). shape is (M, 4).

        Example:

        .. code-block:: python

            >>> # get all rigid prim poses with respect to the world's frame.
            >>> # Returned shape is position (5, 3) and orientation (5, 4) for the example: 5 envs
            >>> positions, orientations = prims.get_world_poses()
            >>> positions
            [[ 1.4999989e+00 -7.4999851e-01 -1.5118626e-07]
             [ 1.4999989e+00  7.5000149e-01 -2.5988294e-07]
             [-1.0017333e-06 -7.4999845e-01  7.6070329e-08]
             [-9.5906785e-07  7.5000149e-01  1.0593490e-07]
             [-1.5000011e+00 -7.4999851e-01  1.9655154e-07]]
            >>> orientations
            [[ 9.9999994e-01 -8.8168377e-07 -4.1946004e-07 -1.5067183e-08]
             [ 9.9999994e-01 -8.8691013e-07 -4.2665880e-07 -2.7188951e-09]
             [ 1.0000000e+00 -9.5171310e-07 -2.2615541e-07  5.5922797e-08]
             [ 1.0000000e+00 -8.9923367e-07 -1.4408238e-07  1.3476099e-08]
             [ 1.0000000e+00 -7.9806580e-07 -1.3064776e-07  5.3154917e-08]]
            >>>
            >>> # get only the rigid prim poses with respect to the world's frame for the first, middle and last of the 5 envs.
            >>> # Returned shape is position (3, 3) and orientation (3, 4) for the example: 3 envs selected
            >>> positions, orientations = prims.get_world_poses(indices=np.array([0, 2, 4]))
            >>> positions
            [[ 1.4999989e+00 -7.4999851e-01 -1.5118626e-07]
             [-1.0017333e-06 -7.4999845e-01  7.6070329e-08]
             [-1.5000011e+00 -7.4999851e-01  1.9655154e-07]]
            >>> orientations
            [[ 9.9999994e-01 -8.8168377e-07 -4.1946004e-07 -1.5067183e-08]
             [ 1.0000000e+00 -9.5171310e-07 -2.2615541e-07  5.5922797e-08]
             [ 1.0000000e+00 -7.9806580e-07 -1.3064776e-07  5.3154917e-08]]
        """
        if not self._is_valid:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))
        if self.is_physics_handle_valid():
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            pose = self._physics_view.get_transforms()
            if clone:
                pose = self._backend_utils.clone_tensor(pose, device=self._device)
            pos = pose[indices, 0:3]

            # We AVOID native self._backend_utils.xyzw2wxyz(pose[indices, 3:7]) because it's slow!!
            rot = pose[:, [6, 3, 4, 5]][indices]
            return pos, rot
        else:
            return _XFormPrimView.get_world_poses(self, indices=indices, usd=usd)

    def enable_gravities(self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None) -> None:
        """Enable gravity on rigid bodies (enabled by default).

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        if not self._is_valid:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))
        if self.is_physics_handle_valid():
            indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
            data = self._physics_view.get_disable_gravities().reshape(self._count)
            data = self._backend_utils.assign(
                self._backend_utils.create_tensor_from_list([False] * len(indices), dtype="uint8"), data, indices
            )
            self._physics_view.set_disable_gravities(data, indices)
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                if self._physx_rigid_body_apis[i] is None:
                    if self._prims[i].HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI(self._prims[i])
                    else:
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(self._prims[i])
                    self._physx_rigid_body_apis[i] = rigid_api
                self._physx_rigid_body_apis[i].GetDisableGravityAttr().Set(False)

    def disable_gravities(self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None) -> None:
        """Disable gravity on rigid bodies (enabled by default).

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        if not self._is_valid:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))
        indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
        if self.is_physics_handle_valid():
            data = self._physics_view.get_disable_gravities().reshape(self._count)
            data = self._backend_utils.assign(
                self._backend_utils.create_tensor_from_list([True] * len(indices), dtype="uint8"), data, indices
            )
            self._physics_view.set_disable_gravities(data, indices)
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                if self._physx_rigid_body_apis[i] is None:
                    if self._prims[i].HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI(self._prims[i])
                    else:
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(self._prims[i])
                    self._physx_rigid_body_apis[i] = rigid_api
                self._physx_rigid_body_apis[i].GetDisableGravityAttr().Set(True)
            return

    def get_coms(
        self, indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None, clone: bool = True
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Get rigid body center of mass (COM) of bodies in the view.

        Args:
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: rigid body center of mass positions and orientations of prims in the view.
            position shape is (M, 1, 3), orientation shape is (M, 1, 4).

        Example:

        .. code-block:: python

            >>> # get all rigid body center of mass.
            >>> # Returned shape is (5, 1, 3) for positions and (5, 1, 4) for orientations for the example: 5 envs
            >>> positions, orientations = prims.get_coms()
            >>> positions
            [[[0. 0. 0.]]
             [[0. 0. 0.]]
             [[0. 0. 0.]]
             [[0. 0. 0.]]
             [[0. 0. 0.]]]
            >>> orientations
            [[[1. 0. 0. 0.]]
             [[1. 0. 0. 0.]]
             [[1. 0. 0. 0.]]
             [[1. 0. 0. 0.]]
             [[1. 0. 0. 0.]]]
            >>>
            >>> # get rigid body center of mass for the first, middle and last of the 5 envs.
            >>> # Returned shape is (3, 1, 3) for positions and (3, 1, 4) for orientations
            >>> positions, orientations = prims.get_coms(indices=np.array([0, 2, 4]))
            >>> positions
            [[[0. 0. 0.]]
             [[0. 0. 0.]]
             [[0. 0. 0.]]]
            >>> orientations
            [[[1. 0. 0. 0.]]
             [[1. 0. 0. 0.]]
             [[1. 0. 0. 0.]]]
        """
        if not self._is_valid:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))
        if self.is_physics_handle_valid():
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            current_values = self._backend_utils.move_data(
                self._physics_view.get_coms().reshape((self.count, 7)), self._device
            )
            if clone:
                current_values = self._backend_utils.clone_tensor(current_values, device=self._device)
            positions = current_values[
                self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, 0:3
            ]
            orientations = self._backend_utils.xyzw2wxyz(
                current_values[self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, 3:7]
            )
            return positions, orientations
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            positions = np.zeros((indices.shape[0], 1, 3), dtype=np.float32)
            orientations = np.zeros((indices.shape[0], 1, 4), dtype=np.float32)
            indices = self._backend_utils.to_list(indices)
            write_idx = 0
            for i in indices:
                positions[write_idx][0] = np.array(self._prims[i].GetAttribute("physics:centerOfMass").Get())
                orientations[write_idx][0] = np.array([1, 0, 0, 0])
                write_idx += 1
            positions = self._backend_utils.convert(positions, device=self._device, dtype="float32", indexed=True)
            orientations = self._backend_utils.convert(orientations, device=self._device, dtype="float32", indexed=True)
            return positions, orientations

    def set_coms(
        self,
        positions: Union[np.ndarray, torch.Tensor, wp.array] = None,
        orientations: Union[np.ndarray, torch.Tensor, wp.array] = None,
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set body center of mass (COM) positions and orientations for bodies in the view.

        Args:
            positions (Union[np.ndarray, torch.Tensor, wp.array]): body center of mass positions for bodies in the view. shape (M, 1, 3).
            orientations (Union[np.ndarray, torch.Tensor, wp.array]): body center of mass orientations for bodies in the view. shape (M, 1, 4).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Example:

        .. code-block:: python

            >>> # set the center of mass for all the rigid bodies to the specified values.
            >>> # Since there are 5 envs, the inertias are repeated 5 times
            >>> positions = np.tile(np.array([0.01, 0.02, 0.03]), (num_envs, 1, 1))
            >>> orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_envs, 1, 1))
            >>> prims.set_coms(positions, orientations)
            >>>
            >>> # set the rigid bodies center of mass for the first, middle and last of the 5 envs
            >>> positions = np.tile(np.array([0.01, 0.02, 0.03]), (3, 1, 1))
            >>> orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1, 1))
            >>> prims.set_coms(positions, orientations, indices=np.array([0, 2, 4]))
        """
        if not self._is_valid:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))
        indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
        if self.is_physics_handle_valid():
            coms = self._physics_view.get_coms().reshape((self.count, 7))
            if positions is not None:
                if self._backend == "warp":
                    coms = self._backend_utils.assign(
                        self._backend_utils.move_data(positions, device="cpu"),
                        coms,
                        [indices, wp.array([0, 1, 2], dtype=wp.int32, device="cpu")],
                    )
                else:
                    coms[self._backend_utils.expand_dims(indices, 1), 0:3] = self._backend_utils.move_data(
                        positions, device="cpu"
                    )
            if orientations is not None:
                if self._backend == "warp":
                    coms = self._backend_utils.assign(
                        self._backend_utils.move_data(self._backend_utils.wxyz2xyzw(orientations), device="cpu"),
                        coms,
                        [indices, wp.array([3, 4, 5, 6], dtype=wp.int32, device="cpu")],
                    )
                else:
                    coms[self._backend_utils.expand_dims(indices, 1), 3:7] = self._backend_utils.move_data(
                        orientations[:, :, [1, 2, 3, 0]], device="cpu"
                    )
            self._physics_view.set_coms(coms, indices)
        else:
            # Note: this does NOT set orientation, as it is not supported in USD
            positions = self._backend_utils.to_list(positions)
            write_idx = 0
            for i in indices:
                properties = self._prims[i].GetPropertyNames()
                position = Gf.Vec3f(*positions[write_idx][0])
                if "physics:centerOfMass" not in properties:
                    carb.log_error(
                        "physics:centerOfMass property needs to be set for {} before setting its position".format(
                            self.name
                        )
                    )
                xform_op = self._prims[i].GetAttribute("physics:centerOfMass")
                xform_op.Set(position)
                write_idx += 1


def colorize_bboxes(bboxes_2d_data, bboxes_2d_rgb, num_channels=3):
    """Colorizes 2D bounding box data for visualization.

    We are overriding the replicator native version of this function to fix a bug.
    In their version of this function, the ordering of the rectangle corners is incorrect and we fix it here.

    Args:
        bboxes_2d_data (numpy.ndarray): 2D bounding box data from the sensor.
        bboxes_2d_rgb (numpy.ndarray): RGB data from the sensor to embed bounding box.
        num_channels (int): Specify number of channels i.e. 3 or 4.
    """
    semantic_id_list = []
    bbox_2d_list = []
    rgb_img = Image.fromarray(bboxes_2d_rgb)
    rgb_img_draw = ImageDraw.Draw(rgb_img)
    for bbox_2d in bboxes_2d_data:
        semantic_id_list.append(bbox_2d[0])
        bbox_2d_list.append(bbox_2d)
    semantic_id_list_np = np.unique(np.array(semantic_id_list))
    color_list = random_colours(len(semantic_id_list_np.tolist()), True, num_channels)
    for bbox_2d in bbox_2d_list:
        index = np.where(semantic_id_list_np == bbox_2d[0])[0][0]
        bbox_color = color_list[index]
        outline = (bbox_color[0], bbox_color[1], bbox_color[2])
        if num_channels == 4:
            outline = (
                bbox_color[0],
                bbox_color[1],
                bbox_color[2],
                bbox_color[3],
            )
        rgb_img_draw.rectangle([(bbox_2d[1], bbox_2d[2]), (bbox_2d[3], bbox_2d[4])], outline=outline, width=2)
    bboxes_2d_rgb = np.array(rgb_img)
    return bboxes_2d_rgb


# This is a faster version than the native implementation, as it avoids pre-processing initially
def _get_world_pose_transform_w_scale(fabric_prim):
    # This will return a transformation matrix with translation as the last row and scale included
    xformable_prim = usdrt.Rt.Xformable(fabric_prim)
    if xformable_prim.HasWorldXform():
        world_pos_attr = xformable_prim.GetWorldPositionAttr()
        if not world_pos_attr.IsValid():
            world_pos = usdrt.Gf.Vec3d(0)
        else:
            world_pos = world_pos_attr.Get(usdrt.Usd.TimeCode.Default())
        world_orientation_attr = xformable_prim.GetWorldOrientationAttr()
        if not world_orientation_attr.IsValid():
            world_orientation = usdrt.Gf.Quatf(1)
        else:
            world_orientation = world_orientation_attr.Get(usdrt.Usd.TimeCode.Default())
        world_scale_attr = xformable_prim.GetWorldScaleAttr()
        if not world_scale_attr.IsValid():
            world_scale = usdrt.Gf.Vec3d(1)
        else:
            world_scale = world_scale_attr.Get(usdrt.Usd.TimeCode.Default())
        scale = usdrt.Gf.Matrix4d()
        rot = usdrt.Gf.Matrix4d()
        scale.SetScale(usdrt.Gf.Vec3d(world_scale))
        rot.SetRotate(usdrt.Gf.Quatd(world_orientation))
        result = scale * rot
        result.SetTranslateOnly(world_pos)
        return result
    elif xformable_prim.HasLocalXform():
        local_transform = xformable_prim.GetLocalMatrixAttr().Get(usdrt.Usd.TimeCode.Default())
        parent_prim = fabric_prim.GetParent()
        parent_world_transform = usdrt.Gf.Matrix4d(1.0)
        if parent_prim:
            parent_world_transform = _get_world_pose_transform_w_scale(parent_prim)
        return local_transform * parent_world_transform
    else:
        usd_prim = get_prim_at_path(prim_path=fabric_prim.GetPrimPath().pathString, fabric=False)
        local_transform = usdrt.Gf.Matrix4d(UsdGeom.Xformable(usd_prim).GetLocalTransformation(Usd.TimeCode.Default()))
        parent_prim = fabric_prim.GetParent()
        parent_world_transform = usdrt.Gf.Matrix4d(1.0)
        if parent_prim:
            parent_world_transform = _get_world_pose_transform_w_scale(parent_prim)
        return local_transform * parent_world_transform


# This is a faster version than the native implementation, as it avoids pre-processing initially and also avoids
# re-ordering the quaternion
def get_world_pose(fabric_prim):
    result_transform = _get_world_pose_transform_w_scale(fabric_prim)
    result_transform.Orthonormalize()
    result_transform = np.transpose(result_transform)
    return result_transform[:3, 3], R.from_matrix(result_transform[:3, :3]).as_quat()
