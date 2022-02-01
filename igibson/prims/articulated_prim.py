# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional, Tuple, Union, List
import numpy as np
from collections import OrderedDict
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.types import DOFInfo
from omni.isaac.core.utils.types import JointsState, ArticulationAction
from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from omni.isaac.core.controllers.articulation_controller import ArticulationController
import carb
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_property, set_prim_property, \
    get_prim_parent, get_prim_at_path

from igibson.prims.xform_prim import XFormPrim
from igibson.prims.rigid_prim import RigidPrim
from igibson.prims.joint_prim import JointPrim


class ArticulatedPrim(XFormPrim):
    """
    Provides high level functions to deal with an articulation prim and its attributes/ properties. Note that this
    type of prim cannot be created from scratch, and assumes there is already a pre-existing prim tree that should
    be converted into an articulation!

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that this is only needed if the prim does not already exist at
            @prim_path -- it will be ignored if it already exists. For this articulated prim, no values should be
            specified by default because this prim cannot be created from scratch!
        """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        # Grab dynamic control reference and store initialized values
        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        # Other values that will be filled in at runtime
        self._handle = None                     # Handle to this articulation
        self._root_handle = None                # Handle to the root rigid body of this articulation
        self._dofs_infos = None
        self._num_dof = None
        self._articulation_controller = None
        self._default_joints_state = None
        self._rigid_api = None
        self._links = None
        self._joints = None

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _setup_references(self):
        # Run super method
        super()._setup_references()

        # Get dynamic control info
        self._handle = self._dc.get_articulation(self.prim_path)
        self._root_handle = self._dc.get_articulation_root_body(self._handle)
        self._root_prim = get_prim_at_path(self._dc.get_rigid_body_path(self._root_handle))
        self._num_dof = self._dc.get_articulation_dof_count(self._handle)

        # Create rigid api for the root prim
        self._rigid_api = UsdPhysics.RigidBodyAPI(self._root_prim)

        # Add additional DOF info if this is an articulated object
        if self._num_dof > 0:
            self._articulation_controller = ArticulationController()
            self._dofs_infos = OrderedDict()
            # Grab DOF info
            for index in range(self._num_dof):
                dof_handle = self._dc.get_articulation_dof(self._handle, index)
                dof_name = self._dc.get_dof_name(dof_handle)
                # add dof to list
                prim_path = self._dc.get_dof_path(dof_handle)
                self._dofs_infos[dof_name] = DOFInfo(prim_path=prim_path, handle=dof_handle, prim=self.prim, index=index)
            # Initialize articulation controller
            self._articulation_controller.initialize(self._handle, self._dofs_infos)
            # Default joints state is the info from the USD
            default_actions = self._articulation_controller.get_applied_action()
            self._default_joints_state = JointsState(
                positions=np.array(default_actions.joint_positions),
                velocities=np.array(default_actions.joint_velocities),
                efforts=np.zeros_like(default_actions.joint_positions),
            )

        # Setup links and joints info
        self._links = OrderedDict()
        for i in range(self._dc.get_articulation_body_count(self._handle)):
            link_handle = self._dc.get_articulation_body(self._handle, i)
            link_name = self._dc.get_rigid_body_name(link_handle)
            link_path = self._dc.get_rigid_body_path(link_handle)
            self._links[link_name] = RigidPrim(
                prim_path=link_path,
                name=f"{self._name}:{link_name}",
            )

        self._joints = OrderedDict()
        for i in range(self._dc.get_articulation_joint_count(self._handle)):
            joint_handle = self._dc.get_articulation_joint(self._handle, i)
            joint_name = self._dc.get_joint_name(joint_handle)
            joint_path = self._dc.get_joint_path(joint_handle)
            self._joints[joint_name] = JointPrim(
                prim_path=joint_path,
                name=f"{self._name}:{joint_name}",
                articulation=self._handle,
            )

    def _load(self, simulator=None):
        # This should not be called, because this prim cannot be instantiated from scratch!
        raise NotImplementedError("By default, an articulated prim cannot be created from scratch.")

    @property
    def handle(self):
        """[summary]

        Returns:
            int: [description]
        """
        return self._handle

    @property
    def num_dof(self):
        """[summary]

        Returns:
            int: [description]
        """
        return self._num_dof

    @property
    def num_joints(self):
        """
        Returns:
            int: Number of joints owned by this articulation
        """
        return len(list(self._joints.keys()))

    @property
    def num_links(self):
        """
        Returns:
            int: Number of links owned by this articulation
        """
        return len(list(self._links.keys()))

    @property
    def joints(self):
        """
        Returns:
            OrderedDict: Dictionary mapping joint names (str) to joint prims (JointPrim) owned by this articulation
        """
        return self._joints

    @property
    def links(self):
        """
        Returns:
            OrderedDict: Dictionary mapping link names (str) to link prims (RigidPrim) owned by this articulation
        """
        return self._links

    @property
    def dof_properties(self):
        """[summary]

        Returns:
            np.ndarray: [description]
        """
        return self._dc.get_articulation_dof_properties(self._handle)

    def get_dof_index(self, dof_name: str) -> int:
        """[summary]

        Args:
            dof_name (str): [description]

        Returns:
            int: [description]
        """
        return self._dofs_infos[dof_name].index

    def read_kinematic_hierarchy(self) -> None:
        """[summary]
        """
        print("Articulation handle: {self._handle}")
        print("--- Hierarchy:\n", self._read_kinematic_hierarchy())
        return

    def _read_kinematic_hierarchy(self, body_index: Optional[int] = None, indent_level: int = 0) -> str:
        if body_index is None:
            body_index = self._dc.get_articulation_root_body(self._handle)
        indent = "|" + "-" * indent_level
        body_name = self._dc.get_rigid_body_name(body_index)
        str_output = f"{indent}Body: {body_name}\n", "blue"
        for i in range(self._dc.get_rigid_body_child_joint_count(body_index)):
            joint = self._dc.get_rigid_body_child_joint(body_index, i)
            joint_name = self._dc.get_joint_name(joint)
            child = self._dc.get_joint_child_body(joint)
            child_name = self._dc.get_rigid_body_name(child)
            str_output += f"{indent}>>Joint: {joint_name} -> {child_name}\n", "green"
            str_output += self._read_kinematic_hierarchy(child, indent_level + 4)
        return str_output

    def disable_gravity(self) -> None:
        """[summary]
        """
        for body_index in range(self._dc.get_articulation_body_count(self._handle)):
            body = self._dc.get_articulation_body(self._handle, body_index)
            self._dc.set_rigid_body_disable_gravity(body, False)
        return

    def set_joint_positions(self, positions: np.ndarray, indices: Optional[Union[List, np.ndarray]] = None) -> None:
        """[summary]

        Args:
            positions (np.ndarray): [description]
            indices (Optional[Union[list, np.ndarray]], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        dof_states = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_POS)
        if indices is None:
            new_joint_positions = positions
        else:
            new_joint_positions = self.get_joint_positions()
            for i in range(len(indices)):
                new_joint_positions[indices[i]] = positions[i]
        dof_states["pos"] = new_joint_positions
        self._dc.set_articulation_dof_states(self._handle, dof_states, _dynamic_control.STATE_POS)
        self._articulation_controller.apply_action(
            ArticulationAction(joint_positions=new_joint_positions, joint_velocities=None, joint_efforts=None)
        )
        return

    def set_joint_velocities(self, velocities: np.ndarray, indices: Optional[Union[List, np.ndarray]] = None) -> None:
        """[summary]

        Args:
            velocities (np.ndarray): [description]
            indices (Optional[Union[list, np.ndarray]], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        dof_states = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_VEL)
        if indices is None:
            new_joint_velocities = velocities
        else:
            new_joint_velocities = self.get_joint_velocities()
            for i in range(len(indices)):
                new_joint_velocities[indices[i]] = velocities[i]
        dof_states["vel"] = new_joint_velocities
        self._dc.set_articulation_dof_states(self._handle, dof_states, _dynamic_control.STATE_VEL)
        self._articulation_controller.apply_action(
            ArticulationAction(joint_positions=None, joint_velocities=new_joint_velocities, joint_efforts=None)
        )
        return

    def set_joint_efforts(self, efforts: np.ndarray, indices: Optional[Union[List, np.ndarray]] = None) -> None:
        """[summary]

        Args:
            efforts (np.ndarray): [description]
            indices (Optional[Union[list, np.ndarray]], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        dof_states = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_EFFORT)
        if indices is None:
            new_joint_efforts = efforts
        else:
            new_joint_efforts = [0] * self.num_dof
            for i in range(len(indices)):
                new_joint_efforts[indices[i]] = efforts[i]
        dof_states["effort"] = new_joint_efforts
        self._dc.set_articulation_dof_states(self._handle, dof_states, _dynamic_control.STATE_EFFORT)
        self._articulation_controller.apply_action(
            ArticulationAction(joint_positions=None, joint_velocities=None, joint_efforts=new_joint_efforts)
        )
        return

    def get_joint_positions(self) -> np.ndarray:
        """[summary]

        Raises:
            Exception: [description]

        Returns:
            np.ndarray: [description]
        """
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        joint_positions = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_POS)
        joint_positions = [joint_positions[i][0] for i in range(len(joint_positions))]
        return np.array(joint_positions)

    def get_joint_velocities(self) -> np.ndarray:
        """[summary]

        Raises:
            Exception: [description]

        Returns:
            np.ndarray: [description]
        """
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        joint_velocities = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_VEL)
        joint_velocities = [joint_velocities[i][1] for i in range(len(joint_velocities))]
        return np.array(joint_velocities)

    def get_joint_efforts(self) -> np.ndarray:
        """[summary]

        Raises:
            Exception: [description]

        Returns:
            np.ndarray: [description]
        """
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        joint_efforts = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_EFFORT)
        joint_efforts = [joint_efforts[i][2] for i in range(len(joint_efforts))]
        return joint_efforts

    def set_joints_default_state(
        self,
        positions: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        efforts: Optional[np.ndarray] = None,
    ) -> None:
        """[summary]

        Args:
            positions (Optional[np.ndarray], optional): [description]. Defaults to None.
            velocities (Optional[np.ndarray], optional): [description]. Defaults to None.
            efforts (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        if positions is not None:
            self._default_joints_state.positions = positions
        if velocities is not None:
            self._default_joints_state.velocities = velocities
        if efforts is not None:
            self._default_joints_state.efforts = efforts
        return

    def get_joints_state(self) -> JointsState:
        """[summary]

        Returns:
            JointsState: [description]
        """
        return JointsState(
            positions=self.get_joint_positions(),
            velocities=self.get_joint_velocities(),
            efforts=self.get_joint_efforts(),
        )

    def reset(self):
        # Run super reset first to reset this articulation's pose
        super().reset()

        # Reset state
        self.set_joint_positions(self._default_joints_state.positions)
        self.set_joint_velocities(self._default_joints_state.velocities)
        self.set_joint_efforts(self._default_joints_state.efforts)

    def get_articulation_controller(self) -> ArticulationController:
        """
        Returns:
            ArticulationController: PD Controller of all degrees of freedom of an articulation, can apply position targets, velocity targets and efforts.
        """
        return self._articulation_controller

    def set_linear_velocity(self, velocity: np.ndarray):
        """Sets the linear velocity of the prim in stage.

        Args:
            velocity (np.ndarray):linear velocity to set the rigid prim to. Shape (3,).
        """

        if self._root_handle is not None and self._dc.is_simulating():
            self._dc.set_rigid_body_linear_velocity(self._root_handle, velocity)
        else:
            self._rigid_api.GetVelocityAttr().Set(Gf.Vec3f(velocity.tolist()))
        return

    def get_linear_velocity(self) -> np.ndarray:
        """[summary]

        Returns:
            np.ndarray: [description]
        """
        if self._root_handle is not None and self._dc.is_simulating():
            return self._dc.get_rigid_body_linear_velocity(self._root_handle)
        else:
            return np.array(self._rigid_api.GetVelocityAttr().Get())

    def set_angular_velocity(self, velocity: np.ndarray) -> None:
        """[summary]

        Args:
            velocity (np.ndarray): [description]
        """
        if self._root_handle is not None and self._dc.is_simulating():
            self._dc.set_rigid_body_angular_velocity(self._root_handle, velocity)
        else:
            self._rigid_api.GetAngularVelocityAttr().Set(Gf.Vec3f(velocity.tolist()))
        return

    def get_angular_velocity(self) -> np.ndarray:
        """[summary]

        Returns:
            np.ndarray: [description]
        """
        if self._root_handle is not None and self._dc.is_simulating():
            return self._dc.get_rigid_body_angular_velocity(self._root_handle)
        else:
            return np.array(self._rigid_api.GetAngularVelocityAttr().Get())

    def set_position_orientation(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None) -> None:
        """Sets prim's pose with respect to the world's frame.

        Args:
            position (Optional[np.ndarray], optional): position in the world frame of the prim. shape is (3, ).
                                                       Defaults to None, which means left unchanged.
            orientation (Optional[np.ndarray], optional): quaternion orientation in the world frame of the prim.
                                                          quaternion is scalar-last (x, y, z, w). shape is (4, ).
                                                          Defaults to None, which means left unchanged.
        """
        if self._root_handle is not None and self._dc.is_simulating():
            current_position, current_orientation = self.get_position_orientation()
            if position is None:
                position = current_position
            if orientation is None:
                orientation = current_orientation
            pose = _dynamic_control.Transform(position, orientation)
            self._dc.set_rigid_body_pose(self._root_handle, pose)
        else:
            super().set_position_orientation(position=position, orientation=orientation)

    def get_position_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets prim's pose with respect to the world's frame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: first index is position in the world frame of the prim. shape is (3, ).
                                           second index is quaternion orientation in the world frame of the prim.
                                           quaternion is scalar-last (x, y, z, w). shape is (4, ).
        """
        if self._root_handle is not None and self._dc.is_simulating():
            pose = self._dc.get_rigid_body_pose(self._root_handle)
            return np.asarray(pose.p), np.asarray(pose.p)
        else:
            return super().get_position_orientation()

    def set_local_pose(
        self, translation: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None
    ) -> None:
        """Sets prim's pose with respect to the local frame (the prim's parent frame).

        Args:
            translation (Optional[np.ndarray], optional): translation in the local frame of the prim
                                                          (with respect to its parent prim). shape is (3, ).
                                                          Defaults to None, which means left unchanged.
            orientation (Optional[np.ndarray], optional): quaternion orientation in the world frame of the prim.
                                                          quaternion is scalar-last (x, y, z, w). shape is (4, ).
                                                          Defaults to None, which means left unchanged.
        """
        if self._root_handle is not None and self._dc.is_simulating():
            current_translation, current_orientation = self.get_local_pose()
            if translation is None:
                translation = current_translation
            if orientation is None:
                orientation = current_orientation
            orientation = orientation[[3, 0, 1, 2]]
            local_transform = tf_matrix_from_pose(translation=translation, orientation=orientation)
            parent_world_tf = UsdGeom.Xformable(get_prim_parent(self._prim)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            my_world_transform = np.matmul(parent_world_tf, local_transform)
            transform = Gf.Transform()
            transform.SetMatrix(Gf.Matrix4d(np.transpose(my_world_transform)))
            calculated_position = transform.GetTranslation()
            calculated_orientation = transform.GetRotation().GetQuat()
            self.set_position_orientation(
                position=np.array(calculated_position),
                orientation=gf_quat_to_np_array(calculated_orientation)[[1, 2, 3, 0]],
            )
            return
        else:
            super().set_local_pose(translation=translation, orientation=orientation)
            return

    def get_local_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets prim's pose with respect to the local frame (the prim's parent frame).

        Returns:
            Tuple[np.ndarray, np.ndarray]: first index is position in the local frame of the prim. shape is (3, ).
                                           second index is quaternion orientation in the local frame of the prim.
                                           quaternion is scalar-last (x, y, z, w). shape is (4, ).
        """
        if self._root_handle is not None and self._dc.is_simulating():
            parent_world_tf = UsdGeom.Xformable(get_prim_parent(self._prim)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            world_position, world_orientation = self.get_position_orientation()
            my_world_transform = tf_matrix_from_pose(translation=world_position, orientation=world_orientation[[3, 0, 1, 2]])
            local_transform = np.matmul(np.linalg.inv(np.transpose(parent_world_tf)), my_world_transform)
            transform = Gf.Transform()
            transform.SetMatrix(Gf.Matrix4d(np.transpose(local_transform)))
            calculated_translation = transform.GetTranslation()
            calculated_orientation = transform.GetRotation().GetQuat()
            return np.array(calculated_translation), gf_quat_to_np_array(calculated_orientation)[[1, 2, 3, 0]]
        else:
            return super().get_local_pose()

    def apply_action(
        self, control_actions: ArticulationAction, indices: Optional[Union[List, np.ndarray]] = None
    ) -> None:
        """[summary]

        Args:
            control_actions (ArticulationAction): actions to be applied for next physics step.
            indices (Optional[Union[list, np.ndarray]], optional): degree of freedom indices to apply actions to.
                                                                   Defaults to all degrees of freedom.

        Raises:
            Exception: [description]
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        self._articulation_controller.apply_action(control_actions=control_actions, indices=indices)
        return

    def get_applied_action(self) -> ArticulationAction:
        """[summary]

        Raises:
            Exception: [description]

        Returns:
            ArticulationAction: [description]
        """
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        return self._articulation_controller.get_applied_action()

    def set_solver_position_iteration_count(self, count: int) -> None:
        """[summary]

        Args:
            count (int): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:solverPositionIterationCount", count)
        return

    def get_solver_position_iteration_count(self) -> int:
        """[summary]

        Returns:
            int: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:solverPositionIterationCount")

    def set_solver_velocity_iteration_count(self, count: int):
        """[summary]

        Args:
            count (int): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:solverVelocityIterationCount", count)
        return

    def get_solver_velocity_iteration_count(self) -> int:
        """[summary]

        Returns:
            int: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:solverVelocityIterationCount")

    def set_stabilization_threshold(self, threshold: float) -> None:
        """[summary]

        Args:
            threshold (float): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:stabilizationThreshold", threshold)
        return

    def get_stabilization_threshold(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:stabilizationThreshold")

    def set_enabled_self_collisions(self, flag: bool) -> None:
        """[summary]

        Args:
            flag (bool): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:enabledSelfCollisions", flag)
        return

    def get_enabled_self_collisions(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:enabledSelfCollisions")

    def set_sleep_threshold(self, threshold: float) -> None:
        """[summary]

        Args:
            threshold (float): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:sleepThreshold", threshold)
        return

    def get_sleep_threshold(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:sleepThreshold")

    def wake(self):
        """
        Enable physics for this articulation
        """
        self._dc.wake_up_articulation(self._handle)

    def sleep(self):
        """
        Disable physics for this articulation
        """
        self._dc.sleep_articulation(self._handle)
