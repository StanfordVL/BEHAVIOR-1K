# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional, Tuple
from omni.isaac.core.utils.types import DynamicState
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_parent
from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from pxr import Gf, UsdPhysics, Usd, UsdGeom
import numpy as np
from omni.isaac.dynamic_control import _dynamic_control
import carb

from igibson.prims.xform_prim import XFormPrim


class RigidPrim(XFormPrim):
    """
    Provides high level functions to deal with a rigid body prim and its attributes/ properties.
    If there is an prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created.

    Notes: if the prim does not already have a rigid body api applied to it before it is loaded,
        it will apply it.

    Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime. Note that this is only needed if the prim does not already exist at
                @prim_path -- it will be ignored if it already exists. For this joint prim, the below values can be
                specified:

                scale (None or float or 3-array): If specified, sets the scale for this object. A single number corresponds
                    to uniform scaling along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
                mass (None or float): If specified, mass of this body in kg
    """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        # Get dynamic control interface and store initialized values
        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        # Other values that will be filled in at runtime
        self._handle = None
        self._body_name = None
        self._rigid_api = None
        self._mass_api = None
        self._default_state = None

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _load(self, simulator=None):
        # Run super first
        super()._load(simulator=simulator)

        # Apply rigid body and mass APIs
        self._rigid_api = UsdPhysics.RigidBodyAPI.Apply(self._prim)
        self._mass_api = UsdPhysics.MassAPI.Apply(self._prim)

        # Possibly set the mass
        if "mass" in self._load_config:
            self.mass = self._load_config["mass"]

    def _setup_references(self):
        # Run super method first
        super()._setup_references()

        # Grab handle to this rigid body and get name
        self._handle = self._dc.get_rigid_body(self._prim_path)
        self._body_name = self._dc.get_rigid_body_name(self._handle)

        # Create rigid and mass apis if they weren't already created during loading
        self._rigid_api = UsdPhysics.RigidBodyAPI(self._prim) if self._rigid_api is None else self._rigid_api
        self._mass_api = UsdPhysics.MassAPI(self._prim) if self._mass_api is None else self._mass_api

        # Add enabled attribute for the rigid body
        self._rigid_api.CreateRigidBodyEnabledAttr(True)

        # Set the default state
        pos, ori = self.get_position_orientation()
        lin_vel = self.get_linear_velocity()
        ang_vel = self.get_angular_velocity()
        self._default_state = DynamicState(
            position=pos,
            orientation=ori,
            linear_velocity=lin_vel,
            angular_velocity=ang_vel,
        )

    def set_linear_velocity(self, velocity):
        """Sets the linear velocity of the prim in stage.

        Args:
            velocity (np.ndarray): linear velocity to set the rigid prim to. Shape (3,).
        """
        if self._handle is not None and self._dc.is_simulating():
            self._dc.set_rigid_body_linear_velocity(self._handle, velocity)
        else:
            self._rigid_api.GetVelocityAttr().Set(Gf.Vec3f(velocity.tolist()))
        return

    def get_linear_velocity(self):
        """
        Returns:
            np.ndarray: current linear velocity of the the rigid prim. Shape (3,).
        """
        if self._handle is not None and self._dc.is_simulating():
            return self._dc.get_rigid_body_linear_velocity(self._handle)
        else:
            return np.array(self._rigid_api.GetVelocityAttr().Get())

    def set_angular_velocity(self, velocity):
        """Sets the angular velocity of the prim in stage.

        Args:
            velocity (np.ndarray): angular velocity to set the rigid prim to. Shape (3,).
        """
        if self._handle is not None and self._dc.is_simulating():
            self._dc.set_rigid_body_angular_velocity(self._handle, velocity)
        else:
            self._rigid_api.GetAngularVelocityAttr().Set(Gf.Vec3f(velocity.tolist()))
        return

    def get_angular_velocity(self):
        """
        Returns:
            np.ndarray: current angular velocity of the the rigid prim. Shape (3,).
        """
        if self._handle is not None and self._dc.is_simulating():
            return self._dc.get_rigid_body_angular_velocity(self._handle)
        else:
            return np.array(self._rigid_api.GetAngularVelocityAttr().Get())

    def set_position_orientation(self, position=None, orientation=None):
        """
        Sets prim's pose with respect to the world's frame.

        Args:
            position (Optional[np.ndarray], optional): position in the world frame of the prim. shape is (3, ).
                                                       Defaults to None, which means left unchanged.
            orientation (Optional[np.ndarray], optional): quaternion orientation in the world frame of the prim.
                                                          quaternion is scalar-last (x, y, z, w). shape is (4, ).
                                                          Defaults to None, which means left unchanged.
        """
        if self._handle is not None and self._dc.is_simulating():
            current_position, current_orientation = self.get_position_orientation()
            if position is None:
                position = current_position
            if orientation is None:
                orientation = current_orientation
            pose = _dynamic_control.Transform(position, orientation)
            self._dc.set_rigid_body_pose(self._handle, pose)
        else:
            # Call super method by default
            super().set_position_orientation(position=position, orientation=orientation)

    def get_position_orientation(self):
        """
        Gets prim's pose with respect to the world's frame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: first index is position in the world frame of the prim. shape is (3, ).
                                           second index is quaternion orientation in the world frame of the prim.
                                           quaternion is scalar-last (x, y, z, w). shape is (4, ).
        """
        if self._handle is not None and self._dc.is_simulating():
            pose = self._dc.get_rigid_body_pose(self._handle)
            pos, ori = pose.p, pose.r
        else:
            # Call super method by default
            pos, ori = super().get_position_orientation()

        return pos, ori

    def set_local_pose(self, translation=None, orientation=None):
        """
        Sets prim's pose with respect to the local frame (the prim's parent frame).

        Args:
            translation (Optional[np.ndarray], optional): translation in the local frame of the prim
                                                          (with respect to its parent prim). shape is (3, ).
                                                          Defaults to None, which means left unchanged.
            orientation (Optional[np.ndarray], optional): quaternion orientation in the world frame of the prim.
                                                          quaternion is scalar-last (x, y, z, w). shape is (4, ).
                                                          Defaults to None, which means left unchanged.
        """
        if self._handle is not None and self._dc.is_simulating():
            current_translation, current_orientation = self.get_local_pose()
            translation = current_translation if translation is None else translation
            orientation = current_orientation if orientation is None else orientation
            orientation = orientation[[3, 0, 1, 2]]  # Flip from x,y,z,w to w,x,y,z
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
                position=np.array(calculated_position), orientation=gf_quat_to_np_array(calculated_orientation)
            )
        else:
            # Call super method by default
            super().set_local_pose(translation=translation, orientation=orientation)

    def get_local_pose(self):
        """
        Gets prim's pose with respect to the local frame (the prim's parent frame).

        Returns:
            Tuple[np.ndarray, np.ndarray]: first index is position in the local frame of the prim. shape is (3, ).
                                           second index is quaternion orientation in the local frame of the prim.
                                           quaternion is scalar-last (x, y, z, w). shape is (4, ).
        """
        if self._handle is not None and self._dc.is_simulating():
            parent_world_tf = UsdGeom.Xformable(get_prim_parent(self._prim)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            world_position, world_orientation = self.get_position_orientation()
            world_orientation = world_orientation[[3, 0, 1, 2]]  # Flip from x,y,z,w to w,x,y,z
            my_world_transform = tf_matrix_from_pose(translation=world_position, orientation=world_orientation)
            local_transform = np.matmul(np.linalg.inv(np.transpose(parent_world_tf)), my_world_transform)
            transform = Gf.Transform()
            transform.SetMatrix(Gf.Matrix4d(np.transpose(local_transform)))
            calculated_translation = transform.GetTranslation()
            calculated_orientation = transform.GetRotation().GetQuat()
            pos, ori = np.array(calculated_translation), gf_quat_to_np_array(calculated_orientation)[[1, 2, 3, 0]] # Flip from w,x,y,z to x,y,z,w to
        else:
            # Call super method by default
            pos, ori = super().get_local_pose()

        return pos, ori

    @property
    def body_name(self):
        """
        Returns:
            str: Name of this body
        """
        return self._body_name

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        return self._mass_api.GetMassAttr().Get()

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        self._mass_api.GetMassAttr().Set(mass)

    def set_default_state(
        self,
        position=None,
        orientation=None,
        linear_velocity=None,
        angular_velocity=None,
    ):
        """Sets the default state of the prim, that will be used after each reset.

        Args:
            position (np.ndarray): position in the world frame of the prim. shape is (3, ).
                                   Defaults to None, which means left unchanged.
            orientation (np.ndarray): quaternion orientation in the world frame of the prim.
                                      quaternion is scalar-last (x, y, z, w). shape is (4, ).
                                      Defaults to None, which means left unchanged.
            linear_velocity (np.ndarray): linear velocity to set the rigid prim to. Shape (3,).
            angular_velocity (np.ndarray): angular velocity to set the rigid prim to. Shape (3,).
        """
        if position is not None:
            self._default_state.position = position
        if orientation is not None:
            self._default_state.orientation = orientation
        if linear_velocity is not None:
            self._default_state.linear_velocity = linear_velocity
        if angular_velocity is not None:
            self._default_state.angular_velocity = angular_velocity
        return

    def get_default_state(self):
        """
        Returns:
            DynamicState: returns the default state of the prim (position, orientation, linear_velocity and
                          angular_velocity) that is used after each reset.
        """
        return self._default_state

    def reset(self):
        """
        Resets the prim to its default state.
        """
        # Call super method to reset pose
        super().reset()

        # Also reset the velocity values
        self.set_linear_velocity(velocity=self._default_state.linear_velocity)
        self.set_angular_velocity(velocity=self._default_state.angular_velocity)

    def get_current_dynamic_state(self):
        """
        Returns:
            DynamicState: the dynamic state of the rigid body including position, orientation, linear_velocity and
                angular_velocity.
        """
        position, orientation = self.get_position_orientation()
        return DynamicState(
            position=position,
            orientation=orientation,
            linear_velocity=self.get_linear_velocity(),
            angular_velocity=self.get_angular_velocity(),
        )

    def wake(self):
        """
        Enable physics for this rigid body
        """
        self._dc.wake_up_rigid_body(self._handle)

    def sleep(self):
        """
        Disable physics for this rigid body
        """
        self._dc.sleep_rigid_body(self._handle)
