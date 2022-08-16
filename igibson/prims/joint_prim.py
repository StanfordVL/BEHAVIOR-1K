# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from collections import Iterable, OrderedDict
from typing import Optional, Tuple
from pxr import Gf, Usd, Sdf, UsdGeom, UsdShade, UsdPhysics, PhysxSchema
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.materials import PreviewSurface, OmniGlass, OmniPBR, VisualMaterial
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.core.utils.prims import (
    get_prim_at_path,
    move_prim,
    query_parent_path,
    is_prim_path_valid,
    define_prim,
    get_prim_parent,
    get_prim_object_type,
)
import numpy as np
import carb
from omni.isaac.core.utils.stage import get_current_stage
from igibson.macros import create_module_macros
from igibson.prims.prim_base import BasePrim
from igibson.utils.usd_utils import create_joint
from igibson.utils.omni_types import JointsState
from igibson.utils.transform_utils import quat_inverse, quat_multiply
from igibson.utils.constants import JointType
from igibson.utils.python_utils import assert_valid_key
import igibson.utils.transform_utils as T
from igibson.controllers.controller_base import ControlType


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_MAX_POS = 1000.0
m.DEFAULT_MAX_PRISMATIC_VEL = 1.0
m.DEFAULT_MAX_REVOLUTE_VEL = 15.0
m.DEFAULT_MAX_EFFORT = 100.0
m.COMPONENT_SUFFIXES = ["x", "y", "z", "rx", "ry", "rz"]

# TODO: Split into non-articulated / articulated Joint Prim classes?


# TODO: Add logic for non Prismatic / Revolute joints (D6, spherical)


class JointPrim(BasePrim):
    """
    Provides high level functions to deal with a joint prim and its attributes/ properties.
    If there is an joint prim present at the path, it will use it. Otherwise, a new joint prim at
    the specified prim path will be created when self.load(...) is called.

        Note: the prim will have "xformOp:orient", "xformOp:translate" and "xformOp:scale" only post init,
                unless it is a non-root articulation link.

        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime. For this joint prim, the below values can be specified:

                joint_type (str): If specified, should be the joint type to create. Valid options are:
                    {"Joint", "FixedJoint", "PrismaticJoint", "RevoluteJoint", "SphericalJoint"}
                    (equivalently, one of JointType)
                body0 (None or str): If specified, should be the absolute prim path to the parent body that this joint
                    is connected to. None can also be valid, which corresponds to cases where only a single body may be
                    specified (e.g.: fixed joints)
                body1 (None or str): If specified, should be the absolute prim path to the child body that this joint
                    is connected to. None can also be valid, which corresponds to cases where only a single body may be
                    specified (e.g.: fixed joints)

            articulation (None or int): if specified, should be handle to pre-existing articulation. This will enable
                additional features for this joint prim, e.g.: polling / setting this joint's state. Note that in this
                case, the joint must already exist prior to this class instance. Default is None,
                which corresponds to a non-articulated joint.
    """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
        articulation=None,
    ):
        # Grab dynamic control reference and set properties
        self._art = articulation

        # Other values that will be filled in at runtime
        self._joint_type = None
        self._control_type = None
        self._dof_properties = None

        # The following values will only be valid if this joint is part of an articulation
        self._dc = None
        self._handle = None
        self._n_dof = None
        self._joint_name = None
        self._dof_handles = None
        self._default_state = None

        # Run super method
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _load(self, simulator=None):
        # Make sure this joint isn't articulated
        assert not self.articulated, "Joint cannot be created, since this is an articulated joint! We are assuming" \
                                     "the joint already exists in the stage."

        # Define a joint prim at the current stage
        stage = get_current_stage()
        prim = create_joint(
            prim_path=self._prim_path,
            joint_type=self._load_config.get("joint_type", JointType.JOINT),
            stage=stage,
        )

        return prim

    def _post_load(self):
        # run super first
        super()._post_load()

        # Possibly set the bodies
        if "body0" in self._load_config and self._load_config["body0"] is not None:
            self.body0 = self._load_config["body0"]
        if "body1" in self._load_config and self._load_config["body1"] is not None:
            self.body1 = self._load_config["body1"]

    def _initialize(self):
        # Always run super first
        super()._initialize()

        # Get joint info
        self._joint_type = JointType.get_type(self._prim.GetTypeName().split("Physics")[-1])

        # Initialize dynamic control references if this joint is articulated
        if self.articulated:
            self._dc = _dynamic_control.acquire_dynamic_control_interface()
            # TODO: A bit hacky way to get the joint handle, ideally we'd simply do dc.get_joint(), but this doesn't seem to work as expected?
            for i in range(self._dc.get_articulation_joint_count(self._art)):
                joint_handle = self._dc.get_articulation_joint(self._art, i)
                joint_path = self._dc.get_joint_path(joint_handle)
                if joint_path == self._prim_path:
                    self._handle = joint_handle
                    break
            assert self._handle is not None, f"Did not find valid articulated joint with path: {self._prim_path}"

            # Grab DOF info / handles
            self._joint_name = self._dc.get_joint_name(self._handle)
            self._n_dof = self._dc.get_joint_dof_count(self._handle)
            self._dof_handles = []
            self._dof_properties = []
            control_types = []
            for i in range(self._n_dof):
                dof_handle = self._dc.get_joint_dof(self._handle, i)
                dof_props = self._dc.get_dof_properties(dof_handle)
                self._dof_handles.append(dof_handle)
                self._dof_properties.append(dof_props)
                # Infer control type based on whether kp and kd are 0 or not
                kp, kd = dof_props.stiffness, dof_props.damping
                if kp == 0.0:
                    control_type = ControlType.EFFORT if kd == 0.0 else ControlType.VELOCITY
                else:
                    control_type = ControlType.POSITION
                control_types.append(control_type)

            # Make sure all the control types are the same -- if not, we had something go wrong!
            assert len(set(control_types)) == 1, f"Got multiple control types for this single joint: {control_types}"
            self._control_type = control_types[0]

            # Grab default state
            default_pos, default_vel, default_effort = self.get_state()
            self._default_state = JointsState(positions=default_pos, velocities=default_vel, efforts=default_effort)

    # def reset(self):
    #     """
    #     Resets the prim to its default state (position and orientation).
    #     """
    #     if self.articulated:
    #         # TODO: Do we also need to reset targets here?
    #         self.set_pos(pos=self._default_state.positions, target=False)
    #         self.set_vel(vel=self._default_state.velocities, target=False)
    #         self.set_effort(effort=self._default_state.efforts)

    def update_handles(self):
        """
        Updates all internal handles for this prim, in case they change since initialization
        """
        self._handle = self._dc.get_joint(self._prim_path)

    def get_default_state(self):
        """
        Returns:
            JointsState: returns the default state of the joint prim (positions, velocities, efforts)
                that is used after each reset.
        """
        self.assert_articulated()
        return self._default_state

    def set_default_state(self, positions=None, velocities=None, efforts=None):
        """
        Sets the default state of the joint prim (positions, velocities, efforts), that will be used after each reset.

        Args:
            positions (None or n-array): positions for all DOFs corresponding to this joint. Should be
                an n-array if specified, where n is the number of DOF for this joint. Defaults to None,
                which means left unchanged.
            velocities (None or n-array): velocities for all DOFs corresponding to this joint. Should be
                an n-array if specified, where n is the number of DOF for this joint. Defaults to None,
                which means left unchanged.
            efforts (None or n-array): efforts for all DOFs corresponding to this joint. Should be
                an n-array if specified, where n is the number of DOF for this joint. Defaults to None,
                which means left unchanged.
        """
        self.assert_articulated()
        if positions is not None:
            self._default_state.positions = np.array(positions)
        if velocities is not None:
            self._default_state.velocities = np.array(velocities)
        if efforts is not None:
            self._default_state.efforts = np.array(efforts)

    def update_default_state(self):
        self.set_default_state(*self.get_state())

    def set_control_type(self, control_type, kp=None, kd=None):
        """
        Sets the control type for this joint.

        Args:
            control_type (ControlType): What type of control to use for this joint.
                Valid options are: {ControlType.POSITION, ControlType.VELOCITY, ControlType.EFFORT}
            kp (None or float): If specified, sets the kp gain value for this joint. Should only be set if
                setting ControlType.POSITION
            kd (None or float): If specified, sets the kd gain value for this joint. Should only be set if
                setting ControlType.VELOCITY
        """
        # Sanity check inputs
        assert_valid_key(key=control_type, valid_keys=ControlType.VALID_TYPES, name="control type")
        if control_type == ControlType.POSITION:
            assert kp is not None, "kp gain must be specified for setting POSITION control!"
            assert kd is None, "kd gain must not be specified for setting POSITION control!"
            kd = 0.0

        elif control_type == ControlType.VELOCITY:
            assert kp is None, "kp gain must not be specified for setting VELOCITY control!"
            assert kd is not None, "kd gain must be specified for setting VELOCITY control!"
            kp = 0.0
        else:   # Efforts
            assert kp is None, "kp gain must not be specified for setting EFFORT control!"
            assert kd is None, "kd gain must not be specified for setting EFFORT control!"
            kp, kd = 0.0, 0.0

        # Set values
        if self._dc:
            for dof_handle, dof_property in zip(self._dof_handles, self._dof_properties):
                dof_property.stiffness = kp
                dof_property.damping = kd
                self._dc.set_dof_properties(dof_handle, dof_property)

        # Update control type
        self._control_type = control_type

    @property
    def body0(self):
        """
        Gets this joint's body0 relationship.

        Returns:
            None or str: Absolute prim path to the body prim to set as this joint's parent link, or None if there is
                no body0 specified.
        """
        targets = self._prim.GetRelationship("physics:body0").GetTargets()
        return targets[0].__str__() if len(targets) > 0 else None

    @body0.setter
    def body0(self, body0):
        """
        Sets this joint's body0 relationship.

        Args:
            body0 (str): Absolute prim path to the body prim to set as this joint's parent link.
        """
        # Make sure prim path is valid
        assert is_prim_path_valid(body0), f"Invalid body0 path specified: {body0}"
        self._prim.GetRelationship("physics:body0").SetTargets([Sdf.Path(body0)])

    @property
    def body1(self):
        """
        Gets this joint's body1 relationship.

        Returns:
            None or str: Absolute prim path to the body prim to set as this joint's child link, or None if there is
                no body1 specified.
        """
        targets = self._prim.GetRelationship("physics:body1").GetTargets()
        return targets[0].__str__()

    @body1.setter
    def body1(self, body1):
        """
        Sets this joint's body1 relationship.

        Args:
            body1 (str): Absolute prim path to the body prim to set as this joint's child link.
        """
        # Make sure prim path is valid
        assert is_prim_path_valid(body1), f"Invalid body1 path specified: {body1}"
        self._prim.GetRelationship("physics:body1").SetTargets([Sdf.Path(body1)])

    @property
    def parent_name(self):
        """
        Gets this joint's parent body name, if it exists

        Returns:
            str: Joint's parent body name
        """
        return self._dc.get_rigid_body_name(self._dc.get_joint_parent_body(self._handle))

    @property
    def child_name(self):
        """
        Gets this joint's child body name, if it exists

        Returns:
            str: Joint's child body name
        """
        return self._dc.get_rigid_body_name(self._dc.get_joint_child_body(self._handle))

    @property
    def local_orientation(self):
        """
        Returns:
            4-array: (x,y,z,w) local quaternion orientation of this joint, relative to the parent link
        """
        # Grab local rotation to parent and child links
        quat0 = gf_quat_to_np_array(self.get_attribute("physics:localRot0"))[[1, 2, 3, 0]]
        quat1 = gf_quat_to_np_array(self.get_attribute("physics:localRot1"))[[1, 2, 3, 0]]

        # Invert the child link relationship, and multiply the two rotations together to get the final rotation
        return quat_multiply(quaternion1=quat_inverse(quat1), quaternion0=quat0)

    @property
    def joint_name(self):
        """
        Returns:
            str: Name of this joint
        """
        return self._joint_name

    @property
    def joint_type(self):
        """
        Gets this joint's type (ignoring the "Physics" prefix)

        Returns:
            str: Joint's type. Should be one of:
                {"FixedJoint", "Joint", "PrismaticJoint", "RevoluteJoint", "SphericalJoint"}
                    (equivalently, one of JointType)
        """
        return self._joint_type

    @property
    def control_type(self):
        """
        Gets the control types for this joint

        Returns:
            ControlType: control type for this joint
        """
        return self._control_type

    @property
    def dof_properties(self):
        """
        Returns:
            list of DOFProperties: Per-DOF properties for this joint.
                See https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.dynamic_control/docs/index.html#omni.isaac.dynamic_control._dynamic_control.DofProperties
                for more information.
        """
        return self._dof_properties

    @property
    def max_velocity(self):
        """
        Gets this joint's maximum velocity

        Returns:
            float: maximum velocity for this joint
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        # We either return the raw value or a default value if there is no max specified
        raw_vel = self._dof_properties[0].max_velocity
        default_max_vel = m.DEFAULT_MAX_REVOLUTE_VEL if self.joint_type == "RevoluteJoint" else m.DEFAULT_MAX_PRISMATIC_VEL
        return default_max_vel if raw_vel in {None, np.inf} else raw_vel

    @max_velocity.setter
    def max_velocity(self, vel):
        """
        Sets this joint's maximum velocity

        Args:
            vel (float): Velocity to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._dof_properties[0].max_velocity = vel
        self._dc.set_dof_properties(self._dof_handles[0], self._dof_properties[0])

    @property
    def max_force(self):
        """
        Gets this joint's maximum force

        Returns:
            float: maximum force for this joint
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        # We either return the raw value or a default value if there is no max specified
        raw_force = self._dof_properties[0].max_effort
        return m.DEFAULT_MAX_EFFORT if raw_force in {None, np.inf} else raw_force

    @max_force.setter
    def max_force(self, force):
        """
        Sets this joint's maximum force

        Args:
            force (float): Force to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._dof_properties[0].max_effort = force
        self._dc.set_dof_properties(self._dof_handles[0], self._dof_properties[0])

    @property
    def stiffness(self):
        """
        Gets this joint's stiffness

        Returns:
            float: stiffness for this joint
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        return self._dof_properties[0].stiffness

    @stiffness.setter
    def stiffness(self, stiffness):
        """
        Sets this joint's stiffness

        Args:
            stiffness (float): stiffness to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._dof_properties[0].stiffness = stiffness
        self._dc.set_dof_properties(self._dof_handles[0], self._dof_properties[0])

    @property
    def damping(self):
        """
        Gets this joint's damping

        Returns:
            float: damping for this joint
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        return self._dof_properties[0].damping

    @damping.setter
    def damping(self, damping):
        """
        Sets this joint's damping

        Args:
            damping (float): damping to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._dof_properties[0].stiffness = damping
        self._dc.set_dof_properties(self._dof_handles[0], self._dof_properties[0])

    @property
    def friction(self):
        """
        Gets this joint's friction

        Returns:
            float: friction for this joint
        """
        return self.get_attribute("physxJoint:jointFriction")

    @friction.setter
    def friction(self, friction):
        """
        Sets this joint's friction

        Args:
            friction (float): friction to set
        """
        self.set_attribute("physxJoint:jointFriction", friction)

    @property
    def lower_limit(self):
        """
        Gets this joint's lower_limit

        Returns:
            float: lower_limit for this joint
        """
        # TODO: Add logic for non Prismatic / Revolute joints (D6, spherical)
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        # We either return the raw value or a default value if there is no max specified
        raw_pos = self._dof_properties[0].lower
        return -m.DEFAULT_MAX_POS if raw_pos in {None, -np.inf} else raw_pos

    @lower_limit.setter
    def lower_limit(self, lower_limit):
        """
        Sets this joint's lower_limit

        Args:
            lower_limit (float): lower_limit to set
        """
        # Can't use DC because it's read only property
        lower_limit = T.rad2deg(lower_limit) if self.is_revolute else lower_limit
        self.set_attribute("physics:lowerLimit", lower_limit)

    @property
    def upper_limit(self):
        """
        Gets this joint's upper_limit

        Returns:
            float: upper_limit for this joint
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        # We either return the raw value or a default value if there is no max specified
        raw_pos = self._dof_properties[0].upper
        return m.DEFAULT_MAX_POS if raw_pos in {None, np.inf} else raw_pos

    @upper_limit.setter
    def upper_limit(self, upper_limit):
        """
        Sets this joint's upper_limit

        Args:
            upper_limit (float): upper_limit to set
        """
        # Can't use DC because it's read only property
        upper_limit = T.rad2deg(upper_limit) if self.is_revolute else upper_limit
        self.set_attribute("physics:upperLimit", upper_limit)

    @property
    def has_limit(self):
        """
        Returns:
            bool: True if this joint has a limit, else False
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        # We either return the raw value or a default value if there is no max specified
        return self._dof_properties[0].has_limits

    @property
    def n_dof(self):
        """
        Returns:
            int: Number of degrees of freedom this joint has
        """
        return self._n_dof

    @property
    def articulated(self):
        """
        Returns:
             bool: Whether this joint is articulated or not
        """
        return self._art is not None

    @property
    def is_revolute(self):
        """
        Returns:
            bool: Whether this joint is revolute or  not
        """
        return self._joint_type == "RevoluteJoint"

    @property
    def is_single_dof(self):
        """
        Returns:
            bool: Whether this joint has a single DOF or not
        """
        return self._joint_type in {"RevoluteJoint", "PrismaticJoint"}

    def assert_articulated(self):
        """
        Sanity check to make sure this joint is articulated. Used as a gatekeeping function to prevent non-intended
        behavior (e.g.: trying to grab this joint's state if it's not articulated)
        """
        assert self.articulated, "Tried to call method not intended for non-articulated joint!"

    def get_state(self, normalized=False):
        """
        (pos, vel, effort) state of this joint

        Args:
            normalized (bool): If True, will return normalized state of this joint, where pos, vel, and effort values
                are in range [-1, 1].

        Returns:
            Tuple:
                - n-array: position of this joint, where n = number of DOF for this joint
                - n-array: velocity of this joint, where n = number of DOF for this joint
                - n-array: effort of this joint, where n = number of DOF for this joint
        """
        # Make sure we only call this if we're an articulated joint
        self.assert_articulated()

        # Grab raw states
        pos, vel, effort = np.zeros(self.n_dof), np.zeros(self.n_dof), np.zeros(self.n_dof)
        for i, dof_handle in enumerate(self._dof_handles):
            dof_state = self._dc.get_dof_state(dof_handle, _dynamic_control.STATE_ALL)
            pos[i] = dof_state.pos
            vel[i] = dof_state.vel
            effort[i] = dof_state.effort

        # Potentially normalize if requested
        if normalized:
            pos, vel, effort = self._normalize_pos(pos), self._normalize_vel(vel), self._normalize_effort(effort)

        return pos, vel, effort

    def _normalize_pos(self, pos):
        """
        Normalizes raw joint positions @pos

        Args:
            pos (n-array): n-DOF raw positions to normalize

        Returns:
            n-array: n-DOF normalized positions in range [-1, 1]
        """
        low, high = self.lower_limit, self.upper_limit
        mean = (low + high) / 2.0
        magnitude = (high - low) / 2.0
        pos = (pos - mean) / magnitude

        return pos

    def _denormalize_pos(self, pos):
        """
        De-normalizes joint positions @pos

        Args:
            pos (n-array): n-DOF normalized positions in range [-1, 1]

        Returns:
            n-array: n-DOF de-normalized positions
        """
        low, high = self.lower_limit, self.upper_limit
        mean = (low + high) / 2.0
        magnitude = (high - low) / 2.0
        pos = pos * magnitude + mean

        return pos

    def _normalize_vel(self, vel):
        """
        Normalizes raw joint velocities @vel

        Args:
            vel (n-array): n-DOF raw velocities to normalize

        Returns:
            n-array: n-DOF normalized velocities in range [-1, 1]
        """
        return vel / self.max_velocity

    def _denormalize_vel(self, vel):
        """
        De-normalizes joint velocities @vel

        Args:
            vel (n-array): n-DOF normalized velocities in range [-1, 1]

        Returns:
            n-array: n-DOF de-normalized velocities
        """
        return vel * self.max_velocity

    def _normalize_effort(self, effort):
        """
        Normalizes raw joint effort @effort

        Args:
            effort (n-array): n-DOF raw effort to normalize

        Returns:
            n-array: n-DOF normalized effort in range [-1, 1]
        """
        return effort / self.max_force

    def _denormalize_effort(self, effort):
        """
        De-normalizes joint effort @effort

        Args:
            effort (n-array): n-DOF normalized effort in range [-1, 1]

        Returns:
            n-array: n-DOF de-normalized effort
        """
        return effort * self.max_force

    def set_pos(self, pos, normalized=False, target=False):
        """
        Set the position of this joint in metric space

        Args:
            pos (float or n-array of float): Set the position(s) for this joint. Can be a single float or 1-array of
                float if the joint only has a single DOF, otherwise it should be an n-array of floats.
            normalized (bool): Whether the input is normalized to [-1, 1] (in this case, the values will be
                de-normalized first before being executed). Default is False
            target (bool): Whether the position being set is a target value or manual value to immediately set. Default
                is False, corresponding to an instantaneous setting of the position
        """
        # Sanity checks -- make sure we're the correct control type if we're setting a target and that we're articulated
        self.assert_articulated()
        if target:
            assert self._control_type == ControlType.POSITION, \
                "Trying to set joint position target, but control type is not position!"

        # Standardize input
        pos = np.array([pos]) if self._n_dof == 1 and not isinstance(pos, Iterable) else np.array(pos)

        # Potentially de-normalize if the input is normalized
        if normalized:
            pos = self._denormalize_pos(pos)

        # Set the DOF(s) in this joint
        for dof_handle, p in zip(self._dof_handles, pos):
            if not target:
                self._dc.set_dof_position(dof_handle, p)
            # We set the position in either case
            self._dc.set_dof_position_target(dof_handle, p)

    def set_vel(self, vel, normalized=False, target=False):
        """
        Set the velocity of this joint in metric space

        Args:
            vel (float or n-array of float): Set the velocity(s) for this joint. Can be a single float or 1-array of
                float if the joint only has a single DOF, otherwise it should be an n-array of floats.
            normalized (bool): Whether the input is normalized to [-1, 1] (in this case, the values will be
                de-normalized first before being executed). Default is False
            target (bool): Whether the velocity being set is a target value or manual value to immediately set. Default
                is False, corresponding to an instantaneous setting of the velocity
        """
        # Sanity checks -- make sure we're the correct control type if we're setting a target and that we're articulated
        self.assert_articulated()
        if target:
            assert self._control_type == ControlType.VELOCITY, \
                f"Trying to set joint velocity target for joint {self.name}, but control type is not velocity!"

        # Standardize input
        vel = np.array([vel]) if self._n_dof == 1 and not isinstance(vel, Iterable) else np.array(vel)

        # Potentially de-normalize if the input is normalized
        if normalized:
            vel = self._denormalize_vel(vel)

        # Set the DOF(s) in this joint
        for dof_handle, v in zip(self._dof_handles, vel):
            if not target:
                self._dc.set_dof_velocity(dof_handle, v)
            # We set the target in either case
            self._dc.set_dof_velocity_target(dof_handle, v)

    def set_effort(self, effort, normalized=False):
        """
        Set the effort of this joint in metric space

        Args:
            effort (float or n-array of float): Set the effort(s) for this joint. Can be a single float or 1-array of
                float if the joint only has a single DOF, otherwise it should be an n-array of floats.
            normalized (bool): Whether the input is normalized to [-1, 1] (in this case, the values will be
                de-normalized first before being executed). Default is False
        """
        # Sanity checks -- make sure that we're articulated (no control type check like position and velocity
        # because we can't set effort targets)
        self.assert_articulated()

        # Standardize input
        effort = np.array([effort]) if self._n_dof == 1 and not isinstance(effort, Iterable) else np.array(effort)

        # Potentially de-normalize if the input is normalized
        if normalized:
            effort = self._denormalize_effort(effort)

        # Set the DOF(s) in this joint
        for dof_handle, e in zip(self._dof_handles, effort):
            self._dc.set_dof_effort(dof_handle, e)

    def keep_still(self):
        """
        Zero out all velocities for this prim
        """
        self.set_vel(np.zeros(self.n_dof))

    def _dump_state(self):
        pos, vel, effort = self.get_state() if self.articulated else (np.array([]), np.array([]), np.array([]))
        return OrderedDict(
            pos=pos,
            vel=vel,
            effort=effort,
        )

    def _load_state(self, state):
        if self.articulated:
            self.set_pos(state["pos"], target=False)
            self.set_vel(state["vel"], target=False)
            self.set_effort(state["effort"])

    def _serialize(self, state):
        # We serialize by iterating over the keys and adding them to a list that's concatenated at the end
        # This is a deterministic mapping because we assume the state is an OrderedDict
        return np.concatenate(list(state.values()))

    def _deserialize(self, state):
        # We deserialize deterministically by knowing the order of values -- pos, vel, effort
        return OrderedDict(
            pos=state[0:self.n_dof],
            vel=state[self.n_dof:2*self.n_dof],
            effort=state[2*self.n_dof:3*self.n_dof],
        ), 3*self.n_dof

    def duplicate(self, simulator, prim_path):
        # Cannot directly duplicate a joint prim
        raise NotImplementedError("Cannot directly duplicate a joint prim!")


class VirtualJointPrim(JointPrim):
    """
    Virtual joint prim inherited from joint prim, provide callback functions to get and set the joint state

        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            joint_type (str): type of joint
            get_state_callback (Callable): callback function to get the joint states (pos, orn)
            control_type (ControlType): control type of the joint. Default is ControlType.POSITION
            set_pos_callback (Callable): callback function to set the joint's position
            set_vel_callback (Callable): callback function to set the joint's velocity
            set_effort_callback (Callable): callback function to set the joint's effort
            lower_limit (float): lower limit for the joint
            upper_limit (float): upper limit for the joint
    """
    def __init__(
        self,
        prim_path,
        name,
        joint_type,
        get_state_callback,
        set_pos_callback=None,
        set_vel_callback=None,
        set_effort_callback=None,
        lower_limit=None,
        upper_limit=None
    ):

        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config={},
            articulation=None,
        )

        self._joint_name = name
        self._joint_type = joint_type
        self._n_dof = 1  # currently only support 1 dof joint
        self.get_state_callback = get_state_callback
        self.set_pos_callback = set_pos_callback
        self.set_vel_callback = set_vel_callback
        self.set_effort_callback = set_effort_callback

        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._max_velocity = None
        self._max_force = None

    def _initialize(self):
        # no need to run super initialize
        # set control type to position as default
        self._control_type = ControlType.POSITION

    def get_state(self, normalized=False):
        return self.get_state_callback()

    def set_pos(self, pos, normalized=False, target=False):
        if normalized:
            pos = self._denormalize_pos(pos)
        if self.set_pos_callback is not None:
            assert self._control_type == ControlType.POSITION, \
                f"Trying to set joint position target, but control type is {self.control_type}!"
            self.set_pos_callback(pos)

    def set_vel(self, vel, normalized=False, target=False):
        if normalized:
            vel = self._denormalize_vel(vel)
        if self.set_vel_callback is not None:
            assert self._control_type == ControlType.VELOCITY, \
                "Trying to set joint velocity target, but control type is not velocity!"
            self.set_vel_callback(vel)

    def set_effort(self, effort, normalized=False):
        if normalized:
            effort = self._denormalize_effort(effort)
        if self.set_effort_callback is not None:
            assert self._control_type == ControlType.EFFORT, \
                f"Trying to set joint effort target, but control type is not effort!"
            self.set_effort_callback(effort)

    def duplicate(self, simulator, prim_path):
        raise NotImplementedError("Cannot directly duplicate a joint prim!")

    def keep_still(self):
        # virtual joints will by default stay still
        pass

    def update_handles(self):
        # virtual joints does not possess dc handle
        self._handle = _dynamic_control.INVALID_HANDLE

    @property
    def dof_properties(self):
        raise NotImplementedError("Virtual joint does not support dof properties!")

    @property
    def max_velocity(self):
        default_max_vel = m.DEFAULT_MAX_REVOLUTE_VEL if self.joint_type == "RevoluteJoint" else m.DEFAULT_MAX_PRISMATIC_VEL
        return default_max_vel if self._max_velocity in {None, np.inf} else self._max_velocity

    @max_velocity.setter
    def max_velocity(self, vel):
        self._max_velocity = vel

    @property
    def max_force(self):
        return m.DEFAULT_MAX_EFFORT if self._max_force in {None, np.inf} else self._max_force

    @max_force.setter
    def max_force(self, force):
        self._max_force = force

    @property
    def stiffness(self):
        print("Virtual joint does not support stiffness!")
        return np.nan

    @stiffness.setter
    def stiffness(self, stiffness):
        raise NotImplementedError("Virtual joint does not support stiffness!")

    @property
    def damping(self):
        raise NotImplementedError("Virtual joint does not support damping!")

    @damping.setter
    def damping(self, damping):
        raise NotImplementedError("Virtual joint does not support damping!")

    @property
    def friction(self):
        print("Virtual joint does not support friction!")
        return np.nan

    @friction.setter
    def friction(self, friction):
        raise NotImplementedError("Virtual joint does not support friction!")

    @property
    def lower_limit(self):
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        return -m.DEFAULT_MAX_POS if self._lower_limit in {None, -np.inf} else self._lower_limit

    @lower_limit.setter
    def lower_limit(self, lower_limit):
        self._lower_limit = T.rad2deg(lower_limit) if self.is_revolute else lower_limit

    @property
    def upper_limit(self):
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        return m.DEFAULT_MAX_POS if self._upper_limit in {None, np.inf} else self._upper_limit

    @upper_limit.setter
    def upper_limit(self, upper_limit):
        self.upper_limit = T.rad2deg(upper_limit) if self.is_revolute else upper_limit

    @property
    def has_limit(self):
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        return self.upper_limit not in {None, np.inf}


class Virtual6DOFJoint:
    """
    A Virtual 6DOF Joint interface that wraps up 6 1DOF VirtualJointPrim and provide set and get state callbacks

        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            joint_name (str): Name for the object. Names need to be unique per scene.
            dof (List[str]): sublist of COMPONENT_SUFFIXES, list of dof for this joint
            get_state_callback (Callable): callback function to get the joint states (pos, orn)
            command_pos_callback (Callable): callback function to set the joint's position
            reset_pos_callback (Callable): callback function to reset the joint's position
            command_vel_callback (Callable): callback function to set the joint's velocity
            command_effort_callback (Callable): callback function to set the joint's effort
            lower_limits (List[float]): lower limits for each dof of the joint
            upper_limits (List[float]): upper limits for each dof of the joint
    """
    def __init__(
        self,
        prim_path,
        joint_name,
        dof=tuple(m.COMPONENT_SUFFIXES),
        get_state_callback=None,
        command_pos_callback=None,
        reset_pos_callback=None,
        command_vel_callback=None,
        command_effort_callback=None,
        lower_limits=None,
        upper_limits=None,
    ):
        self.joint_name = joint_name
        self._get_state_callback = get_state_callback
        self._command_pos_callback = command_pos_callback
        self._reset_pos_callback = reset_pos_callback
        self._command_vel_callback = command_vel_callback
        self._command_effort_callback = command_effort_callback

        self.dof = [m.COMPONENT_SUFFIXES.index(d) for d in dof]
        # Initialize joints dictionary
        self._joints = OrderedDict()
        for i, name in enumerate(m.COMPONENT_SUFFIXES):
            self._joints[f"{self.joint_name}_{name}"] = VirtualJointPrim(
                prim_path=prim_path,
                name=f"{self.joint_name}_{name}",
                joint_type="PrismaticJoint" if i < 3 else "RevoluteJoint",
                get_state_callback=lambda: self.get_state()[i],
                set_pos_callback=lambda pos, i_dof=i: self.set_pos(i_dof, pos),
                set_vel_callback=lambda vel, i_dof=i: self.set_vel(i_dof, vel),
                set_effort_callback=lambda effort, i_dof=i: self.set_effort(i_dof, effort),
                lower_limit=lower_limits[i] if lower_limits is not None else None,
                upper_limit=upper_limits[i] if upper_limits is not None else None,
            )

        self._get_actuated_indices = lambda lst: [lst[idx] for idx in self.dof]

        self._reset_stored_control()
        self._reset_stored_reset()

    def get_state(self):
        """ get the current state (pos/orn) of the joint"""
        if self._get_state_callback is None:
            raise NotImplementedError("get state callback is not implemented for this virtual joint!")
        return self._get_state_callback()

    @property
    def joints(self):
        """Gets the 1DOF VirtualJoints belonging to this 6DOF joint."""
        return self._joints

    def set_pos(self, dof, val):
        """Calls the command callback with position values for the DOF."""
        self._stored_control[dof] = val
        if all(ctrl is not None for ctrl in self._get_actuated_indices(self._stored_control)):
            if self._command_pos_callback:
                self._command_pos_callback(self._stored_control)
                self._reset_stored_control()
            else:
                raise NotImplementedError(f"set position callback for {self.joint_name} is not implemented!")

    def set_vel(self, dof, val):
        """Calls the command callback with velocity values for the DOF."""
        self._stored_control[dof] = val
        if all(ctrl is not None for ctrl in self._get_actuated_indices(self._stored_control)):
            if self._command_vel_callback:
                self._command_vel_callback(self._stored_control)
                self._reset_stored_control()
            else:
                raise NotImplementedError(f"set velocity callback for {self.joint_name} is not implemented!")

    def set_effort(self, dof, effort):
        """Calls the command callback with velocity values for the dof"""
        self._stored_control[dof] = effort
        if all(ctrl is not None for ctrl in self._get_actuated_indices(self._stored_control)):
            if self._command_effort_callback:
                self._command_effort_callback(self._stored_control)
                self._reset_stored_control()
            else:
                raise NotImplementedError(f"set effort callback for {self.joint_name} is not implemented!")

    def reset_pos(self, dof, val):
        """Calls the reset callback with position values for the DOF"""
        self._stored_reset[dof] = val
        if all(reset_val is not None for reset_val in self._get_actuated_indices(self._stored_reset)):
            self._reset_pos_callback(self._stored_reset)
            self._reset_stored_reset()

    def _reset_stored_control(self):
        self._stored_control = [None] * len(self._joints)

    def _reset_stored_reset(self):
        self._stored_reset = [None] * len(self._joints)
