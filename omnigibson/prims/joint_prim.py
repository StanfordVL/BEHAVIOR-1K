from collections.abc import Iterable

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.controllers.controller_base import ControlType
from omnigibson.macros import create_module_macros
from omnigibson.prims.prim_base import BasePrim
from omnigibson.utils.constants import JointAxis, JointType
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.usd_utils import PoseAPI, create_joint

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_MAX_POS = 1000.0
m.DEFAULT_MAX_PRISMATIC_VEL = 1.0
m.DEFAULT_MAX_REVOLUTE_VEL = 15.0
m.DEFAULT_MAX_EFFORT = 100.0
m.INF_POS_THRESHOLD = 1e5
m.INF_VEL_THRESHOLD = 1e5
m.INF_EFFORT_THRESHOLD = 1e10
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
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
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
        relative_prim_path,
        name,
        load_config=None,
        articulation_view=None,
    ):
        # Grab dynamic control reference and set properties
        self._articulation_view_direct = articulation_view

        # Other values that will be filled in at runtime
        self._joint_type = None
        self._control_type = None
        self._driven = None

        # The following values will only be valid if this joint is part of an articulation
        self._n_dof = None  # The number of degrees of freedom this joint provides
        self._joint_idx = None  # The index of this joint in the parent articulation's joint array
        self._joint_dof_offset = (
            None  # The starting index of the DOFs for this joint in the parent articulation's DOF array
        )
        self._joint_name = None  # The name of this joint in the parent's articulation tree

        # Run super method
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _load(self):
        # Make sure this joint isn't articulated
        assert not self.articulated, (
            "Joint cannot be created, since this is an articulated joint! We are assuming"
            "the joint already exists in the stage."
        )

        # Define a joint prim at the current stage
        prim = create_joint(
            prim_path=self.prim_path,
            joint_type=self._load_config.get("joint_type", JointType.JOINT),
        )

        return prim

    def _post_load(self):
        # run super first
        super()._post_load()

        # Check whether this joint is driven or not
        self._driven = self._prim.HasAPI(lazy.pxr.UsdPhysics.DriveAPI)

        # Add joint state API if this is a revolute or prismatic joint
        self._joint_type = JointType.get_type(self._prim.GetTypeName().split("Physics")[-1])
        if self.is_single_dof:
            # We MUST already have the joint state API defined beforehand in the USD
            # This is because physx complains if we try to add physx APIs AFTER a simulation step occurs, which
            # happens because joint prims are usually created externally during an EntityPrim's initialization phase
            assert self._prim.HasAPI(
                lazy.pxr.PhysxSchema.JointStateAPI
            ), "Revolute or Prismatic joints must already have JointStateAPI added!"

        # Possibly set the bodies
        if "body0" in self._load_config and self._load_config["body0"] is not None:
            self.body0 = self._load_config["body0"]
        if "body1" in self._load_config and self._load_config["body1"] is not None:
            self.body1 = self._load_config["body1"]

    def _initialize(self):
        # Always run super first
        super()._initialize()

        # Update the joint indices etc.
        self.update_handles()

        # Get control type
        if self.articulated:
            control_types = []
            stiffnesses, dampings = self._articulation_view.get_gains(joint_indices=self.dof_indices)
            for i, (kp, kd) in enumerate(zip(stiffnesses[0], dampings[0])):
                # Infer control type based on whether kp and kd are 0 or not, as well as whether this joint is driven or not
                # TODO: Maybe assert mutual exclusiveness here?
                if not self._driven:
                    control_type = ControlType.NONE
                elif kp == 0.0:
                    control_type = ControlType.EFFORT if kd == 0.0 else ControlType.VELOCITY
                else:
                    control_type = ControlType.POSITION
                control_types.append(control_type)

            # Make sure all the control types are the same -- if not, we had something go wrong!
            assert len(set(control_types)) == 1, f"Got multiple control types for this single joint: {control_types}"
            self._control_type = control_types[0]

    def update_handles(self):
        """
        Updates all internal handles for this prim, in case they change since initialization
        """
        # It's a bit tricky to get the joint index here. We need to find the first dof at this prim path
        # first, then get the corresponding joint index from that dof offset.
        self._joint_dof_offset = list(self._articulation_view._dof_paths[0]).index(self.prim_path)
        joint_dof_offsets = self._articulation_view._metadata.joint_dof_offsets
        # Note that we are finding the last occurrence of the dof offset, since that corresponds to the joint index
        # The first occurrence can be a fixed link that is 0-dof, meaning the offset will be repeated.
        self._joint_idx = next(
            i for i in reversed(range(len(joint_dof_offsets))) if joint_dof_offsets[i] == self._joint_dof_offset
        )
        self._joint_name = self._articulation_view._metadata.joint_names[self._joint_idx]
        self._n_dof = self._articulation_view._metadata.joint_dof_counts[self._joint_idx]

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
            if kd is None:
                # kd could have bene optionally set, if not, then set 0 as default
                kd = 0.0
        elif control_type == ControlType.VELOCITY:
            assert kp is None, "kp gain must not be specified for setting VELOCITY control!"
            assert kd is not None, "kd gain must be specified for setting VELOCITY control!"
            kp = 0.0
        else:  # Efforts
            assert kp is None, "kp gain must not be specified for setting EFFORT control!"
            assert kd is None, "kd gain must not be specified for setting EFFORT control!"
            kp, kd = 0.0, 0.0

        # Set values
        kps = th.full((1, self._n_dof), kp)
        kds = th.full((1, self._n_dof), kd)
        self._articulation_view.set_gains(kps=kps, kds=kds, joint_indices=self.dof_indices)

        # Update control type
        self._control_type = control_type

    @property
    def _articulation_view(self):
        if self._articulation_view_direct is None:
            return None

        # Validate that the articulation view is initialized and that if physics is running, the
        # view is valid.
        if og.sim.is_playing() and self.initialized:
            assert (
                self._articulation_view_direct.is_physics_handle_valid()
                and self._articulation_view_direct._physics_view.check()
            ), "Articulation view must be valid if physics is running!"

        return self._articulation_view_direct

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
        assert lazy.omni.isaac.core.utils.prims.is_prim_path_valid(body0), f"Invalid body0 path specified: {body0}"
        self._prim.GetRelationship("physics:body0").SetTargets([lazy.pxr.Sdf.Path(body0)])

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
        assert lazy.omni.isaac.core.utils.prims.is_prim_path_valid(body1), f"Invalid body1 path specified: {body1}"
        self._prim.GetRelationship("physics:body1").SetTargets([lazy.pxr.Sdf.Path(body1)])

    @property
    def local_orientation(self):
        """
        Returns:
            4-array: (x,y,z,w) local quaternion orientation of this joint, relative to the parent link
        """
        # Grab local rotation to parent and child links
        quat0 = lazy.omni.isaac.core.utils.rotations.gf_quat_to_np_array(self.get_attribute("physics:localRot0"))[
            [1, 2, 3, 0]
        ]
        quat1 = lazy.omni.isaac.core.utils.rotations.gf_quat_to_np_array(self.get_attribute("physics:localRot1"))[
            [1, 2, 3, 0]
        ]

        # Invert the child link relationship, and multiply the two rotations together to get the final rotation
        return T.quat_multiply(quaternion1=T.quat_inverse(quat1), quaternion0=quat0)

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
            JointType: Joint's type. Should be one corresponding to:
                {JOINT_PRISMATIC, JOINT_REVOLUTE, JOINT_FIXED, JOINT_SPHERICAL}
        """
        return self._joint_type

    @property
    def driven(self):
        """
        Returns:
            bool: Whether this joint can be driven by a motor or not
        """
        return self._driven

    @property
    def control_type(self):
        """
        Gets the control types for this joint

        Returns:
            ControlType: control type for this joint
        """
        return self._control_type

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
        raw_vel = self._articulation_view.get_max_velocities(joint_indices=self.dof_indices)[0][0]
        default_max_vel = (
            m.DEFAULT_MAX_REVOLUTE_VEL if self.joint_type == JointType.JOINT_REVOLUTE else m.DEFAULT_MAX_PRISMATIC_VEL
        )
        return default_max_vel if raw_vel is None or th.abs(raw_vel) > m.INF_VEL_THRESHOLD else raw_vel

    @max_velocity.setter
    def max_velocity(self, vel):
        """
        Sets this joint's maximum velocity

        Args:
            vel (float): Velocity to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._articulation_view.set_max_velocities(th.tensor([[vel]]), joint_indices=self.dof_indices)

    @property
    def max_effort(self):
        """
        Gets this joint's maximum effort

        Returns:
            float: maximum effort for this joint
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        # We either return the raw value or a default value if there is no max specified
        raw_effort = self._articulation_view.get_max_efforts(joint_indices=self.dof_indices)[0][0]
        return m.DEFAULT_MAX_EFFORT if raw_effort is None or th.abs(raw_effort) > m.INF_EFFORT_THRESHOLD else raw_effort

    @max_effort.setter
    def max_effort(self, effort):
        """
        Sets this joint's maximum effort

        Args:
            effort (float): effort to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._articulation_view.set_max_efforts(th.tensor([[effort]], dtype=th.float32), joint_indices=self.dof_indices)

    @property
    def stiffness(self):
        """
        Gets this joint's stiffness

        Returns:
            float: stiffness for this joint
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        stiffnesses = self._articulation_view.get_gains(joint_indices=self.dof_indices)[0]
        return stiffnesses[0][0]

    @stiffness.setter
    def stiffness(self, stiffness):
        """
        Sets this joint's stiffness

        Args:
            stiffness (float): stiffness to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._articulation_view.set_gains(kps=th.tensor([[stiffness]]), joint_indices=self.dof_indices)

    @property
    def damping(self):
        """
        Gets this joint's damping

        Returns:
            float: damping for this joint
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        dampings = self._articulation_view.get_gains(joint_indices=self.dof_indices)[1]
        return dampings[0][0]

    @damping.setter
    def damping(self, damping):
        """
        Sets this joint's damping

        Args:
            damping (float): damping to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._articulation_view.set_gains(kds=th.tensor([[damping]]), joint_indices=self.dof_indices)

    @property
    def friction(self):
        """
        Gets this joint's friction

        Returns:
            float: friction for this joint
        """
        return (
            self._articulation_view.get_friction_coefficients(joint_indices=self.dof_indices)[0][0]
            if og.sim.is_playing()
            else self.get_attribute("physxJoint:jointFriction")
        )

    @friction.setter
    def friction(self, friction):
        """
        Sets this joint's friction

        Args:
            friction (float): friction to set
        """
        self.set_attribute("physxJoint:jointFriction", friction)
        if og.sim.is_playing():
            self._articulation_view.set_friction_coefficients(th.tensor([[friction]]), joint_indices=self.dof_indices)

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
        raw_pos_lower, raw_pos_upper = self._articulation_view.get_joint_limits(
            joint_indices=self.dof_indices
        ).flatten()
        return (
            -m.DEFAULT_MAX_POS
            if raw_pos_lower is None or raw_pos_lower == raw_pos_upper or th.abs(raw_pos_lower) > m.INF_POS_THRESHOLD
            else raw_pos_lower
        )

    @lower_limit.setter
    def lower_limit(self, lower_limit):
        """
        Sets this joint's lower_limit

        Args:
            lower_limit (float): lower_limit to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._articulation_view.set_joint_limits(
            th.tensor([[lower_limit, self.upper_limit]]), joint_indices=self.dof_indices
        )

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
        raw_pos_lower, raw_pos_upper = self._articulation_view.get_joint_limits(
            joint_indices=self.dof_indices
        ).flatten()
        return (
            m.DEFAULT_MAX_POS
            if raw_pos_upper is None or raw_pos_lower == raw_pos_upper or th.abs(raw_pos_upper) > m.INF_POS_THRESHOLD
            else raw_pos_upper
        )

    @upper_limit.setter
    def upper_limit(self, upper_limit):
        """
        Sets this joint's upper_limit

        Args:
            upper_limit (float): upper_limit to set
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        self._articulation_view.set_joint_limits(
            th.tensor([[self.lower_limit, upper_limit]]), joint_indices=self.dof_indices
        )

    @property
    def has_limit(self):
        """
        Returns:
            bool: True if this joint has a limit, else False
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        return th.all(
            th.abs(self._articulation_view.get_joint_limits(joint_indices=self.dof_indices)) < m.INF_POS_THRESHOLD
        )

    @property
    def axis(self):
        """
        Gets this joint's axis

        Returns:
            str: axis for this joint, one of "X", "Y, "Z"
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        return self.get_attribute("physics:axis")

    @axis.setter
    def axis(self, axis):
        """
        Sets this joint's axis

        Args:
            str: axis for this joint, one of "X", "Y, "Z"
        """
        # Only support revolute and prismatic joints for now
        assert self.is_single_dof, "Joint properties only supported for a single DOF currently!"
        assert axis in JointAxis, f"Invalid joint axis specified: {axis}!"
        self.set_attribute("physics:axis", axis)

    @property
    def n_dof(self):
        """
        Returns:
            int: Number of degrees of freedom this joint has
        """
        return self._n_dof

    @property
    def dof_indices(self):
        """
        Returns:
            list of int: Indices of this joint's DOFs in the parent articulation's DOF array
        """
        assert self.articulated, "Can only get DOF indices for articulated joints!"
        return list(range(self._joint_dof_offset, self._joint_dof_offset + self._n_dof))

    @property
    def articulated(self):
        """
        Returns:
             bool: Whether this joint is articulated or not
        """
        return self._articulation_view is not None

    @property
    def is_revolute(self):
        """
        Returns:
            bool: Whether this joint is revolute or  not
        """
        return self._joint_type == JointType.JOINT_REVOLUTE

    @property
    def is_single_dof(self):
        """
        Returns:
            bool: Whether this joint has a single DOF or not
        """
        return self._joint_type in {JointType.JOINT_REVOLUTE, JointType.JOINT_PRISMATIC}

    def get_state(self, normalized=False):
        """
        (pos, vel, effort) state of this joint

        Args:
            normalized (bool): If True, will return normalized state of this joint, where pos, vel, and effort values
                are in range [-1, 1].

        Returns:
            3-tuple:
                - n-array: position of this joint, where n = number of DOF for this joint
                - n-array: velocity of this joint, where n = number of DOF for this joint
                - n-array: effort of this joint, where n = number of DOF for this joint
        """
        # Make sure we only call this if we're an articulated joint
        assert self.articulated, "Can only get state for articulated joints!"

        # Grab raw states
        pos = self._articulation_view.get_joint_positions(joint_indices=self.dof_indices)[0]
        vel = self._articulation_view.get_joint_velocities(joint_indices=self.dof_indices)[0]
        effort = self._articulation_view.get_measured_joint_efforts(joint_indices=self.dof_indices)[0]

        # Potentially normalize if requested
        if normalized:
            pos, vel, effort = self._normalize_pos(pos), self._normalize_vel(vel), self._normalize_effort(effort)

        return pos, vel, effort

    def get_target(self, normalized=False):
        """
        (pos, vel) target of this joint

        Args:
            normalized (bool): If True, will return normalized target of this joint

        Returns:
            2-tuple:
                - n-array: target position of this joint, where n = number of DOF for this joint
                - n-array: target velocity of this joint, where n = number of DOF for this joint
        """
        # Make sure we only call this if we're an articulated joint
        assert self.articulated, "Can only get targets for articulated joints!"

        # Grab raw states
        targets = self._articulation_view.get_applied_actions()
        pos = targets.joint_positions[0][self.dof_indices]
        vel = targets.joint_velocities[0][self.dof_indices]

        # Potentially normalize if requested
        if normalized:
            pos, vel = self._normalize_pos(pos), self._normalize_vel(vel)

        return pos, vel

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
        return effort / self.max_effort

    def _denormalize_effort(self, effort):
        """
        De-normalizes joint effort @effort

        Args:
            effort (n-array): n-DOF normalized effort in range [-1, 1]

        Returns:
            n-array: n-DOF de-normalized effort
        """
        return effort * self.max_effort

    def set_pos(self, pos, normalized=False, drive=False):
        """
        Set the position of this joint in metric space

        Args:
            pos (float or n-array of float): Set the position(s) for this joint. Can be a single float or 1-array of
                float if the joint only has a single DOF, otherwise it should be an n-array of floats.
            normalized (bool): Whether the input is normalized to [-1, 1] (in this case, the values will be
                de-normalized first before being executed). Default is False
            drive (bool): Whether the joint should be driven naturally via its motor to the position being set or
                whether it should be instantaneously set. Default is False, corresponding to an
                instantaneous setting of the position
        """
        # Sanity checks -- make sure we're the correct control type if we're setting a target and that we're articulated
        assert self.articulated, "Can only set position for articulated joints!"
        if drive:
            assert self._driven, "Can only use set_pos with drive=True if this joint is driven!"
            assert (
                self._control_type == ControlType.POSITION
            ), "Trying to set joint position target, but control type is not position!"

        # Standardize input
        pos = (
            pos
            if isinstance(pos, th.Tensor)
            else (
                th.tensor([pos], dtype=th.float32)
                if self._n_dof == 1 and not isinstance(pos, Iterable)
                else th.tensor(pos, dtype=th.float32)
            )
        )

        # Potentially de-normalize if the input is normalized
        if normalized:
            pos = self._denormalize_pos(pos)

        # Set the DOF(s) in this joint
        if not drive:
            self._articulation_view.set_joint_positions(positions=pos, joint_indices=self.dof_indices)
            PoseAPI.invalidate()

        # Also set the target
        self._articulation_view.set_joint_position_targets(positions=pos, joint_indices=self.dof_indices)

    def set_vel(self, vel, normalized=False, drive=False):
        """
        Set the velocity of this joint in metric space

        Args:
            vel (float or n-array of float): Set the velocity(s) for this joint. Can be a single float or 1-array of
                float if the joint only has a single DOF, otherwise it should be an n-array of floats.
            normalized (bool): Whether the input is normalized to [-1, 1] (in this case, the values will be
                de-normalized first before being executed). Default is False
            drive (bool): Whether the joint should be driven naturally via its motor to the velocity being set or
                whether it should be instantaneously set. Default is False, corresponding to an
                instantaneous setting of the velocity
        """
        # Sanity checks -- make sure we're the correct control type if we're setting a target and that we're articulated
        assert self.articulated, "Can only set velocity for articulated joints!"
        if drive:
            assert self._driven, "Can only use set_vel with drive=True if this joint is driven!"
            assert (
                self._control_type == ControlType.VELOCITY
            ), f"Trying to set joint velocity target for joint {self.name}, but control type is not velocity!"

        # Standardize input
        vel = (
            vel
            if isinstance(vel, th.Tensor)
            else (
                th.tensor([vel], dtype=th.float32)
                if self._n_dof == 1 and not isinstance(vel, Iterable)
                else th.tensor(vel, dtype=th.float32)
            )
        )

        # Potentially de-normalize if the input is normalized
        if normalized:
            vel = self._denormalize_vel(vel)

        # Set the DOF(s) in this joint
        if not drive:
            self._articulation_view.set_joint_velocities(velocities=vel, joint_indices=self.dof_indices)

        # Also set the target
        self._articulation_view.set_joint_velocity_targets(velocities=vel, joint_indices=self.dof_indices)

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
        # because we can't set effort targets) and that we're driven
        assert self.articulated, "Can only set effort for articulated joints!"

        # Standardize input
        effort = (
            effort
            if isinstance(effort, th.Tensor)
            else (
                th.tensor([effort], dtype=th.float32)
                if self._n_dof == 1 and not isinstance(effort, Iterable)
                else th.tensor(effort, dtype=th.float32)
            )
        )

        # Potentially de-normalize if the input is normalized
        if normalized:
            effort = self._denormalize_effort(effort)

        # Set the DOF(s) in this joint
        self._articulation_view.set_joint_efforts(efforts=effort, joint_indices=self.dof_indices)

    def keep_still(self):
        """
        Zero out all velocities for this prim
        """
        self.set_vel(th.zeros(self.n_dof))
        # If not driven, set torque equal to zero as well
        if not self.driven:
            self.set_effort(th.zeros(self.n_dof))

    def _dump_state(self):
        pos, vel, effort = self.get_state() if self.articulated else (th.empty(0), th.empty(0), th.empty(0))
        target_pos, target_vel = self.get_target() if self.articulated else (th.empty(0), th.empty(0))
        return dict(
            pos=pos,
            vel=vel,
            effort=effort,
            target_pos=target_pos,
            target_vel=target_vel,
        )

    def _load_state(self, state):
        if self.articulated:
            self.set_pos(state["pos"], drive=False)
            self.set_vel(state["vel"], drive=False)
            if self.driven:
                self.set_effort(state["effort"])
            if self._control_type == ControlType.POSITION:
                self.set_pos(state["target_pos"], drive=True)
            elif self._control_type == ControlType.VELOCITY:
                self.set_vel(state["target_vel"], drive=True)

    def serialize(self, state):
        return th.cat(
            [
                state["pos"],
                state["vel"],
                state["effort"],
                state["target_pos"],
                state["target_vel"],
            ]
        )

    def deserialize(self, state):
        # We deserialize deterministically by knowing the order of values -- pos, vel, effort
        return (
            dict(
                pos=state[0 : self.n_dof],
                vel=state[self.n_dof : 2 * self.n_dof],
                effort=state[2 * self.n_dof : 3 * self.n_dof],
                target_pos=state[3 * self.n_dof : 4 * self.n_dof],
                target_vel=state[4 * self.n_dof : 5 * self.n_dof],
            ),
            5 * self.n_dof,
        )
