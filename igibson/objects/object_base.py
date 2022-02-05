from abc import ABCMeta, abstractmethod
from collections import Iterable, OrderedDict

import numpy as np

from future.utils import with_metaclass

from igibson.utils.constants import (
    ALL_COLLISION_GROUPS_MASK,
    DEFAULT_COLLISION_GROUP,
    SPECIAL_COLLISION_GROUPS,
    SemanticClass,
)
from igibson.utils.semantics_utils import CLASS_NAME_TO_CLASS_ID
from igibson.utils.usd_utils import get_prim_nested_children, create_joint, CollisionAPI
from igibson.prims.articulated_prim import ArticulatedPrim


class BaseObject(ArticulatedPrim, metaclass=ABCMeta):
    """This is the interface that all iGibson objects must implement."""

    def __init__(
            self,
            prim_path,
            name=None,
            category="object",
            class_id=None,
            scale=1.0,
            rendering_params=None,
            visible=True,
            fixed_base=False,
            load_config=None,
    ):
        """
        Create an object instance with the minimum information of class ID and rendering parameters.

        @param prim_path: str, global path in the stage to this object
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        """
        # Generate a name if necessary. Note that the generation order & set of these names is not deterministic.
        if name is None:
            address = "%08X" % id(self)
            name = "{}_{}".format(category, address)

        # Store values
        self.category = category
        self.fixed_base = fixed_base

        # TODO
        # This sets the collision group of the object. In igibson, objects are only permitted to be part of a single
        # collision group, e.g. collisions are only enabled within a single group
        self.collision_group = SPECIAL_COLLISION_GROUPS.get(self.category, DEFAULT_COLLISION_GROUP)

        # category_based_rendering_params = {}
        # if category in ["walls", "floors", "ceilings"]:
        #     category_based_rendering_params["use_pbr"] = False
        #     category_based_rendering_params["use_pbr_mapping"] = False
        # if category == "ceilings":
        #     category_based_rendering_params["shadow_caster"] = False
        #
        # if rendering_params:  # Use the input rendering params as an override.
        #     category_based_rendering_params.update(rendering_params)

        if class_id is None:
            class_id = CLASS_NAME_TO_CLASS_ID.get(category, SemanticClass.USER_ADDED_OBJS)

        self.class_id = class_id
        self.renderer_instances = []
        # self._rendering_params = dict(self.DEFAULT_RENDERING_PARAMS)
        # self._rendering_params.update(category_based_rendering_params)

        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["scale"] = scale
        load_config["visible"] = visible

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _initialize(self):
        """
        Parse this object's articulation hierarchy to get properties including joint information and mass
        """
        # Run super method
        super()._initialize()

        # Set visibility
        if "visible" in self._load_config and self._load_config["visible"] is not None:
            self.visible = self._load_config["visible"]

        # Add fixed joint if we're fixing the base
        print(f"obj {self.name} is fixed base: {self.fixed_base}")
        if self.fixed_base:
            # Create fixed joint, and set Body0 to be this object's root prim
            create_joint(
                prim_path=f"{self._prim_path}/rootJoint",
                joint_type="FixedJoint",
                body1=self._dc.get_rigid_body_path(self._root_handle),
            )

        # TODO: Do we need to explicitly add all links? or is adding articulation root itself sufficient?
        # Set the collision group
        CollisionAPI.add_to_collision_group(
            col_group=self.name + "_col_group", #self.collision_group,
            prim_path=self.prim_path,
            create_if_not_exist=True,
        )

    @property
    def mass(self):
        """
        Returns:
             float: Cumulative mass of this potentially articulated object.
        """
        mass = 0.0
        for link in self._links.values():
            mass += link.mass

        return mass

    def get_velocities(self):
        """Get this object's root body velocity in the format of Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]"""
        return self.get_linear_velocity(), self.get_angular_velocity()

    def set_velocities(self, velocities):
        """Set this object's root body velocity in the format of Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]"""
        lin_vel, ang_vel = velocities

        print(f"obj: {self.name}, old root handle: {self._root_handle}, current root handle: {self._dc.get_rigid_body(f'/World/{self.name}/base_link')}, art handle: {self._dc.get_articulation(f'/World/{self.name}')}")
        self.set_linear_velocity(velocity=lin_vel)
        self.set_angular_velocity(velocity=ang_vel)

    def set_joint_states(self, joint_states):
        """Set object joint states in the format of Dict[String: (q, q_dot)]]"""
        # Make sure this object is articulated
        assert self._num_dof > 0, "Can only set joint states for objects that have > 0 DOF!"
        pos = np.zeros(self._num_dof)
        vel = np.zeros(self._num_dof)
        for i, joint_name in enumerate(self._dofs_infos.keys()):
            pos[i], vel[i] = joint_states[joint_name]

        # Set the joint positions and velocities
        self.set_joint_positions(positions=pos)
        self.set_joint_velocities(velocities=vel)

    def get_joint_states(self):
        """Get object joint states in the format of Dict[String: (q, q_dot)]]"""
        # Make sure this object is articulated
        assert self._num_dof > 0, "Can only get joint states for objects that have > 0 DOF!"
        pos = self.get_joint_positions()
        vel = self.get_joint_velocities()
        joint_states = dict()
        for i, joint_name in enumerate(self._dofs_infos.keys()):
            joint_states[joint_name] = (pos[i], vel[i])

        return joint_states

    def dump_state(self):
        """Dump the state of the object other than what's not included in pybullet state."""
        return None

    def load_state(self, dump):
        """Load the state of the object other than what's not included in pybullet state."""
        return

    # TODO
    def highlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(True)

    def unhighlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(False)

#
#
#
# class Link:
#     """
#     Body part (link) of object
#     """
#
#     def __init__(self, obj, link_prim):
#         """
#         :param obj: BaseObject, the object this link belongs to.
#         :param link_prim: Usd.Prim, prim object corresponding to this link
#         """
#         # Store args and initialize state
#         self.obj = obj
#         self.prim = link_prim
#         self._dc = obj.dc_interface
#         self.handle = self._dc.get_rigid_body(get_prim_path(self.prim))
#         self.initial_pos, self.initial_quat = self.get_position_orientation()
#
#     def get_name(self):
#         """
#         Get name of this link
#         """
#         return self.prim.GetName()
#
#     def get_position_orientation(self):
#         """
#         Get pose of this link
#         :return Tuple[Array[float], Array[float]]: pos (x,y,z) cartesian coordinates, quat (x,y,z,w)
#             orientation in quaternion form of this link
#         """
#         if self.link_id == -1:
#             pos, quat = p.getBasePositionAndOrientation(self.body_id)
#         else:
#             _, _, _, _, pos, quat = p.getLinkState(self.body_id, self.link_id)
#         return np.array(pos), np.array(quat)
#
#     def get_position(self):
#         """
#         :return Array[float]: (x,y,z) cartesian coordinates of this link
#         """
#         return self.get_position_orientation()[0]
#
#     def get_orientation(self):
#         """
#         :return Array[float]: (x,y,z,w) orientation in quaternion form of this link
#         """
#         return self.get_position_orientation()[1]
#
#     def get_local_position_orientation(self):
#         """
#         Get pose of this link in the robot's base frame.
#         :return Tuple[Array[float], Array[float]]: pos (x,y,z) cartesian coordinates, quat (x,y,z,w)
#             orientation in quaternion form of this link
#         """
#         base = self.robot.base_link
#         return p.multiplyTransforms(
#             *p.invertTransform(*base.get_position_orientation()), *self.get_position_orientation()
#         )
#
#     def get_rpy(self):
#         """
#         :return Array[float]: (r,p,y) orientation in euler form of this link
#         """
#         return np.array(p.getEulerFromQuaternion(self.get_orientation()))
#
#     def set_position(self, pos):
#         """
#         Sets the link's position
#         :param pos: Array[float], corresponding to (x,y,z) cartesian coordinates to set
#         """
#         old_quat = self.get_orientation()
#         self.set_position_orientation(pos, old_quat)
#
#     def set_orientation(self, quat):
#         """
#         Set the link's global orientation
#         :param quat: Array[float], corresponding to (x,y,z,w) quaternion orientation to set
#         """
#         old_pos = self.get_position()
#         self.set_position_orientation(old_pos, quat)
#
#     def set_position_orientation(self, pos, quat):
#         """
#         Set model's global position and orientation. Note: only supported if this is the base link (ID = -1!)
#         :param pos: Array[float], corresponding to (x,y,z) global cartesian coordinates to set
#         :param quat: Array[float], corresponding to (x,y,z,w) global quaternion orientation to set
#         """
#         assert self.link_id == -1, "Can only set pose for a base link (id = -1)! Got link id: {}.".format(self.link_id)
#         p.resetBasePositionAndOrientation(self.body_id, pos, quat)
#
#     def get_velocity(self):
#         """
#         Get velocity of this link
#         :return Tuple[Array[float], Array[float]]: linear (x,y,z) velocity, angular (ax,ay,az)
#             velocity of this link
#         """
#         if self.link_id == -1:
#             lin, ang = p.getBaseVelocity(self.body_id)
#         else:
#             _, _, _, _, _, _, lin, ang = p.getLinkState(self.body_id, self.link_id, computeLinkVelocity=1)
#         return np.array(lin), np.array(ang)
#
#     def get_linear_velocity(self):
#         """
#         Get linear velocity of this link
#         :return Array[float]: linear (x,y,z) velocity of this link
#         """
#         return self.get_velocity()[0]
#
#     def get_angular_velocity(self):
#         """
#         Get angular velocity of this link
#         :return Array[float]: angular (ax,ay,az) velocity of this link
#         """
#         return self.get_velocity()[1]
#
#     def contact_list(self):
#         """
#         Get contact points of the body part
#         :return Array[ContactPoints]: list of contact points seen by this link
#         """
#         return p.getContactPoints(self.body_id, -1, self.link_id, -1)
#
#     def force_wakeup(self):
#         """
#         Forces a wakeup for this robot. Defaults to no-op.
#         """
#         p.changeDynamics(self.body_id, self.link_id, activationState=p.ACTIVATION_STATE_WAKE_UP)
#
#
# class RobotJoint(with_metaclass(ABCMeta, object)):
#     """
#     Joint of a robot
#     """
#
#     @property
#     @abstractmethod
#     def joint_name(self):
#         pass
#
#     @property
#     @abstractmethod
#     def joint_type(self):
#         pass
#
#     @property
#     @abstractmethod
#     def lower_limit(self):
#         pass
#
#     @property
#     @abstractmethod
#     def upper_limit(self):
#         pass
#
#     @property
#     @abstractmethod
#     def max_velocity(self):
#         pass
#
#     @property
#     @abstractmethod
#     def max_torque(self):
#         pass
#
#     @property
#     @abstractmethod
#     def damping(self):
#         pass
#
#     @abstractmethod
#     def get_state(self):
#         """
#         Get the current state of the joint
#         :return Tuple[float, float, float]: (joint_pos, joint_vel, joint_tor) observed for this joint
#         """
#         pass
#
#     @abstractmethod
#     def get_relative_state(self):
#         """
#         Get the normalized current state of the joint
#         :return Tuple[float, float, float]: Normalized (joint_pos, joint_vel, joint_tor) observed for this joint
#         """
#         pass
#
#     @abstractmethod
#     def set_pos(self, pos):
#         """
#         Set position of joint (in metric space)
#         :param pos: float, desired position for this joint, in metric space
#         """
#         pass
#
#     @abstractmethod
#     def set_vel(self, vel):
#         """
#         Set velocity of joint (in metric space)
#         :param vel: float, desired velocity for this joint, in metric space
#         """
#         pass
#
#     @abstractmethod
#     def set_torque(self, torque):
#         """
#         Set torque of joint (in metric space)
#         :param torque: float, desired torque for this joint, in metric space
#         """
#         pass
#
#     @abstractmethod
#     def reset_state(self, pos, vel):
#         """
#         Reset pos and vel of joint in metric space
#         :param pos: float, desired position for this joint, in metric space
#         :param vel: float, desired velocity for this joint, in metric space
#         """
#         pass
#
#     @property
#     def has_limit(self):
#         """
#         :return bool: True if this joint has a limit, else False
#         """
#         return self.lower_limit < self.upper_limit
#
#
# class PhysicalJoint(RobotJoint):
#     """
#     A robot joint that exists in the physics simulation (e.g. in pybullet).
#     """
#
#     def __init__(self, joint_name, joint_id, body_id):
#         """
#         :param joint_name: str, name of the joint corresponding to @joint_id
#         :param joint_id: int, ID of this joint within the joint(s) found in the body corresponding to @body_id
#         :param body_id: Robot body ID containing this link
#         """
#         # Store args and initialize state
#         self._joint_name = joint_name
#         self.joint_id = joint_id
#         self.body_id = body_id
#
#         # read joint type and joint limit from the URDF file
#         # lower_limit, upper_limit, max_velocity, max_torque = <limit lower=... upper=... velocity=... effort=.../>
#         # "effort" is approximately torque (revolute) / force (prismatic), but not exactly (ref: http://wiki.ros.org/pr2_controller_manager/safety_limits).
#         # if <limit /> does not exist, the following will be the default value
#         # lower_limit, upper_limit, max_velocity, max_torque = 0.0, -1.0, 0.0, 0.0
#         info = get_joint_info(self.body_id, self.joint_id)
#         self._joint_type = info.jointType
#         self._lower_limit = info.jointLowerLimit
#         self._upper_limit = info.jointUpperLimit
#         self._max_torque = info.jointMaxForce
#         self._max_velocity = info.jointMaxVelocity
#         self._damping = info.jointDamping
#
#         # if joint torque and velocity limits cannot be found in the model file, set a default value for them
#         if self._max_torque == 0.0:
#             self._max_torque = 100.0
#         if self._max_velocity == 0.0:
#             # if max_velocity and joint limit are missing for a revolute joint,
#             # it's likely to be a wheel joint and a high max_velocity is usually supported.
#             self._max_velocity = 15.0 if self._joint_type == p.JOINT_REVOLUTE and not self.has_limit else 1.0
#
#     @property
#     def joint_name(self):
#         return self._joint_name
#
#     @property
#     def joint_type(self):
#         return self._joint_type
#
#     @property
#     def lower_limit(self):
#         return self._lower_limit
#
#     @property
#     def upper_limit(self):
#         return self._upper_limit
#
#     @property
#     def max_velocity(self):
#         return self._max_velocity
#
#     @property
#     def max_torque(self):
#         return self._max_torque
#
#     @property
#     def damping(self):
#         return self._damping
#
#     def __str__(self):
#         return "idx: {}, name: {}".format(self.joint_id, self.joint_name)
#
#     def get_state(self):
#         """
#         Get the current state of the joint
#         :return Tuple[float, float, float]: (joint_pos, joint_vel, joint_tor) observed for this joint
#         """
#         x, vx, _, trq = p.getJointState(self.body_id, self.joint_id)
#         return x, vx, trq
#
#     def get_relative_state(self):
#         """
#         Get the normalized current state of the joint
#         :return Tuple[float, float, float]: Normalized (joint_pos, joint_vel, joint_tor) observed for this joint
#         """
#         pos, vel, trq = self.get_state()
#
#         # normalize position to [-1, 1]
#         if self.has_limit:
#             mean = (self.lower_limit + self.upper_limit) / 2.0
#             magnitude = (self.upper_limit - self.lower_limit) / 2.0
#             pos = (pos - mean) / magnitude
#
#         # (trying to) normalize velocity to [-1, 1]
#         vel /= self.max_velocity
#
#         # (trying to) normalize torque / force to [-1, 1]
#         trq /= self.max_torque
#
#         return pos, vel, trq
#
#     def set_pos(self, pos):
#         """
#         Set position of joint (in metric space)
#         :param pos: float, desired position for this joint, in metric space
#         """
#         if self.has_limit:
#             pos = np.clip(pos, self.lower_limit, self.upper_limit)
#         p.setJointMotorControl2(self.body_id, self.joint_id, p.POSITION_CONTROL, targetPosition=pos)
#
#     def set_vel(self, vel):
#         """
#         Set velocity of joint (in metric space)
#         :param vel: float, desired velocity for this joint, in metric space
#         """
#         vel = np.clip(vel, -self.max_velocity, self.max_velocity)
#         p.setJointMotorControl2(self.body_id, self.joint_id, p.VELOCITY_CONTROL, targetVelocity=vel)
#
#     def set_torque(self, torque):
#         """
#         Set torque of joint (in metric space)
#         :param torque: float, desired torque for this joint, in metric space
#         """
#         torque = np.clip(torque, -self.max_torque, self.max_torque)
#         p.setJointMotorControl2(
#             bodyIndex=self.body_id,
#             jointIndex=self.joint_id,
#             controlMode=p.TORQUE_CONTROL,
#             force=torque,
#         )
#
#     def reset_state(self, pos, vel):
#         """
#         Reset pos and vel of joint in metric space
#         :param pos: float, desired position for this joint, in metric space
#         :param vel: float, desired velocity for this joint, in metric space
#         """
#         p.resetJointState(self.body_id, self.joint_id, targetValue=pos, targetVelocity=vel)
#         self.disable_motor()
#
#     def disable_motor(self):
#         """
#         Disable the motor of this joint
#         """
#         p.setJointMotorControl2(
#             self.body_id,
#             self.joint_id,
#             controlMode=p.POSITION_CONTROL,
#             targetPosition=0,
#             targetVelocity=0,
#             positionGain=0.1,
#             velocityGain=0.1,
#             force=0,
#         )
#
#
# class VirtualJoint(RobotJoint):
#     """A virtual joint connecting two bodies of the same robot that does not exist in the physics simulation.
#     Such a joint must be handled manually by the owning robot class by providing the appropriate callback functions
#     for getting and setting joint positions.
#     Such a joint can also be used as a way of controlling an arbitrary non-joint mechanism on the robot.
#     """
#
#     def __init__(self, joint_name, joint_type, get_pos_callback, set_pos_callback, lower_limit=None, upper_limit=None):
#         self._joint_name = joint_name
#
#         assert joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
#         self._joint_type = joint_type
#
#         self._get_pos_callback = get_pos_callback
#         self._set_pos_callback = set_pos_callback
#
#         self._lower_limit = lower_limit if lower_limit is not None else 0
#         self._upper_limit = upper_limit if upper_limit is not None else -1
#
#     @property
#     def joint_name(self):
#         return self._joint_name
#
#     @property
#     def joint_type(self):
#         return self._joint_type
#
#     @property
#     def lower_limit(self):
#         return self._lower_limit
#
#     @property
#     def upper_limit(self):
#         return self._upper_limit
#
#     @property
#     def max_velocity(self):
#         raise NotImplementedError("This feature is not available for virtual joints.")
#
#     @property
#     def max_torque(self):
#         raise NotImplementedError("This feature is not available for virtual joints.")
#
#     @property
#     def damping(self):
#         raise NotImplementedError("This feature is not available for virtual joints.")
#
#     def get_state(self):
#         return self._get_pos_callback()
#
#     def get_relative_state(self):
#         pos, _, _ = self.get_state()
#
#         # normalize position to [-1, 1]
#         if self.has_limit:
#             mean = (self.lower_limit + self.upper_limit) / 2.0
#             magnitude = (self.upper_limit - self.lower_limit) / 2.0
#             pos = (pos - mean) / magnitude
#
#         return pos, None, None
#
#     def set_pos(self, pos):
#         self._set_pos_callback(pos)
#
#     def set_vel(self, vel):
#         raise NotImplementedError("This feature is not implemented yet for virtual joints.")
#
#     def set_torque(self, torque):
#         raise NotImplementedError("This feature is not available for virtual joints.")
#
#     def reset_state(self, pos, vel):
#         raise NotImplementedError("This feature is not implemented yet for virtual joints.")
#
#     def __str__(self):
#         return "Virtual Joint name: {}".format(self.joint_name)
#
#
# class Virtual6DOFJoint(object):
#     """A wrapper for a floating (e.g. 6DOF) virtual joint between two robot body parts.
#     This wrapper generates the 6 separate VirtualJoint instances needed for such a mechanism, and accumulates their
#     set_pos calls to provide a single callback with a 6-DOF pose callback. Note that all 6 joints must be set for this
#     wrapper to trigger its callback - partial control not allowed.
#     """
#
#     COMPONENT_SUFFIXES = ["x", "y", "z", "rx", "ry", "rz"]
#
#     def __init__(self, joint_name, parent_link, child_link, command_callback, lower_limits=None, upper_limits=None):
#         self.joint_name = joint_name
#         self.parent_link = parent_link
#         self.child_link = child_link
#         self._command_callback = command_callback
#
#         self._joints = [
#             VirtualJoint(
#                 joint_name="%s_%s" % (self.joint_name, name),
#                 joint_type=p.JOINT_PRISMATIC if i < 3 else p.JOINT_REVOLUTE,
#                 get_pos_callback=lambda dof=i: (self.get_state()[dof], None, None),
#                 set_pos_callback=lambda pos, dof=i: self.set_pos(dof, pos),
#                 lower_limit=lower_limits[i] if lower_limits is not None else None,
#                 upper_limit=upper_limits[i] if upper_limits is not None else None,
#             )
#             for i, name in enumerate(Virtual6DOFJoint.COMPONENT_SUFFIXES)
#         ]
#
#         self._reset_stored_control()
#
#     def get_state(self):
#         pos, orn = self.child_link.get_position_orientation()
#
#         if self.parent_link is not None:
#             pos, orn = p.multiplyTransforms(*p.invertTransform(*self.parent_link.get_position_orientation()), pos, orn)
#
#         # Stack the position and the Euler orientation
#         return list(pos) + list(p.getEulerFromQuaternion(orn))
#
#     def get_joints(self):
#         """Gets the 1DOF VirtualJoints belonging to this 6DOF joint."""
#         return tuple(self._joints)
#
#     def set_pos(self, dof, val):
#         """Calls the command callback with values for all 6 DOF once the setter has been called for each of them."""
#         self._stored_control[dof] = val
#
#         if all(ctrl is not None for ctrl in self._stored_control):
#             self._command_callback(self._stored_control)
#             self._reset_stored_control()
#
#     def _reset_stored_control(self):
#         self._stored_control = [None] * len(self._joints)