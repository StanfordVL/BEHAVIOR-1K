from abc import ABCMeta, abstractmethod

import numpy as np

from future.utils import with_metaclass

from igibson.utils.constants import (
    ALL_COLLISION_GROUPS_MASK,
    DEFAULT_COLLISION_GROUP,
    SPECIAL_COLLISION_GROUPS,
    SemanticClass,
)
from igibson.utils.semantics_utils import CLASS_NAME_TO_CLASS_ID


class BaseObject(with_metaclass(ABCMeta, object)):
    """This is the interface that all iGibson objects must implement."""

    DEFAULT_RENDERING_PARAMS = {
        "use_pbr": True,
        "use_pbr_mapping": True,
        "shadow_caster": True,
    }

    def __init__(self, name=None, category="object", class_id=None, rendering_params=None):
        """
        Create an object instance with the minimum information of class ID and rendering parameters.

        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param rendering_params: Any keyword arguments to be passed into simulator.load_object_into_renderer(...).
        """
        # Generate a name if necessary. Note that the generation order & set of these names is not deterministic.
        if name is None:
            address = "%08X" % id(self)
            name = "{}_{}".format(category, address)

        self.name = name
        self.category = category

        # This sets the collision group of the object. In igibson, objects are only permitted to be part of a single
        # collision group, e.g. the collision group bitvector should only have one bit set to 1.
        self.collision_group = 1 << (
            SPECIAL_COLLISION_GROUPS[self.category]
            if self.category in SPECIAL_COLLISION_GROUPS
            else DEFAULT_COLLISION_GROUP
        )

        category_based_rendering_params = {}
        if category in ["walls", "floors", "ceilings"]:
            category_based_rendering_params["use_pbr"] = False
            category_based_rendering_params["use_pbr_mapping"] = False
        if category == "ceilings":
            category_based_rendering_params["shadow_caster"] = False

        if rendering_params:  # Use the input rendering params as an override.
            category_based_rendering_params.update(rendering_params)

        if class_id is None:
            class_id = CLASS_NAME_TO_CLASS_ID.get(category, SemanticClass.USER_ADDED_OBJS)

        self.class_id = class_id
        self.renderer_instances = []
        self._rendering_params = dict(self.DEFAULT_RENDERING_PARAMS)
        self._rendering_params.update(category_based_rendering_params)

        self._loaded = False
        self._body_ids = None

    def load(self, simulator):
        """Load object into pybullet and return list of loaded body ids."""
        if self._loaded:
            raise ValueError("Cannot load a single object multiple times.")
        self._loaded = True
        self._body_ids = self._load(simulator)

        # Set the collision groups.
        for body_id in self._body_ids:
            for link_id in [-1] + list(range(p.getNumJoints(body_id))):
                p.setCollisionFilterGroupMask(body_id, link_id, self.collision_group, ALL_COLLISION_GROUPS_MASK)

        return self._body_ids

    @property
    def loaded(self):
        return self._loaded

    def get_body_ids(self):
        """
        Gets the body IDs belonging to this object.
        """
        return self._body_ids

    @abstractmethod
    def _load(self, simulator):
        pass

    def get_position(self):
        """Get object position in the format of Array[x, y, z]"""
        return self.get_position_orientation()[0]

    def get_orientation(self):
        """Get object orientation as a quaternion in the format of Array[x, y, z, w]"""
        return self.get_position_orientation()[1]

    def get_position_orientation(self):
        """Get object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        assert len(self.get_body_ids()) == 1, "Base implementation only works with single-body objects."
        pos, orn = p.getBasePositionAndOrientation(self.get_body_ids()[0])
        return np.array(pos), np.array(orn)

    def set_position(self, pos):
        """Set object position in the format of Array[x, y, z]"""
        old_orn = self.get_orientation()
        self.set_position_orientation(pos, old_orn)

    def set_orientation(self, orn):
        """Set object orientation as a quaternion in the format of Array[x, y, z, w]"""
        old_pos = self.get_position()
        self.set_position_orientation(old_pos, orn)

    def set_position_orientation(self, pos, orn):
        """Set object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        assert len(self.get_body_ids()) == 1, "Base implementation only works with single-body objects."
        p.resetBasePositionAndOrientation(self.get_body_ids()[0], pos, orn)

    def set_base_link_position_orientation(self, pos, orn):
        """Set object base link position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        dynamics_info = p.getDynamicsInfo(self.get_body_ids()[0], -1)
        inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
        pos, orn = p.multiplyTransforms(pos, orn, inertial_pos, inertial_orn)
        self.set_position_orientation(pos, orn)

    def get_velocities(self):
        """Get object bodies' velocity in the format of List[Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]]"""
        velocities = []
        for body_id in self.get_body_ids():
            lin, ang = p.getBaseVelocity(body_id)
            velocities.append((np.array(lin), np.array(ang)))

        return velocities

    def set_velocities(self, velocities):
        """Set object base velocity in the format of List[Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]]"""
        assert len(velocities) == len(self.get_body_ids()), "Number of velocities should match number of bodies."

        for bid, (linear_velocity, angular_velocity) in zip(self.get_body_ids(), velocities):
            p.resetBaseVelocity(bid, linear_velocity, angular_velocity)

    def set_joint_states(self, joint_states):
        """Set object joint states in the format of Dict[String: (q, q_dot)]]"""
        for body_id in self.get_body_ids():
            for j in range(p.getNumJoints(body_id)):
                info = p.getJointInfo(body_id, j)
                joint_type = info[2]
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    joint_name = info[1].decode("UTF-8")
                    joint_position, joint_velocity = joint_states[joint_name]
                    p.resetJointState(body_id, j, joint_position, targetVelocity=joint_velocity)

    def get_joint_states(self):
        """Get object joint states in the format of Dict[String: (q, q_dot)]]"""
        joint_states = {}
        for body_id in self.get_body_ids():
            for j in range(p.getNumJoints(body_id)):
                info = p.getJointInfo(body_id, j)
                joint_type = info[2]
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    joint_name = info[1].decode("UTF-8")
                    joint_states[joint_name] = p.getJointState(body_id, j)[:2]
        return joint_states

    def dump_state(self):
        """Dump the state of the object other than what's not included in pybullet state."""
        return None

    def load_state(self, dump):
        """Load the state of the object other than what's not included in pybullet state."""
        return

    def highlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(True)

    def unhighlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(False)

    def force_wakeup(self):
        """
        Force wakeup sleeping objects
        """
        for body_id in self.get_body_ids():
            for joint_id in range(p.getNumJoints(body_id)):
                p.changeDynamics(body_id, joint_id, activationState=p.ACTIVATION_STATE_WAKE_UP)
            p.changeDynamics(body_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
