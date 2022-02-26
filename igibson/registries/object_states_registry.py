from collections import OrderedDict, namedtuple
import numpy as np

# from igibson.object_states import OBJECT_STATES_TO_ID


# TODO: Remove all of this! Rely on automatic object state registration in the registry functionality

ObjectSceneState = namedtuple("SceneState", [
    "bbox_center_pose",                     # ([x, y, z], [x, y, z, w])
    "base_com_pose",                        # ([x, y, z], [x, y, z, w])
    "base_velocities",                      # ([vx, vy, vz], [wx, wy, wz])
    "joint_states",                         # {joint_name: (q, q_dot)}
    "non_kinematic_states",                 # dict()
])


class ObjectStatesRegistry:
    """
    Simple structured class to organize object states
    """
    def __init__(self):
        # Create states
        self._states = OrderedDict()

    def add_object(self, obj_name, bbox_center_pose, base_com_pose, base_velocities, joint_states, non_kinematic_states):
        """
        Adds a set of object states to this registry
        """
        self._states[obj_name] = ObjectSceneState(
            bbox_center_pose=bbox_center_pose,
            base_com_pose=base_com_pose,
            base_velocities=base_velocities,
            joint_states=joint_states,
            non_kinematic_states=non_kinematic_states,
        )

    def __call__(self, obj_name):
        """
        Grabs the object's state(s) corresponding to @obj_name.

        Args:
            obj_name (str): Object to grab state for
        Returns:
            None or ObjectSceneState: state corresponding to object @obj_name if the object exists, else None
        """
        return self._states.get(obj_name, None)

    def serialize(self):
        """
        Serialize all the states within this registry into a single 1D numpy float array

        Returns:
            n-array: Serialized object states contained in this registry.
        """
        obj_states_flat = []
        for obj_state in self._states.values():
            obj_state_flat = [
                obj_state.bbox_center_pose[0],
                obj_state.bbox_center_pose[1],
                obj_state.base_com_pose[0],
                obj_state.base_com_pose[1],
                obj_state.base_velocities[0],
                obj_state.base_velocities[1],
                np.concatenate(list(obj_state.joint_states.values())),
                self._encode
            ]

    @staticmethod
    def _serialize_non_kinematic_states(non_kinematic_state):
        """
        Serializes the non kinematic state, which consists of a nested dictionary of info

        Args:
            non_kinematic_state (dict): Dictionary mapping non kinematic state type names to corresponding values,
                which may possibly be nested

        Returns:
            n-array: flattened, 1D serialized array of non_kinematic_states
        """
        state_flat = []
        # We iterate over all the properties in the non kinematic states

