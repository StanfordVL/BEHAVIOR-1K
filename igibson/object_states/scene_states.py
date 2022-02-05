from collections import OrderedDict, namedtuple

ObjectSceneState = namedtuple("SceneState", [
    "bbox_center_pose",                     # ([x, y, z], [x, y, z, w])
    "base_com_pose",                        # ([x, y, z], [x, y, z, w])
    "base_velocities",                      # ([vx, vy, vz], [wx, wy, wz])
    "joint_states",                         # {joint_name: (q, q_dot)}
    "non_kinematic_states",                 # dict()
])


class ObjectSceneStatesRegistry:
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
        Returns:
            None or ObjectSceneState: state corresponding to object @obj_name
        """
        return self._states.get(obj_name, None)
