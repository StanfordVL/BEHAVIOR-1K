import numpy as np
import omnigibson as og
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin


_IN_REACH_DISTANCE_THRESHOLD = 2.0

_IN_FOV_PIXEL_FRACTION_THRESHOLD = 0.05


def _get_robot():
    from omnigibson.robots import ManipulationRobot
    valid_robots = [robot for robot in og.sim.scene.robots if isinstance(robot, ManipulationRobot)]
    if not valid_robots:
        return None

    if len(valid_robots) > 1:
        raise ValueError("Multiple robots found.")

    return valid_robots[0]


# class InReachOfRobot(AbsoluteObjectState, BooleanStateMixin):
#     def _compute_value(self):
#         robot = _get_robot(self.simulator)
#         if not robot:
#             return False

#         robot_pos = robot.get_position()
#         object_pos = self.obj.get_position()
#         return np.linalg.norm(object_pos - np.array(robot_pos)) < _IN_REACH_DISTANCE_THRESHOLD


class InHandOfRobot(AbsoluteObjectState, BooleanStateMixin):
    def _get_value(self):
        robot = _get_robot()
        if not robot:
            return False

        return any(
            robot._ag_obj_in_hand[arm] == self.obj 
            for arm in robot.arm_names
        )


# class InFOVOfRobot(AbsoluteObjectState, BooleanStateMixin):
#     @staticmethod
#     def get_optional_dependencies():
#         return AbsoluteObjectState.get_optional_dependencies() + [ObjectsInFOVOfRobot]

#     def _get_value(self):
#         robot = _get_robot(self.simulator)
#         if not robot:
#             return False

#         body_ids = set(self.obj.get_body_ids())
#         return not body_ids.isdisjoint(robot.states[ObjectsInFOVOfRobot].get_value())


# class ObjectsInFOVOfRobot(AbsoluteObjectState):
#     def _get_value(self):
#         # Pass the FOV through the instance-to-body ID mapping.
#         seg = self.simulator.renderer.render_single_robot_camera(self.obj, modes="ins_seg")[0][:, :, 0]
#         seg = np.round(seg * MAX_INSTANCE_COUNT).astype(int)
#         body_ids = self.simulator.renderer.get_pb_ids_for_instance_ids(seg)

#         # Pixels that don't contain an object are marked -1 but we don't want to include that
#         # as a body ID.
#         return set(np.unique(body_ids)) - {-1}