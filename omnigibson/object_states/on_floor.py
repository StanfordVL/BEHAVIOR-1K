
from IPython import embed

import omnigibson
from omnigibson.object_states.kinematics import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanState, RelativeObjectState
from omnigibson.object_states.touching import Touching
from omnigibson.utils.object_state_utils import get_center_extent, sample_kinematics


# TODO: remove after split floors
class RoomFloor(object):
    def __init__(self, category, name, scene, room_instance, floor_obj):
        self.category = category
        self.name = name
        self.scene = scene
        self.room_instance = room_instance
        self.floor_obj = floor_obj

    def __getattr__(self, item):
        if item == "states":
            self.floor_obj.set_room_floor(self)
        return getattr(self.floor_obj, item)


class OnFloor(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + [Touching]

    def _set_value(self, other, new_value):
        if not isinstance(other, RoomFloor):
            return False

        state = self._simulator.dump_state(serialized=False)

        for _ in range(10):
            sampling_success = sample_kinematics("onFloor", self.obj, other, new_value)
            if sampling_success:
                self.obj.clear_cached_states()
                if self.get_value(other) != new_value:
                    sampling_success = False
                if omnigibson.debug_sampling:
                    print("OnFloor checking", sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                self._simulator.load_state(state, serialized=False)

        return sampling_success

    def _get_value(self, other):
        if not isinstance(other, RoomFloor):
            return False

        room_instance = other.scene.seg_map.get_room_instance_by_point(self.obj.get_position()[:2])
        is_in_room = room_instance == other.room_instance

        floors = other.scene.object_registry("category", "floors")
        assert len(floors) == 1, "has more than one floor object"
        # Use the floor object in the scene to detect contact points
        scene_floor = list(floors)[0]

        # Special case: the BehaviorRobot does not need to actually touch the floor of a room to be considered
        # OnFloor in that room. As a hovering robot, BehaviorRobot won't actually touch the floor during operation.
        # TODO: Same
        # from omnigibson.robots import BehaviorRobot
        #
        # if isinstance(self.obj, BehaviorRobot):
        #     return is_in_room

        touching = self.obj.states[Touching].get_value(scene_floor)
        return is_in_room and touching
