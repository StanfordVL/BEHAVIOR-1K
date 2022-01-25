
from IPython import embed

import igibson
from igibson.object_states.kinematics import KinematicsMixin
from igibson.object_states.object_state_base import BooleanState, RelativeObjectState
from igibson.object_states.touching import Touching
from igibson.object_states.utils import clear_cached_states, get_center_extent, sample_kinematics
from igibson.utils.utils import restoreState


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


class OnFloor(RelativeObjectState, KinematicsMixin, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + [Touching]

    def _set_value(self, other, new_value):
        if not isinstance(other, RoomFloor):
            return False

        state_id = p.saveState()
        for _ in range(10):
            sampling_success = sample_kinematics("onFloor", self.obj, other, new_value)
            if sampling_success:
                clear_cached_states(self.obj)
                if self.get_value(other) != new_value:
                    sampling_success = False
                if igibson.debug_sampling:
                    print("OnFloor checking", sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                restoreState(state_id)

        p.removeState(state_id)

        return sampling_success

    def _get_value(self, other):
        if not isinstance(other, RoomFloor):
            return False

        objA_states = self.obj.states
        center, extent = get_center_extent(objA_states)
        room_instance = other.scene.get_room_instance_by_point(center[:2])
        is_in_room = room_instance == other.room_instance

        floors = other.scene.objects_by_category["floors"]
        assert len(floors) == 1, "has more than one floor object"
        # Use the floor object in the scene to detect contact points
        scene_floor = floors[0]

        # Special case: the BehaviorRobot does not need to actually touch the floor of a room to be considered
        # OnFloor in that room. As a hovering robot, BehaviorRobot won't actually touch the floor during operation.
        from igibson.robots import BehaviorRobot

        if isinstance(self.obj, BehaviorRobot):
            return is_in_room

        touching = self.obj.states[Touching].get_value(scene_floor)
        return is_in_room and touching
