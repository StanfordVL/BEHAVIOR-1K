import numpy as np

from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState, CachingEnabledObjectState, NONE


class InsideRoomTypes(CachingEnabledObjectState):
    """The value of this state is the list of rooms that the object currently is in."""

    def _compute_value(self):
        if hasattr(self.obj, "fixed_base") and self.obj.fixed_base:
            # For fixed objects, we can use the in_rooms attribute.
            if hasattr(self.obj, "in_rooms") and self.obj.in_rooms:
                return self.obj.in_rooms

        # Otherwise we need to calculate using room segmentation function. Check that it exists.
        if not hasattr(self._simulator.scene, "get_room_type_by_point"):
            return ["undefined"]

        pose = self.obj.get_position()
        return [self._simulator.scene.get_room_type_by_point(np.array(pose[:2]))]

    def _set_value(self, new_value):
        raise NotImplementedError("Room state currently does not support setting.")


class IsInRoomTemplate(AbsoluteObjectState, BooleanState):
    ROOM_TYPE = None

    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + [InsideRoomTypes]

    def _get_value(self):
        assert self.ROOM_TYPE, "IsInRoomTemplate should only be used through subclasses."
        return self.ROOM_TYPE in self.obj.states[InsideRoomTypes].get_value()

    def _set_value(self, new_value):
        raise NotImplementedError("IsInRoom states currently does not support setting.")


# TODO: Can we make this automatic in any way?
IsInBathroom = type("IsInBathroom", (IsInRoomTemplate,), {"ROOM_TYPE": "bathroom"})
IsInBedroom = type("IsInBedroom", (IsInRoomTemplate,), {"ROOM_TYPE": "bedroom"})
IsInChildsRoom = type("IsInChildsRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "childs_room"})
IsInCloset = type("IsInCloset", (IsInRoomTemplate,), {"ROOM_TYPE": "closet"})
IsInCorridor = type("IsInCorridor", (IsInRoomTemplate,), {"ROOM_TYPE": "corridor"})
IsInDiningRoom = type("IsInDiningRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "dining_room"})
IsInEmptyRoom = type("IsInEmptyRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "empty_room"})
IsInExerciseRoom = type("IsInExerciseRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "exercise_room"})
IsInGarage = type("IsInGarage", (IsInRoomTemplate,), {"ROOM_TYPE": "garage"})
IsInHomeOffice = type("IsInHomeOffice", (IsInRoomTemplate,), {"ROOM_TYPE": "home_office"})
IsInKitchen = type("IsInKitchen", (IsInRoomTemplate,), {"ROOM_TYPE": "kitchen"})
IsInLivingRoom = type("IsInLivingRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "living_room"})
IsInLobby = type("IsInLobby", (IsInRoomTemplate,), {"ROOM_TYPE": "lobby"})
IsInPantryRoom = type("IsInPantryRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "pantry_room"})
IsInPlayroom = type("IsInPlayroom", (IsInRoomTemplate,), {"ROOM_TYPE": "playroom"})
IsInStaircase = type("IsInStaircase", (IsInRoomTemplate,), {"ROOM_TYPE": "staircase"})
IsInStorageRoom = type("IsInStorageRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "storage_room"})
IsInTelevisionRoom = type("IsInTelevisionRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "television_room"})
IsInUtilityRoom = type("IsInUtilityRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "utility_room"})
IsInBalcony = type("IsInBalcony", (IsInRoomTemplate,), {"ROOM_TYPE": "balcony"})
IsInLibrary = type("IsInLibrary", (IsInRoomTemplate,), {"ROOM_TYPE": "library"})
IsInAuditorium = type("IsInAuditorium", (IsInRoomTemplate,), {"ROOM_TYPE": "auditorium"})
IsInUndefined = type("IsInUndefined", (IsInRoomTemplate,), {"ROOM_TYPE": "undefined"})

ROOM_STATES = [
    IsInBathroom,
    IsInBedroom,
    IsInChildsRoom,
    IsInCloset,
    IsInCorridor,
    IsInDiningRoom,
    IsInEmptyRoom,
    IsInExerciseRoom,
    IsInGarage,
    IsInHomeOffice,
    IsInKitchen,
    IsInLivingRoom,
    IsInLobby,
    IsInPantryRoom,
    IsInPlayroom,
    IsInStaircase,
    IsInStorageRoom,
    IsInTelevisionRoom,
    IsInUtilityRoom,
    IsInBalcony,
    IsInLibrary,
    IsInAuditorium,
    IsInUndefined,
]
