from igibson.object_states.object_state_base import REGISTERED_OBJECT_STATES
from igibson.object_states.aabb import AABB
from igibson.object_states.adjacency import HorizontalAdjacency, VerticalAdjacency
from igibson.object_states.attachment import (
    Attached,
    StickyAttachment,
    MagneticAttachment,
    MaleAttachment,
    FemaleAttachment,
    HungMaleAttachment,
    HungFemaleAttachment,
)
from igibson.object_states.burnt import Burnt
from igibson.object_states.cleaning_tool import CleaningTool
from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.cooked import Cooked
from igibson.object_states.dirty import Dusty, Stained
from igibson.object_states.frozen import Frozen
from igibson.object_states.heat_source_or_sink import HeatSourceOrSink
from igibson.object_states.heated import Heated
from igibson.object_states.inside import Inside
from igibson.object_states.max_temperature import MaxTemperature
from igibson.object_states.next_to import NextTo
from igibson.object_states.on_floor import OnFloor
from igibson.object_states.on_top import OnTop
from igibson.object_states.open import Open
from igibson.object_states.pose import Pose
from igibson.object_states.robot_related_states import (
    InFOVOfRobot,
    InHandOfRobot,
    InReachOfRobot,
    InSameRoomAsRobot,
    ObjectsInFOVOfRobot,
)
from igibson.object_states.room_states import (
    ROOM_STATES,
    InsideRoomTypes,
    IsInAuditorium,
    IsInBalcony,
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
    IsInLibrary,
    IsInLivingRoom,
    IsInLobby,
    IsInPantryRoom,
    IsInPlayroom,
    IsInStaircase,
    IsInStorageRoom,
    IsInTelevisionRoom,
    IsInUndefined,
    IsInUtilityRoom,
)
from igibson.object_states.sliced import Sliced
from igibson.object_states.slicer import Slicer
from igibson.object_states.soaked import Soaked
from igibson.object_states.temperature import Temperature
from igibson.object_states.toggle import ToggledOn
from igibson.object_states.touching import Touching
from igibson.object_states.under import Under
from igibson.object_states.water_source import WaterSource
from igibson.object_states.water_sink import WaterSink
from igibson.object_states.filled import Filled
