from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.object_state_base import BaseObjectState
from omnigibson.object_states.pose import Pose


class KinematicsMixin(BaseObjectState):
    """
    This class is a subclass of BaseObjectState that adds dependencies
    on the default kinematics states.
    """

    @staticmethod
    def get_dependencies():
        return BaseObjectState.get_dependencies() + [Pose, AABB, ContactBodies]
