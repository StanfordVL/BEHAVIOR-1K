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

    def _has_changed(self, get_value_args, t):
        # Import here to avoid circular imports
        from omnigibson.objects.stateful_object import StatefulObject

        # Only clear cache if at least one object has moved
        for arg in get_value_args:
            if isinstance(arg, StatefulObject) and arg.states[Pose].has_changed(get_value_args=(), t=t):
                # We've changed because at least one relevant object's position has changed
                return True

        # Otherwise, nothing has moved, return False
        return False

    def _get_value(self, *args, **kwargs):
        # Import here to avoid circular imports
        from omnigibson.objects.stateful_object import StatefulObject

        # Make sure all poses are cached so we can check in the future if values have changed
        get_value_args = (*args, *tuple(kwargs.values()))
        for arg in get_value_args:
            if isinstance(arg, StatefulObject):
                arg.states[Pose].get_value()
