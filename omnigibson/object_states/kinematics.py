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

    def _cache_info(self, get_value_args):
        # Import here to avoid circular imports
        from omnigibson.objects.stateful_object import StatefulObject

        # Run super first
        info = super()._cache_info(get_value_args=get_value_args)

        # Store this object as well as any other objects from @get_value_args
        info[self.obj.name] = self.obj
        for arg in get_value_args:
            if isinstance(arg, StatefulObject):
                info[arg.name] = arg

        return info

    def _should_clear_cache(self, get_value_args, cache_info):
        # Only clear cache if all the objects have not moved
        for obj in cache_info.values():
            if obj.states[Pose].has_moved:
                # We need to clear the cache because at least one relevant object's position has changed
                return True

        # Otherwise, nothing has moved, return False
        return False
