from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.joint_state import Joint
from omnigibson.object_states.object_state_base import BaseObjectState
from omnigibson.object_states.pose import Pose
from omnigibson.utils.python_utils import classproperty


class KinematicsMixin(BaseObjectState):
    """
    This class is a subclass of BaseObjectState that adds dependencies
    on the default kinematics states.
    """

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({Pose, Joint, AABB, ContactBodies})
        return deps

    def cache_info(self, get_value_args):
        # Import here to avoid circular imports
        from omnigibson.objects.stateful_object import StatefulObject

        # Run super first
        info = super().cache_info(get_value_args=get_value_args)

        # Store this object as well as any other objects from @get_value_args
        info[self.obj] = {"q": self.obj.states[Joint].get_value(), "p": self.obj.states[Pose].get_value()}
        for arg in get_value_args:
            if isinstance(arg, StatefulObject):
                info[arg] = {"q": arg.states[Joint].get_value(), "p": arg.states[Pose].get_value()}

        return info

    def _cache_is_valid(self, get_value_args):
        # Import here to avoid circular imports
        from omnigibson.objects.stateful_object import StatefulObject

        # Cache is valid if and only if all of our cached objects have not changed
        t = self._cache[get_value_args]["t"]
        for obj, info in self._cache[get_value_args]["info"].items():
            # If the object is asleep, assume this object hasn't changed, so continue
            if obj.is_asleep:
                continue
            if isinstance(obj, StatefulObject):
                # If pose has changed, return False
                if obj.states[Pose].has_changed(get_value_args=(), value=info["p"], info={}, t=t):
                    return False
                # If obj's joints have changed, return False
                if obj.states[Joint].has_changed(get_value_args=(), value=info["q"], info={}, t=t):
                    return False
        return True

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("KinematicsMixin")
        return classes
