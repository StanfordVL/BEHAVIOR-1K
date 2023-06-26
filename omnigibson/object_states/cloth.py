from omnigibson.object_states.object_state_base import BaseObjectState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import classproperty


class ClothState(BaseObjectState):
    """
    This class is a subclass of BaseObjectState that adds dependencies
    on the default kinematics states.
    """

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Only compatible with cloth objects
        compatible, reason = super().is_compatible(obj, **kwargs)
        if not compatible:
            return compatible, reason

        return (True, None) if obj.prim_type == PrimType.CLOTH else \
            (False, f"Cannot use ClothState {cls.__name__} with rigid object, make sure object is created "
                    f"with prim_type=PrimType.CLOTH!")

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("ClothState")
        return classes
