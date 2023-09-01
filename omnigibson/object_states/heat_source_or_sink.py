from omnigibson.macros import create_module_macros, macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.utils.python_utils import classproperty
import omnigibson.utils.transform_utils as T


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.HEATSOURCE_LINK_PREFIX = "heatsource"
m.HEATING_ELEMENT_MARKER_SCALE = [1.0] * 3

# TODO: Delete default values for this and make them required.
m.DEFAULT_TEMPERATURE = 200
m.DEFAULT_HEATING_RATE = 0.04
m.DEFAULT_DISTANCE_THRESHOLD = 0.2


class HeatSourceOrSink(AbsoluteObjectState, LinkBasedStateMixin):
    """
    This state indicates the heat source or heat sink state of the object.

    Currently, if the object is not an active heat source/sink, this returns (False, None).
    Otherwise, it returns True and the position of the heat source element, or (True, None) if the heat source has no
    heating element / only checks for Inside.
    E.g. on a stove object, True and the coordinates of the heating element will be returned.
    on a microwave object, True and None will be returned.
    """

    def __init__(
        self,
        obj,
        temperature=m.DEFAULT_TEMPERATURE,
        heating_rate=m.DEFAULT_HEATING_RATE,
        distance_threshold=m.DEFAULT_DISTANCE_THRESHOLD,
        requires_toggled_on=False,
        requires_closed=False,
        requires_inside=False,
    ):
        """
        Args:
            obj (StatefulObject): The object with the heat source ability.
            temperature (float): The temperature of the heat source.
            heating_rate (float): Fraction in [0, 1] of the temperature difference with the
                heat source temperature should be received every step, per second.
            distance_threshold (float): The distance threshold which an object needs
                to be closer than in order to receive heat from this heat source.
            requires_toggled_on (bool): Whether the heat source object needs to be
                toggled on to emit heat. Requires toggleable ability if set to True.
            requires_closed (bool): Whether the heat source object needs to be
                closed (e.g. in terms of the joints) to emit heat. Requires openable
                ability if set to True.
            requires_inside (bool): Whether an object needs to be `inside` the
                heat source to receive heat. See the Inside state for details. This
                will mean that the "heating element" link for the object will be
                ignored.
        """
        super(HeatSourceOrSink, self).__init__(obj)
        self._temperature = temperature
        self._heating_rate = heating_rate
        self.distance_threshold = distance_threshold

        # If the heat source needs to be toggled on, we assert the presence
        # of that ability.
        if requires_toggled_on:
            assert ToggledOn in self.obj.states
        self.requires_toggled_on = requires_toggled_on

        # If the heat source needs to be closed, we assert the presence
        # of that ability.
        if requires_closed:
            assert Open in self.obj.states
        self.requires_closed = requires_closed

        # If the heat source needs to contain an object inside to heat it,
        # we record that for use in the heat transfer process.
        self.requires_inside = requires_inside

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible(obj, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this state has toggledon if required or open if required
        for kwarg, state_type in zip(("requires_toggled_on", "requires_closed"), (ToggledOn, Open)):
            if kwargs.get(kwarg, False) and state_type not in obj.states:
                return False, f"{cls.__name__} has {kwarg} but obj has no {state_type.__name__} state!"

        return True, None

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible_asset(prim, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this state has toggledon if required or open if required
        for kwarg, state_type in zip(("requires_toggled_on", "requires_closed"), (ToggledOn, Open)):
            if kwargs.get(kwarg, False) and not state_type.is_compatible_asset(prim=prim, **kwargs)[0]:
                return False, f"{cls.__name__} has {kwarg} but obj has no {state_type.__name__} state!"

        return True, None

    @classproperty
    def metalink_prefix(cls):
        return m.HEATSOURCE_LINK_PREFIX

    @classmethod
    def requires_metalink(cls, **kwargs):
        # No metalink required if inside
        return not kwargs.get("requires_inside", False)

    @property
    def _default_link(self):
        # Only supported if we require inside
        return self.obj.root_link if self.requires_inside else super()._default_link

    @property
    def heating_rate(self):
        """
        Returns:
            float: Temperature changing rate of this heat source / sink
        """
        return self._heating_rate

    @property
    def temperature(self):
        """
        Returns:
            float: Temperature of this heat source / sink
        """
        return self._temperature

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({AABB, Inside})
        return deps

    @classmethod
    def get_optional_dependencies(cls):
        deps = super().get_optional_dependencies()
        deps.update({ToggledOn, Open})
        return deps

    def _initialize(self):
        # Run super first
        super()._initialize()
        self.initialize_link_mixin()

    def _get_value(self):
        # Check the toggle state.
        if self.requires_toggled_on and not self.obj.states[ToggledOn].get_value():
            return False

        # Check the open state.
        if self.requires_closed and self.obj.states[Open].get_value():
            return False

        return True

    def affects_obj(self, obj):
        """
        Computes whether this heat source or sink object is affecting object @obj
        Computes the temperature delta that may be applied to object @obj. NOTE: This value is agnostic to simulation
        stepping speed, and should be scaled accordingly

        Args:
            obj (StatefulObject): Object whose temperature delta should be computed

        Returns:
            bool: Whether this heat source or sink is currently affecting @obj's temperature
        """
        # No change if we're not on
        if not self.get_value():
            return False

        # Otherwise, check for other edge cases
        # If we require the object to be inside, make sure the object is inside, otherwise, we return 0
        # Otherwise, make sure the object is within close proximity of this heat source
        if self.requires_inside:
            if not obj.states[Inside].get_value(self.obj):
                return False
        else:
            aabb_lower, aabb_upper = obj.states[AABB].get_value()
            obj_pos = (aabb_lower + aabb_upper) / 2.0
            # Position is either the AABB center of the default link or the metalink position itself
            heat_source_pos = self.link.aabb_center if self.link == self._default_link else self.link.get_position()
            if T.l2_distance(heat_source_pos, obj_pos) > self.distance_threshold:
                return False

        # If all checks pass, we're actively influencing the object!
        return True

    # Nothing needs to be done to save/load HeatSource
