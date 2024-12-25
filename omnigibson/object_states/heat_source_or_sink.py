import torch as th

import omnigibson as og
from omnigibson.macros import create_module_macros, macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import classproperty

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.HEATSOURCE_LINK_PREFIX = "heatsource"
m.HEATING_ELEMENT_MARKER_SCALE = [1.0] * 3

# TODO: Delete default values for this and make them required.
m.DEFAULT_TEMPERATURE = 200
m.DEFAULT_HEATING_RATE = 0.04
m.DEFAULT_DISTANCE_THRESHOLD = 0.2


class HeatSourceOrSink(AbsoluteObjectState, LinkBasedStateMixin, UpdateStateMixin):
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
        temperature=None,
        heating_rate=None,
        distance_threshold=None,
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
        self._temperature = temperature if temperature is not None else m.DEFAULT_TEMPERATURE
        self._heating_rate = heating_rate if heating_rate is not None else m.DEFAULT_HEATING_RATE
        self.distance_threshold = distance_threshold if distance_threshold is not None else m.DEFAULT_DISTANCE_THRESHOLD

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

        # Internal state that gets cached
        self._affected_objects = None

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

        # If the object is not affected, we return False
        if obj not in self._affected_objects:
            return False

        # If all checks pass, we're actively influencing the object!
        return True

    def _update(self):
        # Avoid circular imports
        from omnigibson.object_states.temperature import Temperature
        from omnigibson.objects.stateful_object import StatefulObject

        # Update the internally tracked nearby objects to accelerate filtering for affects_obj
        affected_objects = set()

        # Only update if we're valid
        if self.get_value():

            def overlap_callback(hit):
                nonlocal affected_objects
                # global affected_objects
                obj = self.obj.scene.object_registry("prim_path", "/".join(hit.rigid_body.split("/")[:-1]))
                if obj is not None and obj != self.obj and obj in Temperature.OBJ_IDXS:
                    affected_objects.add(obj)
                # Always continue traversal
                return True

            if self.requires_inside:
                # Use overlap_box check to check for objects inside the box!
                aabb_lower, aabb_upper = self.obj.states[AABB].get_value()
                half_extent = (aabb_upper - aabb_lower) / 2.0
                aabb_center = (aabb_upper + aabb_lower) / 2.0

                og.sim.psqi.overlap_box(
                    halfExtent=half_extent.tolist(),
                    pos=aabb_center.tolist(),
                    rot=[0, 0, 0, 1.0],
                    reportFn=overlap_callback,
                )

                # Cloth isn't subject to overlap checks, so we also have to manually check their poses as well
                cloth_objs = tuple(self.obj.scene.object_registry("prim_type", PrimType.CLOTH, []))
                n_cloth_objs = len(cloth_objs)
                if n_cloth_objs > 0:
                    cloth_positions = th.zeros((n_cloth_objs, 3))
                    for i, obj in enumerate(cloth_objs):
                        cloth_positions[i] = obj.get_position_orientation()[0]
                    for idx in th.where(
                        th.all(
                            (aabb_lower.reshape(1, 3) < cloth_positions) & (cloth_positions < aabb_upper.reshape(1, 3)),
                            dim=-1,
                        )
                    )[0]:
                        # Only add if object has temperature
                        if cloth_objs[idx] in Temperature.OBJ_IDXS:
                            affected_objects.add(cloth_objs[idx])

                # Additionally prune objects based on Temperature / Inside requirement -- cast to avoid in-place operations
                for obj in tuple(affected_objects):
                    if not obj.states[Inside].get_value(self.obj):
                        affected_objects.remove(obj)

            else:
                # Position is either the AABB center of the default link or the metalink position itself
                heat_source_pos = (
                    self.link.aabb_center
                    if self.link == self._default_link
                    else self.link.get_position_orientation()[0]
                )

                # Use overlap_sphere check!
                og.sim.psqi.overlap_sphere(
                    radius=self.distance_threshold,
                    pos=heat_source_pos.tolist(),
                    reportFn=overlap_callback,
                )

                # Cloth isn't subject to overlap checks, so we also have to manually check their poses as well
                cloth_objs = tuple(self.obj.scene.object_registry("prim_type", PrimType.CLOTH, []))
                n_cloth_objs = len(cloth_objs)
                if n_cloth_objs > 0:
                    cloth_positions = th.zeros((n_cloth_objs, 3))
                    for i, obj in enumerate(cloth_objs):
                        cloth_positions[i] = obj.get_position_orientation()[0]
                    for idx in th.where(
                        th.norm(heat_source_pos.reshape(1, 3) - cloth_positions, dim=-1) <= self.distance_threshold
                    )[0]:
                        # Only add if object has temperature
                        if cloth_objs[idx] in Temperature.OBJ_IDXS:
                            affected_objects.add(cloth_objs[idx])

        # Remove self (we cannot affect ourselves) and update the internal set of objects, and remove self
        if self.obj in affected_objects:
            affected_objects.remove(self.obj)
        self._affected_objects = affected_objects

        # Propagate the affected objects' temperatures
        if len(self._affected_objects) > 0:
            Temperature.update_temperature_from_heatsource_or_sink(
                objs=self._affected_objects,
                temperature=self.temperature,
                rate=self.heating_rate,
            )

    # Nothing needs to be done to save/load HeatSource
