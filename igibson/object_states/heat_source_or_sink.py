import os

import numpy as np


import igibson
from igibson.object_states.aabb import AABB
from igibson.object_states.inside import Inside
from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.object_states.object_state_base import AbsoluteObjectState, NONE
from igibson.object_states.open import Open
from igibson.object_states.toggle import ToggledOn

# The name of the heat source link inside URDF files.
from igibson.objects.visual_marker import VisualMarker
from igibson.utils.constants import SemanticClass

_HEATING_ELEMENT_LINK_NAME = "heat_source"

_HEATING_ELEMENT_MARKER_SCALE = [1.0] * 3
_HEATING_ELEMENT_MARKER_FILENAME = os.path.join(igibson.assets_path, "models/fire/fire.obj")

# TODO: Delete default values for this and make them required.
_DEFAULT_TEMPERATURE = 200
_DEFAULT_HEATING_RATE = 0.04
_DEFAULT_DISTANCE_THRESHOLD = 0.2


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
        temperature=_DEFAULT_TEMPERATURE,
        heating_rate=_DEFAULT_HEATING_RATE,
        distance_threshold=_DEFAULT_DISTANCE_THRESHOLD,
        requires_toggled_on=False,
        requires_closed=False,
        requires_inside=False,
    ):
        """
        Initialize a heat source state.

        :param obj: The object with the heat source ability.
        :param temperature: The temperature of the heat source.
        :param heating_rate: Fraction of the temperature difference with the
            heat source temperature should be received every step, per second.
        :param distance_threshold: The distance threshold which an object needs
            to be closer than in order to receive heat from this heat source.
        :param requires_toggled_on: Whether the heat source object needs to be
            toggled on to emit heat. Requires toggleable ability if set to True.
        :param requires_closed: Whether the heat source object needs to be
            closed (e.g. in terms of the joints) to emit heat. Requires openable
            ability if set to True.
        :param requires_inside: Whether an object needs to be `inside` the
            heat source to receive heat. See the Inside state for details. This
            will mean that the "heating element" link for the object will be
            ignored.
        """
        super(HeatSourceOrSink, self).__init__(obj)
        self.temperature = temperature
        self.heating_rate = heating_rate
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

        self.marker = None
        self.status = None
        self.position = None

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB, Inside]

    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + [ToggledOn, Open]

    @staticmethod
    def get_state_link_name():
        return _HEATING_ELEMENT_LINK_NAME

    def _compute_state_and_position(self):
        # Find the link first. Note that the link is only required
        # if the object is not in self.requires_inside mode.
        heating_element_position = self.get_link_position()
        if not self.requires_inside and heating_element_position is None:
            return False, None

        # Check the toggle state.
        if self.requires_toggled_on and not self.obj.states[ToggledOn].get_value():
            return False, None

        # Check the open state.
        if self.requires_closed and self.obj.states[Open].get_value():
            return False, None

        # Return True and the heating element position (or None if not required).
        return True, (heating_element_position if not self.requires_inside else None)

    def _initialize(self):
        super(HeatSourceOrSink, self)._initialize()
        if not self.requires_inside:
            self.initialize_link_mixin()

        # Import at runtime to prevent circular imports
        from igibson.objects.usd_object import USDObject
        self.marker = USDObject(
            prim_path=f"{self.obj.prim_path}/heat_source_marker",
            usd_path=_HEATING_ELEMENT_MARKER_FILENAME,
            name="heat_source_marker",
            class_id=SemanticClass.HEAT_SOURCE_MARKER,
            scale=_HEATING_ELEMENT_MARKER_SCALE,
            visible=True,
            fixed_base=False,
            visual_only=True,
        )

        self.marker.load(simulator=self.simulator)
        self.marker.set_position(np.array([0, 0, -100]))

    def _update(self):
        self.status, self.position = self._compute_state_and_position()

        # Move the marker.
        marker_position = np.array([0, 0, -100])
        if self.position is not None:
            marker_position = self.position

        # Lazy update of marker position
        if not np.all(np.isclose(marker_position, self.marker.get_position())):
            self.marker.set_position(marker_position)

    def _get_value(self):
        return self.status, self.position

    def _set_value(self, new_value):
        raise NotImplementedError("Setting heat source capability is not supported.")

    # Nothing needs to be done to save/load HeatSource since it's stateless except for
    # the marker.
