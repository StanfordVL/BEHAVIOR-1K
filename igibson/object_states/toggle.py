import numpy as np
from collections import OrderedDict

from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.object_states.texture_change_state_mixin import TextureChangeStateMixin
from igibson.utils.constants import SemanticClass, SimulatorMode
from igibson.utils.utils import brighten_texture

_TOGGLE_DISTANCE_THRESHOLD = 0.1
_TOGGLE_LINK_NAME = "toggle_button"
_TOGGLE_BUTTON_SCALE = 0.05
_CAN_TOGGLE_STEPS = 5


class ToggledOn(AbsoluteObjectState, BooleanState, TextureChangeStateMixin, LinkBasedStateMixin):
    def __init__(self, obj):
        super(ToggledOn, self).__init__(obj)
        self.value = False
        self.robot_can_toggle_steps = 0

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    @staticmethod
    def get_state_link_name():
        return _TOGGLE_LINK_NAME

    def _initialize(self):
        super(ToggledOn, self)._initialize()
        if self.initialize_link_mixin():
            # Import at runtime to prevent circular imports
            from igibson.objects.primitive_object import PrimitiveObject
            self.visual_marker_on = PrimitiveObject(
                prim_path=f"{self.obj.prim_path}/visual_marker_on",
                primitive_type="Sphere",
                name=f"{self.obj.name}_visual_marker_on",
                class_id=SemanticClass.TOGGLE_MARKER,
                scale=_TOGGLE_BUTTON_SCALE,
                visible=True,
                fixed_base=False,
                visual_only=True,
                rgba=[0, 1, 0, 0.5],
            )
            self.visual_marker_off = PrimitiveObject(
                prim_path=f"{self.obj.prim_path}/visual_marker_off",
                primitive_type="Sphere",
                name=f"{self.obj.name}_visual_marker_off",
                class_id=SemanticClass.TOGGLE_MARKER,
                scale=_TOGGLE_BUTTON_SCALE,
                visible=True,
                fixed_base=False,
                visual_only=True,
                rgba=[1, 0, 0, 0.5],
            )
            self.simulator.import_object(self.visual_marker_on)
            self.visual_marker_on.visible = False
            self.simulator.import_object(self.visual_marker_off)
            self.visual_marker_off.visible = False

    def _update(self):
        button_position_on_object = self.get_link_position()
        if button_position_on_object is None:
            return

        robot_can_toggle = False
        # detect marker and hand interaction
        for robot in self.simulator.scene.robots:
            robot_can_toggle = robot.can_toggle(button_position_on_object, _TOGGLE_DISTANCE_THRESHOLD)
            if robot_can_toggle:
                break

        if robot_can_toggle:
            self.robot_can_toggle_steps += 1
        else:
            self.robot_can_toggle_steps = 0

        if self.robot_can_toggle_steps == _CAN_TOGGLE_STEPS:
            self.value = not self.value

        # Choose which marker to put on object vs which to put away
        show_marker = self.visual_marker_on if self.get_value() else self.visual_marker_off
        hidden_marker = self.visual_marker_off if self.get_value() else self.visual_marker_on

        # update toggle button position
        show_marker.visible = True
        hidden_marker.visible = False

        self.update_texture()

    @staticmethod
    def create_transformed_texture(diffuse_tex_filename, diffuse_tex_filename_transformed):
        # make the texture 1.5x brighter
        brighten_texture(diffuse_tex_filename, diffuse_tex_filename_transformed, brightness=1.5)

    @property
    def settable(self):
        return True

    # For this state, we simply store its value and the robot_can_toggle steps.
    def _dump_state(self):
        return OrderedDict(value=self.value, hand_in_marker_steps=self.robot_can_toggle_steps)

    def _load_state(self, state):
        # Nothing special to do here when initialized vs. uninitialized
        self.value = state["value"]
        self.robot_can_toggle_steps = state["hand_in_marker_steps"]

    @classmethod
    def _serialize(cls, state):
        return np.array([state["value"], state["hand_in_marker_steps"]])

    @classmethod
    def _deserialize(cls, state):
        return OrderedDict(value=state[0], hand_in_marker_steps=state[1]), 2
