

from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.object_states.texture_change_state_mixin import TextureChangeStateMixin
from igibson.objects.visual_marker import VisualMarker
from igibson.utils.constants import PyBulletSleepState, SemanticClass, SimulatorMode
from igibson.utils.utils import brighten_texture

_TOGGLE_DISTANCE_THRESHOLD = 0.1
_TOGGLE_LINK_NAME = "toggle_button"
_TOGGLE_BUTTON_RADIUS = 0.05
_TOGGLE_MARKER_OFF_POSITION = [0, 0, -100]
_CAN_TOGGLE_STEPS = 5


class ToggledOn(AbsoluteObjectState, BooleanState, LinkBasedStateMixin, TextureChangeStateMixin):
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
            self.visual_marker_on = VisualMarker(
                rgba_color=[0, 1, 0, 0.5],
                radius=_TOGGLE_BUTTON_RADIUS,
                class_id=SemanticClass.TOGGLE_MARKER,
                rendering_params={"use_pbr": True, "use_pbr_mapping": True},
            )
            self.visual_marker_off = VisualMarker(
                rgba_color=[1, 0, 0, 0.5],
                radius=_TOGGLE_BUTTON_RADIUS,
                class_id=SemanticClass.TOGGLE_MARKER,
                rendering_params={"use_pbr": True, "use_pbr_mapping": True},
            )
            self.simulator.import_object(self.visual_marker_on)
            self.visual_marker_on.set_position(_TOGGLE_MARKER_OFF_POSITION)
            self.simulator.import_object(self.visual_marker_off)
            self.visual_marker_off.set_position(_TOGGLE_MARKER_OFF_POSITION)

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

        # swap two types of markers when toggled
        # when hud overlay is on, we show the toggle buttons, otherwise the buttons are hidden
        if self.simulator.mode == SimulatorMode.VR:
            hud_overlay_show_state = self.simulator.get_hud_show_state()
        else:
            hud_overlay_show_state = False

        # Choose which marker to put on object vs which to put away
        show_marker = self.visual_marker_on if self.get_value() else self.visual_marker_off
        hidden_marker = self.visual_marker_off if self.get_value() else self.visual_marker_on

        # update toggle button position depending if parent is awake
        dynamics_info = p.getDynamicsInfo(self.body_id, -1)

        if len(dynamics_info) == 13:
            activation_state = dynamics_info[12]
        else:
            activation_state = PyBulletSleepState.AWAKE

        if activation_state in [PyBulletSleepState.AWAKE, PyBulletSleepState.ISLAND_AWAKE]:
            show_marker.set_position(button_position_on_object)
            hidden_marker.set_position(button_position_on_object)

        if hud_overlay_show_state:
            for instance in show_marker.renderer_instances:
                instance.hidden = False
        else:
            for instance in show_marker.renderer_instances:
                instance.hidden = True

        for instance in hidden_marker.renderer_instances:
            instance.hidden = True

        self.update_texture()

    @staticmethod
    def create_transformed_texture(diffuse_tex_filename, diffuse_tex_filename_transformed):
        # make the texture 1.5x brighter
        brighten_texture(diffuse_tex_filename, diffuse_tex_filename_transformed, brightness=1.5)

    # For this state, we simply store its value and the robot_can_toggle steps.
    def _dump(self):
        return {"value": self.value, "hand_in_marker_steps": self.robot_can_toggle_steps}

    def load(self, data):
        # Nothing special to do here when initialized vs. uninitialized
        self.value = data["value"]
        self.robot_can_toggle_steps = data["hand_in_marker_steps"]
