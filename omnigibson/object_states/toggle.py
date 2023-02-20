import numpy as np

from omnigibson.macros import create_module_macros
from omnigibson.prims.geom_prim import VisualGeomPrim
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
from omnigibson.utils.usd_utils import create_primitive_mesh
from omni.isaac.core.utils.prims import get_prim_at_path


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.TOGGLE_DISTANCE_THRESHOLD = 0.1
m.TOGGLE_LINK_NAME = "togglebutton_link"
m.TOGGLE_BUTTON_SCALE = 0.05
m.CAN_TOGGLE_STEPS = 5


class ToggledOn(AbsoluteObjectState, BooleanState, LinkBasedStateMixin, UpdateStateMixin):
    def __init__(self, obj):
        self.value = False
        self.robot_can_toggle_steps = 0
        self.visual_marker = None
        super(ToggledOn, self).__init__(obj)

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    @staticmethod
    def get_state_link_name():
        return m.TOGGLE_LINK_NAME

    def _initialize(self):
        super(ToggledOn, self)._initialize()
        if self.initialize_link_mixin():
            mesh_prim_path = f"{self.link.prim_path}/visual_marker"
            # Create a primitive mesh if it doesn't already exist
            if not get_prim_at_path(mesh_prim_path):
                mesh = create_primitive_mesh(
                    prim_path=mesh_prim_path,
                    primitive_type="Sphere",
                    extents=m.TOGGLE_BUTTON_SCALE,
                )

            # Create the visual geom instance referencing the generated mesh prim
            self.visual_marker = VisualGeomPrim(prim_path=mesh_prim_path, name=f"{self.obj.name}_visual_marker")
            self.visual_marker.initialize()

            # Make sure the marker isn't translated at all
            self.visual_marker.set_local_pose(translation=np.zeros(3), orientation=np.array([0, 0, 0, 1.0]))

    def _update(self):
        button_position_on_object = self.get_link_position()
        if button_position_on_object is None:
            return

        robot_can_toggle = False
        # detect marker and hand interaction
        for robot in self._simulator.scene.robots:
            robot_can_toggle = robot.can_toggle(button_position_on_object, m.TOGGLE_DISTANCE_THRESHOLD)
            if robot_can_toggle:
                break

        if robot_can_toggle:
            self.robot_can_toggle_steps += 1
        else:
            self.robot_can_toggle_steps = 0

        if self.robot_can_toggle_steps == m.CAN_TOGGLE_STEPS:
            self.value = not self.value

        # Choose which color to apply to the toggle marker
        self.visual_marker.color = np.array([0, 1.0, 0]) if self.get_value() else np.array([1.0, 0, 0])

    @staticmethod
    def get_texture_change_params():
        # By default, it keeps the original albedo unchanged.
        albedo_add = 0.0
        diffuse_tint = (1.0, 1.0, 1.0)
        return albedo_add, diffuse_tint

    @property
    def state_size(self):
        return 2

    # For this state, we simply store its value and the robot_can_toggle steps.
    def _dump_state(self):
        return dict(value=self.value, hand_in_marker_steps=self.robot_can_toggle_steps)

    def _load_state(self, state):
        # Nothing special to do here when initialized vs. uninitialized
        self.value = state["value"]
        self.robot_can_toggle_steps = state["hand_in_marker_steps"]

    def _serialize(self, state):
        return np.array([state["value"], state["hand_in_marker_steps"]], dtype=float)

    def _deserialize(self, state):
        return dict(value=bool(state[0]), hand_in_marker_steps=int(state[1])), 2
