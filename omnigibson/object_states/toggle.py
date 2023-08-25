import numpy as np

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.prims.geom_prim import VisualGeomPrim
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
from omni.isaac.core.utils.bounds import recompute_extents
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import create_primitive_mesh
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysicsSchemaTools, UsdGeom, Sdf, Gf, Vt


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.TOGGLE_LINK_PREFIX = "togglebutton"
m.DEFAULT_SCALE = 0.1
m.CAN_TOGGLE_STEPS = 5


class ToggledOn(AbsoluteObjectState, BooleanStateMixin, LinkBasedStateMixin, UpdateStateMixin):
    def __init__(self, obj, scale=None):
        self.scale = scale
        self.value = False
        self.robot_can_toggle_steps = 0
        self.visual_marker = None
        self._check_overlap = None
        self._robot_link_paths = None

        # We also generate the function for checking overlaps at runtime

        super().__init__(obj)

    @classproperty
    def metalink_prefix(cls):
        return m.TOGGLE_LINK_PREFIX

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()
        mesh_prim_path = f"{self.link.prim_path}/mesh_0"
        pre_existing_mesh = get_prim_at_path(mesh_prim_path)
        # Create a primitive mesh if it doesn't already exist
        if not pre_existing_mesh:
            self.scale = m.DEFAULT_SCALE if self.scale is None else self.scale
            # Note: We have to create a mesh (instead of a sphere shape) because physx complains about non-uniform
            # scaling for non-meshes
            mesh = create_primitive_mesh(
                prim_path=mesh_prim_path,
                primitive_type="Sphere",
                extents=1.0,
            )
        else:
            # Infer radius from mesh if not specified as an input
            recompute_extents(prim=pre_existing_mesh)
            self.scale = np.array(pre_existing_mesh.GetAttribute("xformOp:scale").Get())

        # Create the visual geom instance referencing the generated mesh prim
        self.visual_marker = VisualGeomPrim(prim_path=mesh_prim_path, name=f"{self.obj.name}_visual_marker")
        self.visual_marker.scale = self.scale
        self.visual_marker.initialize()
        self.visual_marker.visible = True

        # Make sure the marker isn't translated at all
        self.visual_marker.set_local_pose(translation=np.zeros(3), orientation=np.array([0, 0, 0, 1.0]))

        # Store the projection mesh's IDs
        projection_mesh_ids = PhysicsSchemaTools.encodeSdfPath(self.visual_marker.prim_path)

        # Define function for checking overlap
        valid_hit = False

        def overlap_callback(hit):
            nonlocal valid_hit
            valid_hit = hit.rigid_body in self._robot_link_paths
            # Continue traversal only if we don't have a valid hit yet
            return not valid_hit

        def check_overlap():
            nonlocal valid_hit
            valid_hit = False
            og.sim.psqi.overlap_mesh(*projection_mesh_ids, reportFn=overlap_callback)
            return valid_hit

        self._check_overlap = check_overlap

    def _update(self):
        # Avoid circular imports
        from omnigibson.robots.manipulation_robot import ManipulationRobot
        # detect marker and hand interaction
        self._robot_link_paths = set(link.prim_path
                                     for robot in og.sim.scene.robots if isinstance(robot, ManipulationRobot)
                                     for finger_links in robot.finger_links.values()
                                     for link in finger_links)

        # Check overlap
        robot_can_toggle = self._check_overlap() if len(self._robot_link_paths) > 0 else False

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
