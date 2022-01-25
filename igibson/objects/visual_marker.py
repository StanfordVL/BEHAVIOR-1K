

from igibson.objects.object_base import BaseObject


class VisualMarker(BaseObject):
    """
    Visual shape created with shape primitives
    """

    DEFAULT_RENDERING_PARAMS = {
        "use_pbr": False,
        "use_pbr_mapping": False,
        "shadow_caster": False,
    }

    def __init__(
        self,
        visual_shape=p.GEOM_SPHERE,
        rgba_color=[1, 0, 0, 0.5],
        radius=1.0,
        half_extents=[1, 1, 1],
        length=1,
        initial_offset=[0, 0, 0],
        filename=None,
        scale=[1.0] * 3,
        **kwargs
    ):
        """
        create a visual shape to show in pybullet and MeshRenderer

        :param visual_shape: pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER, pybullet.GEOM_CAPSULE or pybullet.GEOM_SPHERE
        :param rgba_color: color
        :param radius: radius (for sphere)
        :param half_extents: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param length: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param initial_offset: visualFramePosition for the marker
        :param filename: mesh file name for p.GEOM_MESH
        :param scale: scale for p.GEOM_MESH
        """
        super(VisualMarker, self).__init__(**kwargs)
        self.visual_shape = visual_shape
        self.rgba_color = rgba_color
        self.radius = radius
        self.half_extents = half_extents
        self.length = length
        self.initial_offset = initial_offset
        self.filename = filename
        self.scale = scale

    def _load(self, simulator):
        """
        Load the object into pybullet
        """
        if self.visual_shape == p.GEOM_MESH:
            shape = p.createVisualShape(self.visual_shape, fileName=self.filename, meshScale=self.scale)
        elif self.visual_shape == p.GEOM_BOX:
            shape = p.createVisualShape(
                self.visual_shape,
                rgbaColor=self.rgba_color,
                halfExtents=self.half_extents,
                visualFramePosition=self.initial_offset,
            )
        elif self.visual_shape in [p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
            shape = p.createVisualShape(
                self.visual_shape,
                rgbaColor=self.rgba_color,
                radius=self.radius,
                length=self.length,
                visualFramePosition=self.initial_offset,
            )
        else:
            shape = p.createVisualShape(
                self.visual_shape,
                rgbaColor=self.rgba_color,
                radius=self.radius,
                visualFramePosition=self.initial_offset,
            )
        body_id = p.createMultiBody(
            baseVisualShapeIndex=shape, baseCollisionShapeIndex=-1, flags=p.URDF_ENABLE_SLEEPING
        )

        simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params)

        return [body_id]

    def set_color(self, color):
        """
        Set the color of the marker

        :param color: normalized rgba color
        """
        p.changeVisualShape(self.get_body_ids()[0], -1, rgbaColor=color)

    def force_sleep(self, body_id=None):
        if body_id is None:
            body_id = self.get_body_ids()[0]

        activationState = p.ACTIVATION_STATE_SLEEP + p.ACTIVATION_STATE_DISABLE_WAKEUP
        p.changeDynamics(body_id, -1, activationState=activationState)

    def set_position(self, pos):
        self.force_wakeup()
        super(VisualMarker, self).set_position(pos)

    def set_orientation(self, orn):
        self.force_wakeup()
        super(VisualMarker, self).set_orientation(orn)

    def set_position_orientation(self, pos, orn):
        self.force_wakeup()
        super(VisualMarker, self).set_position_orientation(pos, orn)
