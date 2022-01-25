

from igibson.objects.stateful_object import StatefulObject


class Cube(StatefulObject):
    """
    Cube shape primitive
    """

    def __init__(self, pos=[1, 2, 3], dim=[1, 2, 3], visual_only=False, mass=1000, color=[1, 1, 1, 1], **kwargs):
        super(Cube, self).__init__(**kwargs)
        self.basePos = pos
        self.dimension = dim
        self.visual_only = visual_only
        self.mass = mass
        self.color = color

    def _load(self, simulator):
        """
        Load the object into pybullet
        """
        baseOrientation = [0, 0, 0, 1]
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.dimension)
        visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=self.dimension, rgbaColor=self.color)
        if self.visual_only:
            body_id = p.createMultiBody(baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId)
        else:
            body_id = p.createMultiBody(
                baseMass=self.mass, baseCollisionShapeIndex=colBoxId, baseVisualShapeIndex=visualShapeId
            )

        p.resetBasePositionAndOrientation(body_id, self.basePos, baseOrientation)

        simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params)

        return [body_id]
