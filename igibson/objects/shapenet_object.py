import numpy as np


from igibson.objects.stateful_object import StatefulObject


class ShapeNetObject(StatefulObject):
    """
    ShapeNet object
    Reference: https://www.shapenet.org/
    """

    def __init__(self, path, scale=1.0, position=[0, 0, 0], orientation=[0, 0, 0], **kwargs):
        super(ShapeNetObject, self).__init__(**kwargs)
        self.filename = path
        self.scale = scale
        self.position = position
        self.orientation = orientation

        self._default_mass = 3.0
        self._default_transform = {
            "position": [0, 0, 0],
            "orientation_quat": [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)],
        }
        pose = p.multiplyTransforms(
            positionA=self.position,
            orientationA=p.getQuaternionFromEuler(self.orientation),
            positionB=self._default_transform["position"],
            orientationB=self._default_transform["orientation_quat"],
        )
        self.pose = {
            "position": pose[0],
            "orientation_quat": pose[1],
        }

    def _load(self, simulator):
        """
        Load the object into pybullet
        """
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.filename, meshScale=self.scale)
        body_id = p.createMultiBody(
            basePosition=self.pose["position"],
            baseOrientation=self.pose["orientation_quat"],
            baseMass=self._default_mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=-1,
        )
        simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params)

        return [body_id]
