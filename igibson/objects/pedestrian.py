import os



import igibson
from igibson.objects.stateful_object import StatefulObject


class Pedestrian(StatefulObject):
    """
    Pedestiran object
    """

    def __init__(self, style="standing", pos=[0, 0, 0], **kwargs):
        super(Pedestrian, self).__init__(**kwargs)
        self.collision_filename = os.path.join(
            igibson.assets_path, "models", "person_meshes", "person_{}".format(style), "meshes", "person_vhacd.obj"
        )
        self.visual_filename = os.path.join(
            igibson.assets_path, "models", "person_meshes", "person_{}".format(style), "meshes", "person.obj"
        )
        self.cid = None
        self.pos = pos

    def _load(self, simulator):
        """
        Load the object into pybullet
        """
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.collision_filename)
        visual_id = p.createVisualShape(p.GEOM_MESH, fileName=self.visual_filename)
        body_id = p.createMultiBody(
            basePosition=[0, 0, 0], baseMass=60, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id
        )
        p.resetBasePositionAndOrientation(body_id, self.pos, [-0.5, -0.5, -0.5, 0.5])
        self.cid = p.createConstraint(
            body_id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            self.pos,
            parentFrameOrientation=[-0.5, -0.5, -0.5, 0.5],
        )  # facing x axis

        simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params)

        return [body_id]

    def reset_position_orientation(self, pos, orn):
        """
        Reset pedestrian position and orientation by changing constraint
        """
        p.changeConstraint(self.cid, pos, orn)
