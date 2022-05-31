import logging
import os

import numpy as np

import pybullet_data

from igibson.scenes.scene_base import Scene
from igibson.utils.constants import SemanticClass
from igibson.utils.utils import l2_distance


class StadiumScene(Scene):
    """
    A simple stadium scene for debugging
    """

    def __init__(self):
        super(StadiumScene, self).__init__()
        self.objects = []

    def _load(self, simulator):
        """
        Load the scene into pybullet
        """
        filename = os.path.join(pybullet_data.getDataPath(), "stadium_no_collision.sdf")
        self.stadium = p.loadSDF(filename)
        plane_file = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.floor_body_ids += [p.loadMJCF(plane_file)[0]]
        pos, orn = p.getBasePositionAndOrientation(self.floor_body_ids[0])
        p.resetBasePositionAndOrientation(self.floor_body_ids[0], [pos[0], pos[1], pos[2] - 0.005], orn)
        p.changeVisualShape(self.floor_body_ids[0], -1, rgbaColor=[1, 1, 1, 0.5])

        # Only load the stadium mesh into the renderer, not the extra floor plane
        for id in list(self.stadium):
            simulator.load_object_in_renderer(None, id, SemanticClass.SCENE_OBJS, use_pbr=False, use_pbr_mapping=False)

        # Load additional objects & merge body IDs
        additional_object_body_ids = [x for obj in self.objects for x in obj.load(simulator)]
        return list(self.stadium) + self.floor_body_ids + additional_object_body_ids

    def get_random_point(self, floor=None):
        """
        Get a random point in the region of [-5, 5] x [-5, 5]
        """
        return floor, np.array(
            [
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                0.0,
            ]
        )

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get a trivial shortest path because the scene is empty
        """
        logging.warning("WARNING: trying to compute the shortest path in StadiumScene (assuming empty space)")
        shortest_path = np.stack((source_world, target_world))
        geodesic_distance = l2_distance(source_world, target_world)
        return shortest_path, geodesic_distance
