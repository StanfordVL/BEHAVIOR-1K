import logging
import os

import numpy as np

from igibson.scenes.scene_base import Scene
from igibson.utils.constants import SemanticClass
from igibson.utils.utils import l2_distance


class EmptyScene(Scene):
    """
    An empty scene for debugging.
    """

    def __init__(
            self,
            floor_plane_visible=False,
            floor_plane_color=(1.0, 1.0, 1.0),
    ):
        super(EmptyScene, self).__init__()
        self.floor_plane_visible = floor_plane_visible
        self.floor_plane_color = floor_plane_color

    def _load(self, simulator):
        # Load ground plane
        self.add_ground_plane(color=self.floor_plane_color, visible=self.floor_plane_visible)

    def get_random_point(self, floor=None):
        """
        Get a random point in the region of [-5, 5] x [-5, 5].
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
        Get a trivial shortest path because the scene is empty.
        """
        logging.warning("WARNING: trying to compute the shortest path in EmptyScene (assuming empty space)")
        shortest_path = np.stack((source_world, target_world))
        geodesic_distance = l2_distance(source_world, target_world)
        return shortest_path, geodesic_distance
