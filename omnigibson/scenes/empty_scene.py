import logging

import numpy as np

from omnigibson.scenes.scene_base import Scene
from omnigibson.objects.primitive_object import PrimitiveObject
import omnigibson.utils.transform_utils as T


class EmptyScene(Scene):
    """
    An empty scene for debugging.
    """

    def __init__(
            self,
            floor_plane_visible=True,
            floor_plane_color=(1.0, 1.0, 1.0),
    ):
        self.floor_plane_visible = floor_plane_visible
        self.floor_plane_color = floor_plane_color
        self.ground_plane = None

        # Run super
        super().__init__()


    def _load(self, simulator):
        # Initialize systems
        self.initialize_systems(simulator)

        # Load ground plane
        self.ground_plane = PrimitiveObject(
            prim_path="/World/ground_plane",
            name="ground_plane",
            primitive_type="Cube",
            category="floors",
            model="ground_plane",
            scale=[2500, 2500, 0.05],
            visible=self.floor_plane_visible,
            fixed_base=True,
            rgba=[*self.floor_plane_color, 1.0],
        )

        simulator.import_object(self.ground_plane)
        self.ground_plane.set_position([0, 0, -self.ground_plane.scale[2] / 2.0])

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
        geodesic_distance = T.l2_distance(source_world, target_world)
        return shortest_path, geodesic_distance
