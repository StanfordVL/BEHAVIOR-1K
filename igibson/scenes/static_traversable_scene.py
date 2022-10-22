import logging
import os

import numpy as np

from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.prims.geom_prim import CollisionVisualGeomPrim
from omnigibson.utils.asset_utils import get_scene_path
from omnigibson.utils.usd_utils import add_asset_to_stage


class StaticTraversableScene(TraversableScene):
    """
    Static traversable scene class for OmniGibson, where scene is defined by a singular mesh (no intereactable objects).
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(
        self,
        scene_model,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_with_objects=True,
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
        floor_plane_visible=False,
    ):
        """
        Load a building scene and compute traversability

        # TODO: Update
        :param scene_model: Scene model, e.g.: Rs_int
        # TODO: Update doc -- usd_file / usd_path naming convention is too ambiguous / similar
        # :param pybullet_filename: optional specification of which pybullet file to restore after initialization
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_with_objects: whether to use objects or not when constructing graph
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        :param floor_plane_visible: whether to render the additionally added floor planes
        """
        # Run super init first
        super().__init__(
            scene_model=scene_model,
            trav_map_resolution=trav_map_resolution,
            trav_map_erosion=trav_map_erosion,
            trav_map_with_objects=trav_map_with_objects,
            build_graph=build_graph,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
        )

        # Store and initialize additional variables
        self.floor_plane_visible = floor_plane_visible
        self._floor_heights = None
        self._scene_mesh = None

    def _load(self, simulator):
        # Load the scene mesh (use downsampled one if available)
        filename = os.path.join(get_scene_path(self.scene_model), "mesh_z_up_downsampled.obj")
        if not os.path.isfile(filename):
            filename = os.path.join(get_scene_path(self.scene_model), "mesh_z_up.obj")

        scene_prim = add_asset_to_stage(
            asset_path=filename,
            prim_path=f"/World/scene_{self.scene_model}",
        )

        # Grab the actual mesh prim
        self._scene_mesh = CollisionVisualGeomPrim(
            prim_path=f"/World/scene_{self.scene_model}/mesh_z_up/{self.scene_model}_mesh_texture",
            name=f"{self.scene_model}_mesh",
        )

        # Load floor metadata
        floor_height_path = os.path.join(get_scene_path(self.scene_model), "floors.txt")
        assert os.path.isfile(floor_height_path), f"floor_heights.txt cannot be found in model: {self.scene_model}"
        with open(floor_height_path, "r") as f:
            self.floor_heights = sorted(list(map(float, f.readlines())))
            logging.debug("Floors {}".format(self.floor_heights))

        # Load in additional floor planes, setting it to the first floor by default
        self.add_ground_plane(visible=self.floor_plane_visible)
        self.move_floor_plane(floor=0)

        # Filter the collision between the scene mesh and the floor plane
        self._scene_mesh.add_filtered_collision_pair(prim=self._floor_plane)

        # Load the traversability map
        self._trav_map.load_trav_map(get_scene_path(self.scene_model))

        # Initialize omnigibson systems
        self.initialize_systems(simulator)

    def move_floor_plane(self, floor=0, additional_elevation=0.02, height=None):
        """
        Resets the floor plane to a new floor

        :param floor: Integer identifying the floor to move the floor plane to
        :param additional_elevation: Additional elevation with respect to the height of the floor
        :param height: Alternative parameter to control directly the height of the ground plane
        """
        height = height if height is not None else self.floor_heights[floor] + additional_elevation
        self._floor_plane.set_position(np.array([0, 0, height]))

    def get_floor_height(self, floor=0):
        """
        Return the current floor height (in meter)

        :return: current floor height
        """
        return self.floor_heights[floor]

    @property
    def n_floors(self):
        return len(self._floor_heights)
