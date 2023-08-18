import os
import pickle
import sys

import cv2
import networkx as nx
import numpy as np
from PIL import Image

from omnigibson.maps.map_base import BaseMap
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class TraversableMap(BaseMap):
    """
    Traversable scene class.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(
        self,
        map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_with_objects=True,
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
    ):
        """
        Args:
            map_resolution (float): map resolution
            trav_map_erosion (float): erosion radius of traversability areas, should be robot footprint radius
            trav_map_with_objects (bool): whether to use objects or not when constructing graph
            build_graph (bool): build connectivity graph
            num_waypoints (int): number of way points returned
            waypoint_resolution (float): resolution of adjacent way points
        """
        # Set internal values
        self.map_default_resolution = 0.01  # each pixel represents 0.01m
        self.trav_map_erosion = trav_map_erosion
        self.trav_map_with_objects = trav_map_with_objects
        self.build_graph = build_graph
        self.num_waypoints = num_waypoints
        self.waypoint_interval = int(waypoint_resolution / map_resolution)

        # Values loaded at runtime
        self.trav_map_original_size = None
        self.trav_map_size = None
        self.mesh_body_id = None
        self.floor_heights = None
        self.floor_map = None
        self.floor_graph = None

        # Run super method
        super().__init__(map_resolution=map_resolution)

    def _load_map(self, maps_path, floor_heights=(0.0,)):
        """
        Loads the traversability maps for all floors

        Args:
            maps_path (str): Path to the folder containing the traversability maps
            floor_heights (n-array): Height(s) of the floors for this map

        Returns:
            int: Size of the loaded map
        """
        if not os.path.exists(maps_path):
            log.warning("trav map does not exist: {}".format(maps_path))
            return

        self.floor_heights = floor_heights
        self.floor_map = []
        map_size = None
        for floor in range(len(self.floor_heights)):
            if self.trav_map_with_objects:
                # TODO: Shouldn't this be generated dynamically?
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_{}.png".format(floor))))
            else:
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_obj_{}.png".format(floor))))

            # If we do not initialize the original size of the traversability map, we obtain it from the image
            # Then, we compute the final map size as the factor of scaling (default_resolution/resolution) times the
            # original map size
            if self.trav_map_original_size is None:
                height, width = trav_map.shape
                assert height == width, "trav map is not a square"
                self.trav_map_original_size = height
                map_size = int(
                    self.trav_map_original_size * self.map_default_resolution / self.map_resolution
                )

            # We resize the traversability map to the new size computed before
            trav_map = cv2.resize(trav_map, (map_size, map_size))

            # We then erode the image. This is needed because the code that computes shortest path uses the global map
            # and a point robot
            if self.trav_map_erosion != 0:
                trav_map = cv2.erode(trav_map, np.ones((self.trav_map_erosion, self.trav_map_erosion)))

            # We make the pixels of the image to be either 0 or 255
            trav_map[trav_map < 255] = 0

            # We search for the largest connected areas
            if self.build_graph:
                # Directly set map size
                self.floor_graph = self.build_trav_graph(map_size, maps_path, floor, trav_map)

            self.floor_map.append(trav_map)

        return map_size

    # TODO: refactor into C++ for speedup
    @staticmethod
    def build_trav_graph(map_size, maps_path, floor, trav_map):
        """
        Build traversibility graph and only take the largest connected component

        Args:
            map_size (int): Size of the map being generated
            maps_path (str): Path to the folder containing the traversability maps
            floor (int): floor number
            trav_map ((H, W)-array): traversability map in image form
        """
        floor_graph = []
        graph_file = os.path.join(
            maps_path, "floor_trav_{}_py{}{}.p".format(floor, sys.version_info.major, sys.version_info.minor)
        )
        if os.path.isfile(graph_file):
            log.info("Loading traversable graph")
            with open(graph_file, "rb") as pfile:
                g = pickle.load(pfile)
        else:
            log.info("Building traversable graph")
            g = nx.Graph()
            for i in range(map_size):
                for j in range(map_size):
                    if trav_map[i, j] == 0:
                        continue
                    g.add_node((i, j))
                    # 8-connected graph
                    neighbors = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1), (i - 1, j)]
                    for n in neighbors:
                        if (
                            0 <= n[0] < map_size
                            and 0 <= n[1] < map_size
                            and trav_map[n[0], n[1]] > 0
                        ):
                            g.add_edge(n, (i, j), weight=T.l2_distance(n, (i, j)))

            # only take the largest connected component
            largest_cc = max(nx.connected_components(g), key=len)
            g = g.subgraph(largest_cc).copy()
            with open(graph_file, "wb") as pfile:
                pickle.dump(g, pfile)

        floor_graph.append(g)

        # update trav_map accordingly
        # This overwrites the traversability map loaded before
        # It sets everything to zero, then only sets to one the points where we have graph nodes
        # Dangerous! if the traversability graph is not computed from the loaded map but from a file, it could overwrite
        # it silently.
        trav_map[:, :] = 0
        for node in g.nodes:
            trav_map[node[0], node[1]] = 255

        return floor_graph

    @property
    def n_floors(self):
        """
        Returns:
            int: Number of floors belonging to this map's associated scene
        """
        return len(self.floor_heights)

    def get_random_point(self, floor=None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.

        Args:
            floor (None or int): floor number. None means the floor is randomly sampled

        Returns:
            2-tuple:
                - int: floor number. This is the sampled floor number if @floor is None
                - 3-array: (x,y,z) randomly sampled point
        """
        if floor is None:
            floor = np.random.randint(0, self.n_floors)
        trav = self.floor_map[floor]
        trav_space = np.where(trav == 255)
        idx = np.random.randint(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        x, y = self.map_to_world(xy_map)
        z = self.floor_heights[floor]
        return floor, np.array([x, y, z])

    def has_node(self, floor, world_xy):
        """
        Return whether the traversability graph contains a point

            floor: floor number
            world_xy: 2D location in world reference frame (metric)
        """
        map_xy = tuple(self.world_to_map(world_xy))
        g = self.floor_graph[floor]
        return g.has_node(map_xy)

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get the shortest path from one point to another point.
        If any of the given point is not in the graph, add it to the graph and
        create an edge between it to its closest node.

        Args:
            floor (int): floor number
            source_world (2-array): (x,y) 2D source location in world reference frame (metric)
            target_world (2-array): (x,y) 2D target location in world reference frame (metric)
            entire_path (bool): whether to return the entire path

        Returns:
            2-tuple:
                - (N, 2) array: array of path waypoints, where N is the number of generated waypoints
                - float: geodesic distance of the path
        """
        assert self.build_graph, "cannot get shortest path without building the graph"
        source_map = tuple(self.world_to_map(source_world))
        target_map = tuple(self.world_to_map(target_world))

        g = self.floor_graph[floor]

        if not g.has_node(target_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
            g.add_edge(closest_node, target_map, weight=T.l2_distance(closest_node, target_map))

        if not g.has_node(source_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - source_map, axis=1))])
            g.add_edge(closest_node, source_map, weight=T.l2_distance(closest_node, source_map))

        path_map = np.array(nx.astar_path(g, source_map, target_map, heuristic=T.l2_distance))

        path_world = self.map_to_world(path_map)
        geodesic_distance = np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1))
        path_world = path_world[:: self.waypoint_interval]

        if not entire_path:
            path_world = path_world[: self.num_waypoints]
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]
            if num_remaining_waypoints > 0:
                remaining_waypoints = np.tile(target_world, (num_remaining_waypoints, 1))
                path_world = np.concatenate((path_world, remaining_waypoints), axis=0)

        return path_world, geodesic_distance
