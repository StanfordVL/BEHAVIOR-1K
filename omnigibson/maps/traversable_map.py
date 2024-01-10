import os
import pickle
import sys
import heapq

import cv2
import networkx as nx
import numpy as np
from PIL import Image

from omnigibson.maps.map_base import BaseMap
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import create_module_logger

import matplotlib.pyplot as plt

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

            # TODO: do we still need this parameter?
            """
            # We search for the largest connected areas
            if self.build_graph:
                # Directly set map size
                self.floor_graph = self.build_trav_graph(map_size, maps_path, floor, trav_map)
            """

            self.floor_map.append(trav_map)

        return map_size

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
    
    def astar(self, trav_map, start, goal):
        def heuristic(node):
            # Calculate the Euclidean distance from node to goal
            return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
        
        def get_neighbors(cell):
            # Define neighbors of cell
            return [(cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]), (cell[0], cell[1] + 1), (cell[0], cell[1] - 1), 
                    (cell[0] + 1, cell[1] + 1), (cell[0] - 1, cell[1] - 1), (cell[0] + 1, cell[1] - 1), (cell[0] - 1, cell[1] + 1)]
        
        def is_valid(cell):
            # Check if cell is within the map and traversable
            return (0 <= cell[0] < trav_map.shape[0] and
                    0 <= cell[1] < trav_map.shape[1] and
                    trav_map[cell] != 0)

        def cost(cell1, cell2):
            # Define the cost of moving from cell1 to cell2
            # Return 1 for adjacent cells and square root of 2 for diagonal cells in an 8-connected grid.
            if cell1[0] == cell2[0] or cell1[1] == cell2[1]:
                return 1
            else:
                return np.sqrt(2)

        open_set = [(0, start)]
        came_from = {}
        visited = set()
        g_score = {cell: float('inf') for cell in np.ndindex(trav_map.shape)}
        g_score[start] = 0

        while open_set:
            _, current = heapq.heappop(open_set)

            visited.add(current)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.insert(0, current)
                    current = came_from[current]
                path.insert(0, start)
                return np.array(path)

            for neighbor in get_neighbors(current):
                # Skip neighbors that are not valid or have already been visited
                if not is_valid(neighbor) or neighbor in visited:
                    continue
                tentative_g_score = g_score[current] + cost(current, neighbor)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))

        # throw error
        raise ValueError('No path found')

    def get_closest_traversible_point(self, trav_map, point):
        """
        Get the closest traversible point to the given point in map frame

        Args:
            trav_map (np.ndarray): traversibility map
            point (2-array): (x,y) 2D point in world reference frame (metric)

        Returns:
            2-array: (x,y) 2D point in world reference frame (metric)
        """
        trav_map = trav_map.copy()
        trav_map[trav_map == 0] = 255
        trav_map[point[0], point[1]] = 0
        trav_map = cv2.erode(trav_map, np.ones((self.trav_map_erosion, self.trav_map_erosion)))
        trav_map[trav_map < 255] = 0
        trav_space = np.where(trav_map == 255)
        idx = np.argmin(np.linalg.norm(np.array(trav_space).T - point, axis=1))
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        return xy_map

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
        real_source_map = tuple(self.world_to_map(source_world))
        real_target_map = tuple(self.world_to_map(target_world))

        source_map = real_source_map
        target_map = real_target_map

        trav_map = self.floor_map[floor]

        if trav_map[source_map] == 0:
            # find the closest 255 cell in the map to the source
            source_map = self.get_closest_traversible_point(trav_map, source_map)
        if trav_map[target_map] == 0:
            # find the closest 255 cell in the map to the target
            target_map = self.get_closest_traversible_point(trav_map, target_map)
        
        def visualize_map(trav_map):
            plt.imshow(trav_map, cmap='gray')
            plt.scatter(source_map[1], source_map[0], color='blue')  # Plot source in blue
            plt.scatter(target_map[1], target_map[0], color='red')  # Plot goal in red
            plt.colorbar()
            plt.show()
        
        visualize_map(trav_map)
                
        path_map = self.astar(trav_map, source_map, target_map)
        # attach the real source and target to the path
        path_map = np.concatenate((np.array([real_source_map]), path_map, np.array([real_target_map])), axis=0)
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
