import math
import os

import cv2
import torch as th

from omnigibson.maps.map_base import BaseMap
from omnigibson.utils.motion_planning_utils import astar
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
        default_erosion_radius=0.0,
        trav_map_with_objects=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
    ):
        """
        Args:
            map_resolution (float): map resolution in meters, each pixel represents this many meters;
                                    normally, this should be between 0.01 and 0.1
            default_erosion_radius (float): default map erosion radius in meters
            trav_map_with_objects (bool): whether to use objects or not when constructing graph
            num_waypoints (int): number of way points returned
            waypoint_resolution (float): resolution of adjacent way points
        """
        # Set internal values
        self.map_default_resolution = 0.01  # each pixel == 0.01m in the dataset representation
        self.default_erosion_radius = default_erosion_radius
        self.trav_map_with_objects = trav_map_with_objects
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
                trav_map = th.tensor(
                    cv2.imread(os.path.join(maps_path, "floor_trav_{}.png".format(floor)), cv2.IMREAD_GRAYSCALE)
                )
            else:
                trav_map = th.tensor(
                    cv2.imread(os.path.join(maps_path, "floor_trav_no_obj_{}.png".format(floor)), cv2.IMREAD_GRAYSCALE)
                )

            # If we do not initialize the original size of the traversability map, we obtain it from the image
            # Then, we compute the final map size as the factor of scaling (default_resolution/resolution) times the
            # original map size
            if self.trav_map_original_size is None:
                height, width = trav_map.shape
                assert height == width, "trav map is not a square"
                self.trav_map_original_size = height
                map_size = int(self.trav_map_original_size * self.map_default_resolution / self.map_resolution)

            # We resize the traversability map to the new size computed before
            trav_map = th.tensor(cv2.resize(trav_map.cpu().numpy(), (map_size, map_size)))

            # We make the pixels of the image to be either 0 or 255
            trav_map[trav_map < 255] = 0

            self.floor_map.append(trav_map)

        return map_size

    @property
    def n_floors(self):
        """
        Returns:
            int: Number of floors belonging to this map's associated scene
        """
        return len(self.floor_heights)

    def _erode_trav_map(self, trav_map, robot=None):
        # Erode the traversability map to account for the robot's size
        if robot:
            robot_chassis_extent = robot.reset_joint_pos_aabb_extent[:2]
            radius = th.norm(robot_chassis_extent) / 2.0 + 0.2
        else:
            radius = self.default_erosion_radius
        radius_pixel = int(math.ceil(radius / self.map_resolution))
        trav_map = th.tensor(cv2.erode(trav_map.cpu().numpy(), th.ones((radius_pixel, radius_pixel)).cpu().numpy()))
        return trav_map

    def get_random_point(self, floor=None, reference_point=None, robot=None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.
        If @reference_point is given, sample a point in the same connected component as the previous point.

        Args:
            floor (None or int): floor number. None means the floor is randomly sampled
                                 Warning: if @reference_point is given, @floor must be given;
                                          otherwise, this would lead to undefined behavior
            reference_point (3-array): (x,y,z) if given, sample a point in the same connected component as this point

        Returns:
            2-tuple:
                - int: floor number. This is the sampled floor number if @floor is None
                - 3-array: (x,y,z) randomly sampled point
        """
        if reference_point is not None:
            assert floor is not None, "floor must be given if reference_point is given"

        # If nothing is given, sample a random floor and a random point on that floor
        if floor is None and reference_point is None:
            floor = th.randint(0, self.n_floors)

        # create a deep copy so that we don't erode the original map
        trav_map = th.clone(self.floor_map[floor])
        trav_map = self._erode_trav_map(trav_map, robot=robot)

        if reference_point is not None:
            # Find connected component
            _, component_labels = cv2.connectedComponents(trav_map.cpu().numpy(), connectivity=4)
            component_labels = th.tensor(component_labels)

            # If previous point is given, sample a point in the same connected component
            prev_xy_map = self.world_to_map(reference_point[:2])
            prev_label = component_labels[prev_xy_map[0]][prev_xy_map[1]]
            trav_space = th.where(component_labels == prev_label)
        else:
            trav_space = th.where(trav_map == 255)
        idx = th.randint(0, high=trav_space[0].shape[0], size=(1,)).item()
        xy_map = th.tensor([trav_space[0][idx], trav_space[1][idx]])
        x, y = self.map_to_world(xy_map)
        z = self.floor_heights[floor]
        return floor, th.tensor([x, y, z])

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False, robot=None):
        """
        Get the shortest path from one point to another point.
        If any of the given point is not in the graph, add it to the graph and
        create an edge between it to its closest node.

        Args:
            floor (int): floor number
            source_world (2-array): (x,y) 2D source location in world reference frame (metric)
            target_world (2-array): (x,y) 2D target location in world reference frame (metric)
            entire_path (bool): whether to return the entire path
            robot (None or BaseRobot): if given, erode the traversability map to account for the robot's size

        Returns:
            2-tuple:
                - (N, 2) array: array of path waypoints, where N is the number of generated waypoints
                - float: geodesic distance of the path
        """
        source_map = tuple(self.world_to_map(source_world).tolist())
        target_map = tuple(self.world_to_map(target_world).tolist())

        # create a deep copy so that we don't erode the original map
        trav_map = th.clone(self.floor_map[floor])

        trav_map = self._erode_trav_map(trav_map, robot=robot)

        path_map = astar(trav_map, source_map, target_map)
        if path_map is None:
            # No traversable path found
            return None, None
        path_world = self.map_to_world(path_map)
        geodesic_distance = th.sum(th.norm(path_world[1:] - path_world[:-1], dim=1))
        path_world = path_world[:: self.waypoint_interval]

        if not entire_path:
            path_world = path_world[: self.num_waypoints]
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]
            if num_remaining_waypoints > 0:
                remaining_waypoints = th.tile(target_world, (num_remaining_waypoints, 1))
                path_world = th.cat((path_world, remaining_waypoints), dim=0)

        return path_world, geodesic_distance
