import logging
import os
import pickle
import sys
from abc import ABCMeta

import cv2
import networkx as nx
import numpy as np
from future.utils import with_metaclass
from PIL import Image

from igibson.scenes.scene_base import Scene
from igibson.maps.traversable_map import TraversableMap
from igibson.utils.utils import l2_distance


class TraversableScene(Scene, metaclass=ABCMeta):
    """
    Traversable scene class.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(
        self,
        scene_id,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_with_objects=True,
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
    ):
        """
        Load a traversable scene and compute traversability

        :param scene_id: Scene id
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_with_objects: whether to use objects or not when constructing graph
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        """
        super().__init__()
        logging.info("TraversableScene model: {}".format(scene_id))
        self.scene_id = scene_id

        # Create traversable map
        self._trav_map = TraversableMap(
            trav_map_resolution=trav_map_resolution,
            trav_map_erosion=trav_map_erosion,
            trav_map_with_objects=trav_map_with_objects,
            build_graph=build_graph,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
        )

    @property
    def trav_map(self):
        """
        Returns:
            TraversableMap: Map for computing connectivity between nodes for this scene
        """
        return self._trav_map

    @property
    def has_connectivity_graph(self):
        # Connectivity graph is determined by travserable map
        return self._trav_map.build_graph

    def get_random_point(self, floor=None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.

        :param floor: floor number
        :return floor: floor number
        :return point: randomly sampled point in [x, y, z]
        """
        return self._trav_map.get_random_point(floor=floor)

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get the shortest path from one point to another point.
        If any of the given point is not in the graph, add it to the graph and
        create an edge between it to its closest node.

        :param floor: floor number
        :param source_world: 2D source location in world reference frame (metric)
        :param target_world: 2D target location in world reference frame (metric)
        :param entire_path: whether to return the entire path
        """
        assert self._trav_map.build_graph, "cannot get shortest path without building the graph"

        return self._trav_map.get_shortest_path(
            floor=floor,
            source_world=source_world,
            target_world=target_world,
            entire_path=entire_path,
        )