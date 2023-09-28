from omnigibson.scenes.scene_base import Scene
from omnigibson.maps.traversable_map import TraversableMap
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class TraversableScene(Scene):
    """
    Traversable scene class.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(
        self,
        scene_model,
        scene_file=None,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_with_objects=True,
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
        use_floor_plane=True,
        floor_plane_visible=True,
        floor_plane_color=(1.0, 1.0, 1.0),
    ):
        """
        Args:
            scene_model (str): Scene model name, e.g.: Adrian or Rs_int
            scene_file (None or str): If specified, full path of JSON file to load (with .json).
                None results in no additional objects being loaded into the scene
            trav_map_resolution (float): traversability map resolution
            trav_map_erosion (float): erosion radius of traversability areas, should be robot footprint radius
            trav_map_with_objects (bool): whether to use objects or not when constructing graph
            build_graph (bool): build connectivity graph
            num_waypoints (int): number of way points returned
            waypoint_resolution (float): resolution of adjacent way points
            use_floor_plane (bool): whether to load a flat floor plane into the simulator
            floor_plane_visible (bool): whether to render the additionally added floor plane
            floor_plane_color (3-array): if @floor_plane_visible is True, this determines the (R,G,B) color assigned
                to the generated floor plane
        """
        log.info("TraversableScene model: {}".format(scene_model))
        self.scene_model = scene_model

        # Create traversable map
        self._trav_map = TraversableMap(
            map_resolution=trav_map_resolution,
            trav_map_erosion=trav_map_erosion,
            trav_map_with_objects=trav_map_with_objects,
            build_graph=build_graph,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
        )
        # Run super init
        super().__init__(
            scene_file=scene_file,
            use_floor_plane=use_floor_plane,
            floor_plane_visible=floor_plane_visible,
            floor_plane_color=floor_plane_color,
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
        return self._trav_map.get_random_point(floor=floor)

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        assert self._trav_map.build_graph, "cannot get shortest path without building the graph"

        return self._trav_map.get_shortest_path(
            floor=floor,
            source_world=source_world,
            target_world=target_world,
            entire_path=entire_path,
        )
