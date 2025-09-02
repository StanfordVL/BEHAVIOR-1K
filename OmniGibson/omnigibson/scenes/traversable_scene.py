from omnigibson.maps.traversable_map import TraversableMap
from omnigibson.scenes.scene_base import Scene
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
        default_erosion_radius=0.0,
        trav_map_with_objects=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
        use_floor_plane=True,
        include_robots=True,
    ):
        """
        Args:
            scene_model (str): Scene model name, e.g.: Adrian or Rs_int
            scene_file (None or str): If specified, full path of JSON file to load (with .json).
                None results in no additional objects being loaded into the scene
            trav_map_resolution (float): traversability map resolution
            default_erosion_radius (float): default map erosion radius in meters
            trav_map_with_objects (bool): whether to use objects or not when constructing graph
            num_waypoints (int): number of way points returned
            waypoint_resolution (float): resolution of adjacent way points
            use_floor_plane (bool): whether to load a flat floor plane into the simulator
            include_robots (bool): whether to also include the robot(s) defined in the scene
        """
        log.info("TraversableScene model: {}".format(scene_model))
        self.scene_model = scene_model

        # Create traversable map
        self._trav_map = TraversableMap(
            map_resolution=trav_map_resolution,
            default_erosion_radius=default_erosion_radius,
            trav_map_with_objects=trav_map_with_objects,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
        )
        # Run super init
        super().__init__(
            scene_file=scene_file,
            use_floor_plane=use_floor_plane,
            include_robots=include_robots,
        )

    @property
    def trav_map(self):
        """
        Returns:
            TraversableMap: Map for computing connectivity between nodes for this scene
        """
        return self._trav_map

    def get_random_point(self, floor=None, reference_point=None, robot=None):
        return self._trav_map.get_random_point(floor=floor, reference_point=reference_point, robot=robot)

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False, robot=None):
        return self._trav_map.get_shortest_path(
            floor=floor,
            source_world=source_world,
            target_world=target_world,
            entire_path=entire_path,
            robot=robot,
        )
