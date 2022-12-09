import logging
import os
from omnigibson.robots.robot_base import m as robot_macros
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.maps.segmentation_map import SegmentationMap
from omnigibson.utils.asset_utils import (
    get_3dfront_scene_path,
    get_cubicasa_scene_path,
    get_og_scene_path,
)

SCENE_SOURCE_PATHS = {
    "OG": get_og_scene_path,
    "CUBICASA": get_cubicasa_scene_path,
    "THREEDFRONT": get_3dfront_scene_path,
}


class InteractiveTraversableScene(TraversableScene):
    """
    Create an interactive scene defined from a scene json file.
    In general, this supports curated, pre-defined scene layouts with annotated objects.
    This adds semantic support via a segmentation map generated for this specific scene.
    """
    def __init__(
        self,
        scene_model,
        scene_instance=None,
        scene_file=None,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_with_objects=True,
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
        load_object_categories=None,
        not_load_object_categories=None,
        load_room_types=None,
        load_room_instances=None,
        seg_map_resolution=0.1,
        scene_source="OG",
        include_robots=True,
    ):
        """
        Args:
            scene_model (str): Scene model name, e.g.: Rs_int
            scene_instance (None or str): name of json file to load (without .json); if None,
                defaults to og_dataset/scenes/<scene_model>/json/<scene_instance>.urdf
            scene_file (None or str): If specified, full path of JSON file to load (with .json).
                This will override scene_instance and scene_model!
            trav_map_resolution (float): traversability map resolution
            trav_map_erosion (float): erosion radius of traversability areas, should be robot footprint radius
            trav_map_with_objects (bool): whether to use objects or not when constructing graph
            build_graph (bool): build connectivity graph
            num_waypoints (int): number of way points returned
            waypoint_resolution (float): resolution of adjacent way points
            load_object_categories (None or list): if specified, only load these object categories into the scene
            not_load_object_categories (None or list): if specified, do not load these object categories into the scene
            load_room_types (None or list): only load objects in these room types into the scene
            load_room_instances (None or list): if specified, only load objects in these room instances into the scene
            seg_map_resolution (float): room segmentation map resolution
            scene_source (str): source of scene data; options are: {OG, CUBICASA, THREEDFRONT}
            include_robots (bool): whether to also include the robot(s) defined in the scene
        """

        # Store attributes from inputs
        self.scene_source = scene_source
        self.include_robots = include_robots

        # Other values that will be loaded at runtime
        self.scene_dir = None
        self.load_object_categories = None
        self.not_load_object_categories = None
        self.load_room_instances = None

        # Get scene information
        if scene_file is None:
            scene_file = self.get_scene_loading_info(
                scene_model=scene_model,
                scene_instance=scene_instance,
            )

        # Load room semantic and instance segmentation map (must occur AFTER inferring scene directory)
        self._seg_map = SegmentationMap(scene_dir=self.scene_dir, seg_map_resolution=seg_map_resolution)

        # Decide which room(s) and object categories to load
        self.filter_rooms_and_object_categories(
            load_object_categories, not_load_object_categories, load_room_types, load_room_instances
        )

        # Run super init first
        super().__init__(
            scene_model=scene_model,
            scene_file=scene_file,
            trav_map_resolution=trav_map_resolution,
            trav_map_erosion=trav_map_erosion,
            trav_map_with_objects=trav_map_with_objects,
            build_graph=build_graph,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
            use_floor_plane=False,
        )

    def get_scene_loading_info(self, scene_model, scene_instance=None):
        """
        Gets scene loading info to know what single USD file to load, either specified indirectly via @usd_file or
        directly by the fpath from @json_path. Note that if both are specified, @json_path takes precidence.
        If neither are specified, then a file will automatically be chosen based on self.scene_model and
        self.object_randomization

        Args:
            scene_model (str): Name of the scene to load, e.g, Rs_int, etc.
            scene_instance (None or str): If specified, should be name of json file to load. (without .json), default to
                og_dataset/scenes/<scene_model>/json/<scene_instance>.json

        Returns:
            str: Absolute path to the desired scene file (.json) to load
        """
        # Grab scene source path
        # TODO: Extend once we support the other scene sources
        assert self.scene_source == "OG", "Currently, only OG interactive traversable scenes are supported!"
        assert self.scene_source in SCENE_SOURCE_PATHS, f"Unsupported scene source: {self.scene_source}"
        self.scene_dir = SCENE_SOURCE_PATHS[self.scene_source](scene_model)

        # Infer scene file from model and directory
        fname = "{}_best".format(scene_model) if scene_instance is None else scene_instance
        return os.path.join(self.scene_dir, "json", "{}.json".format(fname))

    def filter_rooms_and_object_categories(
        self, load_object_categories, not_load_object_categories, load_room_types, load_room_instances
    ):
        """
        Handle partial scene loading based on object categories, room types or room instances

        Args:
            load_object_categories (None or list): if specified, only load these object categories into the scene
            not_load_object_categories (None or list): if specified, do not load these object categories into the scene
            load_room_types (None or list): only load objects in these room types into the scene
            load_room_instances (None or list): if specified, only load objects in these room instances into the scene
        """
        self.load_object_categories = [load_object_categories] if \
            isinstance(load_object_categories, str) else load_object_categories

        self.not_load_object_categories = [not_load_object_categories] if \
            isinstance(not_load_object_categories, str) else not_load_object_categories

        if load_room_instances is not None:
            if isinstance(load_room_instances, str):
                load_room_instances = [load_room_instances]
            load_room_instances_filtered = []
            for room_instance in load_room_instances:
                if room_instance in self._seg_map.room_ins_name_to_ins_id:
                    load_room_instances_filtered.append(room_instance)
                else:
                    logging.warning("room_instance [{}] does not exist.".format(room_instance))
            self.load_room_instances = load_room_instances_filtered
        elif load_room_types is not None:
            if isinstance(load_room_types, str):
                load_room_types = [load_room_types]
            load_room_instances_filtered = []
            for room_type in load_room_types:
                if room_type in self._seg_map.room_sem_name_to_ins_name:
                    load_room_instances_filtered.extend(self._seg_map.room_sem_name_to_ins_name[room_type])
                else:
                    logging.warning("room_type [{}] does not exist.".format(room_type))
            self.load_room_instances = load_room_instances_filtered
        else:
            self.load_room_instances = None

    def _load(self, simulator):
        # Run super first
        super()._load(simulator=simulator)

        # Load the traversability map if we have the connectivity graph
        maps_path = os.path.join(self.scene_dir, "layout")
        if self.has_connectivity_graph:
            self._trav_map.load_trav_map(maps_path)

    def _should_load_object(self, obj_info):
        category = obj_info["args"]["category"]
        in_rooms = obj_info["args"]["in_rooms"]

        # Do not load these object categories (can blacklist building structures as well)
        not_blacklisted = self.not_load_object_categories is None or category not in self.not_load_object_categories

        # Only load these object categories (no need to white list building structures)
        whitelisted = self.load_object_categories is None or category in self.load_object_categories

        # This object is not located in one of the selected rooms, skip
        valid_room = self.load_room_instances is None or len(set(self.load_room_instances) & set(in_rooms)) >= 0

        # Check whether this is an agent and we allow agents
        agent_ok = self.include_robots or category != robot_macros.ROBOT_CATEGORY

        # We only load this model if all the above conditions are met
        return not_blacklisted and whitelisted and valid_room and agent_ok

    @property
    def seg_map(self):
        """
        Returns:
            SegmentationMap: Map for segmenting this scene
        """
        return self._seg_map
