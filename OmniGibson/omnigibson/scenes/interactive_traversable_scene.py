import os

import omnigibson as og
from omnigibson.maps.segmentation_map import SegmentationMap
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.utils.asset_utils import get_scene_path, get_task_instance_path
from omnigibson.utils.constants import STRUCTURE_CATEGORIES, GROUND_CATEGORIES
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


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
        dataset_name="behavior-1k-assets",
        trav_map_resolution=0.1,
        default_erosion_radius=0.0,
        trav_map_with_objects=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
        load_object_categories=None,
        not_load_object_categories=None,
        load_room_types=None,
        load_room_instances=None,
        load_task_relevant_only=False,
        seg_map_resolution=0.1,
        include_robots=True,
    ):
        """
        Args:
            scene_model (str): Scene model name, e.g.: Rs_int
            scene_instance (None or str): name of json file to load (without .json); if None,
                defaults to <gm.DATA_PATH>/<dataset_name>/scenes/<scene_model>/json/<scene_instance>.urdf
            scene_file (None or str): If specified, full path of JSON file to load (with .json).
                This will override scene_instance and scene_model!
            dataset_name (str): Type of dataset to load the scene from, e.g.: "behavior-1k-assets"
            trav_map_resolution (float): traversability map resolution
            default_erosion_radius (float): default map erosion radius in meters
            trav_map_with_objects (bool): whether to use objects or not when constructing graph
            num_waypoints (int): number of way points returned
            waypoint_resolution (float): resolution of adjacent way points
            load_object_categories (None or list): if specified, only load these object categories into the scene
            not_load_object_categories (None or list): if specified, do not load these object categories into the scene
            load_room_types (None or list): only load objects in these room types into the scene
            load_room_instances (None or list): if specified, only load objects in these room instances into the scene
            load_task_relevant_only (bool): Whether only task relevant objects (and building structure) should be loaded
            seg_map_resolution (float): room segmentation map resolution (m)
            include_robots (bool): whether to also include the robot(s) defined in the scene
        """

        # Infer scene directory
        self.scene_dir = get_scene_path(scene_model, dataset_name=dataset_name)
        self.task_dir = get_task_instance_path(scene_model)

        # Other values that will be loaded at runtime
        self.load_object_categories = None
        self.not_load_object_categories = None
        self.load_room_instances = None
        self.load_task_relevant_only = load_task_relevant_only

        # Get scene information
        if scene_file is None:
            scene_file = self.get_scene_loading_info(
                scene_model=scene_model,
                scene_instance=scene_instance,
            )

        # Load room semantic and instance segmentation map (must occur AFTER inferring scene directory)
        self._seg_map = SegmentationMap(scene_dir=self.scene_dir, map_resolution=seg_map_resolution)

        # Decide which room(s) and object categories to load
        self.filter_rooms_and_object_categories(
            load_object_categories, not_load_object_categories, load_room_types, load_room_instances
        )

        # Assert that the floor plane is not enabled
        assert not og.sim.floor_plane, "Floor plane should not be enabled for interactive scenes!"

        # Run super init first
        super().__init__(
            scene_model=scene_model,
            scene_file=scene_file,
            trav_map_resolution=trav_map_resolution,
            default_erosion_radius=default_erosion_radius,
            trav_map_with_objects=trav_map_with_objects,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
            use_floor_plane=False,
            include_robots=include_robots,
        )

    def get_scene_loading_info(self, scene_model, scene_instance=None):
        """
        Gets scene loading info to know what single USD file to load, specified indirectly via @scene_instance if it
        is specified, otherwise, will grab the "best" scene file to load.

        Args:
            scene_model (str): Name of the scene to load, e.g, Rs_int, etc.
            scene_instance (None or str): If specified, should be name of json file to load. (without .json), default to
                behavior-1k-assets/scenes/<scene_model>/json/<scene_instance>.json

        Returns:
            str: Absolute path to the desired scene file (.json) to load
        """
        if scene_instance is None:
            # Load default "best" scene from scene directory
            fname = f"{scene_model}_best"
            return os.path.join(self.scene_dir, "json", f"{fname}.json")
        else:
            # Load specific instance from task instance directory
            return os.path.join(self.task_dir, "json", f"{scene_instance}.json")

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
        self.load_object_categories = (
            [load_object_categories] if isinstance(load_object_categories, str) else load_object_categories
        )

        self.not_load_object_categories = (
            [not_load_object_categories] if isinstance(not_load_object_categories, str) else not_load_object_categories
        )

        if load_room_instances is not None:
            if isinstance(load_room_instances, str):
                load_room_instances = [load_room_instances]
            load_room_instances_filtered = []
            for room_instance in load_room_instances:
                if room_instance not in self._seg_map.room_ins_name_to_ins_id:
                    log.warning("room_instance [{}] does not exist.".format(room_instance))
                load_room_instances_filtered.append(room_instance)
            self.load_room_instances = load_room_instances_filtered
        elif load_room_types is not None:
            if isinstance(load_room_types, str):
                load_room_types = [load_room_types]
            load_room_instances_filtered = []
            for room_type in load_room_types:
                if room_type in self._seg_map.room_sem_name_to_ins_name:
                    load_room_instances_filtered.extend(self._seg_map.room_sem_name_to_ins_name[room_type])
                else:
                    log.warning("room_type [{}] does not exist.".format(room_type))
            self.load_room_instances = load_room_instances_filtered
        else:
            self.load_room_instances = None

    def _load(self):
        # Load the traversability map if we have the connectivity graph
        maps_path = os.path.join(self.scene_dir, "layout")
        self._trav_map.load_map(maps_path)

    def _should_load_object(self, obj_info, task_metadata):
        agent_ok = super()._should_load_object(obj_info, task_metadata)

        name = obj_info["args"]["name"]
        category = obj_info["args"].get("category", "object")
        in_rooms = obj_info["args"].get("in_rooms", [])

        if isinstance(in_rooms, str):
            assert "," not in in_rooms
        in_rooms = [in_rooms] if isinstance(in_rooms, str) else in_rooms

        # Do not load these object categories (can blacklist building structures as well)
        not_blacklisted = self.not_load_object_categories is None or category not in self.not_load_object_categories

        # Only load these object categories (no need to white list building structures)
        task_relevant_names = set(task_metadata["inst_to_name"].values()) if "inst_to_name" in task_metadata else set()
        is_task_relevant = name in task_relevant_names or category in STRUCTURE_CATEGORIES
        whitelisted = (
            # Either no whitelisting-only mode is on
            (self.load_object_categories is None and not self.load_task_relevant_only)
            or
            # Or the object is in the whitelist
            (self.load_object_categories is not None and category in self.load_object_categories)
            or
            # Or it's in the task relevant list
            (self.load_task_relevant_only and is_task_relevant)
        )

        # This object is not located in one of the selected rooms, skip
        valid_room = self.load_room_instances is None or len(set(self.load_room_instances) & set(in_rooms)) > 0

        # Always load building structures for now because walls are missing room assignments
        # TODO (Wensi): below is changed to include doors to allow partial scene demo replay.
        # We need to think of a better way to handle this.
        is_building_structure = category in (STRUCTURE_CATEGORIES - GROUND_CATEGORIES) or category in [
            "door",
            "sliding_door",
        ]

        # We only load this model if all the above conditions are met
        return (not_blacklisted and whitelisted and valid_room and agent_ok) or is_building_structure

    @property
    def seg_map(self):
        """
        Returns:
            SegmentationMap: Map for segmenting this scene
        """
        return self._seg_map

    @classmethod
    def modify_init_info_for_restoring(cls, init_info):
        # Run super first
        super().modify_init_info_for_restoring(init_info=init_info)

        # We also make sure we load in any robots, and also pop any filters that were stored
        init_info["args"]["include_robots"] = True
        init_info["args"]["load_object_categories"] = None
        init_info["args"]["not_load_object_categories"] = None
        init_info["args"]["load_room_types"] = None
        init_info["args"]["load_room_instances"] = None
