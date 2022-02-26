import json
import logging
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.dom import minidom
from itertools import combinations
from collections import OrderedDict

import numpy as np

from pxr.Sdf import ValueTypeNames as VT
from omni.isaac.core.utils.rotations import gf_quat_to_np_array

from igibson.registries.object_states_registry import ObjectStatesRegistry
# from igibson.object_states.factory import get_state_from_name, get_state_name
# from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.objects.dataset_object import DatasetObject
# from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
# from igibson.robots import REGISTERED_ROBOTS
# from igibson.robots.behavior_robot import BehaviorRobot
# from igibson.robots.robot_base import BaseRobot
from igibson.scenes.traversable_scene import TraversableScene
from igibson.maps.segmentation_map import SegmentationMap
from igibson.utils.assets_utils import (
    get_3dfront_scene_path,
    get_cubicasa_scene_path,
    get_ig_category_ids,
    get_ig_category_path,
    get_ig_model_path,
    get_ig_scene_path,
)
from igibson.utils.utils import NumpyEncoder, restoreState, rotate_vector_3d
from igibson.utils.constants import JointType

SCENE_SOURCE_PATHS = {
    "IG": get_ig_scene_path,
    "CUBICASA": get_cubicasa_scene_path,
    "THREEDFRONT": get_3dfront_scene_path,
}


class InteractiveTraversableScene(TraversableScene):
    """
    Create an interactive scene defined with iGibson Scene Description Format (iGSDF).
    iGSDF is an extension of URDF that we use to define an interactive scene.
    It has support for URDF scaling, URDF nesting and randomization.
    InteractiveIndoorScene inherits from TraversableScene the functionalities to compute shortest path and other
    navigation functionalities.
    """

    def __init__(
        self,
        scene_id,
        usd_file=None,
        usd_path=None,
        # pybullet_filename=None,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_with_objects=True,
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
        texture_randomization=False,
        link_collision_tolerance=0.03,
        object_randomization=False,
        object_randomization_idx=None,
        should_open_all_doors=False,
        load_object_categories=None,
        not_load_object_categories=None,
        load_room_types=None,
        load_room_instances=None,
        seg_map_resolution=0.1,
        scene_source="IG",
        # merge_fixed_links=True,
        rendering_params=None,
        include_robots=True,
    ):
        """
        # TODO: Update
        :param scene_id: Scene id, e.g.: Rs_int
        :param usd_file: name of ursd file to load (without .urdf), default to ig_dataset/scenes/<scene_id>/urdf/<urdf_file>.urdf
        :param usd_path: full path of URDF file to load (with .urdf)
        # :param pybullet_filename: optional specification of which pybullet file to restore after initialization
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_with_objects: whether to use objects or not when constructing graph
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        :param texture_randomization: whether to randomize material/texture
        :param link_collision_tolerance: tolerance of the percentage of links that cannot be fully extended after object randomization
        :param object_randomization: whether to randomize object
        :param object_randomization_idx: index of a pre-computed object randomization model that guarantees good scene quality
        :param should_open_all_doors: whether to open all doors after episode reset (usually required for navigation tasks)
        :param load_object_categories: only load these object categories into the scene (a list of str)
        :param not_load_object_categories: do not load these object categories into the scene (a list of str)
        :param load_room_types: only load objects in these room types into the scene (a list of str)
        :param load_room_instances: only load objects in these room instances into the scene (a list of str)
        :param seg_map_resolution: room segmentation map resolution
        :param scene_source: source of scene data; among IG, CUBICASA, THREEDFRONT
        :param merge_fixed_links: whether to merge fixed links in pybullet
        :param rendering_params: additional rendering params to be passed into object initializers (e.g. texture scale)
        :param include_robots: whether to also include the robot(s) defined in the scene
        """
        # Run super init first
        super().__init__(
            scene_id=scene_id,
            trav_map_resolution=trav_map_resolution,
            trav_map_erosion=trav_map_erosion,
            trav_map_with_objects=trav_map_with_objects,
            build_graph=build_graph,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
        )

        # Store attributes from inputs
        self.texture_randomization = texture_randomization
        self.object_randomization = object_randomization
        self.object_randomization_idx = object_randomization_idx
        self.rendering_params = rendering_params
        self.should_open_all_doors = should_open_all_doors
        self.scene_source = scene_source

        # Other values that will be loaded at runtime
        self.fname = None
        self.scene_file = None
        self.scene_dir = None
        self.load_object_categories = None
        self.not_load_object_categories = None
        self.load_room_instances = None
        self._stage = None

        # self.scene_tree = ET.parse(self.scene_file)
        # self.pybullet_filename = pybullet_filename
        self.random_groups = {}
        self.category_ids = get_ig_category_ids()
        # self.merge_fixed_links = merge_fixed_links
        self.include_robots = include_robots

        # Get scene information
        self.get_scene_loading_info(usd_file=usd_file, usd_path=usd_path)

        # Load room semantic and instance segmentation map (must occur AFTER inferring scene directory)
        self._seg_map = SegmentationMap(scene_dir=self.scene_dir, seg_map_resolution=seg_map_resolution)

        # Decide which room(s) and object categories to load
        self.filter_rooms_and_object_categories(
            load_object_categories, not_load_object_categories, load_room_types, load_room_instances
        )

        # load overlapping bboxes in scene annotation
        self.overlapped_bboxes = self.load_overlapped_bboxes()

        # percentage of objects allowed that CANNOT extend their joints by >66%
        self.link_collision_tolerance = link_collision_tolerance

        # Agent placeholder
        self.agent = {}

        # ObjectMultiplexer
        self.object_multiplexers = defaultdict(dict)

        # ObjectGrouper
        self.object_groupers = defaultdict(dict)

        # Store the original states retrieved from the USD
        self.object_states = ObjectStatesRegistry()

    def get_scene_loading_info(self, usd_file=None, usd_path=None):
        """
        Gets scene loading info to know what single USD file to load, either specified indirectly via @usd_file or
        directly by the fpath from @usd_path. Note that if both are specified, @usd_path takes precidence.
        If neither are specified, then a file will automatically be chosen based on self.scene_id and
        self.object_randomization

        Args:
            usd_file (None or str): If specified, should be name of usd file to load. (without .usd), default to
                ig_dataset/scenes/<scene_id>/usd/<usd_file>.usd
            usd_path (None or str): If specified, should be absolute filepath to the USD file to load (with .usd)
        """
        # Grab scene source path
        assert self.scene_source in SCENE_SOURCE_PATHS, f"Unsupported scene source: {self.scene_source}"
        self.scene_dir = SCENE_SOURCE_PATHS[self.scene_source](self.scene_id)

        fname = None
        scene_file = usd_path
        # Possibly grab the USD directly from a specified fpath
        if usd_path is None:
            if usd_file is not None:
                fname = usd_file
            else:
                if not self.object_randomization:
                    fname = "{}_best".format(self.scene_id)
                else:
                    if self.object_randomization_idx is None:
                        fname = self.scene_id
                    else:
                        fname = "{}_random_{}".format(self.scene_id, self.object_randomization_idx)
            fname = fname
            scene_file = os.path.join(self.scene_dir, "usd", "{}.usd".format(fname))

        # Store values internally
        self.fname = fname
        self.scene_file = scene_file

    def get_objects_with_state(self, state):
        # We overload this method to provide a faster implementation.
        return self.object_registry("states", state, [])

    def filter_rooms_and_object_categories(
        self, load_object_categories, not_load_object_categories, load_room_types, load_room_instances
    ):
        """
        Handle partial scene loading based on object categories, room types or room instances

        :param load_object_categories: only load these object categories into the scene (a list of str)
        :param not_load_object_categories: do not load these object categories into the scene (a list of str)
        :param load_room_types: only load objects in these room types into the scene (a list of str)
        :param load_room_instances: only load objects in these room instances into the scene (a list of str)
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

    def load_overlapped_bboxes(self):
        """
        Load overlapped bounding boxes in scene definition.
        E.g. a dining table usually has overlaps with the surrounding dining chairs
        """
        bbox_overlap_file = os.path.join(self.scene_dir, "misc", "bbox_overlap.json")
        if os.path.isfile(bbox_overlap_file):
            with open(bbox_overlap_file) as f:
                return json.load(f)
        else:
            return []

    def randomize_texture(self):
        """
        Randomize texture/material for all objects in the scene
        """
        if not self.texture_randomization:
            logging.warning("calling randomize_texture while texture_randomization is False during initialization.")
            return
        for obj in self.objects:
            obj.randomize_texture()

    # TODO
    def check_collision(self, body_a, body_b=None, link_a=None, fixed_body_ids=None):
        """
        Helper function to check for collision for scene quality
        """
        if body_b is None:
            assert link_a is not None
            pts = p.getContactPoints(bodyA=body_a, linkIndexA=link_a)
        else:
            assert body_b is not None
            pts = p.getContactPoints(bodyA=body_a, bodyB=body_b)

        # contactDistance < 0 means actual penetration
        pts = [elem for elem in pts if elem[8] < 0.0]

        # only count collision with fixed body ids if provided
        if fixed_body_ids is not None:
            pts = [elem for elem in pts if elem[2] in fixed_body_ids]

        return len(pts) > 0

    # TODO
    def check_scene_quality(self, simulator):
        """
        Helper function to check for scene quality.
        1) Objects should have no collision with each other.
        2) Fixed, articulated objects that cannot fully extend their joints should be less than self.link_collision_tolerance

        Args:
            simulator (Simulator): Active simulator object

        :return: whether scene passes quality check
        """
        quality_check = True

        body_body_collision = []
        body_link_collision = []

        # # build mapping from body_id to object name for debugging
        # body_id_to_name = {}
        # for name in self.objects_by_name:
        #     for body_id in self.objects_by_name[name].get_body_ids():
        #         body_id_to_name[body_id] = name
        # self.body_id_to_name = body_id_to_name

        # collect body ids for overlapped bboxes (e.g. tables and chairs,
        # sofas and coffee tables)
        overlapped_objs = []
        for obj1_name, obj2_name in self.overlapped_bboxes:
            if obj1_name not in self.object_registry or obj2_name not in self.object_registry:
                # This could happen if only part of the scene is loaded (e.g. only a subset of rooms)
                continue
            overlapped_objs.append((self.object_registry("name", obj1_name), self.object_registry("name", obj2_name)))

        # TODO -- current method is probably insufficient
        # cache pybullet initial state
        state = self.save_state()

        # check if these overlapping bboxes have collision
        simulator.step()
        for obj_a, obj_b in overlapped_objs:
            has_collision = obj_a.in_contact(objects=obj_b)
            quality_check = quality_check and (not has_collision)
            if has_collision:
                body_body_collision.append((obj_a, obj_b))

        # check if fixed, articulated objects can extend their joints
        # without collision with other fixed objects
        joint_collision_allowed = int(self.get_num_objects() * self.link_collision_tolerance)
        joint_collision_so_far = 0
        for fixed_obj in self.fixed_objects.values():
            # This only applies if the object is articulated
            if fixed_obj.articulated:
                joint_quality = True
                for joint in fixed_obj.joints.values():
                    j_low, j_high = joint.lower_limit, joint.upper_limit
                    if joint.joint_type not in [JointType.JOINT_REVOLUTE, JointType.JOINT_PRISMATIC]:
                        continue
                    # this is the continuous joint (e.g. wheels for office chairs)
                    if j_low >= j_high:
                        continue

                    # usually j_low and j_high includes j_default = 0.0
                    # if not, set j_default to be j_low
                    j_default = 0.0
                    if not (j_low <= j_default <= j_high):
                        j_default = j_low

                    # check three joint positions, 0%, 33% and 66%
                    j_range = j_high - j_low
                    j_low_perc = j_range * 0.33 + j_low
                    j_high_perc = j_range * 0.66 + j_low

                    # check if j_default, j_low_per, or j_high_perc has collision
                    for j_pos in (j_default, j_low_perc, j_high_perc):
                        self.restore_state(state)
                        joint.set_pos(pos=j_pos)
                        simulator.step()
                        # TODO: I don't think this is working properly -- we currently don't check for self collision between fixed_obj and joint
                        has_collision = fixed_obj.in_contact(objects=self.fixed_objects, links=joint)
                        joint_quality = joint_quality and (not has_collision)

                if not joint_quality:
                    joint_collision_so_far += 1
                    body_link_collision.append(fixed_obj)

        quality_check = quality_check and (joint_collision_so_far <= joint_collision_allowed)

        # restore state to the initial state before testing collision
        self.restore_state(state)

        body_collision_set = set()
        for obj_a, obj_b in body_body_collision:
            logging.warning(f"scene quality check: {obj_a.name} and {obj_b.name} has collision.")
            body_collision_set.add(obj_a.name)
            body_collision_set.add(obj_b.name)

        link_collision_set = set()
        for obj in body_link_collision:
            logging.warning(f"scene quality check: {obj.name} has joint that cannot extend for >66%.")
            link_collision_set.add(obj.name)

        return quality_check

    def _set_first_n_objects(self, first_n_objects):
        """
        Only load the first N objects. Hidden API for debugging purposes.

        :param first_n_objects: only load the first N objects (integer)
        """
        raise ValueError(
            "The _set_first_n_object function is now deprecated due to "
            "incompatibility with recent object state features. Please "
            "use the load_object_categories method for limiting the "
            "objects to be loaded from the scene."
        )

    def _set_obj_names_to_load(self, obj_name_list):
        """
        Only load in objects with the given string names. Hidden API as is only
        used internally in the VR benchmark. This function automatically
        adds walls, floors and ceilings to the room.

        :param obj_name_list: list of string object names. These names must
            all be in the scene URDF file.
        """
        raise ValueError(
            "The _set_obj_names_to_load function is now deprecated due "
            "to incompatibility with recent object state features. Please "
            "use the load_object_categories method for limiting the "
            "objects to be loaded from the scene."
        )

    # TODO
    def open_one_obj(self, body_id, mode="random"):
        """
        Attempt to open one object without collision

        :param body_id: body id of the object
        :param mode: opening mode (zero, max, or random)
        """
        body_joint_pairs = []
        for joint_id in range(p.getNumJoints(body_id)):
            # cache current physics state
            state_id = p.saveState()

            j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
            j_type = p.getJointInfo(body_id, joint_id)[2]
            parent_idx = p.getJointInfo(body_id, joint_id)[-1]
            if j_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                p.removeState(state_id)
                continue
            # this is the continuous joint
            if j_low >= j_high:
                p.removeState(state_id)
                continue
            # this is the 2nd degree joint, ignore for now
            if parent_idx != -1:
                p.removeState(state_id)
                continue

            if mode == "max":
                # try to set the joint to the maxr value until no collision
                # step_size is 5cm for prismatic joint and 5 degrees for revolute joint
                step_size = np.pi / 36.0 if j_type == p.JOINT_REVOLUTE else 0.05
                for j_pos in np.arange(0.0, j_high + step_size, step=step_size):
                    p.resetJointState(body_id, joint_id, j_high - j_pos)
                    p.stepSimulation()
                    has_collision = self.check_collision(body_a=body_id, link_a=joint_id)
                    restoreState(state_id)
                    if not has_collision:
                        p.resetJointState(body_id, joint_id, j_high - j_pos)
                        break

            elif mode == "random":
                # try to set the joint to a random value until no collision
                reset_success = False
                # make 10 attemps
                for _ in range(10):
                    j_pos = np.random.uniform(j_low, j_high)
                    p.resetJointState(body_id, joint_id, j_pos)
                    p.stepSimulation()
                    has_collision = self.check_collision(body_a=body_id, link_a=joint_id)
                    restoreState(state_id)
                    if not has_collision:
                        p.resetJointState(body_id, joint_id, j_pos)
                        reset_success = True
                        break

                # if none of the random values work, set it to 0.0 by default
                if not reset_success:
                    p.resetJointState(body_id, joint_id, 0.0)
            elif mode == "zero":
                p.resetJointState(body_id, joint_id, 0.0)
            else:
                assert False

            body_joint_pairs.append((body_id, joint_id))
            # Remove cached state to avoid memory leak.
            p.removeState(state_id)

        return body_joint_pairs

    def open_all_objs_by_category(self, category, mode="random", prob=1.0):
        """
        Attempt to open all objects of a certain category without collision

        :param category: object category (str)
        :param mode: opening mode (zero, max, or random)
        :param prob: opening probability
        """
        body_joint_pairs = []
        if category not in self.object_registry.get_ids("category"):
            return body_joint_pairs
        for obj in self.object_registry("category", category):
            # open probability
            if np.random.random() > prob:
                continue
            for body_id in obj.get_body_ids():
                body_joint_pairs += self.open_one_obj(body_id, mode=mode)
        return body_joint_pairs

    def open_all_objs_by_categories(self, categories, mode="random", prob=1.0):
        """
        Attempt to open all objects of a number of categories without collision

        :param categories: object categories (a list of str)
        :param mode: opening mode (zero, max, or random)
        :param prob: opening probability
        """
        body_joint_pairs = []
        for category in categories:
            body_joint_pairs += self.open_all_objs_by_category(category, mode=mode, prob=prob)
        return body_joint_pairs

    def open_all_doors(self):
        """
        Attempt to open all doors to maximum values without collision
        """
        return self.open_all_objs_by_category("door", mode="max")

    def restore_object_states_single_object(self, obj, obj_state):
        """
        Restores object @obj to the state defined by @object_state.

        Args:
            obj (DatasetObject): Object to restore state
            object_state (ObjectSceneState): namedtuple with named parameters corresponding to different states of
                object @obj
        """
        print(f"restoring obj: {obj.name} state")
        # If the object isn't loaded, skip
        if not obj.loaded:
            return

        # If the object state is empty (which happens if an object is added after the scene URDF is parsed), skip
        if not obj_state:
            return

        # TODO: For velocities, we are now storing each body's com. Should we somehow do the same for positions?
        if obj_state.base_com_pose is not None:
            obj.set_position_orientation(*obj_state.base_com_pose)
        else:
            # TODO:
            # if isinstance(obj, BaseRobot):
            #     # Backward compatibility, existing scene cache saves robot's base link CoM frame as bbox_center_pose
            #     obj.set_position_orientation(*obj_kin_state["bbox_center_pose"])
            # else:
            #     obj.set_bbox_center_position_orientation(*obj_kin_state["bbox_center_pose"])
            # obj.set_position_orientation(*obj_kin_state.bbox_center_pose)
            obj.set_bbox_center_position_orientation(*obj_state.bbox_center_pose)

        # TODO: Change how we do this. Use serialized states instead?
        # if obj_state.base_velocities is not None:
        #     obj.set_velocities(obj_state.base_velocities)
        # else:
        #     obj.set_velocities([np.zeros(3), np.zeros(3)])
        #
        # # Only reset joint states if the object has joint states
        # if obj.articulated:
        #     if obj_state.joint_states is not None:
        #         # TODO: This breaks
        #         obj.set_joint_states(obj_state.joint_states)
        #     else:
        #         obj.reset_joint_states()
        #
        # if obj_state.non_kinematic_states is not None:
        #     obj.load_state(obj_state.non_kinematic_states)

    def restore_object_states(self, object_states=None):
        """
        Restores all scene objects according to the object_states defined in @object_states. If not specified, will
        assume the internal default self.object_states will be used.

        Args:
            object_states (ObjectSceneStatesRegistry): Registry (dict) storing object-specific information. Maps object
                name to a namedtuple of object states.
        """
        object_states = self.object_states if object_states is None else object_states
        for obj in self.objects:
            # TODO
            # if not isinstance(obj, ObjectMultiplexer):
            #     self.restore_object_states_single_object(obj, object_states[obj_name])
            # else:
            #     for sub_obj in obj._multiplexed_objects:
            #         if isinstance(sub_obj, ObjectGrouper):
            #             for obj_part in sub_obj.objects:
            #                 self.restore_object_states_single_object(obj_part, object_states[obj_part.name])
            #         else:
            #             self.restore_object_states_single_object(sub_obj, object_states[sub_obj.name])
            self.restore_object_states_single_object(obj, obj_state=object_states(obj.name))

    def _create_obj_from_template_xform(self, simulator, prim):
        """
        Creates the object specified from a template xform @prim, presumed to be in a template USD file,
        and simultaneously replaces the template prim with the actual object prim


        Args:
            simulator (Simulator): Active simulation object
            prim: Usd.Prim: Object template Xform prim

        Returns:
            DatasetObject: Created iGibson object
        """
        # Extract relevant info from template
        name, prim_path = prim.GetName(), prim.GetPrimPath().__str__()
        category, model, bbox, bbox_center_pos, bbox_center_ori, fixed, in_rooms, random_group, scale, bddl_obj_scope = self._extract_obj_info_from_template_xform(prim=prim)

        # Delete the template prim
        simulator.stage.RemovePrim(prim_path)

        # Skip multiplexer and grouper because they are not real objects
        if category == "multiplexer":
            raise NotImplementedError()
            # self.object_multiplexers[object_name]["current_index"] = link.attrib["current_index"]
            # continue

        elif category == "grouper":
            raise NotImplementedError()
            # self.object_groupers[object_name]["pose_offsets"] = json.loads(link.attrib["pose_offsets"])
            # self.object_groupers[object_name]["multiplexer"] = link.attrib["multiplexer"]
            # self.object_multiplexers[link.attrib["multiplexer"]]["grouper"] = object_name
            # continue

        elif category == "agent" and not self.include_robots:
            raise NotImplementedError()
            # continue

        # Robot object
        elif category == "agent":
            raise NotImplementedError()
            # robot_config = json.loads(link.attrib["robot_config"]) if "robot_config" in link.attrib else {}
            # assert model in REGISTERED_ROBOTS, "Got invalid robot to instantiate: {}".format(model)
            # obj = REGISTERED_ROBOTS[model](name=object_name, **robot_config)

        # Non-robot object
        else:
            # Do not load these object categories (can blacklist building structures as well)
            not_blacklisted = self.not_load_object_categories is None or category not in self.not_load_object_categories

            # Only load these object categories (no need to white list building structures)
            whitelisted = self.load_object_categories is None or category in self.load_object_categories

            # This object is not located in one of the selected rooms, skip
            valid_room = self.load_room_instances is None or len(set(self.load_room_instances) & set(in_rooms)) >= 0

            # We only load this model if all the above conditions are met
            if not_blacklisted and whitelisted and valid_room:

                # Make sure objects exist in the actual requested category
                category_path = get_ig_category_path(category)
                assert len(os.listdir(category_path)) != 0, "No models in category folder {}".format(category_path)

                # Potentially grab random object
                if model == "random":
                    if random_group is None:
                        model = random.choice(os.listdir(category_path))
                    else:
                        # Using random group to assign the same model to a group of objects
                        # E.g. we want to use the same model for a group of chairs around the same dining table
                        # random_group is a unique integer within the category
                        random_group_key = (category, random_group)

                        if random_group_key in self.random_groups:
                            model = self.random_groups[random_group_key]
                        else:
                            # We create a new unique entry for this random group if it doesn't already exist
                            model = random.choice(os.listdir(category_path))
                            self.random_groups[random_group_key] = model

                model_path = get_ig_model_path(category, model)
                # TODO: Remove "usd" in the middle when we simply have the model directory directly contain the USD
                usd_path = os.path.join(model_path, "usd", model + ".usd")

                # Make sure only a bounding box OR scale is specified
                assert bbox is None or scale is None, f"Both scale and bounding box size was defined for a USDObject in the template scene!"

                # Make sure scale is, at the minimum, specified
                if bbox is None and scale is None:
                    scale = np.array([1.0, 1.0, 1.0])

                # Create the object (finally!)
                obj = DatasetObject(
                    prim_path=prim_path,
                    usd_path=usd_path,
                    name=name,
                    category=category,
                    scale=scale,
                    rendering_params=self.rendering_params,
                    fixed_base=fixed,
                    bounding_box=bbox,
                    in_rooms=in_rooms,
                    texture_randomization=self.texture_randomization,
                    bddl_object_scope=bddl_obj_scope,
                )

            # TODO: Are all of these necessary now that we can directly save USD snapshots?
            # bbox_center_pos = np.array([float(val) for val in connecting_joint.find("origin").attrib["xyz"].split(" ")])
            # if "rpy" in connecting_joint.find("origin").attrib:
            #     bbx_center_orn = np.array(
            #         [float(val) for val in connecting_joint.find("origin").attrib["rpy"].split(" ")]
            #     )
            # else:
            #     bbx_center_orn = np.array([0.0, 0.0, 0.0])
            # bbx_center_orn = p.getQuaternionFromEuler(bbx_center_orn)
            #
            # base_com_pose = json.loads(link.attrib["base_com_pose"]) if "base_com_pose" in link.attrib else None
            # base_velocities = json.loads(link.attrib["base_velocities"]) if "base_velocities" in link.attrib else None
            # if "joint_states" in link.keys():
            #     joint_states = json.loads(link.attrib["joint_states"])
            # elif "joint_positions" in link.keys():
            #     # Backward compatibility, assuming multi-sub URDF object don't have any joints
            #     joint_states = {
            #         key: (position, 0.0) for key, position in json.loads(link.attrib["joint_positions"])[0].items()
            #     }
            # else:
            #     joint_states = None
            #
            # if "states" in link.keys():
            #     non_kinematic_states = json.loads(link.attrib["states"])
            # else:
            #     non_kinematic_states = None
            print(f"obj: {name}, bbox center pos: {bbox_center_pos}, bbox center ori: {bbox_center_ori}")

            self.object_states.add_object(
                obj_name=name,
                bbox_center_pose=(bbox_center_pos, bbox_center_ori),
                base_com_pose=None,#(np.zeros(3), np.array([0, 0, 0, 1.0])),
                base_velocities=None,
                joint_states=None,
                non_kinematic_states=None,
            )

            # TODO: Handle multiplexing / groupers
            # if "multiplexer" in link.keys() or "grouper" in link.keys():
            #     if "multiplexer" in link.keys():
            #         self.object_multiplexers[link.attrib["multiplexer"]]["whole_object"] = obj
            #     else:
            #         grouper = self.object_groupers[link.attrib["grouper"]]
            #         if "object_parts" not in grouper:
            #             grouper["object_parts"] = []
            #         grouper["object_parts"].append(obj)
            #
            #         # Once the two halves are added, this multiplexer is ready to be added to the scene
            #         if len(grouper["object_parts"]) == 2:
            #             multiplexer = grouper["multiplexer"]
            #             current_index = int(self.object_multiplexers[multiplexer]["current_index"])
            #             whole_object = self.object_multiplexers[multiplexer]["whole_object"]
            #             object_parts = grouper["object_parts"]
            #             pose_offsets = grouper["pose_offsets"]
            #             grouped_obj_parts = ObjectGrouper(list(zip(object_parts, pose_offsets)))
            #             obj = ObjectMultiplexer(multiplexer, [whole_object, grouped_obj_parts], current_index)
            #             self.add_object(obj, simulator=None)
            # else:
            #     self.add_object(obj, simulator=None)

        return obj

    def _extract_obj_info_from_template_xform(self, prim):
        """
        Extracts relevant iGibson object information from a template xform, presumed to be in a template USD file

        Args:
            prim: Usd.Prim: Object template Xform prim

        Returns:
            6-tuple:
                - category (str): object category
                - model (str): object model ("random" implies should be randomly selected)
                - bbox (None or 3-array): (x,y,z) bounding box for the object in the scene, if specified
                - pos (3-array): (x,y,z) global coordinate for this object
                - ori (4-array): (x,y,z,w) global quaternion orientation for this object
                - fixed (bool): whether to fix this object in the scene via a fixed joint
                - rooms (list of str): room(s) allowed by this object
                - random_group (None or str): if specified, group from which to grab a randomized instance
                    if @model is "random"
                - scale (None or 3-array): (x,y,z) scale for the object in the scene, if specified
                - obj_scope (None or str): If specified, the scope of this object
        """
        category = prim.GetAttribute("ig:category").Get()
        model = prim.GetAttribute("ig:model").Get()
        bbox = prim.GetAttribute("ig:boundingBox").Get()
        bbox = None if bbox is None else np.array(bbox)
        pos = np.array(prim.GetAttribute("ig:position").Get())
        ori = gf_quat_to_np_array(prim.GetAttribute("ig:orientation").Get())[[1, 2, 3, 0]]
        fixed = prim.GetAttribute("ig:fixedJoint").Get()
        rooms = prim.GetAttribute("ig:rooms").Get().split(",")
        random_group = prim.GetAttribute("ig:randomGroup").Get()
        scale = prim.GetAttribute("ig:scale").Get()
        scale = None if scale is None else np.array(scale)
        obj_scope = prim.GetAttribute("ig:objectScope").Get()

        return category, model, bbox, pos, ori, fixed, rooms, random_group, scale, obj_scope

    def _load(self, simulator):
        """
        Load all scene objects into pybullet
        """
        # Notify user we're loading the scene
        logging.info("Clearing stage and loading scene USD: {}".format(self.scene_file))

        # We first load the initial USD file, clearing the stage in the meantime
        simulator.clear()
        simulator.load_stage(usd_path=self.scene_file)

        # Store stage reference
        self._stage = simulator.stage

        # Check if current stage is a template based on ig:isTemplate value, and set the value to False if it does not exist
        is_template = False
        world_prim = simulator.world_prim
        # TODO: Need to set template to false by default after loading everything
        if "ig:isTemplate" in world_prim.GetPropertyNames():
            is_template = world_prim.GetAttribute("ig:isTemplate").Get()
        else:
            # Create the property and set it to False
            world_prim.CreateAttribute("ig:isTemplate", VT.Bool)
            world_prim.GetAttribute("ig:isTemplate").Set(False)

        # Iterate over all the children in the stage world
        for prim in world_prim.GetChildren():
            # Only process prims that are an Xform
            if prim.GetPrimTypeInfo().GetTypeName() == "Xform":
                name = prim.GetName()
                # Skip over the wall, floor, or ceilings (#TODO: Can we make this more elegant?)
                if name in {"walls", "floors", "ceilings"}:
                    continue

                # Check if we're using a template -- if so, we need to load the object, otherwise, we simply
                # add a reference internally
                if is_template:
                    # Create the object and load it into the simulator
                    obj = self._create_obj_from_template_xform(simulator=simulator, prim=prim)
                    # Note that we don't auto-initialize because of some very specific state-setting logic that
                    # has to occur a certain way at the start of scene creation (sigh, Omniverse ): )
                    simulator.import_object(obj, auto_initialize=False)
                    # We also directly set it's bounding box position since this is a known quantity
                    # This is also the only time we'll be able to set fixed object poses
                    obj.set_bbox_center_position_orientation(*self.object_states(obj.name).bbox_center_pose)

        # disable collision between the fixed links of the fixed objects
        fixed_objs = self.object_registry("fixed_base", True)
        # We iterate over all pairwise combinations of fixed objects
        for obj_a, obj_b in combinations(fixed_objs, 2):
            obj_a.root_link.add_filtered_collision_pair(obj_b.root_link)

        # Load the traversability map
        maps_path = os.path.join(self.scene_dir, "layout")
        if self.has_connectivity_graph:
            self._trav_map.load_trav_map(maps_path)

        return list(self.objects)

    def _initialize(self):
        # First, we initialize all of our objects and add the object
        for obj in self.objects:

            # Initialize objects
            obj.initialize()

            # TODO
            # if not isinstance(obj, ObjectMultiplexer):
            #     self.restore_object_states_single_object(obj, object_states[obj_name])
            # else:
            #     for sub_obj in obj._multiplexed_objects:
            #         if isinstance(sub_obj, ObjectGrouper):
            #             for obj_part in sub_obj.objects:
            #                 self.restore_object_states_single_object(obj_part, object_states[obj_part.name])
            #         else:
            #             self.restore_object_states_single_object(sub_obj, object_states[sub_obj.name])
            self.restore_object_states_single_object(obj, obj_state=self.object_states(obj.name))
            obj.keep_still()

        # Re-initialize our scene object registry by handle since now handles are populated
        self.object_registry.update(keys="handle")

        # # TODO: Additional restoring from USD state? Is this even necessary?
        # if self.pybullet_filename is not None:
        #     restoreState(fileName=self.pybullet_filename)

        # TODO: Need to check scene quality
        # self.check_scene_quality(simulator=simulator)

        # force wake up each body once
        self.wake_scene_objects()

    def wake_scene_objects(self):
        """
        Force wakeup sleeping objects
        """
        for obj in self.objects:
            obj.wake()

    def reset_scene_objects(self):
        """
        Reset the pose and joint configuration of all scene objects.
        Also open all doors if self.should_open_all_doors is True
        """
        self.restore_object_states()

        if self.should_open_all_doors:
            self.wake_scene_objects()
            self.open_all_doors()

    def get_num_objects(self):
        """
        Get the number of objects

        :return: number of objects
        """
        return len(self.objects)

    def get_object_handles(self):
        """
        Return the object handles of all scene objects

        :return: object handles
        """
        return [obj.handle for obj in self.objects]

    # TODO
    def save_obj_or_multiplexer(self, obj, tree_root, additional_attribs_by_name):
        if not isinstance(obj, ObjectMultiplexer):
            self.save_obj(obj, tree_root, additional_attribs_by_name)
            return

        multiplexer_link = ET.SubElement(tree_root, "link")

        # Store current index
        multiplexer_link.attrib = {"category": "multiplexer", "name": obj.name, "current_index": str(obj.current_index)}

        for i, sub_obj in enumerate(obj._multiplexed_objects):
            if isinstance(sub_obj, ObjectGrouper):
                grouper_link = ET.SubElement(tree_root, "link")

                # Store pose offset
                grouper_link.attrib = {
                    "category": "grouper",
                    "name": obj.name + "_grouper",
                    "pose_offsets": json.dumps(sub_obj.pose_offsets, cls=NumpyEncoder),
                    "multiplexer": obj.name,
                }
                for group_sub_obj in sub_obj.objects:
                    # Store reference to grouper
                    if group_sub_obj.name not in additional_attribs_by_name:
                        additional_attribs_by_name[group_sub_obj.name] = {}
                    additional_attribs_by_name[group_sub_obj.name]["grouper"] = obj.name + "_grouper"

                    if i == obj.current_index:
                        # Assign object_scope to each object of in the grouper
                        if obj.name in additional_attribs_by_name:
                            for key in additional_attribs_by_name[obj.name]:
                                additional_attribs_by_name[group_sub_obj.name][key] = additional_attribs_by_name[
                                    obj.name
                                ][key]
                    self.save_obj(group_sub_obj, tree_root, additional_attribs_by_name)
            else:
                # Store reference to multiplexer
                if sub_obj.name not in additional_attribs_by_name:
                    additional_attribs_by_name[sub_obj.name] = {}
                additional_attribs_by_name[sub_obj.name]["multiplexer"] = obj.name
                if i == obj.current_index:
                    # Assign object_scope to the whole object
                    if obj.name in additional_attribs_by_name:
                        for key in additional_attribs_by_name[obj.name]:
                            additional_attribs_by_name[sub_obj.name][key] = additional_attribs_by_name[obj.name][key]
                self.save_obj(sub_obj, tree_root, additional_attribs_by_name)

    # TODO
    def save_obj(self, obj, tree_root, additional_attribs_by_name):
        name = obj.name
        link = tree_root.find('link[@name="{}"]'.format(name))

        # Convert from center of mass to base link position
        body_ids = obj.get_body_ids()
        main_body_id = body_ids[0] if len(body_ids) == 1 else body_ids[obj.main_body]

        dynamics_info = p.getDynamicsInfo(main_body_id, -1)
        inertial_pos = dynamics_info[3]
        inertial_orn = dynamics_info[4]

        # TODO: replace this with obj.get_position_orientation() once URDFObject no longer works with multiple body ids
        pos, orn = p.getBasePositionAndOrientation(main_body_id)
        pos, orn = np.array(pos), np.array(orn)
        inv_inertial_pos, inv_inertial_orn = p.invertTransform(inertial_pos, inertial_orn)
        base_link_position, base_link_orientation = p.multiplyTransforms(pos, orn, inv_inertial_pos, inv_inertial_orn)

        # Convert to XYZ position for URDF
        euler = euler_from_quat(orn)
        roll, pitch, yaw = euler
        if hasattr(obj, "scaled_bbxc_in_blf"):
            offset = rotate_vector_3d(obj.scaled_bbxc_in_blf, roll, pitch, yaw, False)
        else:
            offset = np.array([0, 0, 0])
        bbox_pos = base_link_position - offset

        xyz = " ".join([str(p) for p in bbox_pos])
        rpy = " ".join([str(e) for e in euler])

        # The object is already in the scene URDF
        if link is not None:
            if obj.category == "floors":
                floor_names = [obj_name for obj_name in additional_attribs_by_name if "room_floor" in obj_name]
                if len(floor_names) > 0:
                    floor_name = floor_names[0]
                    for key in additional_attribs_by_name[floor_name]:
                        floor_mappings = []
                        for floor_name in floor_names:
                            floor_mappings.append(
                                "{}:{}".format(additional_attribs_by_name[floor_name][key], floor_name)
                            )
                        link.attrib[key] = ",".join(floor_mappings)
            else:
                # Overwrite the pose in the original URDF with the pose
                # from the simulator for floating objects (typically
                # floating objects will fall by a few millimeters due to
                # gravity).
                joint = tree_root.find('joint[@name="{}"]'.format("j_{}".format(name)))
                if joint is not None and joint.attrib["type"] != "fixed":
                    link.attrib["rpy"] = rpy
                    link.attrib["xyz"] = xyz
                    origin = joint.find("origin")
                    origin.attrib["rpy"] = rpy
                    origin.attrib["xyz"] = xyz
        else:
            # We need to add the object to the scene URDF
            category = obj.category
            room = self.get_room_instance_by_point(pos[:2])

            link = ET.SubElement(tree_root, "link")
            link.attrib = {
                "category": category,
                "name": name,
                "rpy": rpy,
                "xyz": xyz,
            }

            if hasattr(obj, "bounding_box"):
                bounding_box = " ".join([str(b) for b in obj.bounding_box])
                link.attrib["bounding_box"] = bounding_box

            if hasattr(obj, "model_name"):
                link.attrib["model"] = obj.model_name
            elif hasattr(obj, "model_path"):
                model = os.path.basename(obj.model_path)
                link.attrib["model"] = model

            if room is not None:
                link.attrib["room"] = room

            if isinstance(obj, BaseRobot):
                link.attrib["robot_config"] = json.dumps(obj.dump_config(), cls=NumpyEncoder)

            new_joint = ET.SubElement(tree_root, "joint")
            new_joint.attrib = {"name": "j_{}".format(name), "type": "floating"}
            new_origin = ET.SubElement(new_joint, "origin")
            new_origin.attrib = {"rpy": rpy, "xyz": xyz}
            new_child = ET.SubElement(new_joint, "child")
            new_child.attrib["link"] = name
            new_parent = ET.SubElement(new_joint, "parent")
            new_parent.attrib["link"] = "world"

        # Common logic for objects that are both in the scene & otherwise.
        base_com_pose = (pos, orn)
        joint_states = obj.get_joint_states()
        link.attrib["base_com_pose"] = json.dumps(base_com_pose, cls=NumpyEncoder)
        link.attrib["base_velocities"] = json.dumps(obj.get_velocities(), cls=NumpyEncoder)
        link.attrib["joint_states"] = json.dumps(joint_states, cls=NumpyEncoder)

        # Add states
        if hasattr(obj, "states"):
            link.attrib["states"] = json.dumps(obj.dump_state(), cls=NumpyEncoder)

        # Add additional attributes.
        if name in additional_attribs_by_name:
            for key in additional_attribs_by_name[name]:
                link.attrib[key] = additional_attribs_by_name[name][key]

    # TODO
    def restore(self, urdf_name=None, urdf_path=None, scene_tree=None, pybullet_filename=None, pybullet_state_id=None):
        """
        Restore a already-loaded scene with a given URDF file plus pybullet_filename or pybullet_state_id (optional)
        The non-kinematic states (e.g. temperature, sliced, dirty) will be loaded from the URDF file.
        The kinematic states (e.g. pose, joint states) will be loaded from the URDF file OR pybullet state / filename (if provided, for better determinism)
        This function assume the given URDF and pybullet_filename or pybullet_state_id contains the exact same objects as the current scene, and only their states will be restored.

        :param urdf_name: name of urdf file to save (without .urdf), default to ig_dataset/scenes/<scene_id>/urdf/<urdf_name>.urdf
        :param urdf_path: full path of URDF file to save (with .urdf)
        :param scene_tree: already-loaded URDF file stored in memory
        :param pybullet_filename: optional specification of which pybullet file to save to
        :param pybullet_save_state: whether to save to pybullet state
        :param additional_attribs_by_name: additional attributes to be added to object link in the scene URDF
        """
        if scene_tree is None:
            assert urdf_name is not None or urdf_path is not None, "need to specify either urdf_name or urdf_path"
            if urdf_path is None:
                urdf_path = os.path.join(self.scene_dir, "urdf", urdf_name + ".urdf")
            scene_tree = ET.parse(urdf_path)

        assert (
            pybullet_filename is None or pybullet_state_id is None
        ), "you can only specify either a pybullet filename or a pybullet state id"

        object_states = defaultdict(dict)
        for link in scene_tree.findall("link"):
            object_name = link.attrib["name"]
            if object_name == "world":
                continue
            category = link.attrib["category"]

            if category == "multiplexer":
                self.object_registry("name", object_name).set_selection(int(link.attrib["current_index"]))

            if category in ["grouper", "multiplexer"]:
                continue

            object_states[object_name]["bbox_center_pose"] = None
            object_states[object_name]["base_com_pose"] = json.loads(link.attrib["base_com_pose"])
            object_states[object_name]["base_velocities"] = json.loads(link.attrib["base_velocities"])
            object_states[object_name]["joint_states"] = json.loads(link.attrib["joint_states"])
            object_states[object_name]["non_kinematic_states"] = json.loads(link.attrib["states"])

        self.restore_object_states()

        if pybullet_filename is not None:
            restoreState(fileName=pybullet_filename)
        elif pybullet_state_id is not None:
            restoreState(stateId=pybullet_state_id)

    # TODO
    def save(
        self,
        urdf_name=None,
        urdf_path=None,
        pybullet_filename=None,
        pybullet_save_state=False,
        additional_attribs_by_name={},
    ):
        """
        Saves a modified URDF file in the scene urdf directory having all objects added to the scene.

        :param urdf_name: name of urdf file to save (without .urdf), default to ig_dataset/scenes/<scene_id>/urdf/<urdf_name>.urdf
        :param urdf_path: full path of URDF file to save (with .urdf), assumes higher priority than urdf_name
        :param pybullet_filename: optional specification of which pybullet file to save to
        :param pybullet_save_state: whether to save to pybullet state
        :param additional_attribs_by_name: additional attributes to be added to object link in the scene URDF
        """
        if urdf_path is None and urdf_name is not None:
            urdf_path = os.path.join(self.scene_dir, "urdf", urdf_name + ".urdf")

        scene_tree = ET.parse(self.scene_file)
        tree_root = scene_tree.getroot()
        for obj in self.objects:
            self.save_obj_or_multiplexer(obj, tree_root, additional_attribs_by_name)

        if urdf_path is not None:
            xmlstr = minidom.parseString(ET.tostring(tree_root).replace(b"\n", b"").replace(b"\t", b"")).toprettyxml()
            with open(urdf_path, "w") as f:
                f.write(xmlstr)

        if pybullet_filename is not None:
            p.saveBullet(pybullet_filename)

        if pybullet_save_state:
            snapshot_id = p.saveState()

        if pybullet_save_state:
            return scene_tree, snapshot_id
        else:
            return scene_tree

    @property
    def seg_map(self):
        """
        Returns:
            SegmentationMap: Map for segmenting this scene
        """
        return self._seg_map

    @property
    def object_registry_unique_keys(self):
        # Grab all inherited keys and return additional ones
        keys = super().object_registry_unique_keys
        return keys + ["handle"]

    @property
    def object_registry_group_keys(self):
        # Grab all inherited keys and return additional ones
        keys = super().object_registry_group_keys
        return keys + ["category", "in_rooms", "states", "fixed_base"]

    @property
    def fixed_objects(self):
        """
        Returns:
            dict: Keyword-mapped objects that are are fixed in the scene. Maps object name to their object class instances
                (DatasetObject)
        """
        return {obj.name: obj for obj in self.object_registry("fixed_base", True)}
