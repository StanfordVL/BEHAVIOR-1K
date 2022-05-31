import json
import logging
import os
import random
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.dom import minidom

import numpy as np

from IPython import embed
from PIL import Image

import igibson
# from igibson.object_states.factory import get_state_from_name, get_state_name
# from igibson.object_states.object_state_base import AbsoluteObjectState
# from igibson.objects.articulated_object import URDFObject
# from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
# from igibson.robots import REGISTERED_ROBOTS
# from igibson.robots.behavior_robot import BehaviorRobot
# from igibson.robots.robot_base import BaseRobot
from igibson.scenes.traversable_scene import TraversableScene
from igibson.utils.assets_utils import (
    get_3dfront_scene_path,
    get_cubicasa_scene_path,
    get_ig_avg_category_specs,
    get_ig_category_ids,
    get_ig_category_path,
    get_ig_model_path,
    get_ig_scene_path,
)
from igibson.utils.utils import NumpyEncoder, restoreState, rotate_vector_3d

SCENE_SOURCE_PATHS = {
    "IG": get_ig_scene_path,
    "CUBICASA": get_cubicasa_scene_path,
    "THREEDFRONT": get_3dfront_scene_path,
}


class SegmentationMap:
    """
    Segmentation map for computing connectivity within the scene
    """

    def __init__(
        self,
        scene_dir,
        seg_map_resolution=0.1,
        floor_heights=(0.0,),
    ):
        """
        :param scene_dir: str, path to the scene directory from which segmentation info will be extracted
        :param seg_map_resolution: room segmentation map resolution
        :param floor_heights: heights of the floors for this segmentation map
        """
        # Store internal values
        self.scene_dir = scene_dir
        self.seg_map_default_resolution = 0.01
        self.seg_map_resolution = seg_map_resolution
        self.floor_heights = floor_heights

        # Other values that will be loaded at runtime
        self.seg_map_size = None
        self.room_sem_name_to_sem_id = None
        self.room_sem_id_to_sem_name = None
        self.room_ins_name_to_ins_id = None
        self.room_ins_id_to_ins_name = None
        self.room_sem_name_to_ins_name = None
        self.room_ins_map = None
        self.room_sem_map = None

        # Load the map
        self.load_room_sem_ins_seg_map()

    def load_room_sem_ins_seg_map(self):
        """
        Load room segmentation map
        """
        layout_dir = os.path.join(self.scene_dir, "layout")
        room_seg_imgs = os.path.join(layout_dir, "floor_insseg_0.png")
        img_ins = Image.open(room_seg_imgs)
        room_seg_imgs = os.path.join(layout_dir, "floor_semseg_0.png")
        img_sem = Image.open(room_seg_imgs)
        height, width = img_ins.size
        assert height == width, "room seg map is not a square"
        assert img_ins.size == img_sem.size, "semantic and instance seg maps have different sizes"
        self.seg_map_size = int(height * self.seg_map_default_resolution / self.seg_map_resolution)
        img_ins = np.array(img_ins.resize((self.seg_map_size, self.seg_map_size), Image.NEAREST))
        img_sem = np.array(img_sem.resize((self.seg_map_size, self.seg_map_size), Image.NEAREST))

        room_categories = os.path.join(igibson.ig_dataset_path, "metadata", "room_categories.txt")
        with open(room_categories, "r") as fp:
            room_cats = [line.rstrip() for line in fp.readlines()]

        sem_id_to_ins_id = {}
        unique_ins_ids = np.unique(img_ins)
        unique_ins_ids = np.delete(unique_ins_ids, 0)
        for ins_id in unique_ins_ids:
            # find one pixel for each ins id
            x, y = np.where(img_ins == ins_id)
            # retrieve the correspounding sem id
            sem_id = img_sem[x[0], y[0]]
            if sem_id not in sem_id_to_ins_id:
                sem_id_to_ins_id[sem_id] = []
            sem_id_to_ins_id[sem_id].append(ins_id)

        room_sem_name_to_sem_id = {}
        room_ins_name_to_ins_id = {}
        room_sem_name_to_ins_name = {}
        for sem_id, ins_ids in sem_id_to_ins_id.items():
            sem_name = room_cats[sem_id - 1]
            room_sem_name_to_sem_id[sem_name] = sem_id
            for i, ins_id in enumerate(ins_ids):
                # valid class start from 1
                ins_name = "{}_{}".format(sem_name, i)
                room_ins_name_to_ins_id[ins_name] = ins_id
                if sem_name not in room_sem_name_to_ins_name:
                    room_sem_name_to_ins_name[sem_name] = []
                room_sem_name_to_ins_name[sem_name].append(ins_name)

        self.room_sem_name_to_sem_id = room_sem_name_to_sem_id
        self.room_sem_id_to_sem_name = {value: key for key, value in room_sem_name_to_sem_id.items()}
        self.room_ins_name_to_ins_id = room_ins_name_to_ins_id
        self.room_ins_id_to_ins_name = {value: key for key, value in room_ins_name_to_ins_id.items()}
        self.room_sem_name_to_ins_name = room_sem_name_to_ins_name
        self.room_ins_map = img_ins
        self.room_sem_map = img_sem

    def get_random_point_by_room_type(self, room_type):
        """
        Sample a random point by room type

        :param room_type: room type (e.g. bathroom)
        :return: floor (always 0), a randomly sampled point in [x, y, z]
        """
        if room_type not in self.room_sem_name_to_sem_id:
            logging.warning("room_type [{}] does not exist.".format(room_type))
            return None, None

        sem_id = self.room_sem_name_to_sem_id[room_type]
        valid_idx = np.array(np.where(self.room_sem_map == sem_id))
        random_point_map = valid_idx[:, np.random.randint(valid_idx.shape[1])]

        x, y = self.seg_map_to_world(random_point_map)
        # assume only 1 floor
        floor = 0
        z = self.floor_heights[floor]
        return floor, np.array([x, y, z])

    def get_random_point_by_room_instance(self, room_instance):
        """
        Sample a random point by room instance

        :param room_instance: room instance (e.g. bathroom_1)
        :return: floor (always 0), a randomly sampled point in [x, y, z]
        """
        if room_instance not in self.room_ins_name_to_ins_id:
            logging.warning("room_instance [{}] does not exist.".format(room_instance))
            return None, None

        ins_id = self.room_ins_name_to_ins_id[room_instance]
        valid_idx = np.array(np.where(self.room_ins_map == ins_id))
        random_point_map = valid_idx[:, np.random.randint(valid_idx.shape[1])]

        x, y = self.seg_map_to_world(random_point_map)
        # assume only 1 floor
        floor = 0
        z = self.floor_heights[floor]
        return floor, np.array([x, y, z])

    # TODO: remove after split floors
    def get_aabb_by_room_instance(self, room_instance):
        """
        Get AABB of the floor by room instance
        :param room_instance: room instance (e.g. bathroom_1)
        """
        if room_instance not in self.room_ins_name_to_ins_id:
            logging.warning("room_instance [{}] does not exist.".format(room_instance))
            return None, None

        ins_id = self.room_ins_name_to_ins_id[room_instance]
        valid_idx = np.array(np.where(self.room_ins_map == ins_id))
        u_min = np.min(valid_idx[0])
        u_max = np.max(valid_idx[0])
        v_min = np.min(valid_idx[1])
        v_max = np.max(valid_idx[1])
        x_a, y_a = self.seg_map_to_world(np.array([u_min, v_min]))
        x_b, y_b = self.seg_map_to_world(np.array([u_max, v_max]))
        x_min = np.min([x_a, x_b])
        x_max = np.max([x_a, x_b])
        y_min = np.min([y_a, y_b])
        y_max = np.max([y_a, y_b])
        # assume only 1 floor
        floor = 0
        z = self.floor_heights[floor]

        return np.array([x_min, y_min, z]), np.array([x_max, y_max, z])

    def seg_map_to_world(self, xy):
        """
        Transforms a 2D point in map reference frame into world (simulator) reference frame

        :param xy: 2D location in seg map reference frame (image)
        :return: 2D location in world reference frame (metric)
        """
        axis = 0 if len(xy.shape) == 1 else 1
        return np.flip((xy - self.seg_map_size / 2.0) * self.seg_map_resolution, axis=axis)

    def world_to_seg_map(self, xy):
        """
        Transforms a 2D point in world (simulator) reference frame into map reference frame

        :param xy: 2D location in world reference frame (metric)
        :return: 2D location in seg map reference frame (image)
        """
        return np.flip((xy / self.seg_map_resolution + self.seg_map_size / 2.0)).astype(np.int)

    def get_room_type_by_point(self, xy):
        """
        Return the room type given a point

        :param xy: 2D location in world reference frame (metric)
        :return: room type that this point is in or None, if this point is not on the room segmentation map
        """
        x, y = self.world_to_seg_map(xy)
        if x < 0 or x >= self.room_sem_map.shape[0] or y < 0 or y >= self.room_sem_map.shape[1]:
            return None
        sem_id = self.room_sem_map[x, y]
        # room boundary
        if sem_id == 0:
            return None
        else:
            return self.room_sem_id_to_sem_name[sem_id]

    def get_room_instance_by_point(self, xy):
        """
        Return the room instance given a point

        :param xy: 2D location in world reference frame (metric)
        :return: room instance that this point is in or None, if this point is not on the room segmentation map
        """

        x, y = self.world_to_seg_map(xy)
        if x < 0 or x >= self.room_ins_map.shape[0] or y < 0 or y >= self.room_ins_map.shape[1]:
            return None
        ins_id = self.room_ins_map[x, y]
        # room boundary
        if ins_id == 0:
            return None
        else:
            return self.room_ins_id_to_ins_name[ins_id]
