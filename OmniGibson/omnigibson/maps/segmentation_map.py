import os

import cv2
import torch as th

from omnigibson.macros import gm
from omnigibson.maps.map_base import BaseMap
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.python_utils import torch_delete
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class SegmentationMap(BaseMap):
    """
    Segmentation map for computing connectivity within the scene
    """

    def __init__(
        self,
        scene_dir,
        map_resolution=0.1,
        floor_heights=(0.0,),
    ):
        """
        Args:
            scene_dir (str): path to the scene directory from which segmentation info will be extracted
            map_resolution (float): map resolution (m)
            floor_heights (list of float): heights of the floors for this segmentation map
        """
        # Store internal values
        self.scene_dir = scene_dir
        self.map_default_resolution = 0.01  # Default floor plan images are in meters
        self.floor_heights = floor_heights

        # Other values that will be loaded at runtime
        self.room_sem_name_to_sem_id = None
        self.room_sem_id_to_sem_name = None
        self.room_ins_name_to_ins_id = None
        self.room_ins_id_to_ins_name = None
        self.room_sem_name_to_ins_name = None
        self.room_ins_map = None
        self.room_sem_map = None

        # Run super call
        super().__init__(map_resolution=map_resolution)

        # Load the map
        self.load_map()

    def _load_map(self):
        layout_dir = os.path.join(self.scene_dir, "layout")
        room_seg_imgs = os.path.join(layout_dir, "floor_insseg_0.png")
        img_ins = cv2.imread(room_seg_imgs, cv2.IMREAD_GRAYSCALE)
        room_seg_imgs = os.path.join(layout_dir, "floor_semseg_0.png")
        img_sem = cv2.imread(room_seg_imgs, cv2.IMREAD_GRAYSCALE)
        height = img_ins.shape[0]
        width = img_ins.shape[1]
        assert height == width, "room seg map is not a square"
        assert img_ins.shape == img_sem.shape, "semantic and instance seg maps have different sizes"
        map_size = int(height * self.map_default_resolution / self.map_resolution)
        img_ins = th.tensor(cv2.resize(img_ins, (map_size, map_size), interpolation=cv2.INTER_NEAREST))
        img_sem = th.tensor(cv2.resize(img_sem, (map_size, map_size), interpolation=cv2.INTER_NEAREST))

        room_categories = os.path.join(get_dataset_path("behavior-1k-assets"), "metadata", "room_categories.txt")
        with open(room_categories, "r") as fp:
            room_cats = [line.rstrip() for line in fp.readlines()]

        sem_id_to_ins_id = {}
        unique_ins_ids = th.unique(img_ins)
        unique_ins_ids = torch_delete(unique_ins_ids, 0)
        for ins_id in unique_ins_ids:
            ins_id = ins_id.item()
            # find one pixel for each ins id
            x, y = th.where(img_ins == ins_id)
            # retrieve the correspounding sem id
            sem_id = img_sem[x[0], y[0]].item()
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

        return map_size

    def get_random_point_by_room_type(self, room_type):
        """
        Sample a random point on the given a specific room type @room_type.

        Args:
            room_type (str): Room type to sample random point (e.g.: "bathroom")

        Returns:
            2-tuple:
                - int: floor number. This is always 0
                - 3-array: (x,y,z) randomly sampled point in a room of type @room_type
        """
        if room_type not in self.room_sem_name_to_sem_id:
            log.warning("room_type [{}] does not exist.".format(room_type))
            return None, None

        sem_id = self.room_sem_name_to_sem_id[room_type]
        valid_idx = th.tensor(th.where(self.room_sem_map == sem_id))
        random_point_map = valid_idx[:, th.randint(valid_idx.shape[1])]

        x, y = self.map_to_world(random_point_map)
        # assume only 1 floor
        floor = 0
        z = self.floor_heights[floor]
        return floor, th.tensor([x, y, z])

    def get_random_point_by_room_instance(self, room_instance):
        """
        Sample a random point on the given a specific room instance @room_instance.

        Args:
            room_instance (str): Room instance to sample random point (e.g.: "bathroom_1")

        Returns:
            2-tuple:
                - int: floor number. This is always 0
                - 3-array: (x,y,z) randomly sampled point in room @room_instance
        """
        if room_instance not in self.room_ins_name_to_ins_id:
            log.warning("room_instance [{}] does not exist.".format(room_instance))
            return None, None

        ins_id = self.room_ins_name_to_ins_id[room_instance]
        valid_idx = th.tensor(th.where(self.room_ins_map == ins_id))
        random_point_map = valid_idx[:, th.randint(valid_idx.shape[1])]

        x, y = self.map_to_world(random_point_map)
        # assume only 1 floor
        floor = 0
        z = self.floor_heights[floor]
        return floor, th.tensor([x, y, z])

    def get_room_type_by_point(self, xy):
        """
        Return the room type given a point

        Args:
            xy (2-array): 2D location in world reference frame (in metric space)

        Returns:
            None or str: room type that this point is in or None, if this point is not on the room segmentation map
        """
        x, y = self.world_to_map(xy)
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
        Return the room type given a point

        Args:
            xy (2-array): 2D location in world reference frame (in metric space)

        Returns:
            None or str: room instance that this point is in or None, if this point is not on the room segmentation map
        """
        x, y = self.world_to_map(xy)
        if x < 0 or x >= self.room_ins_map.shape[0] or y < 0 or y >= self.room_ins_map.shape[1]:
            return None
        ins_id = self.room_ins_map[x, y].item()
        # room boundary
        if ins_id == 0:
            return None
        else:
            return self.room_ins_id_to_ins_name[ins_id]
