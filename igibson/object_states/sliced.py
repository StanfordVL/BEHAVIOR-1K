import numpy as np
from collections import OrderedDict
from igibson import ig_dataset_path
from igibson.macros import create_module_macros
import igibson.utils.transform_utils as T
from igibson.object_states import *
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState


# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: propagate dusty/stained to object parts
m.DEFAULT_SLICE_FORCE = 10
m.STASH_POSITION = [-100, -100, -100]


class Sliced(AbsoluteObjectState, BooleanState):
    def __init__(self, obj, slice_force=m.DEFAULT_SLICE_FORCE):
        super(Sliced, self).__init__(obj)
        self.slice_force = slice_force
        self.value = False

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        if self.value == new_value:
            return True

        if not new_value:
            raise ValueError("Cannot set sliced from True to False")

        self.value = new_value

        # Object parts offset annotation are w.r.t the base link of the whole object
        pos, orn = self.obj.get_position_orientation()

        # load object parts
        for _, part_idx in enumerate(self.obj.metadata["object_parts"]):
            # list of dicts gets replaced by {'0':dict, '1':dict, ...}
            part = self.obj.metadata["object_parts"][part_idx]
            part_category = part["category"]
            part_model = part["model"]
            # Scale the offset accordingly
            part_pos = part["pos"] * self.obj.scale
            part_orn = part["orn"]
            part_obj_name = f"{self.obj.name}_part_{part_idx}"
            model_root_path = f"{ig_dataset_path}/objects/{part_category}/{part_model}"
            usd_path = f"{model_root_path}/usd/{part_model}.usd"

            # calculate global part pose
            part_pos, part_orn = T.pose_transform(pos, orn, np.array(part_pos), np.array(part_orn))

            # circular import
            from igibson.objects.dataset_object import DatasetObject

            part_obj = DatasetObject(
                prim_path=f"/World/{part_obj_name}",
                usd_path=usd_path,
                category=part_category,
                name=part_obj_name,
                scale=self.obj.scale,
                abilities={}
            )
            
            # add to stage
            self._simulator.import_object(part_obj, auto_initialize=True)
            # inherit parent position and orientation
            part_obj.set_position_orientation(position=np.array(part_pos),
                                              orientation=np.array(part_orn))

        # delete original object from simulator
        self._simulator.remove_object(self.obj)

        return True

    @property
    def settable(self):
        return True

    def _dump_state(self):
        return OrderedDict(sliced=self.value)

    def _load_state(self, state):
        self.value = state["sliced"]

    def _serialize(self, state):
        return np.array([float(state["sliced"])])

    def _deserialize(self, state):
        return OrderedDict(sliced=(state[0] == 1.0)), 1
