from abc import ABCMeta, abstractmethod

import numpy as np
from omni.isaac.utils._isaac_utils import math as math_utils

from igibson import ig_dataset_path
from igibson.object_states import *
import igibson.utils.transform_utils as T


class TransitionRule(metaclass=ABCMeta):
    """Defines a set of categories of objects and how to transition their states."""

    @abstractmethod
    def __init__(self, categories):
        """TransitionRule ctor.

        Args:
            categories: (tuple) Categories of objects that this rule cares about.
        """
        self.categories = categories

    @abstractmethod
    def condition(self, simulator, obj_tuple):
        """Returns True if the rule applies to the object tuple."""
        pass

    @abstractmethod
    def transition(self, simulator, obj_tuple):
        """Rule to apply for each tuples satisfying the condition."""
        pass


# TODO: Perhaps the rules should be in a separate file.
class DemoRule(TransitionRule):
    """An example transition rule."""

    def __init__(self):
        super(DemoRule, self).__init__(("apple", "table_knife"))

    def condition(self, simulator, obj_tuple):
        return True

    def transition(self, simulator, obj_tuple):
        print(f"Applied {DemoRule.__name__} to {obj_tuple}")


class SlicingRule(TransitionRule):
    """Transition rule to apply to sliced / slicer object pairs."""

    def __init__(self):
        super(SlicingRule, self).__init__(("apple", "table_knife"))

    def condition(self, simulator, obj_tuple):
        sliced_obj, slicer_obj = obj_tuple

        slicer_position = slicer_obj.states[Slicer].get_link_position()
        if slicer_position is None:
            False

        contact_list = slicer_obj.states[ContactBodies].get_value()
        sliced_link_paths = {link.prim_path for link in sliced_obj.links.values()}
        sliced_c = None
        for c in contact_list:
            if c.body0 in sliced_link_paths or c.body1 in sliced_link_paths:
                sliced_c = c
                break
        if not sliced_c:
            return False

        # Calculate the normal force applied to the contact object.
        normal_force = math_utils.dot(sliced_c.impulse, sliced_c.normal) / sliced_c.dt
        if Sliced in sliced_obj.states:
            if (
                not sliced_obj.states[Sliced].get_value()
                and normal_force > sliced_obj.states[Sliced].slice_force
            ):
                # Slicer may contact the same body in multiple points, so
                # cut once since removing the object from the simulator
                return True
        return False

    def transition(self, simulator, obj_tuple):
        sliced_obj, _ = obj_tuple

        # Object parts offset annotation are w.r.t the base link of the whole object.
        pos, orn = sliced_obj.get_position_orientation()

        # Load object parts.
        for _, part_idx in enumerate(sliced_obj.metadata["object_parts"]):
            # List of dicts gets replaced by {'0':dict, '1':dict, ...}.
            part = sliced_obj.metadata["object_parts"][part_idx]
            part_category = part["category"]
            part_model = part["model"]
            # Scale the offset accordingly.
            part_pos = part["pos"] * sliced_obj.scale
            part_orn = part["orn"]
            part_obj_name = f"{sliced_obj.name}_part_{part_idx}"
            model_root_path = f"{ig_dataset_path}/objects/{part_category}/{part_model}"
            usd_path = f"{model_root_path}/usd/{part_model}.usd"

            # Calculate global part pose.
            part_pos = np.array(part_pos) + pos
            part_orn = T.quat_multiply(np.array(part_orn), orn)

            # Circular import.
            from igibson.objects.dataset_object import DatasetObject

            part_obj = DatasetObject(
                prim_path=f"/World/{part_obj_name}",
                usd_path=usd_path,
                category=part_category,
                name=part_obj_name,
                scale=sliced_obj.scale,
                abilities={}
            )

            # Add to stage.
            simulator.import_object(part_obj, auto_initialize=False)
            # Inherit parent position and orientation.
            part_obj.set_position_orientation(position=np.array(part_pos),
                                              orientation=np.array(part_orn))

        # Delete original object from stage.
        simulator.remove_object(sliced_obj)
        print(f"Applied {SlicingRule.__name__} to {obj_tuple}")


DEFAULT_RULES = [
    # DemoRule(),
    SlicingRule(),
]
