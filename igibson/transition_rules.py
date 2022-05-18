from abc import ABCMeta, abstractmethod

import numpy as np
from omni.isaac.utils._isaac_utils import math as math_utils

from igibson import ig_dataset_path
from igibson.object_states import *
from igibson.object_states.factory import get_states_for_ability
import igibson.utils.transform_utils as T


class BaseFilter(metaclass=ABCMeta):
    """Defines a filter to apply to objects."""

    @abstractmethod
    def __call__(self, obj):
        """Returns true if the given object passes the filter."""
        return False


class CategoryFilter(BaseFilter):
    """Filter for object categories."""

    def __init__(self, category):
        self.category = category

    def __call__(self, obj):
        return obj.category == self.category


class StateFilter(BaseFilter):
    """Filter for object states."""

    def __init__(self, state_type, state_value):
        self.state_type = state_type
        self.state_value = state_value

    def __call__(self, obj):
        if self.state_type not in obj.states:
            return False
        return obj.states[self.state_type].get_value() == self.state_value


class AbilityFilter(BaseFilter):
    """Filter for object abilities."""

    def __init__(self, ability):
        self.ability = ability

    def __call__(self, obj):
        return self.ability in obj._abilities


class OrFilter(BaseFilter):
    """Logical-or of a set of filters."""

    def __init__(self, filters):
        self.filters = filters

    def __call__(self, obj):
        return any(f(obj) for f in self.filters)


class AndFilter(BaseFilter):
    """Logical-and of a set of filters."""

    def __init__(self, filters):
        self.filters = filters

    def __call__(self, obj):
        return all(f(obj) for f in self.filters)


class BaseTransitionRule(metaclass=ABCMeta):
    """Defines a set of categories of objects and how to transition their states."""

    @abstractmethod
    def __init__(self, filters):
        """TransitionRule ctor.

        Args:
            filters: (tuple) Object filters that this rule cares about.
        """
        self.filters = filters

    @abstractmethod
    def condition(self, simulator,  *args):
        """Returns True if the rule applies to the object tuple."""
        pass

    @abstractmethod
    def transition(self, simulator,  *args):
        """Rule to apply for each tuples satisfying the condition."""
        pass


class TransitionRule(BaseTransitionRule):
    """A generic transition rule template used typically for simple rules."""

    def __init__(self, filters, condition_fn, transition_fn):
        super(TransitionRule, self).__init__(filters)
        self.condition_fn = condition_fn
        self.transition_fn = transition_fn

    def condition(self, simulator, *args):
        return self.condition_fn(*args)

    def transition(self, simulator, *args):
        return self.transition_fn(*args)


class SlicingRule(BaseTransitionRule):
    """Transition rule to apply to sliced / slicer object pairs."""

    def __init__(self):
        a_filter_sliceable = AbilityFilter("sliceable")
        a_filter_slicer = AbilityFilter("slicer")
        super(SlicingRule, self).__init__((a_filter_sliceable, a_filter_slicer))

    def condition(self, simulator, sliced_obj, slicer_obj):
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

    def transition(self, simulator, sliced_obj, slicer_obj):
        # Object parts offset annotation are w.r.t the base link of the whole object.
        sliced_obj.states[Sliced].set_value(True)
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
        print(f"Applied {SlicingRule.__name__} to {sliced_obj}")


"""See the following example for writing simple rules.

  TransitionRule(
      filters=(
          AndFilter(CategoryFilter("light"), StateFilter(ToggledOn, True)),
          CategoryFilter("key"),
      ),
      condition_fn=lambda light, key: dist(light, key) < 1,
      transition_fn=lambda light, key: light.states[ToggledOn].set_value(True),
  )
"""
DEFAULT_RULES = [
    SlicingRule(),
]
