from abc import ABCMeta, abstractmethod
from collections import namedtuple, OrderedDict

import numpy as np

import omnigibson as og
from omnigibson import og_dataset_path
from omnigibson.systems import *
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.object_states import *
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import BoundingBoxAPI


# Tuple of attributes of objects created in transitions.
_attrs_fields = ["category", "model", "name", "scale", "obj", "pos", "orn"]
ObjectAttrs = namedtuple(
    "ObjectAttrs", _attrs_fields, defaults=(None,) * len(_attrs_fields))

# Tuple of lists of objects to be added or removed returned from transitions.
TransitionResults = namedtuple(
    "TransitionResults", ["add", "remove"], defaults=([], []))


class BaseFilter(metaclass=ABCMeta):
    """Defines a filter to apply to objects."""
    # Class global variable for maintaining cached state
    # Maps tuple of unique filter inputs to cached output value (T / F)
    state = None

    @classmethod
    def __new__(cls, *args, **kwargs):
        """
        Initializes the cached state for this filter if it doesn't already exist
        """
        if cls.state is None:
            cls.state = OrderedDict()

        return super(BaseFilter, cls).__new__(cls)

    @classmethod
    def update(cls):
        """
        Updates the internal state by checking the filter status on all filter inputs
        """
        raise NotImplementedError()

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
    def __init__(self, individual_filters=None, group_filters=None):
        """TransitionRule ctor.

        Args:
            individual_filters (None or dict): Individual object filters that this filter cares about.
                For each name, filter key-value pair, the global transition rule step will produce tuples of valid
                filtered objects such that the cross product over all individual filter outputs occur.
                For example, if the individual filters are:

                    {"apple": CategoryFilter("apple"), "knife": CategoryFilter("knife")},

                the transition rule step will produce all 2-tuples of valid (apple, knife) combinations:

                    {"apple": apple_i, "knife": knife_j}

                based on the current instances of each object type in the scene and pass them to @self.condition as the
                @individual_objects entry.
                If None is specified, then no filter will be applied

            group_filters (None or dict): Group object filters that this filter cares about. For each name, filter
                key-value pair, the global transition rule step will produce a single dictionary of valid filtered
                objects.
                For example, if the group filters are:

                    {"apple": CategoryFilter("apple"), "knife": CategoryFilter("knife")},

                the transition rule step will produce the following dictionary:

                    {"apple": [apple0, apple1, ...], "knife": [knife0, knife1, ...]}

                based on the current instances of each object type in the scene and pass them to @self.condition
                as the @group_objects entry.
                If None is specified, then no filter will be applied
        """
        # Make sure at least one set of filters is specified -- in general, there should never be a rule
        # where no filter is specified
        assert not (individual_filters is None and group_filters is None),\
            "At least one of individual_filters or group_filters must be specified!"

        # Store the filters
        self.individual_filters = dict() if individual_filters is None else individual_filters
        self.group_filters = dict() if group_filters is None else group_filters

    def process(self, individual_objects, group_objects):
        """
        Processes this transition rule at the current simulator step. If @condition evaluates to True, then
        @transition will be executed.

        Args:
            individual_objects (dict): Dictionary mapping corresponding keys from @individual_filters to individual
                object instances where the filter is satisfied. Note: if @self.individual_filters is None or no values
                satisfy the filter, then this will be an empty dictionary
            group_objects (dict): Dictionary mapping corresponding keys from @group_filters to a list of individual
                object instances where the filter is satisfied. Note: if @self.group_filters is None or no values
                satisfy the filter, then this will be an empty dictionary

        Returns:
            2-tuple:
                - bool: Whether @self.condition is met
                - None or TransitionResults: Output from @self.transition (None if it was never executed)
        """
        should_transition = self.condition(individual_objects=individual_objects, group_objects=group_objects)
        return should_transition, \
               self.transition(individual_objects=individual_objects, group_objects=group_objects) \
                   if should_transition else None

    @property
    def requires_individual_filters(self):
        """
        Returns:
            bool: Whether this transition rule requires any specific filters
        """
        return len(self.individual_filters) > 0

    @property
    def requires_group_filters(self):
        """
        Returns:
            bool: Whether this transition rule requires any group filters
        """
        return len(self.group_filters) > 0

    @abstractmethod
    def condition(self, individual_objects, group_objects):
        """
        Returns True if the rule applies to the object tuple.

        Args:
            individual_objects (dict): Dictionary mapping corresponding keys from @individual_filters to individual
                object instances where the filter is satisfied. Note: if @self.individual_filters is None or no values
                satisfy the filter, then this will be an empty dictionary
            group_objects (dict): Dictionary mapping corresponding keys from @group_filters to a list of individual
                object instances where the filter is satisfied. Note: if @self.group_filters is None or no values
                satisfy the filter, then this will be an empty dictionary

        Returns:
            bool: Whether the condition is met or not
        """
        pass

    @abstractmethod
    def transition(self, individual_objects, group_objects):
        """
        Rule to apply for each set of objects satisfying the condition.

        Args:
            individual_objects (dict): Dictionary mapping corresponding keys from @individual_filters to individual
                object instances where the filter is satisfied. Note: if @self.individual_filters is None or no values
                satisfy the filter, then this will be an empty dictionary
            group_objects (dict): Dictionary mapping corresponding keys from @group_filters to a list of individual
                object instances where the filter is satisfied. Note: if @self.group_filters is None or no values
                satisfy the filter, then this will be an empty dictionary

        Returns:
            TransitionResults: results from the executed transition
        """
        pass


class GenericTransitionRule(BaseTransitionRule):
    """A generic transition rule template used typically for simple rules."""

    def __init__(self, individual_filters, group_filters, condition_fn, transition_fn):
        super(GenericTransitionRule, self).__init__(individual_filters, group_filters)
        self.condition_fn = condition_fn
        self.transition_fn = transition_fn

    def condition(self, individual_objects, group_objects):
        return self.condition_fn(individual_objects, group_objects)

    def transition(self, individual_objects, group_objects):
        return self.transition_fn(individual_objects, group_objects)


class SlicingRule(BaseTransitionRule):
    """Transition rule to apply to sliced / slicer object pairs."""

    def __init__(self):
        # Define an individual filter dictionary so we can track all valid combos of slicer - sliceable
        individual_filters = {ability: AbilityFilter(ability) for ability in ("sliceable", "slicer")}

        # Run super
        super().__init__(individual_filters=individual_filters)

    def condition(self, individual_objects, group_objects):
        slicer_obj, sliced_obj = individual_objects["slicer"], individual_objects["sliceable"]
        slicer_position = slicer_obj.states[Slicer].get_link_position()
        if slicer_position is None:
            return False

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
        normal_force = np.dot(sliced_c.impulse, sliced_c.normal) / sliced_c.dt
        if Sliced in sliced_obj.states:
            if (
                not sliced_obj.states[Sliced].get_value()
                and normal_force > sliced_obj.states[Sliced].slice_force
            ):
                # Slicer may contact the same body in multiple points, so
                # cut once since removing the object from the simulator
                return True
        return False

    def transition(self, individual_objects, group_objects):
        slicer_obj, sliced_obj = individual_objects["slicer"], individual_objects["sliceable"]
        # Object parts offset annotation are w.r.t the base link of the whole object.
        sliced_obj.states[Sliced].set_value(True)
        pos, orn = sliced_obj.get_position_orientation()

        t_results = TransitionResults()

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
            model_root_path = f"{og_dataset_path}/objects/{part_category}/{part_model}"
            usd_path = f"{model_root_path}/usd/{part_model}.usd"

            # Calculate global part pose.
            part_pos = np.array(part_pos) + pos
            part_orn = T.quat_multiply(np.array(part_orn), orn)

            # Circular import.
            from omnigibson.objects.dataset_object import DatasetObject

            part_obj = DatasetObject(
                prim_path=f"/World/{part_obj_name}",
                usd_path=usd_path,
                category=part_category,
                name=part_obj_name,
                scale=sliced_obj.scale,
                abilities={}
            )

            # Add the new object to the results.
            new_obj_attrs = ObjectAttrs(
                obj=part_obj, pos=np.array(part_pos), orn=np.array(part_orn))
            t_results.add.append(new_obj_attrs)

        # Delete original object from stage.
        t_results.remove.append(sliced_obj)
        print(f"Applied {SlicingRule.__name__} to {sliced_obj}")
        return t_results


# TODO: Replace with a more standard API when available.
def _contained_objects(scene, container_obj):
    """Returns a list of all objects ``inside'' the container object."""
    bbox = BoundingBoxAPI.compute_aabb(container_obj.prim_path)
    contained_objs = []
    for obj in scene.objects:
        if obj == container_obj:
            continue
        if BoundingBoxAPI.aabb_contains_point(obj.get_position(), bbox):
            contained_objs.append(obj)
    return contained_objs


class ContainerRule(BaseTransitionRule):
    """Rule to apply to a container and a set of objects that may be inside."""

    def __init__(self, trigger_steps, final_obj_attrs, container_filter, *contained_filters):
        # Should be in this order to have the container object come first.
        super(ContainerRule, self).__init__((container_filter, *contained_filters))
        self.obj_attrs = final_obj_attrs
        self.trigger_steps = trigger_steps
        self._current_steps = 1
        self._counter = 0

    def condition(self, container_obj, *contained_objs):
        if (
            ToggledOn in container_obj.states and
            not container_obj.states[ToggledOn].get_value()
        ):
            return False
        # Check all objects inside the container against the expected objects.
        all_contained_objs = _contained_objects(og.sim.scene, container_obj)
        contained_prim_paths = set(obj.prim_path for obj in contained_objs)
        all_contained_prim_paths = set(obj.prim_path for obj in all_contained_objs)
        if contained_prim_paths != all_contained_prim_paths:
            return False
        # Check if the trigger step has been reached.
        if self._current_steps < self.trigger_steps:
            self._current_steps += 1
            return False
        self._current_steps = 1
        return True

    def transition(self, container_obj, *contained_objs):
        t_results = TransitionResults()

        # Create a new object to be added.
        all_pos, all_orn = [], []
        for contained_obj in contained_objs:
            pos, orn = contained_obj.get_position_orientation()
            all_pos.append(pos)
            all_orn.append(orn)

        category = self.obj_attrs.category
        model = self.obj_attrs.model
        name = f"{self.obj_attrs.name}_{self._counter}"
        self._counter += 1
        scale = self.obj_attrs.scale

        model_root_path = f"{og_dataset_path}/objects/{category}/{model}"
        usd_path = f"{model_root_path}/usd/{model}.usd"

        final_obj = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=category,
            name=f"{name}",
            scale=scale)

        final_obj_attrs = ObjectAttrs(
            obj=final_obj, pos=np.mean(all_pos, axis=0), orn=np.mean(all_orn, axis=0))
        t_results.add.append(final_obj_attrs)

        # Delete all objects inside the container.
        for contained_obj in contained_objs:
            t_results.remove.append(contained_obj)

        # Turn off the container, otherwise things would turn into garbage.
        if ToggledOn in container_obj.states:
            container_obj.states[ToggledOn].set_value(False)
        print(f"Applied {ContainerRule.__name__} to {container_obj}")
        return t_results


class ContainerGarbageRule(BaseTransitionRule):
    """Rule to apply to a container to turn what remain inside into garbage.

    This rule is used as a catch-all rule for containers to turn objects inside
    the container that did not match any other legitimate rules all into a
    single garbage object.
    """

    def __init__(self, garbage_obj_attrs, container_filter):
        """Ctor for ContainerGarbageRule.

        Args:
            garbage_obj_attrs: (ObjectAttrs) a namedtuple containing the
                attributes of garbage objects to be created.
            container_filter: (BaseFilter) a filter for the container.
        """
        super(ContainerGarbageRule, self).__init__((container_filter,))
        self.obj_attrs = garbage_obj_attrs
        self._cached_contained_objs = None
        self._counter = 0

    def condition(self, container_obj):
        if (
            ToggledOn in container_obj.states and
            not container_obj.states[ToggledOn].get_value()
        ):
            return False
        self._cached_contained_objs = _contained_objects(og.sim.scene, container_obj)
        # Skip in case only a garbage object is inside the container.
        if len(self._cached_contained_objs) == 1:
            contained_obj = self._cached_contained_objs[0]
            if (contained_obj.category == self.obj_attrs.category
                    and contained_obj.name.startswith(self.obj_attrs.name)):
                return False
        return bool(self._cached_contained_objs)

    def transition(self, container_obj):
        t_results = TransitionResults()

        # Create a single garbage object to be added.
        all_pos, all_orn = [], []
        for contained_obj in self._cached_contained_objs:
            pos, orn = contained_obj.get_position_orientation()
            all_pos.append(pos)
            all_orn.append(orn)

        category = self.obj_attrs.category
        model = self.obj_attrs.model
        name = f"{self.obj_attrs.name}_{self._counter}"
        self._counter += 1
        scale = self.obj_attrs.scale

        model_root_path = f"{og_dataset_path}/objects/{category}/{model}"
        usd_path = f"{model_root_path}/usd/{model}.usd"

        garbage_obj = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=category,
            name=f"{name}",
            scale=scale)

        garbage_obj_attrs = ObjectAttrs(
            obj=garbage_obj, pos=np.mean(all_pos, axis=0), orn=np.mean(all_orn, axis=0))
        t_results.add.append(garbage_obj_attrs)

        # Remove all contained objects.
        for contained_obj in self._cached_contained_objs:
            t_results.remove.append(contained_obj)

        # Turn off the container after the transition and reset things.
        if ToggledOn in container_obj.states:
            container_obj.states[ToggledOn].set_value(False)
        self._cached_contained_objs = None
        return t_results


"""See the following example for writing simple rules.

  GenericTransitionRule(
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
