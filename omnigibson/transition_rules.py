import operator
from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict
import numpy as np
import json
from copy import copy
import itertools
import os
from collections import defaultdict
import networkx as nx

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.systems import get_system, is_system_active, PhysicalParticleSystem, VisualParticleSystem, REGISTERED_SYSTEMS
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.object_states import *
from omnigibson.object_states.factory import get_system_states
from omnigibson.object_states.object_state_base import AbsoluteObjectState, RelativeObjectState
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import Registerable, classproperty, subclass_factory
from omnigibson.utils.registry_utils import Registry
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import disclaimer, create_module_logger
from omnigibson.utils.usd_utils import RigidContactAPI
import bddl
from bddl.object_taxonomy import ObjectTaxonomy
from omnigibson.utils.bddl_utils import is_substance_synset, get_system_name_by_synset, SUPPORTED_PREDICATES

# Create module logger
log = create_module_logger(module_name=__name__)

# Create object taxonomy
OBJECT_TAXONOMY = ObjectTaxonomy()

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Default melting temperature
m.MELTING_TEMPERATURE = 100.0

# Default "trash" system if an invalid mixing rule transition occurs
m.DEFAULT_GARBAGE_SYSTEM = "sludge"

# Tuple of attributes of objects created in transitions.
# `states` field is dict mapping object state class to arguments to pass to setter for that class
_attrs_fields = ["category", "model", "name", "scale", "obj", "pos", "orn", "bb_pos", "bb_orn", "states", "callback"]
# States: dict: mapping state nameargs to pass to the state setter for @obj in order to set the object state
# callback: function: signature callback(obj) -> None to execute after states are set, if any
ObjectAttrs = namedtuple(
    "ObjectAttrs", _attrs_fields, defaults=(None,) * len(_attrs_fields))

# Tuple of lists of objects to be added or removed returned from transitions, if not None
TransitionResults = namedtuple(
    "TransitionResults", ["add", "remove"], defaults=(None, None))

# Mapping from transition rule json files to rule classe names
_JSON_FILES_TO_RULES = {
    "dicing.json": ["DicingRule"],
    "heat_cook.json": ["CookingObjectRule", "CookingSystemRule"],
    "melting.json": ["MeltingRule"],
    "mixing_stick.json": ["MixingToolRule"],
    "single_toggleable_machine.json": ["ToggleableMachineRule"],
    "slicing.json": ["SlicingRule"],
    "substance_cooking.json": ["CookingPhysicalParticleRule"],
    "substance_watercooking.json": ["CookingPhysicalParticleRule"],
    # TODO: washer and dryer
}
# Global dicts that will contain mappings
REGISTERED_RULES = dict()

class TransitionRuleAPI:
    """
    Monolithic class containing methods to check and execute arbitrary discrete state transitions within the simulator
    """
    # Set of active rules
    ACTIVE_RULES = set()

    # Maps BaseObject instances to dictionary with the following keys:
    # "states": None or dict mapping object states to arguments to set for that state when the object is initialized
    # "callback": None or function to execute when the object is initialized
    _INIT_INFO = dict()

    @classmethod
    def get_rule_candidates(cls, rule, objects):
        """
        Computes valid input object candidates for transition rule @rule, if any exist

        Args:
            rule (BaseTransitionRule): Transition rule whose candidates should be computed
            objects (list of BaseObject): List of objects that will be used to compute object candidates

        Returns:
            None or dict: None if no valid candidates are found, otherwise mapping from filter key to list of object
                instances that satisfy that filter
        """
        obj_candidates = rule.get_object_candidates(objects=objects)
        n_filters_satisfied = sum(len(candidates) > 0 for candidates in obj_candidates.values())
        # Return object candidates if all filters are met, otherwise return None
        return obj_candidates if n_filters_satisfied == len(rule.candidate_filters) else None

    @classmethod
    def prune_active_rules(cls):
        """
        Prunes the active transition rules, removing any whose filter requirements are not satisfied by all current
        objects on the scene. Useful when the current object set changes, e.g.: an object is removed from the simulator
        """
        # Need explicit tuple to iterate over because refresh_rules mutates the ACTIVE_RULES set in place
        cls.refresh_rules(rules=tuple(cls.ACTIVE_RULES))

    @classmethod
    def refresh_all_rules(cls):
        """
        Refreshes all registered rules given the current set of objects in the scene
        """
        global RULES_REGISTRY

        # Clear all active rules
        cls.ACTIVE_RULES = set()

        # Refresh all registered rules
        cls.refresh_rules(rules=RULES_REGISTRY.objects)

    @classmethod
    def refresh_rules(cls, rules):
        """
        Refreshes the specified transition rules @rules based on current set of objects in the simulator.
        This will prune any pre-existing rules in cls.ACTIVE_RULES if no valid candidates are found, or add / update
        the entry if valid candidates are found

        Args:
            rules (list of BaseTransitionRule): List of transition rules whose candidate lists should be refreshed
        """
        objects = og.sim.scene.objects
        for rule in rules:
            # Check if rule is still valid, if so, update its entry
            object_candidates = cls.get_rule_candidates(rule=rule, objects=objects)

            # Update candidates if valid, otherwise pop the entry if it exists in cls.ACTIVE_RULES
            if object_candidates is not None:
                # We have a valid rule which should be active, so grab and initialize all of its conditions
                # NOTE: The rule may ALREADY exist in ACTIVE_RULES, but we still need to refresh its candidates because
                # the relevant candidate set / information for the rule + its conditions may have changed given the
                # new set of objects
                rule.refresh(object_candidates=object_candidates)
                cls.ACTIVE_RULES.add(rule)
            elif rule in cls.ACTIVE_RULES:
                cls.ACTIVE_RULES.remove(rule)

    @classmethod
    def step(cls):
        """
        Steps all active transition rules, checking if any are satisfied, and if so, executing their transition
        """
        # First apply any transition object init states from before, and then clear the dictionary
        for obj, info in cls._INIT_INFO.items():
            if info["states"] is not None:
                for state, args in info["states"].items():
                    obj.states[state].set_value(*args)
            if info["callback"] is not None:
                info["callback"](obj)
        cls._INIT_INFO = dict()

        # Iterate over all active rules and process the rule for every valid object candidate combination
        # Cast to list before iterating since ACTIVE_RULES may get updated mid-iteration
        added_obj_attrs = []
        removed_objs = []
        for rule in tuple(cls.ACTIVE_RULES):
            output = rule.step()
            # Store objects to be added / removed if we have a valid output
            if output is not None:
                added_obj_attrs += output.add
                removed_objs += output.remove

        cls.execute_transition(added_obj_attrs=added_obj_attrs, removed_objs=removed_objs)

    @classmethod
    def execute_transition(cls, added_obj_attrs, removed_objs):
        """
        Executes the transition for the given added and removed objects.

        :param added_obj_attrs: List of ObjectAttrs instances to add to the scene
        :param removed_objs: List of BaseObject instances to remove from the scene
        """
        # Process all transition results
        if len(removed_objs) > 0:
            disclaimer(
                f"We are attempting to remove objects during the transition rule phase of the simulator step.\n"
                f"However, Omniverse currently has a bug when using GPU dynamics where a segfault will occur if an "
                f"object in contact with another object is attempted to be removed.\n"
                f"This bug should be fixed by the next Omniverse release.\n"
                f"In the meantime, we instead teleport these objects to a graveyard location located far outside of "
                f"the scene."
            )
            # First remove pre-existing objects
            for i, removed_obj in enumerate(removed_objs):
                og.sim.remove_object(removed_obj)

        # Then add new objects
        if len(added_obj_attrs) > 0:
            state = og.sim.dump_state()
            for added_obj_attr in added_obj_attrs:
                new_obj = added_obj_attr.obj
                og.sim.import_object(new_obj)
                # By default, added_obj_attr is populated with all Nones -- so these will all be pass-through operations
                # unless pos / orn (or, conversely, bb_pos / bb_orn) is specified
                if added_obj_attr.pos is not None or added_obj_attr.orn is not None:
                    new_obj.set_position_orientation(position=added_obj_attr.pos, orientation=added_obj_attr.orn)
                elif isinstance(new_obj, DatasetObject) and \
                        (added_obj_attr.bb_pos is not None or added_obj_attr.bb_orn is not None):
                    new_obj.set_bbox_center_position_orientation(position=added_obj_attr.bb_pos,
                                                                 orientation=added_obj_attr.bb_orn)
                else:
                    raise ValueError("Expected at least one of pos, orn, bb_pos, or bb_orn to be specified in ObjectAttrs!")
                # Additionally record any requested states if specified to be updated during the next transition step
                if added_obj_attr.states is not None or added_obj_attr.callback is not None:
                    cls._INIT_INFO[new_obj] = {
                        "states": added_obj_attr.states,
                        "callback": added_obj_attr.callback,
                    }

    @classmethod
    def clear(cls):
        """
        Clears any internal state when the simulator is restarted (e.g.: when a new stage is opened)
        """
        global RULES_REGISTRY

        # Clear internal dictionaries
        cls.ACTIVE_RULES = set()
        cls._INIT_INFO = dict()


class ObjectCandidateFilter(metaclass=ABCMeta):
    """
    Defines a filter to apply for inferring which objects are valid candidates for checking a transition rule's
    condition requirements.

    NOTE: These filters should describe STATIC properties about an object -- i.e.: properties that should NOT change
    at runtime, once imported
    """
    @abstractmethod
    def __call__(self, obj):
        """Returns true if the given object passes the filter."""
        return False


class CategoryFilter(ObjectCandidateFilter):
    """Filter for object categories."""

    def __init__(self, category):
        self.category = category

    def __call__(self, obj):
        return obj.category == self.category


class AbilityFilter(ObjectCandidateFilter):
    """Filter for object abilities."""

    def __init__(self, ability):
        self.ability = ability

    def __call__(self, obj):
        return self.ability in obj._abilities


class NameFilter(ObjectCandidateFilter):
    """Filter for object names."""

    def __init__(self, name):
        self.name = name

    def __call__(self, obj):
        return self.name in obj.name


class NotFilter(ObjectCandidateFilter):
    """Logical-not of a filter"""
    def __init__(self, f):
        self.f = f

    def __call__(self, obj):
        return not self.f(obj)


class OrFilter(ObjectCandidateFilter):
    """Logical-or of a set of filters."""

    def __init__(self, filters):
        self.filters = filters

    def __call__(self, obj):
        return any(f(obj) for f in self.filters)


class AndFilter(ObjectCandidateFilter):
    """Logical-and of a set of filters."""

    def __init__(self, filters):
        self.filters = filters

    def __call__(self, obj):
        return all(f(obj) for f in self.filters)


class RuleCondition:
    """
    Defines a transition rule condition for filtering a given set of input object candidates.

    NOTE: These filters should describe DYNAMIC properties about object candidates -- i.e.: properties that MAY change
    at runtime, once imported
    """
    def refresh(self, object_candidates):
        """
        Refreshes any internal state for this rule condition, given set of input object candidates @object_candidates

        Args:
            object_candidates (dict): Maps filter name to valid object(s) that satisfy that filter
        """
        # No-op by default
        pass

    @abstractmethod
    def __call__(self, object_candidates):
        """
        Filters @object_candidates and updates the candidates in-place, returning True if there are still valid
        candidates

        Args:
            object_candidates (dict): Maps filter name to valid object(s) that satisfy that filter

        Returns:
            bool: Whether there are still valid candidates in @object_candidates
        """
        # Default is False
        return False

    @property
    def modifies_filter_names(self):
        """
        Returns:
            set: Filter name(s) whose values may be modified in-place by this condition
        """
        raise NotImplementedError


class TouchingAnyCondition(RuleCondition):
    """
    Rule condition that prunes object candidates from @filter_1_name, only keeping any that are touching any object
    from @filter_2_name
    """
    def __init__(self, filter_1_name, filter_2_name):
        """
        Args:
            filter_1_name (str): Name of the filter whose object candidates will be pruned based on whether or not
                they are touching any object from @filter_2_name
            filter_2_name (str): Name of the filter whose object candidates will be used to prune the candidates from
                @filter_1_name
        """
        self._filter_1_name = filter_1_name
        self._filter_2_name = filter_2_name

        # Will be filled in during self.initialize
        # Maps object to the list of rigid body idxs in the global contact matrix corresponding to filter 1
        self._filter_1_idxs = None

        # If optimized, filter_2_idxs will be used, otherwise filter_2_bodies will be used!
        # Maps object to the list of rigid body idxs in the global contact matrix corresponding to filter 2
        self._filter_2_idxs = None
        # Maps object to set of rigid bodies corresponding to filter 2
        self._filter_2_bodies = None

        # Flag whether optimized call can be used
        self._optimized = None

    def refresh(self, object_candidates):
        # Check whether we can use optimized computation or not -- this is determined by whether or not any objects
        # in our collision set are kinematic only
        self._optimized = not np.any([obj.kinematic_only or obj.prim_type == PrimType.CLOTH
                                  for f in (self._filter_1_name, self._filter_2_name) for obj in object_candidates[f]])

        if self._optimized:
            # Register idx mappings
            self._filter_1_idxs = {obj: [RigidContactAPI.get_body_idx(link.prim_path) for link in obj.links.values()]
                                for obj in object_candidates[self._filter_1_name]}
            self._filter_2_idxs = {obj: [RigidContactAPI.get_body_idx(link.prim_path) for link in obj.links.values()]
                                for obj in object_candidates[self._filter_2_name]}
        else:
            # Register body mappings
            self._filter_2_bodies = {obj: set(obj.links.values()) for obj in object_candidates[self._filter_2_name]}

    def __call__(self, object_candidates):
        # Keep any object that has non-zero impulses between itself and any of the @filter_2_name's objects
        objs = []

        if self._optimized:
            # Get all impulses
            impulses = RigidContactAPI.get_all_impulses()
            idxs_to_check = np.concatenate([self._filter_2_idxs[obj] for obj in object_candidates[self._filter_2_name]])
            # Batch check for each object
            for obj in object_candidates[self._filter_1_name]:
                if np.any(impulses[self._filter_1_idxs[obj]][:, idxs_to_check]):
                    objs.append(obj)
        else:
            # Manually check contact
            filter_2_bodies = set.union(*(self._filter_2_bodies[obj] for obj in object_candidates[self._filter_2_name]))
            for obj in object_candidates[self._filter_1_name]:
                if len(obj.states[ContactBodies].get_value().intersection(filter_2_bodies)) > 0:
                    objs.append(obj)

        # Update candidates
        object_candidates[self._filter_1_name] = objs

        # If objs is empty, return False, otherwise, True
        return len(objs) > 0

    @property
    def modifies_filter_names(self):
        # Only modifies values from filter 1
        return {self._filter_1_name}


class StateCondition(RuleCondition):
    """
    Rule condition that checks all objects from @filter_name whether a state condition is equal to @val for
    """
    def __init__(
            self,
            filter_name,
            state,
            val,
            op=operator.eq,
    ):
        """
        Args:
            filter_name (str): Name of the filter whose object candidates will be pruned based on whether or not
                the state @state's value is equal to @val
            state (BaseObjectState): Object state whose value should be queried as a rule condition
            val (any): The value @state should be in order for this condition to be satisfied
            op (function): Binary operator to apply between @state's getter and @val. Default is operator.eq,
                which does state.get_value() == val.
                Expected signature:
                    def op(state_getter, val) --> bool
        """
        self._filter_name = filter_name
        self._state = state
        self._val = val
        self._op = op

    def __call__(self, object_candidates):
        # Keep any object whose states are satisfied
        object_candidates[self._filter_name] = \
            [obj for obj in object_candidates[self._filter_name] if self._op(obj.states[self._state].get_value(), self._val)]

        # Condition met if any object meets the condition
        return len(object_candidates[self._filter_name]) > 0

    @property
    def modifies_filter_names(self):
        return {self._filter_name}


class ChangeConditionWrapper(RuleCondition):
    """
    Rule condition wrapper that checks whether the output from @condition
    """
    def __init__(
            self,
            condition,
    ):
        """
        Args:
            condition (RuleCondition): Condition whose output will be additionally filtered whether or not its relevant
                values have changed since the previous time this condition was called
        """
        self._condition = condition
        self._last_valid_candidates = None

    def refresh(self, object_candidates):
        # Refresh nested condition
        self._condition.refresh(object_candidates=object_candidates)

        # Clear last valid candidates
        self._last_valid_candidates = {filter_name: set() for filter_name in self.modifies_filter_names}

    def __call__(self, object_candidates):
        # Call wrapped method first
        valid = self._condition(object_candidates=object_candidates)

        # Iterate over all current candidates -- if there's a mismatch in last valid candidates and current,
        # then we store it, otherwise, we don't
        for filter_name in self.modifies_filter_names:
            # Compute current valid candidates
            objs = [obj for obj in object_candidates[filter_name] if obj not in self._last_valid_candidates[filter_name]]
            # Store last valid objects -- these are all candidates that were validated by self._condition at the
            # current timestep
            self._last_valid_candidates[filter_name] = set(object_candidates[filter_name])
            # Update current object candidates with the change-filtered ones
            object_candidates[filter_name] = objs
            valid = valid and len(objs) > 0

        # Valid if any object conditions have changed and we still have valid objects
        return valid

    @property
    def modifies_filter_names(self):
        # Return wrapped names
        return self._condition.modifies_filter_names


class OrConditionWrapper(RuleCondition):
    """
    Logical OR between multiple RuleConditions
    """
    def __init__(self, conditions):
        """
        Args:
            conditions (list of RuleConditions): Conditions to take logical OR over. This will generate
                the UNION of all pruning between the two sets
        """
        self._conditions = conditions

    def refresh(self, object_candidates):
        # Refresh nested conditions
        for condition in self._conditions:
            condition.refresh(object_candidates=object_candidates)

    def __call__(self, object_candidates):
        # Iterate over all conditions and aggregate their results
        pruned_candidates = dict()
        for condition in self._conditions:
            # Copy the candidates because they get modified in place
            pruned_candidates[condition] = copy(object_candidates)
            condition(object_candidates=pruned_candidates[condition])

        # Now, take the union over all keys in the candidates --
        # if the result is empty, then we immediately return False
        for filter_name, old_candidates in object_candidates.keys():
            # If an entry was already empty, we skip it
            if len(old_candidates) == 0:
                continue
            object_candidates[filter_name] = \
                list(set(np.concatenate([candidates[filter_name] for candidates in pruned_candidates.values()])))
            if len(object_candidates[filter_name]) == 0:
                return False

        return True

    @property
    def modifies_filter_names(self):
        # Return all wrapped names
        return set.union(*(condition.modifies_filter_names for condition in self._conditions))


class BaseTransitionRule(Registerable):
    """
    Defines a set of categories of objects and how to transition their states.
    """
    conditions = None
    candidates = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Register this system, and
        # make sure at least one filter is specified -- in general, there should never be a rule
        # where no filter is specified
        # Only run this check for actual rules that are being registered
        if cls.__name__ not in cls._do_not_register_classes:
            global RULES_REGISTRY
            RULES_REGISTRY.add(obj=cls)
            assert len(cls.candidate_filters) > 0, \
                "At least one of individual_filters or group_filters must be specified!"

            # Store conditions
            cls.conditions = cls._generate_conditions()

    @classproperty
    def required_systems(cls):
        """
        Particle systems that this transition rule cares about. Should be specified by subclass.

        Returns:
            list of str: Particle system names which must be active in order for the transition rule to occur
        """
        return []

    @classproperty
    def candidate_filters(cls):
        """
        Object candidate filters that this transition rule cares about.
        For each name, filter key-value pair, the global transition rule step will produce a
        single dictionary of valid filtered objects.
        For example, if the group filters are:

            {"apple": CategoryFilter("apple"), "knife": CategoryFilter("knife")},

        the transition rule step will produce the following dictionary:

            {"apple": [apple0, apple1, ...], "knife": [knife0, knife1, ...]}

        based on the current instances of each object type in the scene and pass them to conditions in @self.conditions

        NOTE: There should always be at least one filter applied for every rule!

        Returns:
            dict: Maps filter name to filter for inferring valid object candidates for this transition rule
        """
        raise NotImplementedError

    @classmethod
    def _generate_conditions(cls):
        """
        Generates rule condition(s)s for this transition rule. These conditions are used to prune object
        candidates at runtime, to determine whether a transition rule should occur at the given timestep

        Returns:
            list of RuleCondition: Condition(s) to enforce to determine whether a transition rule should occur
        """
        raise NotImplementedError

    @classmethod
    def get_object_candidates(cls, objects):
        """
        Given the set of objects @objects, compute the valid object candidate combinations that may be valid for
        this TransitionRule

        Args:
            objects (list of BaseObject): Objects to filter for valid transition rule candidates

        Returns:
            dict: Maps filter name to valid object(s) that satisfy that filter
        """
        # Iterate over all objects and add to dictionary if valid
        filters = cls.candidate_filters
        obj_dict = {filter_name: [] for filter_name in filters.keys()}

        # Only compile candidates if all active system requirements are met
        if np.all([is_system_active(system_name=name) for name in cls.required_systems]):
            for obj in objects:
                for fname, f in filters.items():
                    if f(obj):
                        obj_dict[fname].append(obj)

        return obj_dict

    @classmethod
    def refresh(cls, object_candidates):
        """
        Refresh any internal state for this rule, given set of input object candidates @object_candidates

        Args:
            object_candidates (dict): Maps filter name to valid object(s) that satisfy that filter
        """
        # Store candidates
        cls.candidates = object_candidates

        # Refresh all conditions
        for condition in cls.conditions:
            condition.refresh(object_candidates=object_candidates)


    @classmethod
    def transition(cls, object_candidates):
        """
        Rule to apply for each set of objects satisfying the condition.

        Args:
            object_candidates (dict): Dictionary mapping corresponding keys from @cls.filters to list of individual
                object instances where the filter is satisfied

        Returns:
            TransitionResults: results from the executed transition
        """
        raise NotImplementedError()

    @classmethod
    def step(cls):
        """
        Takes a step for this transition rule, checking if all of @cls.conditions are satisified, and if so, taking
        a transition via @cls.transition()

        Returns:
            None or TransitionResults: If a transition occurs, returns its results, otherwise, returns None
        """
        # Copy the candidates dictionary since it may be mutated in place by @conditions
        object_candidates = {filter_name: candidates.copy() for filter_name, candidates in cls.candidates.items()}
        for condition in cls.conditions:
            if not condition(object_candidates=object_candidates):
                # Condition was not met, so immediately terminate
                return

        # All conditions are met, take the transition
        return cls.transition(object_candidates=object_candidates)

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseTransitionRule")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_RULES
        return REGISTERED_RULES


# Global dicts that will contain mappings. Must be placed here immediately AFTER BaseTransitionRule!
RULES_REGISTRY = Registry(
    name="TransitionRuleRegistry",
    class_types=BaseTransitionRule,
    default_key="__name__",
    group_keys=["required_systems"],
)


class SlicingRule(BaseTransitionRule):
    """
    Transition rule to apply to sliced / slicer object pairs.
    """
    @classproperty
    def candidate_filters(cls):
        # TODO: Remove hacky filter once half category is updated properly
        return {
            "sliceable": AndFilter(filters=[AbilityFilter("sliceable"), NotFilter(NameFilter(name="half"))]),
            "slicer": AbilityFilter("slicer"),
        }

    @classmethod
    def _generate_conditions(cls):
        # sliceables should be touching any slicer
        return [TouchingAnyCondition(filter_1_name="sliceable", filter_2_name="slicer"),
                StateCondition(filter_name="slicer", state=SlicerActive, val=True, op=operator.eq)]

    @classmethod
    def transition(cls, object_candidates):
        objs_to_add, objs_to_remove = [], []
        for sliceable_obj in object_candidates["sliceable"]:
            # Object parts offset annotation are w.r.t the base link of the whole object.
            pos, orn = sliceable_obj.get_position_orientation()

            # If it has no parts, silently fail
            if not sliceable_obj.metadata["object_parts"]:
                continue

            # Load object parts
            for i, part in enumerate(sliceable_obj.metadata["object_parts"].values()):
                # List of dicts gets replaced by {'0':dict, '1':dict, ...}

                # Get bounding box info
                part_bb_pos = np.array(part["bb_pos"])
                part_bb_orn = np.array(part["bb_orn"])

                # Determine the relative scale to apply to the object part from the original object
                # Note that proper (rotated) scaling can only be applied when the relative orientation of
                # the object part is a multiple of 90 degrees wrt the parent object, so we assert that here
                assert T.check_quat_right_angle(part_bb_orn), "Sliceable objects should only have relative object part orientations that are factors of 90 degrees!"

                # Scale the offset accordingly.
                scale = np.abs(T.quat2mat(part_bb_orn) @ sliceable_obj.scale)

                # Calculate global part bounding box pose.
                part_bb_pos = pos + T.quat2mat(orn) @ (part_bb_pos * scale)
                part_bb_orn = T.quat_multiply(orn, part_bb_orn)
                part_obj_name = f"half_{sliceable_obj.name}_{i}"
                part_obj = DatasetObject(
                    prim_path=f"/World/{part_obj_name}",
                    name=part_obj_name,
                    category=part["category"],
                    model=part["model"],
                    bounding_box=part["bb_size"] * scale,   # equiv. to scale=(part["bb_size"] / self.native_bbox) * (scale)
                )

                # Add the new object to the results.
                new_obj_attrs = ObjectAttrs(
                    obj=part_obj,
                    bb_pos=part_bb_pos,
                    bb_orn=part_bb_orn,
                )
                objs_to_add.append(new_obj_attrs)

            # Delete original object from stage.
            objs_to_remove.append(sliceable_obj)

        return TransitionResults(add=objs_to_add, remove=objs_to_remove)


class DicingRule(BaseTransitionRule):
    """
    Transition rule to apply to diceable / slicer object pairs.
    """
    @classproperty
    def candidate_filters(cls):
        return {
            "diceable": AbilityFilter("diceable"),
            "slicer": AbilityFilter("slicer"),
        }

    @classmethod
    def _generate_conditions(cls):
        # sliceables should be touching any slicer
        return [TouchingAnyCondition(filter_1_name="diceable", filter_2_name="slicer"),
                StateCondition(filter_name="slicer", state=SlicerActive, val=True, op=operator.eq)]

    @classmethod
    def transition(cls, object_candidates):
        objs_to_remove = []

        for diceable_obj in object_candidates["diceable"]:
            system = get_system(f"diced__{diceable_obj.category}")
            system.generate_particles_from_link(diceable_obj, diceable_obj.root_link, check_contact=False, use_visual_meshes=False)

            # Delete original object from stage.
            objs_to_remove.append(diceable_obj)

        return TransitionResults(add=[], remove=objs_to_remove)


class MeltingRule(BaseTransitionRule):
    """
    Transition rule to apply to meltable objects to simulate melting
    """
    @classproperty
    def candidate_filters(cls):
        # We want to find all meltable objects
        return {"meltable": AbilityFilter("meltable")}

    @classmethod
    def _generate_conditions(cls):
        return [StateCondition(filter_name="meltable", state=Temperature, val=m.MELTING_TEMPERATURE, op=operator.ge)]

    @classmethod
    def transition(cls, object_candidates):
        objs_to_remove = []

        # Convert the meltable object into its melted substance
        for meltable_obj in object_candidates["meltable"]:
            system = get_system(f"melted_{meltable_obj.category}")
            system.generate_particles_from_link(meltable_obj, meltable_obj.root_link, check_contact=False, use_visual_meshes=False)

            # Delete original object from stage.
            objs_to_remove.append(meltable_obj)

        return TransitionResults(add=[], remove=objs_to_remove)


class RecipeRule(BaseTransitionRule):
    """
    Transition rule to approximate recipe-based transitions
    """
    # Maps recipe name to recipe information
    _RECIPES = None

    # Maps active recipe name to recipe information
    _ACTIVE_RECIPES = None

    # Maps object category name to indices in the flattened object array for efficient computation
    _CATEGORY_IDXS = None

    # Flattened array of all simulator objects, sorted by category
    _OBJECTS = None

    # Maps object to idx within the _OBJECTS array
    _OBJECTS_TO_IDX = None

    def __init_subclass__(cls, **kwargs):
        # Run super first
        super().__init_subclass__(**kwargs)

        # Initialize recipes
        cls._RECIPES = dict()

    @classmethod
    def add_recipe(
        cls,
        name,
        input_synsets,
        output_synsets,
        input_states=None,
        output_states=None,
        fillable_categories=None,
        **kwargs,
    ):
        """
        Adds a recipe to this recipe rule to check against. This defines a valid mapping of inputs that will transform
        into the outputs

        Args:
            name (str): Name of the recipe
            input_synsets (dict): Maps synsets to number of instances required for the recipe
            output_synsets (dict): Maps synsets to number of instances to be spawned in the container when the recipe executes
            input_states (dict or None): Maps input synsets to states that must be satisfied for the recipe to execute,
                or None if no states are required
            otuput_states (dict or None): Map output synsets to states that should be set when spawned when the recipe executes,
                or None if no states are required
            fillable_categories (None or set of str): If specified, set of fillable categories which are allowed
                for this recipe. If None, any fillable is allowed
            kwargs (dict): Any additional keyword-arguments to be stored as part of this recipe
        """
        # Store information for this recipe
        cls._RECIPES[name] = {
            "name": name,
            # Maps object categories to number of instances required for the recipe
            "input_objects": dict(),
            # List of system names required for the recipe
            "input_systems": list(),
            # Maps object categories to number of instances to be spawned in the container when the recipe executes
            "output_objects": dict(),
            # List of system names to be spawned in the container when the recipe executes. Currently the length is 1.
            "output_systems": list(),
            # Maps object categories to ["unary", "bianry_system", "binary_object"] to a list of states that must be satisfied for the recipe to execute
            "input_states": defaultdict(lambda: defaultdict(list)),
            # Maps object categories to ["unary", "bianry_system", "binary_object"] to a list of states that should be set after the output objects are spawned
            "output_states": defaultdict(lambda: defaultdict(list)),
            # Set of fillable categories which are allowed for this recipe
            "fillable_categories": None,
            # networkx DiGraph that represents the kinematic dependency graph of the input objects
            # If input_states has no kinematic states between pairs of objects, this will be None.
            "input_object_tree": None,
            **kwargs,
        }

        # Map input/output synsets into input/output objects and systems.
        for synsets, obj_key, system_key in zip((input_synsets, output_synsets), ("input_objects", "output_objects"), ("input_systems", "output_systems")):
            for synset, count in synsets.items():
                assert OBJECT_TAXONOMY.is_leaf(synset), f"Synset {synset} must be a leaf node in the taxonomy!"
                if is_substance_synset(synset):
                    cls._RECIPES[name][system_key].append(get_system_name_by_synset(synset))
                else:
                    obj_categories = OBJECT_TAXONOMY.get_categories(synset)
                    assert len(obj_categories) == 1, f"Object synset {synset} must map to exactly one object category! Now: {obj_categories}."
                    cls._RECIPES[name][obj_key][obj_categories[0]] = count

        # Assert only one of output_objects or output_systems is not None
        assert len(cls._RECIPES[name]["output_objects"]) == 0 or len(cls._RECIPES[name]["output_systems"]) == 0, \
            "Recipe can only generate output objects or output systems, but not both!"

        # Apply post-processing for input/output states if specified
        for synsets_to_states, states_key in zip((input_states, output_states), ("input_states", "output_states")):
            if synsets_to_states is None:
                continue
            for synsets, states in synsets_to_states.items():
                # For unary/binary states, synsets is a single synset or a comma-separated pair of synsets, respectively
                synset_split = synsets.split(",")
                if len(synset_split) == 1:
                    first_synset = synset_split[0]
                    second_synset = None
                else:
                    first_synset, second_synset = synset_split

                # Assert the first synset is an object because the systems don't have any states.
                assert OBJECT_TAXONOMY.is_leaf(first_synset), f"Input/output state synset {first_synset} must be a leaf node in the taxonomy!"
                assert not is_substance_synset(first_synset), f"Input/output state synset {first_synset} must be applied to an object, not a substance!"
                obj_categories = OBJECT_TAXONOMY.get_categories(first_synset)
                assert len(obj_categories) == 1, f"Input/output state synset {first_synset} must map to exactly one object category! Now: {obj_categories}."
                first_obj_category = obj_categories[0]

                if second_synset is None:
                    # Unary states for the first synset
                    for state_type, state_value in states:
                        state_class = SUPPORTED_PREDICATES[state_type].STATE_CLASS
                        assert issubclass(state_class, AbsoluteObjectState), f"Input/output state type {state_type} must be a unary state!"
                        # Example: (Cooked, True)
                        cls._RECIPES[name][states_key][first_obj_category]["unary"].append((state_class, state_value))
                else:
                    assert OBJECT_TAXONOMY.is_leaf(second_synset), f"Input/output state synset {second_synset} must be a leaf node in the taxonomy!"
                    obj_categories = OBJECT_TAXONOMY.get_categories(second_synset)
                    if is_substance_synset(second_synset):
                        second_obj_category = get_system_name_by_synset(second_synset)
                        is_substance = True
                    else:
                        obj_categories = OBJECT_TAXONOMY.get_categories(second_synset)
                        assert len(obj_categories) == 1, f"Input/output state synset {second_synset} must map to exactly one object category! Now: {obj_categories}."
                        second_obj_category = obj_categories[0]
                        is_substance = False

                    for state_type, state_value in states:
                        state_class = SUPPORTED_PREDICATES[state_type].STATE_CLASS
                        assert issubclass(state_class, RelativeObjectState), f"Input/output state type {state_type} must be a binary state!"
                        assert is_substance == (state_class in get_system_states()), f"Input/output state type {state_type} system state inconsistency found!"
                        if is_substance:
                            # Non-kinematic binary states, e.g. Covered, Saturated, Filled, Contains.
                            # Example: (Covered, "sesame_seed", True)
                            cls._RECIPES[name][states_key][first_obj_category]["binary_system"].append(
                                (state_class, second_obj_category, state_value))
                        else:
                            # Kinematic binary states w.r.t. the second object.
                            # Example: (OnTop, "raw_egg", True)
                            assert cls.is_multi_instance, f"Input/output state type {state_type} can only be used in multi-instance recipes!"
                            assert states_key != "output_states", f"Output state type {state_type} can only be used in input states!"
                            cls._RECIPES[name][states_key][first_obj_category]["binary_object"].append(
                                (state_class, second_obj_category, state_value))

        if cls.is_multi_instance and len(cls._RECIPES[name]["input_objects"]) > 0:
            # Build a tree of input objects according to the kinematic binary states
            # Example: 'raw_egg': {'binary_object': [(OnTop, 'bagel_dough', True)]} results in an edge
            # from 'bagel_dough' to 'raw_egg', i.e. 'bagel_dough' is the parent of 'raw_egg'.
            input_object_tree = nx.DiGraph()
            for obj_category, state_checks in cls._RECIPES[name]["input_states"].items():
                for state_class, second_obj_category, state_value in state_checks["binary_object"]:
                    input_object_tree.add_edge(second_obj_category, obj_category)

            if not nx.is_empty(input_object_tree):
                assert nx.is_tree(input_object_tree), f"Input object tree must be a tree! Now: {input_object_tree}."
                root_nodes = [node for node in input_object_tree.nodes() if input_object_tree.in_degree(node) == 0]
                assert len(root_nodes) == 1, f"Input object tree must have exactly one root node! Now: {root_nodes}."
                assert cls._RECIPES[name]["input_objects"][root_nodes[0]] == 1, f"Input object tree root node must have exactly one instance! Now: {cls._RECIPES[name]['input_objects'][root_nodes[0]]}."
                cls._RECIPES[name]["input_object_tree"] = input_object_tree

        # Map fillable synsets to fillable object categories.
        if fillable_categories is not None:
            cls._RECIPES[name]["fillable_categories"] = set()
            for synset in fillable_categories:
                assert OBJECT_TAXONOMY.is_leaf(synset), f"Synset {synset} must be a leaf node in the taxonomy!"
                assert not is_substance_synset(synset), f"Synset {synset} must be applied to an object, not a substance!"
                for category in OBJECT_TAXONOMY.get_categories(synset):
                    cls._RECIPES[name]["fillable_categories"].add(category)

    @classmethod
    def _validate_recipe_container_is_valid(cls, recipe, container):
        """
        Validates that @container's category satisfies @recipe's fillable_categories

        Args:
            recipe (dict): Recipe whose fillable_categories should be checked against @container
            container (StatefulObject): Container whose category should match one of @recipe's fillable_categories,
                if specified

        Returns:
            bool: True if @container is valid, else False
        """
        fillable_categories = recipe["fillable_categories"]
        return fillable_categories is None or container.category in fillable_categories

    @classmethod
    def _validate_recipe_systems_are_contained(cls, recipe, container):
        """
        Validates whether @recipe's input_systems are all contained in @container or not

        Args:
            recipe (dict): Recipe whose systems should be checked
            container (BaseObject): Container object that should contain all of @recipe's input systems

        Returns:
            bool: True if all the input systems are contained
        """
        for system_name in recipe["input_systems"]:
            system = get_system(system_name=system_name)
            if not container.states[Contains].get_value(system=system):
                return False
        return True

    @classmethod
    def _validate_nonrecipe_systems_not_contained(cls, recipe, container):
        """
        Validates whether all systems not relevant to @recipe are not contained in @container

        Args:
            recipe (dict): Recipe whose systems should be checked
            container (BaseObject): Container object that should contain all of @recipe's input systems

        Returns:
            bool: True if none of the non-relevant systems are contained
        """
        for system in og.sim.scene.system_registry.objects:
            # Skip cloth system
            if system.name == "cloth":
                continue
            if system.name not in recipe["input_systems"] and container.states[Contains].get_value(system=system):
                return False
        return True

    @classmethod
    def _validate_recipe_objects_are_contained_and_states_satisfied(cls, recipe, container_info):
        """
        Validates whether @recipe's input_objects are contained in the container and whether their states are satisfied

        Args:
            recipe (dict): Recipe whose objects should be checked
            container_info (dict): Output of @cls._compute_container_info(); container-specific information which may
                be relevant for computing whether recipe is executable. This will be populated with execution info.

        Returns:
            bool: True if all the input object quantities are contained
        """
        in_volume = container_info["in_volume"]

        container_info["execution_info"] = dict()

        # Filter input objects based on a subset of input states (unary states and binary system states)
        obj_category_to_valid_objs = dict()
        for obj_category in recipe["input_objects"]:
            if obj_category not in recipe["input_states"]:
                # If there are no input states, all objects of this category are valid
                obj_category_to_valid_objs[obj_category] = cls._CATEGORY_IDXS[obj_category]
            else:
                obj_category_to_valid_objs[obj_category] = []
                for idx in cls._CATEGORY_IDXS[obj_category]:
                    obj = cls._OBJECTS[idx]
                    success = True

                    # Check if unary states are satisfied
                    for state_class, state_value in recipe["input_states"][obj_category]["unary"]:
                        if obj.states[state_class].get_value() != state_value:
                            success = False
                            break
                    if not success:
                        continue

                    # Check if binary system states are satisfied
                    for state_class, system_name, state_value in recipe["input_states"][obj_category]["binary_system"]:
                        if obj.states[state_class].get_value(system=get_system(system_name)) != state_value:
                            success = False
                            break
                    if not success:
                        continue

                    obj_category_to_valid_objs[obj_category].append(idx)

                # Convert to numpy array for faster indexing
                obj_category_to_valid_objs[obj_category] = np.array(obj_category_to_valid_objs[obj_category], dtype=int)

        container_info["execution_info"]["obj_category_to_valid_objs"] = obj_category_to_valid_objs
        if not cls.is_multi_instance:
            # Check if sufficiently number of objects are contained
            for obj_category, obj_quantity in recipe["input_objects"].items():
                if np.sum(in_volume[obj_category_to_valid_objs[obj_category]]) < obj_quantity:
                    return False
            return True
        else:
            input_object_tree = recipe["input_object_tree"]
            # If multi-instance is True but doesn't require kinematic states between objects
            if input_object_tree is None:
                num_instances = np.inf
                # Compute how many instances of this recipe can be produced.
                # Example: if a recipe requires 1 apple and 2 bananas, and there are 3 apples and 4 bananas in the
                # container, then 2 instance of the recipe can be produced.
                for obj_category, obj_quantity in recipe["input_objects"].items():
                    quantity_in_volume = np.sum(in_volume[obj_category_to_valid_objs[obj_category]])
                    num_inst = quantity_in_volume // obj_quantity
                    if num_inst < 1:
                        return False
                    num_instances = min(num_instances, num_inst)

                # Map object category to a set of objects that are used in this execution
                relevant_objects = defaultdict(set)
                for obj_category, obj_quantity in recipe["input_objects"].items():
                    quantity_used = num_instances * obj_quantity
                    relevant_objects[obj_category] = set(obj_category_to_valid_objs[obj_category][:quantity_used])

            # If multi-instance is True and requires kinematic states between objects
            else:
                root_node_category = [node for node in input_object_tree.nodes() if input_object_tree.in_degree(node) == 0][0]
                # A list of objects belonging to the root node category
                root_nodes = cls._OBJECTS[cls._CATEGORY_IDXS[root_node_category]]
                input_states = recipe["input_states"]

                # Recursively check if the kinematic tree is satisfied.
                # Return True/False, and a set of objects that belong to the subtree rooted at the current node
                def check_kinematic_tree(obj, should_check_in_volume=False):
                    # Check if obj is in volume
                    if should_check_in_volume and not in_volume[cls._OBJECTS_TO_IDX[obj]]:
                        return False, set()

                    # If the object is a leaf node, return True and the set containing the object
                    if input_object_tree.out_degree(obj.category) == 0:
                        return True, set([obj])

                    children_categories = list(input_object_tree.successors(obj.category))

                    all_subtree_objs = set()
                    for child_cat in children_categories:
                        assert len(input_states[child_cat]["binary_object"]) == 1, \
                            "Each child node should have exactly one binary object state, i.e. one parent in the input_object_tree"
                        state_class, _, state_value = input_states[child_cat]["binary_object"][0]
                        num_valid_children = 0
                        for child_obj in cls._OBJECTS[cls._CATEGORY_IDXS[child_cat]]:
                            # If the child doesn't satisfy the binary object state, skip
                            if child_obj.states[state_class].get_value(obj) != state_value:
                                continue
                            # Recursively check if the subtree rooted at the child is valid
                            subtree_valid, subtree_objs = check_kinematic_tree(child_obj)
                            # If the subtree is valid, increment the number of valid children and aggregate the objects
                            if subtree_valid:
                                num_valid_children += 1
                                all_subtree_objs |= subtree_objs

                        # If there are not enough valid children, return False
                        if num_valid_children < recipe["input_objects"][child_cat]:
                            return False, set()

                    # If all children categories have sufficient number of objects that satisfy the binary object state,
                    # e.g. five pieces of pepperoni and two pieces of basil on the pizza, the subtree rooted at the
                    # current node is valid. Return True and the set of objects in the subtree (all descendants plus
                    # the current node)
                    return True, all_subtree_objs | {obj}

                num_instances = 0
                relevant_objects = defaultdict(set)
                for root_node in root_nodes:
                    # should_check_in_volume is True only for the root nodes.
                    # Example: the bagel dough needs to be in_volume of the container, but the raw egg on top doesn't.
                    tree_valid, relevant_object_set = check_kinematic_tree(root_node, should_check_in_volume=True)
                    if tree_valid:
                        # For each valid tree, increment the number of instances and aggregate the objects
                        num_instances += 1
                        for obj in relevant_object_set:
                            relevant_objects[obj.category].add(obj)

                # If there are no valid trees, return False
                if num_instances == 0:
                    return False

            # Map system name to a set of particle indices that are used in this execution
            relevant_systems = defaultdict(set)
            for obj_category, objs in relevant_objects.items():
                for state_class, system_name, state_value in recipe["input_states"][obj_category]["binary_system"]:
                    if state_class in [Filled, Contains]:
                        for obj in objs:
                            contained_particle_idx = obj.states[ContainedParticles].get_value(get_system(system_name)).in_volume.nonzero()[0]
                            relevant_systems[system_name] |= contained_particle_idx
                    elif state_class in [Covered]:
                        for obj in objs:
                            covered_particle_idx = obj.states[ContactParticles].get_value(get_system(system_name))
                            relevant_systems[system_name] |= covered_particle_idx

            # Now we populate the execution info with the relevant objects and systems as well as the number of
            # instances of the recipe that can be produced.
            container_info["execution_info"]["relevant_objects"] = relevant_objects
            container_info["execution_info"]["relevant_systems"] = relevant_systems
            container_info["execution_info"]["num_instances"] = num_instances
            return True

    @classmethod
    def _validate_nonrecipe_objects_not_contained(cls, recipe, container_info):
        """
        Validates whether all objects not relevant to @recipe are not contained in the container
        represented by @in_volume

        Args:
            recipe (dict): Recipe whose systems should be checked
            container_info (dict): Output of @cls._compute_container_info(); container-specific information
                which may be relevant for computing whether recipe is executable

        Returns:
            bool: True if none of the non-relevant objects are contained
        """
        in_volume = container_info["in_volume"]
        # These are object indices whose objects satisfy the input states
        obj_category_to_valid_objs = container_info["execution_info"]["obj_category_to_valid_objs"]
        nonrecipe_objects_in_volume = in_volume if len(recipe["input_objects"]) == 0 else \
            np.delete(in_volume, np.concatenate([obj_category_to_valid_objs[obj_category]
                                                 for obj_category in obj_category_to_valid_objs]))
        return not np.any(nonrecipe_objects_in_volume)

    @classmethod
    def _validate_recipe_systems_exist(cls, recipe):
        """
        Validates whether @recipe's input_systems are all active or not

        Args:
            recipe (dict): Recipe whose systems should be checked

        Returns:
            bool: True if all the input systems are active
        """
        for system_name in recipe["input_systems"]:
            if not is_system_active(system_name=system_name):
                return False
        return True

    @classmethod
    def _validate_recipe_objects_exist(cls, recipe):
        """
        Validates whether @recipe's input_objects exist in the current scene or not

        Args:
            recipe (dict): Recipe whose objects should be checked

        Returns:
            bool: True if all the input objects exist in the scene
        """
        for obj_category, obj_quantity in recipe["input_objects"].items():
            if len(og.sim.scene.object_registry("category", obj_category, default_val=set())) < obj_quantity:
                return False
        return True

    @classmethod
    def _validate_recipe_fillables_exist(cls, recipe):
        """
        Validates that recipe @recipe's necessary fillable categorie(s) exist in the current scene

        Args:
            recipe (dict): Recipe whose fillable categories should be checked

        Returns:
            bool: True if there is at least a single valid fillable category in the current scene, else False
        """
        fillable_categories = recipe["fillable_categories"]
        if fillable_categories is None:
            # Any is valid
            return True
        # Otherwise, at least one valid type must exist
        for category in fillable_categories:
            if len(og.sim.scene.object_registry("category", category, default_val=set())) > 0:
                return True

        # None found, return False
        return False

    @classmethod
    def _is_recipe_active(cls, recipe):
        """
        Helper function to determine whether a given recipe @recipe should be actively checked for or not.

        Args:
            recipe (dict): Maps relevant keyword to corresponding recipe info

        Returns:
            bool: True if the recipe is active, else False
        """
        # Check valid active systems
        if not cls._validate_recipe_systems_exist(recipe=recipe):
            return False

        # Check valid object quantities
        if not cls._validate_recipe_objects_exist(recipe=recipe):
            return False

        # Check valid fillable categories
        if not cls._validate_recipe_fillables_exist(recipe=recipe):
            return False

        return True

    @classmethod
    def _is_recipe_executable(cls, recipe, container, global_info, container_info):
        """
        Helper function to determine whether a given recipe @recipe should be immediately executed or not.

        Args:
            recipe (dict): Maps relevant keyword to corresponding recipe info
            container (StatefulObject): Container in which @recipe may be executed
            global_info (dict): Output of @cls._compute_global_rule_info(); global information which may be
                relevant for computing whether recipe is executable
            container_info (dict): Output of @cls._compute_container_info(); container-specific information
                which may be relevant for computing whether recipe is executable

        Returns:
            bool: True if the recipe is active, else False
        """
        in_volume = container_info["in_volume"]

        # Verify the container category is valid
        if not cls._validate_recipe_container_is_valid(recipe=recipe, container=container):
            return False

        # Verify all required systems are contained in the container
        if not cls.relax_recipe_systems and not cls._validate_recipe_systems_are_contained(recipe=recipe, container=container):
            return False

        # Verify all required object quantities are contained in the container and their states are satisfied
        if not cls._validate_recipe_objects_are_contained_and_states_satisfied(recipe=recipe, container_info=container_info):
            return False

        # Verify no non-relevant system is contained
        if not cls.ignore_nonrecipe_systems and not cls._validate_nonrecipe_systems_not_contained(recipe=recipe, container=container):
            return False

        # Verify no non-relevant object is contained if we're not ignoring them
        if not cls.ignore_nonrecipe_objects and not cls._validate_nonrecipe_objects_not_contained(recipe=recipe, container_info=container_info):
            return False

        return True

    @classmethod
    def _compute_global_rule_info(cls, object_candidates):
        """
        Helper function to compute global information necessary for checking rules. This is executed exactly
        once per cls.transition() step

        Args:
            object_candidates (dict): Dictionary mapping corresponding keys from @cls.filters to list of individual
                object instances where the filter is satisfied

        Returns:
            dict: Keyword-mapped global rule information
        """
        # Compute all relevant object AABB positions
        obj_positions = np.array([obj.aabb_center for obj in cls._OBJECTS])

        return dict(obj_positions=obj_positions)

    @classmethod
    def _compute_container_info(cls, object_candidates, container, global_info):
        """
        Helper function to compute container-specific information necessary for checking rules. This is executed once
        per container per cls.transition() step

        Args:
            object_candidates (dict): Dictionary mapping corresponding keys from @cls.filters to list of individual
                object instances where the filter is satisfied
            container (StatefulObject): Relevant container object for computing information
            global_info (dict): Output of @cls._compute_global_rule_info(); global information which may be
                relevant for computing container information

        Returns:
            dict: Keyword-mapped container information
        """
        obj_positions = global_info["obj_positions"]
        # Compute in volume for all relevant object positions
        # We check for either the object AABB being contained OR the object being on top of the container, in the
        # case that the container is too flat for the volume to contain the object
        in_volume = container.states[ContainedParticles].check_in_volume(obj_positions) | \
                    np.array([obj.states[OnTop].get_value(container) for obj in cls._OBJECTS])

        # Container itself is never within its own volume
        in_volume[cls._OBJECTS_TO_IDX[container]] = False

        return dict(in_volume=in_volume)

    @classmethod
    def refresh(cls, object_candidates):
        # Run super first
        super().refresh(object_candidates=object_candidates)

        # Cache active recipes given the current set of objects
        cls._ACTIVE_RECIPES = dict()
        cls._CATEGORY_IDXS = dict()
        cls._OBJECTS = []
        cls._OBJECTS_TO_IDX = dict()

        # Prune any recipes whose objects / system requirements are not met by the current set of objects / systems
        objects_by_category = og.sim.scene.object_registry.get_dict("category")

        for name, recipe in cls._RECIPES.items():
            # If all pre-requisites met, add to active recipes
            if cls._is_recipe_active(recipe=recipe):
                cls._ACTIVE_RECIPES[name] = recipe

        # Finally, compute relevant objects and category mapping based on relevant categories
        i = 0
        for category, objects in objects_by_category.items():
            cls._CATEGORY_IDXS[category] = i + np.arange(len(objects))
            cls._OBJECTS += list(objects)
            for obj in objects:
                cls._OBJECTS_TO_IDX[obj] = i
                i += 1

        # Wrap relevant objects as numpy array so we can index into it efficiently
        cls._OBJECTS = np.array(cls._OBJECTS)

    @classproperty
    def candidate_filters(cls):
        # Fillable object required
        return {"container": AbilityFilter(ability="fillable")}

    @classmethod
    def transition(cls, object_candidates):
        objs_to_add, objs_to_remove = [], []

        # Compute global info
        global_info = cls._compute_global_rule_info(object_candidates=object_candidates)

        # Iterate over all fillable objects, to execute recipes for each one
        for container in object_candidates["container"]:
            recipe_results = None
            # Compute container info
            container_info = cls._compute_container_info(
                object_candidates=object_candidates,
                container=container,
                global_info=global_info,
            )

            # Check every recipe to find if any is valid
            for name, recipe in cls._ACTIVE_RECIPES.items():
                if cls._is_recipe_executable(recipe=recipe, container=container, global_info=global_info, container_info=container_info):
                    # Otherwise, all conditions met, we found a valid recipe and so we execute and terminate early
                    og.log.info(f"Executing recipe: {name} in container {container.name}!")

                    # Take the transform and terminate early
                    recipe_results = cls._execute_recipe(
                        container=container,
                        recipe=recipe,
                        container_info=container_info,
                    )
                    objs_to_add += recipe_results.add
                    objs_to_remove += recipe_results.remove
                    break

            # Otherwise, if we didn't find a valid recipe, we execute a garbage transition instead if requested
            if recipe_results is None and cls.use_garbage_fallback_recipe:
                og.log.info(f"Did not find a valid recipe for rule {cls.__name__}; generating {m.DEFAULT_GARBAGE_SYSTEM} in {container.name}!")

                # Generate garbage fluid
                garbage_results = cls._execute_recipe(
                    container=container,
                    recipe=dict(
                        name="garbage",
                        input_objects=dict(),
                        input_systems=[],
                        output_objects=dict(),
                        output_systems=[m.DEFAULT_GARBAGE_SYSTEM],
                        output_states=defaultdict(lambda: defaultdict(list)),
                    ),
                    container_info=container_info,
                )
                objs_to_add += garbage_results.add
                objs_to_remove += garbage_results.remove

        return TransitionResults(add=objs_to_add, remove=objs_to_remove)

    @classmethod
    def _execute_recipe(cls, container, recipe, container_info):
        """
        Transforms all items contained in @container into @output_system, generating volume of @output_system
        proportional to the number of items transformed.

        Args:
            container (BaseObject): Container object which will have its contained elements transformed into
                @output_system
            recipe (dict): Recipe to execute. Should include, at the minimum, "input_objects", "input_systems",
                "output_objects", and "output_systems" keys
            container_info (dict): Output of @cls._compute_container_info(); container-specific information which may
                be relevant for computing whether recipe is executable.

        Returns:
            TransitionResults: Results of the executed recipe transition
        """
        objs_to_add, objs_to_remove = [], []

        in_volume = container_info["in_volume"]
        if cls.is_multi_instance:
            execution_info = container_info["execution_info"]

        # Compute total volume of all contained items
        volume = 0

        if not cls.is_multi_instance:
            # Remove either all systems or only the ones specified in the input systems of the recipe
            contained_particles_state = container.states[ContainedParticles]
            for system in PhysicalParticleSystem.get_active_systems().values():
                if not cls.ignore_nonrecipe_systems or system.name in recipe["input_systems"]:
                    if container.states[Contains].get_value(system):
                        volume += contained_particles_state.get_value(system).n_in_volume * np.pi * (system.particle_radius ** 3) * 4 / 3
                        container.states[Contains].set_value(system, False)
            for system in VisualParticleSystem.get_active_systems().values():
                if not cls.ignore_nonrecipe_systems or system.name in recipe["input_systems"]:
                    group_name = system.get_group_name(container)
                    if group_name in system.groups and system.num_group_particles(group_name) > 0:
                        system.remove_all_group_particles(group=group_name)
        else:
            # Remove the particles that are involved in this execution
            for system_name, particle_idxs in execution_info["relevant_systems"].items():
                system = get_system(system_name)
                volume += len(particle_idxs) * np.pi * (system.particle_radius ** 3) * 4 / 3
                system.remove_particles(idxs=np.array(list(particle_idxs)))

        if not cls.is_multi_instance:
            # Remove either all objects or only the ones specified in the input objects of the recipe
            object_mask = in_volume.copy()
            if cls.ignore_nonrecipe_objects:
                object_category_mask = np.zeros_like(object_mask, dtype=bool)
                for obj_category in recipe["input_objects"].keys():
                    object_category_mask[cls._CATEGORY_IDXS[obj_category]] = True
                object_mask &= object_category_mask
            objs_to_remove.extend(cls._OBJECTS[object_mask])
        else:
            # Remove the objects that are involved in this execution
            for obj_category, objs in execution_info["relevant_objects"].items():
                objs_to_remove.extend(objs)

        volume += sum(obj.volume for obj in objs_to_remove)

        # Define callback for spawning new objects inside container
        def _spawn_object_in_container(obj):
            # For simplicity sake, sample only OnTop
            # TODO: Can we sample inside intelligently?
            state = OnTop
            # TODO: What to do if setter fails?
            if not obj.states[state].set_value(container, True):
                log.warning(f"Failed to spawn object {obj.name} in container {container.name}!")

        # Spawn in new objects
        for category, n_instances in recipe["output_objects"].items():
            # Multiply by number of instances of execution if this is a multi-instance recipe
            if cls.is_multi_instance:
                n_instances *= execution_info["num_instances"]

            output_states = dict()
            for state_type, state_value in recipe["output_states"][category]["unary"]:
                output_states[state_type] = (state_value,)
            for state_type, system_name, state_value in recipe["output_states"][category]["binary_system"]:
                output_states[state_type] = (get_system(system_name), state_value)

            n_category_objs = len(og.sim.scene.object_registry("category", category, []))
            models = get_all_object_category_models(category=category)

            for i in range(n_instances):
                obj = DatasetObject(
                    name=f"{category}_{n_category_objs + i}",
                    category=category,
                    model=np.random.choice(models),
                )
                new_obj_attrs = ObjectAttrs(
                    obj=obj,
                    callback=_spawn_object_in_container,
                    states=output_states,
                    pos=np.ones(3) * (100.0 + i),
                )
                objs_to_add.append(new_obj_attrs)

        # Spawn in new fluid
        if len(recipe["output_systems"]) > 0:
            # Only one system is allowed to be spawned
            assert len(recipe["output_systems"]) == 1, "Only a single output system can be spawned for a given recipe!"
            out_system = get_system(recipe["output_systems"][0])
            out_system.generate_particles_from_link(
                obj=container,
                link=contained_particles_state.link,
                # In these two cases, we don't necessarily have removed all objects in the container.
                check_contact=cls.ignore_nonrecipe_objects or cls.is_multi_instance,
                max_samples=int(volume / (np.pi * (out_system.particle_radius ** 3) * 4 / 3)),
            )

        # Return transition results
        return TransitionResults(add=objs_to_add, remove=objs_to_remove)

    @classproperty
    def relax_recipe_systems(cls):
        """
        Returns:
            bool: Whether to relax the requirement of having all systems in the recipe contained in the container
        """
        raise NotImplementedError("Must be implemented by subclass!")

    @classproperty
    def ignore_nonrecipe_systems(cls):
        """
        Returns:
            bool: Whether contained systems not relevant to the recipe should be ignored or not
        """
        raise NotImplementedError("Must be implemented by subclass!")

    @classproperty
    def ignore_nonrecipe_objects(cls):
        """
        Returns:
            bool: Whether contained rigid objects not relevant to the recipe should be ignored or not
        """
        raise NotImplementedError("Must be implemented by subclass!")

    @classproperty
    def use_garbage_fallback_recipe(cls):
        """
        Returns:
            bool: Whether this recipe rule should use a garbage fallback recipe if all conditions are met but no
                valid recipe is found for a given container
        """
        raise NotImplementedError("Must be implemented by subclass!")

    @classproperty
    def is_multi_instance(cls):
        """
        Returns:
            bool: Whether this rule can be applied multiple times to the same container, e.g. to cook multiple doughs
        """
        return False

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("RecipeRule")
        return classes


class CookingPhysicalParticleRule(RecipeRule):
    """
    Transition rule to apply to "cook" physicl particles
    """
    @classmethod
    def add_recipe(cls, name, input_synsets, output_synsets):
        super().add_recipe(
            name=name,
            input_synsets=input_synsets,
            output_synsets=output_synsets,
            input_states=None,
            output_states=None,
            fillable_categories=None,
        )

        input_objects = cls._RECIPES[name]["input_objects"]
        output_objects = cls._RECIPES[name]["output_objects"]
        assert len(input_objects) == 0, f"No input objects can be specified for {cls.__name__}, recipe: {name}!"
        assert len(output_objects) == 0, f"No output objects can be specified for {cls.__name__}, recipe: {name}!"

        input_systems = cls._RECIPES[name]["input_systems"]
        output_systems = cls._RECIPES[name]["output_systems"]
        assert len(input_systems) == 1 or len(input_systems) == 2, \
            f"Only one or two input systems can be specified for {cls.__name__}, recipe: {name}!"
        if len(input_systems) == 2:
            assert input_systems[1] == "water", \
                f"Second input system must be water for {cls.__name__}, recipe: {name}!"
        assert len(output_systems) == 1, \
            f"Exactly one output system needs to be specified for {cls.__name__}, recipe: {name}!"

    @classproperty
    def candidate_filters(cls):
        # Modify the container filter to include the heatable ability as well
        candidate_filters = super().candidate_filters
        candidate_filters["container"] = AndFilter(filters=[candidate_filters["container"], AbilityFilter(ability="heatable")])
        return candidate_filters

    @classmethod
    def _generate_conditions(cls):
        # Only heated objects are valid
        return [StateCondition(filter_name="container", state=Heated, val=True, op=operator.eq)]

    @classproperty
    def relax_recipe_systems(cls):
        return False

    @classproperty
    def ignore_nonrecipe_systems(cls):
        return True

    @classproperty
    def ignore_nonrecipe_objects(cls):
        return True

    @classproperty
    def use_garbage_fallback_recipe(cls):
        return False

    @classmethod
    def _execute_recipe(cls, container, recipe, in_volume):
        system = get_system(recipe["input_systems"][0])
        contained_particles_state = container.states[ContainedParticles].get_value(system)
        in_volume_idx = np.where(contained_particles_state.in_volume)[0]
        assert len(in_volume_idx) > 0, "No particles found in the container when executing recipe!"

        # Remove uncooked particles
        system.remove_particles(idxs=in_volume_idx)

        # Generate cooked particles
        cooked_system = get_system(recipe["output_systems"][0])
        particle_positions = contained_particles_state.positions[in_volume_idx]
        cooked_system.generate_particles(positions=particle_positions)

        # Remove water if the cooking requires water
        if len(recipe["input_systems"]) > 1:
            water_system = get_system(recipe["input_systems"][1])
            container.states[Contains].set_value(water_system, False)

        return TransitionResults(add=[], remove=[])


class ToggleableMachineRule(RecipeRule):
    """
    Transition mixing rule that leverages a single toggleable machine (e.g. electric mixer, coffee machine, blender),
    which require toggledOn in order to trigger the recipe event
    """

    @classmethod
    def add_recipe(
            cls,
            name,
            input_synsets,
            output_synsets,
            fillable_categories,
            input_states=None,
            output_states=None,
    ):
        """
        Adds a recipe to this cooking recipe rule to check against. This defines a valid mapping of inputs that
        will transform into the outputs

        Args:
            name (str): Name of the recipe
            input_synsets (dict): Maps synsets to number of instances required for the recipe
            output_synsets (dict): Maps synsets to number of instances to be spawned in the container when the recipe executes
            fillable_categories (set of str): Set of toggleable machine categories which are allowed for this recipe
            input_states (dict or None): Maps input synsets to states that must be satisfied for the recipe to execute,
                or None if no states are required
            otuput_states (dict or None): Map output synsets to states that should be set when spawned when the recipe executes,
                or None if no states are required
        """
        super().add_recipe(
            name=name,
            input_synsets=input_synsets,
            output_synsets=output_synsets,
            input_states=input_states,
            output_states=output_states,
            fillable_categories=fillable_categories
        )
        output_objects = cls._RECIPES[name]["output_objects"]
        if len(output_objects) > 0:
            assert len(output_objects) == 1, f"Only one category of output object can be specified for {cls.__name__}, recipe: {name}!"
            assert output_objects[list(output_objects.keys())[0]] == 1, f"Only one instance of output object can be specified for {cls.__name__}, recipe: {name}!"

    @classproperty
    def candidate_filters(cls):
        # Modify the container filter to include toggleable ability as well
        candidate_filters = super().candidate_filters
        candidate_filters["container"] = AndFilter(filters=[candidate_filters["container"], AbilityFilter(ability="toggleable")])
        return candidate_filters

    @classmethod
    def _generate_conditions(cls):
        # Container must be toggledOn, and should only be triggered once
        return [ChangeConditionWrapper(
            condition=StateCondition(filter_name="container", state=ToggledOn, val=True, op=operator.eq)
        )]

    @classproperty
    def relax_recipe_systems(cls):
        return False

    @classproperty
    def ignore_nonrecipe_systems(cls):
        return False

    @classproperty
    def ignore_nonrecipe_objects(cls):
        return False

    @classproperty
    def use_garbage_fallback_recipe(cls):
        return True


class MixingToolRule(RecipeRule):
    """
    Transition mixing rule that leverages "mixingTool" ability objects, which require touching between a mixing tool
    and a container in order to trigger the recipe event
    """
    @classmethod
    def add_recipe(cls, name, input_synsets, output_synsets, input_states=None, output_states=None):
        super().add_recipe(
            name=name,
            input_synsets=input_synsets,
            output_synsets=output_synsets,
            input_states=input_states,
            output_states=output_states,
            fillable_categories=None,
        )

        output_objects = cls._RECIPES[name]["output_objects"]
        assert len(output_objects) == 0, f"No output objects can be specified for {cls.__name__}, recipe: {name}!"

        input_systems = cls._RECIPES[name]["input_systems"]
        output_systems = cls._RECIPES[name]["output_systems"]
        assert len(input_systems) > 0, f"Some input systems need to be specified for {cls.__name__}, recipe: {name}!"
        assert len(output_systems) == 1, \
            f"Exactly one output system needs to be specified for {cls.__name__}, recipe: {name}!"

    @classproperty
    def candidate_filters(cls):
        # Add mixing tool filter as well
        candidate_filters = super().candidate_filters
        candidate_filters["mixingTool"] = AbilityFilter(ability="mixingTool")
        return candidate_filters

    @classmethod
    def _generate_conditions(cls):
        # Mixing tool must be touching the container, and should only be triggered once
        return [ChangeConditionWrapper(
            condition=TouchingAnyCondition(filter_1_name="container", filter_2_name="mixingTool")
        )]

    @classproperty
    def ignore_nonrecipe_systems(cls):
        return False

    @classproperty
    def ignore_nonrecipe_objects(cls):
        return True

    @classproperty
    def use_garbage_fallback_recipe(cls):
        return True


class CookingRule(RecipeRule):
    """
    Transition mixing rule that approximates cooking recipes via a container and heatsource
    """
    # Counter that increments monotonically
    COUNTER = 0

    # Maps recipe name to current number of consecutive heating steps
    _HEAT_STEPS = None

    # Maps recipe name to the last timestep that it was active
    _LAST_HEAT_TIMESTEP = None

    @classmethod
    def refresh(cls, object_candidates):
        # Run super first
        super().refresh(object_candidates=object_candidates)

        # Iterate through all (updated) active recipes and store in internal variables if not already recorded
        cls._HEAT_STEPS = dict() if cls._HEAT_STEPS is None else cls._HEAT_STEPS
        cls._LAST_HEAT_TIMESTEP = dict() if cls._LAST_HEAT_TIMESTEP is None else cls._LAST_HEAT_TIMESTEP

        for name in cls._ACTIVE_RECIPES.keys():
            if name not in cls._HEAT_STEPS:
                cls._HEAT_STEPS[name] = 0
                cls._LAST_HEAT_TIMESTEP[name] = -1

    @classmethod
    def _validate_recipe_fillables_exist(cls, recipe):
        """
        Validates that recipe @recipe's necessary fillable categorie(s) exist in the current scene

        Args:
            recipe (dict): Recipe whose fillable categories should be checked

        Returns:
            bool: True if there is at least a single valid fillable category in the current scene, else False
        """
        fillable_categories = recipe["fillable_categories"]
        if fillable_categories is None:
            # Any is valid
            return True
        # Otherwise, at least one valid type must exist
        for category in fillable_categories:
            if len(og.sim.scene.object_registry("category", category, default_val=set())) > 0:
                return True

        # None found, return False
        return False

    @classmethod
    def _validate_recipe_heatsources_exist(cls, recipe):
        """
        Validates that recipe @recipe's necessary heatsource categorie(s) exist in the current scene

        Args:
            recipe (dict): Recipe whose heatsource categories should be checked

        Returns:
            bool: True if there is at least a single valid heatsource category in the current scene, else False
        """
        heatsource_categories = recipe["heatsource_categories"]
        if heatsource_categories is None:
            # Any is valid
            return True
        # Otherwise, at least one valid type must exist
        for category in heatsource_categories:
            if len(og.sim.scene.object_registry("category", category, default_val=set())) > 0:
                return True

        # None found, return False
        return False

    @classmethod
    def _validate_recipe_heatsource_is_valid(cls, recipe, heatsource_categories):
        """
        Validates that there is a valid heatsource category in @heatsource_categories compatible with @recipe

        Args:
            recipe (dict): Recipe whose heatsource_categories should be checked against @heatsource_categories
            heatsource_categories (set of str): Set of potential heatsource categories

        Returns:
            bool: True if there is a compatible category in @heatsource_categories, else False
        """
        required_heatsource_categories = recipe["heatsource_categories"]
        # Either no specific required and there is at least 1 heatsource or there is at least 1 matching heatsource
        # between the required and available
        return (required_heatsource_categories is None and len(heatsource_categories) > 0) or \
               len(required_heatsource_categories.intersection(heatsource_categories)) > 0

    @classmethod
    def _compute_container_info(cls, object_candidates, container, global_info):
        # Run super first
        info = super()._compute_container_info(object_candidates=object_candidates, container=container, global_info=global_info)

        # Compute whether each heatsource is affecting the container
        info["heatsource_categories"] = set(obj.category for obj in object_candidates["heatSource"] if
                                            obj.states[HeatSourceOrSink].affects_obj(container))

        return info

    @classmethod
    def _is_recipe_active(cls, recipe):
        # Check for heatsource categories first
        if not cls._validate_recipe_heatsources_exist(recipe=recipe):
            return False

        # Otherwise, run super normally
        return super()._is_recipe_active(recipe=recipe)

    @classmethod
    def _is_recipe_executable(cls, recipe, container, global_info, container_info):
        # Check for heatsource compatibility first
        if not cls._validate_recipe_heatsource_is_valid(recipe=recipe, heatsource_categories=container_info["heatsource_categories"]):
            return False

        # Run super
        executable = super()._is_recipe_executable(
            recipe=recipe,
            container=container,
            global_info=global_info,
            container_info=container_info,
        )

        # If executable, increment heat counter by 1, if we were also active last timestep, else, reset to 1
        if executable:
            name = recipe["name"]
            cls._HEAT_STEPS[name] = cls._HEAT_STEPS[name] + 1 if \
                cls._LAST_HEAT_TIMESTEP[name] == cls.COUNTER - 1 else 1
            cls._LAST_HEAT_TIMESTEP[name] = cls.COUNTER

            # If valid number of timesteps met, recipe is indeed executable
            executable = cls._HEAT_STEPS[name] >= recipe["timesteps"]

        return executable

    @classmethod
    def add_recipe(
            cls,
            name,
            input_synsets,
            output_synsets,
            input_states=None,
            output_states=None,
            fillable_categories=None,
            heatsource_categories=None,
            timesteps=None,
    ):
        """
        Adds a recipe to this cooking recipe rule to check against. This defines a valid mapping of inputs that
        will transform into the outputs

        Args:
            name (str): Name of the recipe
            input_synsets (dict): Maps synsets to number of instances required for the recipe
            output_synsets (dict): Maps synsets to number of instances to be spawned in the container when the recipe executes
            input_states (dict or None): Maps input synsets to states that must be satisfied for the recipe to execute,
                or None if no states are required
            otuput_states (dict or None): Map output synsets to states that should be set when spawned when the recipe executes,
                or None if no states are required
            fillable_categories (None or set of str): If specified, set of fillable categories which are allowed
                for this recipe. If None, any fillable is allowed
            heatsource_categories (None or set of str): If specified, set of heatsource categories which are allowed
                for this recipe. If None, any heatsource is allowed
            timesteps (None or int): Number of subsequent heating steps required for the recipe to execute. If None,
                it will be set to be 1, i.e.: instantaneous execution
        """
        if heatsource_categories is not None:
            heatsource_categories_postprocessed = set()
            for synset in heatsource_categories:
                assert OBJECT_TAXONOMY.is_leaf(synset), f"Synset {synset} must be a leaf node in the taxonomy!"
                assert not is_substance_synset(synset), f"Synset {synset} must be applied to an object, not a substance!"
                for category in OBJECT_TAXONOMY.get_categories(synset):
                    heatsource_categories_postprocessed.add(category)
            heatsource_categories = heatsource_categories_postprocessed

        super().add_recipe(
            name=name,
            input_synsets=input_synsets,
            output_synsets=output_synsets,
            input_states=input_states,
            output_states=output_states,
            fillable_categories=fillable_categories,
            heatsource_categories=heatsource_categories,
            timesteps=1 if timesteps is None else timesteps,
        )

    @classproperty
    def candidate_filters(cls):
        # Add mixing tool filter as well
        candidate_filters = super().candidate_filters
        candidate_filters["heatSource"] = AbilityFilter(ability="heatSource")
        return candidate_filters

    @classmethod
    def _generate_conditions(cls):
        # Define a class to increment this class's internal time counter every time it is triggered
        class TimeIncrementCondition(RuleCondition):
            def __init__(self, cls):
                self.cls = cls

            def __call__(self, object_candidates):
                # This is just a pass-through, but also increment the time
                self.cls.COUNTER += 1
                return True

            def modifies_filter_names(self):
                return set()

        # Any heatsource must be active
        return [
            TimeIncrementCondition(cls=cls),
            StateCondition(filter_name="heatSource", state=HeatSourceOrSink, val=True, op=operator.eq),
        ]

    @classproperty
    def use_garbage_fallback_recipe(cls):
        return False

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("CookingRule")
        return classes

class CookingObjectRule(CookingRule):
    @classmethod
    def add_recipe(
            cls,
            name,
            input_synsets,
            output_synsets,
            input_states=None,
            output_states=None,
            fillable_categories=None,
            heatsource_categories=None,
            timesteps=None,
    ):
        """
        Adds a recipe to this cooking recipe rule to check against. This defines a valid mapping of inputs that
        will transform into the outputs

        Args:
            name (str): Name of the recipe
            input_synsets (dict): Maps synsets to number of instances required for the recipe
            output_synsets (dict): Maps synsets to number of instances to be spawned in the container when the recipe executes
            input_states (dict or None): Maps input synsets to states that must be satisfied for the recipe to execute,
                or None if no states are required
            otuput_states (dict or None): Map output synsets to states that should be set when spawned when the recipe executes,
                or None if no states are required
            fillable_categories (None or set of str): If specified, set of fillable categories which are allowed
                for this recipe. If None, any fillable is allowed
            heatsource_categories (None or set of str): If specified, set of heatsource categories which are allowed
                for this recipe. If None, any heatsource is allowed
            timesteps (None or int): Number of subsequent heating steps required for the recipe to execute. If None,
                it will be set to be 1, i.e.: instantaneous execution
        """
        super().add_recipe(
            name=name,
            input_synsets=input_synsets,
            output_synsets=output_synsets,
            input_states=input_states,
            output_states=output_states,
            fillable_categories=fillable_categories,
            heatsource_categories=heatsource_categories,
            timesteps=timesteps,
        )
        output_systems = cls._RECIPES[name]["output_systems"]
        assert len(output_systems) == 0, f"No output systems can be specified for {cls.__name__}, recipe: {name}!"

    @classproperty
    def relax_recipe_systems(cls):
        # We don't require systems like seasoning/cheese/sesame seeds/etc. to be contained in the baking sheet
        return True

    @classproperty
    def ignore_nonrecipe_systems(cls):
        return True

    @classproperty
    def ignore_nonrecipe_objects(cls):
        return True

    @classproperty
    def is_multi_instance(cls):
        return True

class CookingSystemRule(CookingRule):
    @classmethod
    def add_recipe(
            cls,
            name,
            input_synsets,
            output_synsets,
            input_states=None,
            output_states=None,
            fillable_categories=None,
            heatsource_categories=None,
            timesteps=None,
    ):
        """
        Adds a recipe to this cooking recipe rule to check against. This defines a valid mapping of inputs that
        will transform into the outputs

        Args:
            name (str): Name of the recipe
            input_synsets (dict): Maps synsets to number of instances required for the recipe
            output_synsets (dict): Maps synsets to number of instances to be spawned in the container when the recipe executes
            input_states (dict or None): Maps input synsets to states that must be satisfied for the recipe to execute,
                or None if no states are required
            otuput_states (dict or None): Map output synsets to states that should be set when spawned when the recipe executes,
                or None if no states are required
            fillable_categories (None or set of str): If specified, set of fillable categories which are allowed
                for this recipe. If None, any fillable is allowed
            heatsource_categories (None or set of str): If specified, set of heatsource categories which are allowed
                for this recipe. If None, any heatsource is allowed
            timesteps (None or int): Number of subsequent heating steps required for the recipe to execute. If None,
                it will be set to be 1, i.e.: instantaneous execution
        """
        super().add_recipe(
            name=name,
            input_synsets=input_synsets,
            output_synsets=output_synsets,
            input_states=input_states,
            output_states=output_states,
            fillable_categories=fillable_categories,
            heatsource_categories=heatsource_categories,
            timesteps=timesteps,
        )
        output_objects = cls._RECIPES[name]["output_objects"]
        assert len(output_objects) == 0, f"No output objects can be specified for {cls.__name__}, recipe: {name}!"

    @classproperty
    def relax_recipe_systems(cls):
        return False

    @classproperty
    def ignore_nonrecipe_systems(cls):
        return False

    @classproperty
    def ignore_nonrecipe_objects(cls):
        return False

def import_recipes():
    for json_file, rule_names in _JSON_FILES_TO_RULES.items():
        recipe_fpath = os.path.join(os.path.dirname(bddl.__file__), "generated_data", "transition_map", "tm_jsons", json_file)
        if not os.path.exists(recipe_fpath):
            log.warning(f"Cannot find recipe file at {recipe_fpath}. Skipping importing recipes.")
            # return
        with open(recipe_fpath, "r") as f:
            rule_recipes = json.load(f)
            for rule_name in rule_names:
                rule = REGISTERED_RULES[rule_name]
                if issubclass(rule, RecipeRule):
                    for recipe in rule_recipes:
                        if "rule_name" in recipe:
                            recipe["name"] = recipe.pop("rule_name")
                        if "container" in recipe:
                            recipe["fillable_categories"] = set(recipe.pop("container").keys())
                        if "heat_source" in recipe:
                            recipe["heatsource_categories"] = set(recipe.pop("heat_source").keys())
                        if "machine" in recipe:
                            recipe["fillable_categories"] = set(recipe.pop("machine").keys())

                        satisfied = True
                        output_synsets = set(recipe["output_synsets"].keys())
                        has_substance = any([s for s in output_synsets if is_substance_synset(s)])
                        if (rule_name == "CookingObjectRule" and has_substance) or (rule_name == "CookingSystemRule" and not has_substance):
                            satisfied = False
                        if satisfied:
                            print(recipe)
                            rule.add_recipe(**recipe)
                    print(f"All recipes of rule {rule_name} imported successfully.")

import_recipes()