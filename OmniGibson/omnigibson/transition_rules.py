import json
import math
import operator
import os
import random
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from copy import copy

import bddl
import networkx as nx
import torch as th

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.object_states import (
    ContactParticles,
    ContainedParticles,
    Contains,
    Cooked,
    Covered,
    Filled,
    Heated,
    HeatSourceOrSink,
    MaxTemperature,
    OnTop,
    Open,
    Saturated,
    SlicerActive,
    ToggledOn,
)
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.utils.bddl_utils import translate_bddl_recipe_to_og_recipe, translate_bddl_washer_rule_to_og_washer_rule
from omnigibson.utils.python_utils import Registerable, classproperty, torch_delete
from omnigibson.utils.registry_utils import Registry
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.systems.system_base import VisualParticleSystem
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem

# Create module logger
log = create_module_logger(module_name=__name__)

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
ObjectAttrs = namedtuple("ObjectAttrs", _attrs_fields, defaults=(None,) * len(_attrs_fields))

# Tuple of lists of objects to be added or removed returned from transitions, if not None
TransitionResults = namedtuple("TransitionResults", ["add", "remove"], defaults=(None, None))

# Mapping from transition rule json files to rule classe names
_JSON_FILES_TO_RULES = {
    "heat_cook.json": ["CookingObjectRule", "CookingSystemRule"],
    "mixing_stick.json": ["MixingToolRule"],
    "single_toggleable_machine.json": ["ToggleableMachineRule"],
    "substance_cooking.json": ["CookingPhysicalParticleRule"],
    "substance_watercooking.json": ["CookingPhysicalParticleRule"],
    "washer.json": ["WasherRule"],
}
# Global dicts that will contain mappings
REGISTERED_RULES = dict()


class TransitionRuleAPI:
    """
    Containing methods to check and execute arbitrary discrete state transitions within the simulator
    """

    def __init__(self, scene):
        self.scene = scene

        # Set of active rules
        self.active_rules = set()

        # Maps BaseObject instances to dictionary with the following keys:
        # "states": None or dict mapping object states to arguments to set for that state when the object is initialized
        # "callback": None or function to execute when the object is initialized
        self.obj_init_info = dict()

        self.all_rules = set([rule(scene) for rule in RULES_REGISTRY.objects])

    def get_rule_candidates(self, rule, objects):
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

    def prune_active_rules(self):
        """
        Prunes the active transition rules, removing any whose filter requirements are not satisfied by all current
        objects on the scene. Useful when the current object set changes, e.g.: an object is removed from the simulator
        """
        # Need explicit tuple to iterate over because refresh_rules mutates the ACTIVE_RULES set in place
        self.refresh_rules(rules=tuple(self.active_rules))

    def refresh_all_rules(self):
        """
        Refreshes all registered rules given the current set of objects in the scene
        """
        # Clear all active rules
        self.active_rules = set()

        # Refresh all registered rules
        self.refresh_rules(rules=self.all_rules)

    def refresh_rules(self, rules):
        """
        Refreshes the specified transition rules @rules based on current set of objects in the simulator.
        This will prune any pre-existing rules in self.active_rules if no valid candidates are found, or add / update
        the entry if valid candidates are found

        Args:
            rules (list of BaseTransitionRule): List of transition rules whose candidate lists should be refreshed
        """
        for rule in rules:
            # Skip if rule is not enabled
            if not rule.ENABLED:
                continue

            # Check if rule is still valid, if so, update its entry
            object_candidates = self.get_rule_candidates(rule=rule, objects=self.scene.objects)

            # Update candidates if valid, otherwise pop the entry if it exists in self.active_rules
            if object_candidates is not None:
                # We have a valid rule which should be active, so grab and initialize all of its conditions
                # NOTE: The rule may ALREADY exist in ACTIVE_RULES, but we still need to refresh its candidates because
                # the relevant candidate set / information for the rule + its conditions may have changed given the
                # new set of objects
                rule.refresh(object_candidates=object_candidates)
                self.active_rules.add(rule)
            elif rule in self.active_rules:
                self.active_rules.remove(rule)

    def step(self):
        """
        Steps all active transition rules, checking if any are satisfied, and if so, executing their transition
        """
        # First apply any transition object init states from before, and then clear the dictionary
        for obj, info in self.obj_init_info.items():
            if info["states"] is not None:
                for state, args in info["states"].items():
                    obj.states[state].set_value(*args)
            if info["callback"] is not None:
                info["callback"](obj)
        self.obj_init_info = dict()

        # Iterate over all active rules and process the rule for every valid object candidate combination
        # Cast to list before iterating since ACTIVE_RULES may get updated mid-iteration
        added_obj_attrs = []
        removed_objs = []
        for rule in tuple(self.active_rules):
            output = rule.step()
            # Store objects to be added / removed if we have a valid output
            if output is not None:
                added_obj_attrs += output.add
                removed_objs += output.remove

        self.execute_transition(added_obj_attrs=added_obj_attrs, removed_objs=removed_objs)

    def execute_transition(self, added_obj_attrs, removed_objs):
        """
        Executes the transition for the given added and removed objects.

        :param added_obj_attrs: List of ObjectAttrs instances to add to the scene
        :param removed_objs: List of BaseObject instances to remove from the scene
        """
        # Process all transition results
        if len(removed_objs) > 0:
            # First remove pre-existing objects
            og.sim.batch_remove_objects(removed_objs)

        # Then add new objects
        if len(added_obj_attrs) > 0:
            for added_obj_attr in added_obj_attrs:
                new_obj = added_obj_attr.obj
                self.scene.add_object(new_obj)
                # By default, added_obj_attr is populated with all Nones -- so these will all be pass-through operations
                # unless pos / orn (or, conversely, bb_pos / bb_orn) is specified
                if added_obj_attr.pos is not None or added_obj_attr.orn is not None:
                    new_obj.set_position_orientation(position=added_obj_attr.pos, orientation=added_obj_attr.orn)
                elif isinstance(new_obj, DatasetObject) and (
                    added_obj_attr.bb_pos is not None or added_obj_attr.bb_orn is not None
                ):
                    new_obj.set_bbox_center_position_orientation(
                        position=added_obj_attr.bb_pos, orientation=added_obj_attr.bb_orn
                    )
                else:
                    raise ValueError(
                        "Expected at least one of pos, orn, bb_pos, or bb_orn to be specified in ObjectAttrs!"
                    )
                # Additionally record any requested states if specified to be updated during the next transition step
                if added_obj_attr.states is not None or added_obj_attr.callback is not None:
                    self.obj_init_info[new_obj] = {
                        "states": added_obj_attr.states,
                        "callback": added_obj_attr.callback,
                    }

    def clear(self):
        """
        Clears any internal state when the simulator is restarted (e.g.: when a new stage is opened)
        """
        # Clear internal dictionaries
        self.active_rules = set()
        self.obj_init_info = dict()


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


class ObjectPropertyFilter(ObjectCandidateFilter):
    """Filter for arbitrary object properties"""

    def __init__(self, name, value):
        self.property = name
        self.value = value

    def __call__(self, obj):
        return getattr(obj, self.property) == self.value


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

    Note that this condition uses the RigidContactAPI for contact checking. This is not a persistent contact check,
    meaning that if objects get in contact for some time and both fall asleep, the contact will not be detected.
    To get persistent contact checking, please use contact_sensor.
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

    def refresh(self, object_candidates):
        # Register idx mappings
        self._filter_1_idxs = {
            obj: [RigidContactAPI.get_body_row_idx(link.prim_path)[1] for link in obj.links.values()]
            for obj in object_candidates[self._filter_1_name]
        }
        self._filter_2_idxs = {
            obj: th.tensor(
                [RigidContactAPI.get_body_col_idx(link.prim_path)[1] for link in obj.links.values()],
                dtype=th.float32,
            )
            for obj in object_candidates[self._filter_2_name]
        }

    def __call__(self, object_candidates):
        # Keep any object that has non-zero impulses between itself and any of the @filter_2_name's objects
        objs = []

        # Batch check for each object
        for obj in object_candidates[self._filter_1_name]:
            # Get all impulses between @obj and any object in @filter_2_name that are in the same scene
            idxs_to_check = th.cat(
                [
                    self._filter_2_idxs[obj2]
                    for obj2 in object_candidates[self._filter_2_name]
                    if obj2.scene == obj.scene
                ]
            )
            if th.any(
                RigidContactAPI.get_all_impulses(obj.scene.idx)[self._filter_1_idxs[obj]][:, idxs_to_check.tolist()]
            ):
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
        object_candidates[self._filter_name] = [
            obj
            for obj in object_candidates[self._filter_name]
            if self._op(obj.states[self._state].get_value(), self._val)
        ]

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
        self._last_valid_candidates = {filter_name: set() for filter_name in self.modifies_filter_names}

    def refresh(self, object_candidates):
        # Refresh nested condition
        self._condition.refresh(object_candidates=object_candidates)

    def __call__(self, object_candidates):
        # Call wrapped method first
        valid = self._condition(object_candidates=object_candidates)
        # Iterate over all current candidates -- if there's a mismatch in last valid candidates and current,
        # then we store it, otherwise, we don't
        for filter_name in self.modifies_filter_names:
            # Compute current valid candidates
            objs = [
                obj for obj in object_candidates[filter_name] if obj not in self._last_valid_candidates[filter_name]
            ]
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
                the UNION of all candidates.
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

        # For each filter, take the union over object candidates across each condition.
        # If the result is empty, we immediately return False.
        for filter_name in object_candidates:
            object_candidates[filter_name] = list(
                set.union(*[set(candidates[filter_name]) for candidates in pruned_candidates.values()])
            )
            if len(object_candidates[filter_name]) == 0:
                return False

        return True

    @property
    def modifies_filter_names(self):
        # Return all wrapped names
        return set.union(*(condition.modifies_filter_names for condition in self._conditions))


class AndConditionWrapper(RuleCondition):
    """
    Logical AND between multiple RuleConditions
    """

    def __init__(self, conditions):
        """
        Args:
            conditions (list of RuleConditions): Conditions to take logical AND over. This will generate
                the INTERSECTION of all candidates.
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

        # For each filter, take the intersection over object candidates across each condition.
        # If the result is empty, we immediately return False.
        for filter_name in object_candidates:
            object_candidates[filter_name] = list(
                set.intersection(*[set(candidates[filter_name]) for candidates in pruned_candidates.values()])
            )
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

    ENABLED = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Register this system, and
        # make sure at least one filter is specified -- in general, there should never be a rule
        # where no filter is specified
        # Only run this check for actual rules that are being registered
        if cls.__name__ not in cls._do_not_register_classes:
            global RULES_REGISTRY
            RULES_REGISTRY.add(obj=cls)
            assert (
                len(cls.candidate_filters) > 0
            ), "At least one of individual_filters or group_filters must be specified!"

    def __init__(self, scene):
        self.scene = scene
        self.candidates = None
        # Delay condition generation until the first time it's accessed
        self.conditions = None

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

    def _generate_conditions(self):
        """
        Generates rule condition(s)s for this transition rule. These conditions are used to prune object
        candidates at runtime, to determine whether a transition rule should occur at the given timestep

        Returns:
            list of RuleCondition: Condition(s) to enforce to determine whether a transition rule should occur
        """
        raise NotImplementedError

    def get_object_candidates(self, objects):
        """
        Given the set of objects @objects, compute the valid object candidate combinations that may be valid for
        this TransitionRule

        Args:
            objects (list of BaseObject): Objects to filter for valid transition rule candidates

        Returns:
            dict: Maps filter name to valid object(s) that satisfy that filter
        """
        # Iterate over all objects and add to dictionary if valid
        filters = self.candidate_filters
        obj_dict = {filter_name: [] for filter_name in filters.keys()}

        for obj in objects:
            for fname, f in filters.items():
                if f(obj):
                    obj_dict[fname].append(obj)

        return obj_dict

    def refresh(self, object_candidates):
        """
        Refresh any internal state for this rule, given set of input object candidates @object_candidates

        Args:
            object_candidates (dict): Maps filter name to valid object(s) that satisfy that filter
        """
        # Store candidates
        self.candidates = object_candidates

        # Refresh all conditions
        if self.conditions is None:
            self.conditions = self._generate_conditions()
        for condition in self.conditions:
            condition.refresh(object_candidates=object_candidates)

    def transition(self, object_candidates):
        """
        Rule to apply for each set of objects satisfying the condition.

        Args:
            object_candidates (dict): Dictionary mapping corresponding keys from @cls.candidate_filters to list of individual
                object instances where the filter is satisfied

        Returns:
            TransitionResults: results from the executed transition
        """
        raise NotImplementedError()

    def step(self):
        """
        Takes a step for this transition rule, checking if all of @self.conditions are satisified, and if so, taking
        a transition via @self.transition()

        Returns:
            None or TransitionResults: If a transition occurs, returns its results, otherwise, returns None
        """
        # Copy the candidates dictionary since it may be mutated in place by @conditions
        object_candidates = {filter_name: candidates.copy() for filter_name, candidates in self.candidates.items()}
        if self.conditions is None:
            self.conditions = self._generate_conditions()
        for condition in self.conditions:
            if not condition(object_candidates=object_candidates):
                # Condition was not met, so immediately terminate
                return

        # All conditions are met, take the transition
        return self.transition(object_candidates=object_candidates)

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
)


class WasherDryerRule(BaseTransitionRule):
    """
    Transition rule to apply to cloth washers and dryers.
    """

    def _generate_conditions(self):
        assert len(self.candidate_filters.keys()) == 1
        machine_type = list(self.candidate_filters.keys())[0]
        return [
            ChangeConditionWrapper(
                condition=AndConditionWrapper(
                    conditions=[
                        StateCondition(filter_name=machine_type, state=ToggledOn, val=True, op=operator.eq),
                        StateCondition(filter_name=machine_type, state=Open, val=False, op=operator.eq),
                    ]
                )
            )
        ]

    def _compute_global_rule_info(self):
        """
        Helper function to compute global information necessary for checking rules. This is executed exactly
        once per self.transition() step

        Returns:
            dict: Keyword-mapped global rule information
        """
        # Compute all obj collision points (handles both cloth and rigid objects)
        obj_collision_points = [obj.collision_points_world for obj in self.scene.objects]
        return dict(obj_collision_points=obj_collision_points)

    def _compute_container_info(self, object_candidates, container, global_info):
        """
        Helper function to compute container-specific information necessary for checking rules. This is executed once
        per container per self.transition() step

        Args:
            object_candidates (dict): Dictionary mapping corresponding keys from @cls.candidate_filters to list of individual
                object instances where the filter is satisfied
            container (StatefulObject): Relevant container object for computing information
            global_info (dict): Output of @self._compute_global_rule_info(); global information which may be
                relevant for computing container information

        Returns:
            dict: Keyword-mapped container information
        """
        del object_candidates
        obj_collision_points = global_info["obj_collision_points"]

        # Check each object's collision points
        in_volume_objs = []
        for obj, collision_points in zip(self.scene.objects, obj_collision_points):
            # Check if any of the collision points is in volume
            points_in_volume = container.states[ContainedParticles].link.check_points_in_volume(collision_points)
            # If any point is in volume, include this object
            if th.any(points_in_volume):
                in_volume_objs.append(obj)

        # Remove the container itself
        if container in in_volume_objs:
            in_volume_objs.remove(container)

        return dict(in_volume_objs=in_volume_objs)

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("WasherDryerRule")
        return classes


class WasherRule(WasherDryerRule):
    """
    Transition rule to apply to cloth washers.
    1. remove "dirty" particles from the washer if the necessary solvent is present.
    2. wet the objects inside by making them either Saturated with or Covered by water.
    """

    cleaning_conditions = None

    @classmethod
    def register_cleaning_conditions(cls, conditions):
        """
        Register cleaning conditions for this rule.

        Args:
            conditions (dict): ictionary mapping the system name (str) to None or list of system names (str). None
                represents "never", empty list represents "always", or non-empty list represents at least one of the
                systems in the list needs to be present in the washer for the key system to be removed.
                E.g. "rust" -> None: "never remove rust from the washer"
                E.g. "dust" -> []: "always remove dust from the washer"
                E.g. "cooking_oil" -> ["sodium_carbonate", "vinegar"]: "remove cooking_oil from the washer if either
                sodium_carbonate or vinegar is present"
                For keys not present in the dictionary, the default is []: "always remove"
        """
        cls.cleaning_conditions = conditions

    @classproperty
    def candidate_filters(cls):
        return {
            "washer": CategoryFilter("washer"),
        }

    def transition(self, object_candidates):
        global_info = self._compute_global_rule_info()
        for washer in object_candidates["washer"]:
            # Remove the systems if the conditions are met
            systems_to_remove = []
            for system in self.scene.active_systems.values():
                # Never remove
                if system.name in self.cleaning_conditions and self.cleaning_conditions[system.name] is None:
                    continue
                if not washer.states[Contains].get_value(system):
                    continue

                solvents = self.cleaning_conditions.get(system.name, [])
                # Always remove
                if len(solvents) == 0:
                    systems_to_remove.append(system)
                else:
                    solvents = [
                        self.scene.get_system(solvent) for solvent in solvents if self.scene.is_system_active(solvent)
                    ]
                    # If any of the solvents are present
                    if any(washer.states[Contains].get_value(solvent) for solvent in solvents):
                        systems_to_remove.append(system)

            container_info = self._compute_container_info(
                object_candidates=object_candidates, container=washer, global_info=global_info
            )
            in_volume_objs = container_info["in_volume_objs"]
            for system in systems_to_remove:
                if isinstance(system, VisualParticleSystem):
                    # If the system is a visual particle system, remove it from the objects in the washer
                    for obj in in_volume_objs:
                        obj.states[Covered].set_value(system, False)
                else:
                    washer.states[Contains].set_value(system, False)

            if gm.USE_GPU_DYNAMICS:
                # Make the objects wet
                water = self.scene.get_system("water", force_init=True)
                for obj in in_volume_objs:
                    if Saturated in obj.states:
                        obj.states[Saturated].set_value(water, True)
                    else:
                        obj.states[Covered].set_value(water, True)

        return TransitionResults(add=[], remove=[])


class DryerRule(WasherDryerRule):
    """
    Transition rule to apply to cloth dryers.
    1. dry the objects inside by making them not Saturated with water.
    2. remove all water from the dryer.
    """

    @classproperty
    def candidate_filters(cls):
        return {
            "dryer": CategoryFilter("clothes_dryer"),
        }

    def transition(self, object_candidates):
        water = self.scene.get_system("water", force_init=False)
        if water.initialized:
            global_info = self._compute_global_rule_info()
            for dryer in object_candidates["dryer"]:
                container_info = self._compute_container_info(
                    object_candidates=object_candidates, container=dryer, global_info=global_info
                )
                in_volume_objs = container_info["in_volume_objs"]
                for obj in in_volume_objs:
                    if Saturated in obj.states:
                        obj.states[Saturated].set_value(water, False)
                dryer.states[Contains].set_value(water, False)

        return TransitionResults(add=[], remove=[])


class SlicingRule(BaseTransitionRule):
    """
    Transition rule to apply to sliced / slicer object pairs.
    """

    @classproperty
    def candidate_filters(cls):
        return {
            "sliceable": AbilityFilter("sliceable"),
            "slicer": AbilityFilter("slicer"),
        }

    def _generate_conditions(self):
        # sliceables should be touching any slicer
        return [
            TouchingAnyCondition(filter_1_name="sliceable", filter_2_name="slicer"),
            StateCondition(filter_name="slicer", state=SlicerActive, val=True, op=operator.eq),
        ]

    def transition(self, object_candidates):
        objs_to_add, objs_to_remove = [], []

        for sliceable_obj in object_candidates["sliceable"]:
            # Object parts offset annotation are w.r.t the base link of the whole object.
            pos, orn = sliceable_obj.get_position_orientation()

            # Load object parts
            for i, part in enumerate(sliceable_obj.metadata["object_parts"].values()):
                # List of dicts gets replaced by {'0':dict, '1':dict, ...}

                # Get bounding box info
                part_bb_pos = th.tensor(part["bb_pos"], dtype=th.float32)
                part_bb_orn = th.tensor(part["bb_orn"], dtype=th.float32)

                # Scale the offset accordingly.
                # If the scale of the sliceable object is uniform, we can just take its scale
                if th.all(sliceable_obj.scale == sliceable_obj.scale[0]):
                    scale = sliceable_obj.scale
                else:
                    # Determine the relative scale to apply to the object part from the original object
                    # Note that proper (rotated) scaling can only be applied when the relative orientation of
                    # the object part is a multiple of 90 degrees wrt the parent object, so we assert that here
                    assert T.check_quat_right_angle(
                        part_bb_orn
                    ), "Sliceable objects should only have relative object part orientations that are factors of 90 degrees!"
                    scale = th.abs(T.quat2mat(part_bb_orn) @ sliceable_obj.scale)

                # Calculate global part bounding box pose.
                part_bb_pos = pos + T.quat2mat(orn) @ (part_bb_pos * scale)
                part_bb_orn = T.quat_multiply(orn, part_bb_orn)
                part_obj_name = f"half_{sliceable_obj.name}_{i}"
                part_obj = DatasetObject(
                    name=part_obj_name,
                    category=part["category"],
                    model=part["model"],
                    bounding_box=th.tensor(part["bb_size"], dtype=th.float32)
                    * scale,  # equiv. to scale=(part["bb_size"] / self.native_bbox) * (scale)
                )

                sliceable_obj_state = sliceable_obj.dump_state()
                # Propagate non-physical states of the whole object to the half objects, e.g. cooked, saturated, etc.
                # Add the new object to the results.
                new_obj_attrs = ObjectAttrs(
                    obj=part_obj,
                    bb_pos=part_bb_pos,
                    bb_orn=part_bb_orn,
                    callback=lambda obj: obj.load_non_kin_state(sliceable_obj_state),
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

    def _generate_conditions(self):
        # sliceables should be touching any slicer
        return [
            TouchingAnyCondition(filter_1_name="diceable", filter_2_name="slicer"),
            StateCondition(filter_name="slicer", state=SlicerActive, val=True, op=operator.eq),
        ]

    def transition(self, object_candidates):
        objs_to_remove = []

        for diceable_obj in object_candidates["diceable"]:
            # We expect all diced particle systems to follow the naming convention (cooked__)diced__<category>
            system_name = "diced__" + diceable_obj.category.removeprefix("half_")
            if Cooked in diceable_obj.states and diceable_obj.states[Cooked].get_value():
                system_name = "cooked__" + system_name
            system = self.scene.get_system(system_name)
            system.generate_particles_from_link(
                diceable_obj, diceable_obj.root_link, check_contact=False, use_visual_meshes=False
            )

            # Delete original object from stage.
            objs_to_remove.append(diceable_obj)

        return TransitionResults(add=[], remove=objs_to_remove)


class MeltingRule(BaseTransitionRule):
    """
    Transition rule to apply to meltable objects to simulate melting
    Once the object reaches the melting temperature, remove the object and spawn the melted substance in its place.
    """

    @classproperty
    def candidate_filters(cls):
        # We want to find all meltable objects
        return {"meltable": AbilityFilter("meltable")}

    def _generate_conditions(self):
        return [StateCondition(filter_name="meltable", state=MaxTemperature, val=m.MELTING_TEMPERATURE, op=operator.ge)]

    def transition(self, object_candidates):
        objs_to_remove = []

        # Convert the meltable object into its melted substance
        for meltable_obj in object_candidates["meltable"]:
            # All meltable xyz, half_xyz and diced__xyz transform into melted__xyz
            root_category = meltable_obj.category.removeprefix("half_").removeprefix("diced__")
            system_name = f"melted__{root_category}"
            system = self.scene.get_system(system_name)
            system.generate_particles_from_link(
                meltable_obj, meltable_obj.root_link, check_contact=False, use_visual_meshes=False
            )

            # Delete original object from stage.
            objs_to_remove.append(meltable_obj)

        return TransitionResults(add=[], remove=objs_to_remove)


class RecipeRule(BaseTransitionRule):
    """
    Transition rule to approximate recipe-based transitions
    """

    # Maps recipe name to recipe information
    _RECIPES = None

    def __init_subclass__(cls, **kwargs):
        # Run super first
        super().__init_subclass__(**kwargs)

        # Initialize recipes
        cls._recipes = dict()

    def __init__(self, scene):
        super().__init__(scene)

        # Maps active recipe name to recipe information
        self._active_recipes = None

        # Maps object category name to indices in the flattened object array for efficient computation
        self._category_idxs = None

        # Flattened array of all simulator objects, sorted by category
        self._objects = None

        # Maps object to idx within the _objects array
        self._objects_to_idx = None

    @classmethod
    def add_recipe(
        cls,
        name,
        input_objects,
        input_systems,
        output_objects,
        output_systems,
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
            input_objects (dict): Maps object categories to number of instances required for the recipe
            input_systems (list): List of system names required for the recipe
            output_objects (dict): Maps object categories to number of instances to be spawned in the container when the recipe executes
            output_systems (list): List of system names to be spawned in the container when the recipe executes. Currently the length is 1.
            input_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system", "binary_object"] to a list of states that must be satisfied for the recipe to execute
            output_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system"] to a list of states that should be set after the output objects are spawned
            fillable_categories (None or set of str): If specified, set of fillable categories which are allowed
                for this recipe. If None, any fillable is allowed
            kwargs (dict): Any additional keyword-arguments to be stored as part of this recipe
        """

        input_states = input_states if input_states is not None else defaultdict(lambda: defaultdict(list))
        output_states = output_states if output_states is not None else defaultdict(lambda: defaultdict(list))

        input_object_tree = None
        if cls.is_multi_instance and len(input_objects) > 0:
            # Build a tree of input object categories according to the kinematic binary states
            # Example: 'raw_egg': {'binary_object': [(OnTop, 'bagel_dough', True)]} results in an edge
            # from 'bagel_dough' to 'raw_egg', i.e. 'bagel_dough' is the parent of 'raw_egg'.
            input_object_tree = nx.DiGraph()
            for obj_category, state_checks in input_states.items():
                for state_class, second_obj_category, state_value in state_checks["binary_object"]:
                    input_object_tree.add_edge(second_obj_category, obj_category)

            if nx.is_empty(input_object_tree):
                input_object_tree = None
            else:
                assert nx.is_tree(input_object_tree), f"Input object tree must be a tree! Now: {input_object_tree}."
                root_nodes = [node for node in input_object_tree.nodes() if input_object_tree.in_degree(node) == 0]
                assert len(root_nodes) == 1, f"Input object tree must have exactly one root node! Now: {root_nodes}."
                assert (
                    input_objects[root_nodes[0]] == 1
                ), f"Input object tree root node must have exactly one instance! Now: {cls._recipes[name]['input_objects'][root_nodes[0]]}."

        # Store information for this recipe
        cls._recipes[name] = {
            "name": name,
            "input_objects": input_objects,
            "input_systems": input_systems,
            "output_objects": output_objects,
            "output_systems": output_systems,
            "input_states": input_states,
            "output_states": output_states,
            "fillable_categories": fillable_categories,
            "input_object_tree": input_object_tree,
            **kwargs,
        }

    def _validate_recipe_container_is_valid(self, recipe, container):
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

    def _validate_recipe_systems_are_contained(self, recipe, container):
        """
        Validates whether @recipe's input_systems are all contained in @container or not

        Args:
            recipe (dict): Recipe whose systems should be checked
            container (BaseObject): Container object that should contain all of @recipe's input systems

        Returns:
            bool: True if all the input systems are contained
        """
        for system_name in recipe["input_systems"]:
            system = self.scene.get_system(system_name)
            if not container.states[Contains].get_value(system=system):
                return False
        return True

    def _validate_nonrecipe_systems_not_contained(self, recipe, container):
        """
        Validates whether all systems not relevant to @recipe are not contained in @container

        Args:
            recipe (dict): Recipe whose systems should be checked
            container (BaseObject): Container object that should contain all of @recipe's input systems

        Returns:
            bool: True if none of the non-relevant systems are contained
        """
        for system in self.scene.active_systems.values():
            if system.name not in recipe["input_systems"] and container.states[Contains].get_value(system=system):
                return False
        return True

    def _validate_recipe_objects_are_contained_and_states_satisfied(self, recipe, container_info):
        """
        Validates whether @recipe's input_objects are contained in the container and whether their states are satisfied

        Args:
            recipe (dict): Recipe whose objects should be checked
            container_info (dict): Output of @self._compute_container_info(); container-specific information which may
                be relevant for computing whether recipe is executable. This will be populated with execution info.

        Returns:
            bool: True if all the input object quantities are contained
        """
        in_volume = container_info["in_volume"]

        # Store necessary information for execution
        container_info["execution_info"] = dict()
        category_to_valid_indices = self._filter_input_objects_by_unary_and_binary_system_states(recipe=recipe)
        container_info["execution_info"]["category_to_valid_indices"] = category_to_valid_indices

        if not self.is_multi_instance:
            return self._validate_recipe_objects_non_multi_instance(
                recipe=recipe,
                category_to_valid_indices=category_to_valid_indices,
                in_volume=in_volume,
            )
        else:
            return self._validate_recipe_objects_multi_instance(
                recipe=recipe,
                category_to_valid_indices=category_to_valid_indices,
                container_info=container_info,
            )

    def _filter_input_objects_by_unary_and_binary_system_states(self, recipe):
        # Filter input objects based on a subset of input states (unary states and binary system states)
        # Map object categories (str) to valid indices (th.tensor)
        category_to_valid_indices = dict()
        for obj_category in recipe["input_objects"]:
            if obj_category not in recipe["input_states"]:
                # If there are no input states, all objects of this category are valid
                category_to_valid_indices[obj_category] = self._category_idxs[obj_category]
            else:
                category_to_valid_indices[obj_category] = []
                for idx in self._category_idxs[obj_category]:
                    obj = self._objects[idx]
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
                        system = self.scene.get_system(system_name)
                        if obj.states[state_class].get_value(system=system) != state_value:
                            success = False
                            break
                    if not success:
                        continue

                    category_to_valid_indices[obj_category].append(idx)

                # Convert to numpy array for faster indexing
                category_to_valid_indices[obj_category] = th.tensor(
                    category_to_valid_indices[obj_category], dtype=th.int32
                )
        return category_to_valid_indices

    def _validate_recipe_objects_non_multi_instance(self, recipe, category_to_valid_indices, in_volume):
        # Check if sufficiently number of objects are contained
        for obj_category, obj_quantity in recipe["input_objects"].items():
            if th.sum(in_volume[category_to_valid_indices[obj_category]]) < obj_quantity:
                return False
        return True

    def _validate_recipe_objects_multi_instance(self, recipe, category_to_valid_indices, container_info):
        in_volume = container_info["in_volume"]
        input_object_tree = recipe["input_object_tree"]

        # Map object category to a set of objects that are used in this execution
        relevant_objects = defaultdict(set)

        # Map system name to a set of particle indices that are used in this execution
        relevant_systems = defaultdict(set)

        # Number of instances of this recipe that can be produced
        num_instances = 0

        # Define a recursive function to check the kinematic tree
        def check_kinematic_tree(obj, should_check_in_volume=False):
            """
            Recursively check if the kinematic tree is satisfied.
            Return True/False, and a set of objects that belong to the subtree rooted at the current node

            Args:
                obj (BaseObject): Subtree root node to check
                should_check_in_volume (bool): Whether to check if the object is in the volume or not
            Returns:
                bool: True if the subtree rooted at the current node is satisfied
                set: Set of objects that belong to the subtree rooted at the current node
            """

            # Check if obj is in volume
            if should_check_in_volume and not in_volume[self._objects_to_idx[obj]]:
                return False, set()

            # If the object is a leaf node, return True and the set containing the object
            if input_object_tree.out_degree(obj.category) == 0:
                return True, set([obj])

            children_categories = list(input_object_tree.successors(obj.category))

            all_subtree_objs = set()
            for child_cat in children_categories:
                assert (
                    len(input_states[child_cat]["binary_object"]) == 1
                ), "Each child node should have exactly one binary object state, i.e. one parent in the input_object_tree"
                state_class, _, state_value = input_states[child_cat]["binary_object"][0]
                num_valid_children = 0
                children_objs = [self._objects[i] for i in category_to_valid_indices[child_cat]]
                for child_obj in children_objs:
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

        # If multi-instance is True but doesn't require kinematic states between objects
        if input_object_tree is None:
            num_instances = float("inf")
            # Compute how many instances of this recipe can be produced.
            # Example: if a recipe requires 1 apple and 2 bananas, and there are 3 apples and 4 bananas in the
            # container, then 2 instance of the recipe can be produced.
            for obj_category, obj_quantity in recipe["input_objects"].items():
                quantity_in_volume = th.sum(in_volume[category_to_valid_indices[obj_category]])
                num_inst = quantity_in_volume // obj_quantity
                if num_inst < 1:
                    return False
                num_instances = min(num_instances, num_inst)

            # If at least one instance of the recipe can be executed, add all valid objects to be relevant_objects.
            # This can be considered as a special case of below where there are no binary kinematic states required.
            for obj_category in recipe["input_objects"]:
                relevant_objects[obj_category] = set(self._objects[category_to_valid_indices[obj_category]])

        # If multi-instance is True and requires kinematic states between objects
        else:
            root_node_category = [node for node in input_object_tree.nodes() if input_object_tree.in_degree(node) == 0][
                0
            ]
            # A list of objects belonging to the root node category
            root_nodes = [self._objects[i] for i in category_to_valid_indices[root_node_category]]
            input_states = recipe["input_states"]

            for root_node in root_nodes:
                # should_check_in_volume is True only for the root nodes.
                # Example: the bagel dough needs to be in_volume of the container, but the raw egg on top doesn't.
                tree_valid, relevant_object_set = check_kinematic_tree(obj=root_node, should_check_in_volume=True)
                if tree_valid:
                    # For each valid tree, increment the number of instances and aggregate the objects
                    num_instances += 1
                    for obj in relevant_object_set:
                        relevant_objects[obj.category].add(obj)

            # If there are no valid trees, return False
            if num_instances == 0:
                return False

        # Note that for multi instance recipes, the relevant system particles are NOT the ones in the container.
        # Instead, they are the ones that are related to the relevant objects, e.g. salt covering the bagel dough.
        for obj_category, objs in relevant_objects.items():
            for state_class, system_name, state_value in recipe["input_states"][obj_category]["binary_system"]:
                # If the state value is False, skip
                if not state_value:
                    continue
                system = self.scene.get_system(system_name)
                for obj in objs:
                    if state_class in [Filled, Contains]:
                        contained_particle_idx = obj.states[ContainedParticles].get_value(system).in_volume.nonzero()
                        relevant_systems[system_name] |= contained_particle_idx
                    elif state_class in [Covered]:
                        covered_particle_idx = obj.states[ContactParticles].get_value(system)
                        relevant_systems[system_name] |= covered_particle_idx

        # Now we populate the execution info with the relevant objects and systems as well as the number of
        # instances of the recipe that can be produced.
        container_info["execution_info"]["relevant_objects"] = relevant_objects
        container_info["execution_info"]["relevant_systems"] = relevant_systems
        container_info["execution_info"]["num_instances"] = num_instances
        return True

    def _validate_nonrecipe_objects_not_contained(self, recipe, container_info):
        """
        Validates whether all objects not relevant to @recipe are not contained in the container
        represented by @in_volume

        Args:
            recipe (dict): Recipe whose systems should be checked
            container_info (dict): Output of @self._compute_container_info(); container-specific information
                which may be relevant for computing whether recipe is executable

        Returns:
            bool: True if none of the non-relevant objects are contained
        """
        in_volume = container_info["in_volume"]
        # These are object indices whose objects satisfy the input states
        category_to_valid_indices = container_info["execution_info"]["category_to_valid_indices"]
        nonrecipe_objects_in_volume = (
            in_volume
            if len(recipe["input_objects"]) == 0
            else torch_delete(
                in_volume,
                th.cat([category_to_valid_indices[obj_category] for obj_category in category_to_valid_indices]),
            )
        )
        return not th.any(nonrecipe_objects_in_volume)

    def _validate_recipe_systems_exist(self, recipe):
        """
        Validates whether @recipe's input_systems are all active or not

        Args:
            recipe (dict): Recipe whose systems should be checked

        Returns:
            bool: True if all the input systems are active
        """
        for system_name in recipe["input_systems"]:
            if not self.scene.is_system_active(system_name=system_name):
                return False
        return True

    def _validate_recipe_objects_exist(self, recipe):
        """
        Validates whether @recipe's input_objects exist in the current scene or not

        Args:
            recipe (dict): Recipe whose objects should be checked

        Returns:
            bool: True if all the input objects exist in the scene
        """
        for obj_category, obj_quantity in recipe["input_objects"].items():
            if len(self.scene.object_registry("category", obj_category, default_val=set())) < obj_quantity:
                return False
        return True

    def _validate_recipe_fillables_exist(self, recipe):
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
            if len(self.scene.object_registry("category", category, default_val=set())) > 0:
                return True

        # None found, return False
        return False

    def _is_recipe_active(self, recipe):
        """
        Helper function to determine whether a given recipe @recipe should be actively checked for or not.

        Args:
            recipe (dict): Maps relevant keyword to corresponding recipe info

        Returns:
            bool: True if the recipe is active, else False
        """
        # Check valid active systems
        if not self._validate_recipe_systems_exist(recipe=recipe):
            return False

        # Check valid object quantities
        if not self._validate_recipe_objects_exist(recipe=recipe):
            return False

        # Check valid fillable categories
        if not self._validate_recipe_fillables_exist(recipe=recipe):
            return False

        return True

    def _is_recipe_executable(self, recipe, container, global_info, container_info):
        """
        Helper function to determine whether a given recipe @recipe should be immediately executed or not.

        Args:
            recipe (dict): Maps relevant keyword to corresponding recipe info
            container (StatefulObject): Container in which @recipe may be executed
            global_info (dict): Output of @self._compute_global_rule_info(); global information which may be
                relevant for computing whether recipe is executable
            container_info (dict): Output of @self._compute_container_info(); container-specific information
                which may be relevant for computing whether recipe is executable

        Returns:
            bool: True if the recipe is active, else False
        """
        # Verify the container category is valid
        if not self._validate_recipe_container_is_valid(recipe=recipe, container=container):
            return False

        # Verify all required systems are contained in the container
        if not self.relax_recipe_systems and not self._validate_recipe_systems_are_contained(
            recipe=recipe, container=container
        ):
            return False

        # Verify all required object quantities are contained in the container and their states are satisfied
        if not self._validate_recipe_objects_are_contained_and_states_satisfied(
            recipe=recipe, container_info=container_info
        ):
            return False

        # Verify no non-relevant system is contained
        if not self.ignore_nonrecipe_systems and not self._validate_nonrecipe_systems_not_contained(
            recipe=recipe, container=container
        ):
            return False

        # Verify no non-relevant object is contained if we're not ignoring them
        if not self.ignore_nonrecipe_objects and not self._validate_nonrecipe_objects_not_contained(
            recipe=recipe, container_info=container_info
        ):
            return False

        return True

    def _compute_global_rule_info(self):
        """
        Helper function to compute global information necessary for checking rules. This is executed exactly
        once per self.transition() step

        Returns:
            dict: Keyword-mapped global rule information
        """
        # Compute all relevant object AABB positions
        obj_positions = th.stack([obj.aabb_center for obj in self._objects])
        return dict(obj_positions=obj_positions)

    def _compute_container_info(self, object_candidates, container, global_info):
        """
        Helper function to compute container-specific information necessary for checking rules. This is executed once
        per container per self.transition() step

        Args:
            object_candidates (dict): Dictionary mapping corresponding keys from @cls.candidate_filters to list of individual
                object instances where the filter is satisfied
            container (StatefulObject): Relevant container object for computing information
            global_info (dict): Output of @self._compute_global_rule_info(); global information which may be
                relevant for computing container information

        Returns:
            dict: Keyword-mapped container information
        """
        del object_candidates
        obj_positions = global_info["obj_positions"]
        # Compute in volume for all relevant object positions
        # We check for either the object AABB being contained OR the object being on top of the container, in the
        # case that the container is too flat for the volume to contain the object
        in_volume = container.states[ContainedParticles].link.check_points_in_volume(obj_positions) | th.tensor(
            [obj.states[OnTop].get_value(container) for obj in self._objects]
        )

        # Container itself is never within its own volume
        in_volume[self._objects_to_idx[container]] = False

        return dict(in_volume=in_volume)

    def refresh(self, object_candidates):
        # Run super first
        super().refresh(object_candidates=object_candidates)

        # Cache active recipes given the current set of objects
        self._active_recipes = dict()
        self._category_idxs = dict()
        self._objects = []
        self._objects_to_idx = dict()

        # Prune any recipes whose objects / system requirements are not met by the current set of objects / systems
        objects_by_category = defaultdict(list)
        scene_objects_by_category = self.scene.object_registry.get_dict("category")
        for cat, objs in scene_objects_by_category.items():
            objects_by_category[cat].extend(objs)

        for name, recipe in self._recipes.items():
            # If all pre-requisites met, add to active recipes
            if self._is_recipe_active(recipe=recipe):
                self._active_recipes[name] = recipe

        # Finally, compute relevant objects and category mapping based on relevant categories
        i = 0
        for category, objects in objects_by_category.items():
            # Exclude fixed base objects for performance
            objects = [obj for obj in objects if not obj.fixed_base]
            self._category_idxs[category] = i + th.arange(len(objects))
            self._objects += list(objects)
            for obj in objects:
                self._objects_to_idx[obj] = i
                i += 1

    @classproperty
    def candidate_filters(cls):
        # Fillable object required
        # We also will filter out any fixed_base objects, because otherwise all cabinets, fridges, etc. would become valid containers!
        return {"container": AndFilter([AbilityFilter(ability="fillable"), ObjectPropertyFilter("fixed_base", False)])}

    def transition(self, object_candidates):
        objs_to_add, objs_to_remove = [], []

        # Compute global info
        global_info = self._compute_global_rule_info()

        # Iterate over all fillable objects, to execute recipes for each one
        for container in object_candidates["container"]:
            recipe_results = None
            # Compute container info
            container_info = self._compute_container_info(
                object_candidates=object_candidates,
                container=container,
                global_info=global_info,
            )

            # Check every recipe to find if any is valid
            for name, recipe in self._active_recipes.items():
                if self._is_recipe_executable(
                    recipe=recipe, container=container, global_info=global_info, container_info=container_info
                ):
                    # Otherwise, all conditions met, we found a valid recipe and so we execute and terminate early
                    og.log.info(f"Executing recipe: {name} in container {container.name}!")

                    # Take the transform and terminate early
                    recipe_results = self._execute_recipe(
                        container=container,
                        recipe=recipe,
                        container_info=container_info,
                    )
                    objs_to_add += recipe_results.add
                    objs_to_remove += recipe_results.remove
                    break

            # Otherwise, if we didn't find a valid recipe, we execute a garbage transition instead if requested
            if recipe_results is None and self.use_garbage_fallback_recipe:
                og.log.info(
                    f"Did not find a valid recipe for rule {self.__class__.__name__}; generating {m.DEFAULT_GARBAGE_SYSTEM} in {container.name}!"
                )

                # Generate garbage fluid
                garbage_results = self._execute_recipe(
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

    def _execute_recipe(self, container, recipe, container_info):
        """
        Transforms all items contained in @container into @output_system, generating volume of @output_system
        proportional to the number of items transformed.

        Args:
            container (BaseObject): Container object which will have its contained elements transformed into
                @output_system
            recipe (dict): Recipe to execute. Should include, at the minimum, "input_objects", "input_systems",
                "output_objects", and "output_systems" keys
            container_info (dict): Output of @self._compute_container_info(); container-specific information which may
                be relevant for computing whether recipe is executable.

        Returns:
            TransitionResults: Results of the executed recipe transition
        """
        objs_to_add, objs_to_remove = [], []

        in_volume = container_info["in_volume"]
        if self.is_multi_instance:
            execution_info = container_info["execution_info"]

        # Compute total volume of all contained items
        volume = 0

        if not self.is_multi_instance:
            # Remove either all systems or only the ones specified in the input systems of the recipe
            contained_particles_state = container.states[ContainedParticles]
            for system_name, system in self.scene.active_systems.items():
                if not self.ignore_nonrecipe_systems or system_name in recipe["input_systems"]:
                    if container.states[Contains].get_value(system):
                        if self.scene.is_physical_particle_system(system_name):
                            volume += (
                                contained_particles_state.get_value(system).n_in_volume
                                * math.pi
                                * (system.particle_radius**3)
                                * 4
                                / 3
                            )
                        container.states[Contains].set_value(system, False)
        else:
            # Remove the particles that are involved in this execution
            for system_name, particle_idxs in execution_info["relevant_systems"].items():
                system = self.scene.get_system(system_name)
                volume += len(particle_idxs) * math.pi * (system.particle_radius**3) * 4 / 3
                system.remove_particles(idxs=th.tensor(list(particle_idxs)))

        if not self.is_multi_instance:
            # Remove either all objects or only the ones specified in the input objects of the recipe
            object_mask = th.clone(in_volume)
            if self.ignore_nonrecipe_objects:
                object_category_mask = th.zeros_like(object_mask, dtype=bool)
                for obj_category in recipe["input_objects"].keys():
                    object_category_mask[self._category_idxs[obj_category]] = True
                object_mask &= object_category_mask
            mask_indices = object_mask.nonzero().flatten().tolist()
            objs_to_remove.extend([self._objects[i] for i in mask_indices])
        else:
            # Remove the objects that are involved in this execution
            for obj_category, objs in execution_info["relevant_objects"].items():
                objs_to_remove.extend(objs)

        volume += sum(obj.volume for obj in objs_to_remove)

        # Compute full AABB containing all objects in objs_to_remove
        full_aabb = None
        if objs_to_remove:
            aabbs = [obj.aabb for obj in objs_to_remove]
            full_aabb = (
                th.min(th.stack([aabb[0] for aabb in aabbs]), dim=0).values,
                th.max(th.stack([aabb[1] for aabb in aabbs]), dim=0).values,
            )

        # Define callback for spawning new objects inside container
        def _spawn_object_in_container(obj):
            # For simplicity sake, sample only OnTop
            # TODO: Can we sample inside intelligently?
            state = OnTop
            # TODO: What to do if setter fails?
            if not obj.states[state].set_value(container, True):
                log.warning(
                    f"Failed to spawn object {obj.name} in container {container.name}! Directly placing on top instead."
                )
                pos = th.tensor(container.aabb_center, dtype=th.float32) + th.tensor(
                    [0, 0, container.aabb_extent[2] / 2.0 + obj.aabb_extent[2] / 2.0], dtype=th.float32
                )
                obj.set_bbox_center_position_orientation(position=pos)

        # Spawn in new objects
        for category, n_instances in recipe["output_objects"].items():
            # Multiply by number of instances of execution if this is a multi-instance recipe
            if self.is_multi_instance:
                n_instances *= execution_info["num_instances"]

            output_states = dict()
            for state_type, state_value in recipe["output_states"][category]["unary"]:
                output_states[state_type] = (state_value,)
            for state_type, system_name, state_value in recipe["output_states"][category]["binary_system"]:
                output_states[state_type] = (self.scene.get_system(system_name), state_value)
            n_category_objs = len(container.scene.object_registry("category", category, []))
            models = get_all_object_category_models(category=category)

            bounding_box_size = None
            if full_aabb is not None and n_instances > 0:
                # Compute the 3D bounding box size for the new objects
                full_aabb_extent = full_aabb[1] - full_aabb[0]  # Get the full AABB extent

                # Determine how to divide the space in the x-y plane
                # Calculate factors for grid layout
                n_rows = int(math.ceil(n_instances**0.5))
                n_columns = math.ceil(n_instances / n_rows)

                # Calculate the size for each instance in the x-y plane
                bounding_box_size = th.tensor(
                    [full_aabb_extent[0] / n_columns, full_aabb_extent[1] / n_rows, full_aabb_extent[2]],
                    dtype=th.float32,
                )

            for i in range(n_instances):
                obj = DatasetObject(
                    name=f"{category}_{n_category_objs + i}",
                    category=category,
                    model=random.choice(models),
                    bounding_box=bounding_box_size,
                )
                new_obj_attrs = ObjectAttrs(
                    obj=obj,
                    callback=_spawn_object_in_container,
                    states=output_states,
                    pos=th.ones(3) * (100.0 + i),
                )
                objs_to_add.append(new_obj_attrs)

        # Spawn in new fluid
        if len(recipe["output_systems"]) > 0:
            # Only one system is allowed to be spawned
            assert len(recipe["output_systems"]) == 1, "Only a single output system can be spawned for a given recipe!"
            out_system = self.scene.get_system(recipe["output_systems"][0])
            out_system.generate_particles_from_link(
                obj=container,
                link=contained_particles_state.link,
                # When ignore_nonrecipe_objects is True, we don't necessarily remove all objects in the container.
                # Therefore, we need to check for contact when generating output systems.
                check_contact=self.ignore_nonrecipe_objects,
                max_samples=math.ceil(volume / (math.pi * (out_system.particle_radius**3) * 4 / 3)),
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
    Transition rule to apply to "cook" physical particles.
    It comes with two forms of recipes:
    1. xyz -> cooked__xyz, e.g. diced__chicken -> cooked__diced__chicken
    2. xyz + cooked__water -> cooked__xyz, e.g. rice + cooked__water -> cooked__rice
    During execution, we replace the input particles (xyz) with the output particles (cooked__xyz), and remove the
    cooked__water if it was used as an input.
    """

    @classmethod
    def add_recipe(
        cls,
        name,
        input_objects,
        input_systems,
        output_objects,
        output_systems,
        **kwargs,
    ):
        """
        Adds a recipe to this recipe rule to check against. This defines a valid mapping of inputs that will transform
        into the outputs

        Args:
            name (str): Name of the recipe
            input_objects (dict): Maps object categories to number of instances required for the recipe
            input_systems (list): List of system names required for the recipe
            output_objects (dict): Maps object categories to number of instances to be spawned in the container when the recipe executes
            output_systems (list): List of system names to be spawned in the container when the recipe executes. Currently the length is 1.
        """
        assert len(input_objects) == 0, f"No input objects can be specified for {cls.__name__}, recipe: {name}!"
        assert len(output_objects) == 0, f"No output objects can be specified for {cls.__name__}, recipe: {name}!"

        assert (
            len(input_systems) == 1 or len(input_systems) == 2
        ), f"Only one or two input systems can be specified for {cls.__name__}, recipe: {name}!"
        if len(input_systems) == 2:
            assert (
                input_systems[1] == "cooked__water"
            ), f"Second input system must be cooked__water for {cls.__name__}, recipe: {name}!"
        assert (
            len(output_systems) == 1
        ), f"Exactly one output system needs to be specified for {cls.__name__}, recipe: {name}!"

        super().add_recipe(
            name=name,
            input_objects=input_objects,
            input_systems=input_systems,
            output_objects=output_objects,
            output_systems=output_systems,
            **kwargs,
        )

    @classproperty
    def candidate_filters(cls):
        # Modify the container filter to include the heatable ability as well
        candidate_filters = super().candidate_filters
        candidate_filters["container"] = AndFilter(
            filters=[candidate_filters["container"], AbilityFilter(ability="heatable")]
        )
        return candidate_filters

    def _generate_conditions(self):
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

    def _execute_recipe(self, container, recipe, container_info):
        system = self.scene.get_system(recipe["input_systems"][0])
        contained_particles_state = container.states[ContainedParticles].get_value(system)
        in_volume_idx = th.where(contained_particles_state.in_volume)[0]
        assert len(in_volume_idx) > 0, "No particles found in the container when executing recipe!"

        # Remove uncooked particles
        system.remove_particles(idxs=in_volume_idx)

        # Generate cooked particles
        cooked_system = self.scene.get_system(recipe["output_systems"][0])
        pre_gen_count = cooked_system.n_particles
        particle_positions = contained_particles_state.positions[in_volume_idx]
        cooked_system.generate_particles(positions=particle_positions)

        # Stabilize generated particles
        if isinstance(cooked_system, MacroPhysicalParticleSystem):
            new_particle_indices = th.arange(pre_gen_count, cooked_system.n_particles)
            lin_vel, ang_vel = cooked_system.get_particles_velocities()
            lin_vel[new_particle_indices] = 0
            ang_vel[new_particle_indices] = 0
            cooked_system.set_particles_velocities(lin_vels=lin_vel, ang_vels=ang_vel)

        # Remove water if the cooking requires water
        if len(recipe["input_systems"]) > 1:
            cooked_water_system = self.scene.get_system(recipe["input_systems"][1])
            container.states[Contains].set_value(cooked_water_system, False)

        return TransitionResults(add=[], remove=[])


class ToggleableMachineRule(RecipeRule):
    """
    Transition mixing rule that leverages a single toggleable machine (e.g. electric mixer, coffee machine, blender),
    which require toggledOn in order to trigger the recipe event.
    It comes with two forms of recipes:
    1. output is a single object, e.g. flour + butter + sugar -> dough, machine is electric mixer
    2. output is a system, e.g. strawberry + milk -> smoothie, machine is blender
    """

    @classmethod
    def add_recipe(
        cls,
        name,
        input_objects,
        input_systems,
        output_objects,
        output_systems,
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
            input_objects (dict): Maps object categories to number of instances required for the recipe
            input_systems (list): List of system names required for the recipe
            output_objects (dict): Maps object categories to number of instances to be spawned in the container when the recipe executes
            output_systems (list): List of system names to be spawned in the container when the recipe executes. Currently the length is 1.
            input_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system", "binary_object"] to a list of states that must be satisfied for the recipe to execute
            output_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system"] to a list of states that should be set after the output objects are spawned
            fillable_categories (None or set of str): If specified, set of fillable categories which are allowed
                for this recipe. If None, any fillable is allowed
        """
        if len(output_objects) > 0:
            assert (
                len(output_objects) == 1
            ), f"Only one category of output object can be specified for {cls.__name__}, recipe: {name}!"
            assert (
                output_objects[list(output_objects.keys())[0]] == 1
            ), f"Only one instance of output object can be specified for {cls.__name__}, recipe: {name}!"

        super().add_recipe(
            name=name,
            input_objects=input_objects,
            input_systems=input_systems,
            output_objects=output_objects,
            output_systems=output_systems,
            input_states=input_states,
            output_states=output_states,
            fillable_categories=fillable_categories,
            **kwargs,
        )

    @classproperty
    def candidate_filters(cls):
        # Modify the container filter to include toggleable ability as well
        candidate_filters = super().candidate_filters
        candidate_filters["container"] = AndFilter(
            filters=[
                candidate_filters["container"],
                AbilityFilter(ability="toggleable"),
                # Exclude washer and clothes dryer because they are handled by WasherRule and DryerRule
                NotFilter(CategoryFilter("washer")),
                NotFilter(CategoryFilter("clothes_dryer")),
                NotFilter(CategoryFilter("hot_tub")),
                NotFilter(CategoryFilter("oven")),
                NotFilter(CategoryFilter("microwave")),
            ]
        )
        return candidate_filters

    def _generate_conditions(self):
        # Container must be toggledOn, and should only be triggered once
        return [
            ChangeConditionWrapper(
                condition=StateCondition(filter_name="container", state=ToggledOn, val=True, op=operator.eq)
            )
        ]

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
    and a container in order to trigger the recipe event.
    Example: water + lemon_juice + sugar -> lemonade, mixing tool is spoon
    """

    @classmethod
    def add_recipe(
        cls,
        name,
        input_objects,
        input_systems,
        output_objects,
        output_systems,
        input_states=None,
        output_states=None,
        **kwargs,
    ):
        """
        Adds a recipe to this recipe rule to check against. This defines a valid mapping of inputs that will transform
        into the outputs

        Args:
            name (str): Name of the recipe
            input_objects (dict): Maps object categories to number of instances required for the recipe
            input_systems (list): List of system names required for the recipe
            output_objects (dict): Maps object categories to number of instances to be spawned in the container when the recipe executes
            output_systems (list): List of system names to be spawned in the container when the recipe executes. Currently the length is 1.
            input_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system", "binary_object"] to a list of states that must be satisfied for the recipe to execute
            output_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system"] to a list of states that should be set after the output objects are spawned
        """
        assert len(output_objects) == 0, f"No output objects can be specified for {cls.__name__}, recipe: {name}!"
        assert len(input_systems) > 0, f"Some input systems need to be specified for {cls.__name__}, recipe: {name}!"
        assert (
            len(output_systems) == 1
        ), f"Exactly one output system needs to be specified for {cls.__name__}, recipe: {name}!"

        super().add_recipe(
            name=name,
            input_objects=input_objects,
            input_systems=input_systems,
            output_objects=output_objects,
            output_systems=output_systems,
            input_states=input_states,
            output_states=output_states,
            **kwargs,
        )

    @classproperty
    def candidate_filters(cls):
        # Add mixing tool filter as well
        candidate_filters = super().candidate_filters
        candidate_filters["mixingTool"] = AbilityFilter(ability="mixingTool")
        return candidate_filters

    def _generate_conditions(self):
        # Mixing tool must be touching the container, and should only be triggered once
        return [
            ChangeConditionWrapper(
                condition=TouchingAnyCondition(filter_1_name="container", filter_2_name="mixingTool")
            )
        ]

    @classproperty
    def relax_recipe_systems(cls):
        return False

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
    Transition mixing rule that approximates cooking recipes via a container and heatsource.
    It is subclassed by CookingObjectRule and CookingSystemRule.
    """

    def __init__(self, scene):
        super().__init__(scene=scene)

        # Counter that increments monotonically
        self.counter = 0

        # Maps recipe name to current number of consecutive heating steps
        self._heat_steps = None

        # Maps recipe name to the last timestep that it was active
        self._last_heat_timestep = None

    def refresh(self, object_candidates):
        # Run super first
        super().refresh(object_candidates=object_candidates)

        # Iterate through all (updated) active recipes and store in internal variables if not already recorded
        self._heat_steps = dict() if self._heat_steps is None else self._heat_steps
        self._last_heat_timestep = dict() if self._last_heat_timestep is None else self._last_heat_timestep

        for name in self._active_recipes.keys():
            if name not in self._heat_steps:
                self._heat_steps[name] = 0
                self._last_heat_timestep[name] = -1

    def _validate_recipe_fillables_exist(self, recipe):
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
            if len(self.scene.object_registry("category", category, default_val=set())) > 0:
                return True

        # None found, return False
        return False

    def _validate_recipe_heatsources_exist(self, recipe):
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
            if len(self.scene.object_registry("category", category, default_val=set())) > 0:
                return True

        # None found, return False
        return False

    def _validate_recipe_heatsource_is_valid(self, recipe, heatsource_categories):
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
        return (required_heatsource_categories is None and len(heatsource_categories) > 0) or len(
            required_heatsource_categories.intersection(heatsource_categories)
        ) > 0

    def _compute_container_info(self, object_candidates, container, global_info):
        # Run super first
        info = super()._compute_container_info(
            object_candidates=object_candidates, container=container, global_info=global_info
        )

        # Compute whether each heatsource is affecting the container
        info["heatsource_categories"] = set(
            obj.category
            for obj in object_candidates["heatSource"]
            if obj.states[HeatSourceOrSink].affects_obj(container)
        )

        return info

    def _is_recipe_active(self, recipe):
        # Check for heatsource categories first
        if not self._validate_recipe_heatsources_exist(recipe=recipe):
            return False

        # Otherwise, run super normally
        return super()._is_recipe_active(recipe=recipe)

    def _is_recipe_executable(self, recipe, container, global_info, container_info):
        # Check for heatsource compatibility first
        if not self._validate_recipe_heatsource_is_valid(
            recipe=recipe, heatsource_categories=container_info["heatsource_categories"]
        ):
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
            self._heat_steps[name] = (
                self._heat_steps[name] + 1 if self._last_heat_timestep[name] == self.counter - 1 else 1
            )
            self._last_heat_timestep[name] = self.counter

            # If valid number of timesteps met, recipe is indeed executable
            executable = self._heat_steps[name] >= recipe["timesteps"]

        return executable

    @classmethod
    def add_recipe(
        cls,
        name,
        input_objects,
        input_systems,
        output_objects,
        output_systems,
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
            input_objects (dict): Maps object categories to number of instances required for the recipe
            input_systems (list): List of system names required for the recipe
            output_objects (dict): Maps object categories to number of instances to be spawned in the container when the recipe executes
            output_systems (list): List of system names to be spawned in the container when the recipe executes. Currently the length is 1.
            input_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system", "binary_object"] to a list of states that must be satisfied for the recipe to execute
            output_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system"] to a list of states that should be set after the output objects are spawned
            fillable_categories (None or set of str): If specified, set of fillable categories which are allowed
                for this recipe. If None, any fillable is allowed
            heatsource_categories (None or set of str): If specified, set of heatsource categories which are allowed
                for this recipe. If None, any heatsource is allowed
            timesteps (None or int): Number of subsequent heating steps required for the recipe to execute. If None,
                it will be set to be 1, i.e.: instantaneous execution
        """
        super().add_recipe(
            name=name,
            input_objects=input_objects,
            input_systems=input_systems,
            output_objects=output_objects,
            output_systems=output_systems,
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

    def _generate_conditions(self):
        # Define a class to increment this class's internal time counter every time it is triggered
        class TimeIncrementCondition(RuleCondition):
            def __init__(self, rule):
                self.rule = rule

            def __call__(self, object_candidates):
                # This is just a pass-through, but also increment the time
                self.rule.counter += 1
                return True

            def modifies_filter_names(self):
                return set()

        # Any heatsource must be active
        return [
            TimeIncrementCondition(rule=self),
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
    """
    Cooking rule when output is objects (e.g. one dough can produce many bagels as output).
    Example: bagel_dough + egg + sesame_seed -> bagel, heat source is oven, fillable is baking_sheet.
    This is the only rule where is_multi_instance is True, where multiple copies of the recipe can be executed.
    """

    @classmethod
    def add_recipe(
        cls,
        name,
        input_objects,
        input_systems,
        output_objects,
        output_systems,
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
            input_objects (dict): Maps object categories to number of instances required for the recipe
            input_systems (list): List of system names required for the recipe
            output_objects (dict): Maps object categories to number of instances to be spawned in the container when the recipe executes
            output_systems (list): List of system names to be spawned in the container when the recipe executes. Currently the length is 1.
            input_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system", "binary_object"] to a list of states that must be satisfied for the recipe to execute
            output_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system"] to a list of states that should be set after the output objects are spawned
            fillable_categories (None or set of str): If specified, set of fillable categories which are allowed
                for this recipe. If None, any fillable is allowed
            heatsource_categories (None or set of str): If specified, set of heatsource categories which are allowed
                for this recipe. If None, any heatsource is allowed
            timesteps (None or int): Number of subsequent heating steps required for the recipe to execute. If None,
                it will be set to be 1, i.e.: instantaneous execution
        """
        assert len(output_systems) == 0, f"No output systems can be specified for {cls.__name__}, recipe: {name}!"
        super().add_recipe(
            name=name,
            input_objects=input_objects,
            input_systems=input_systems,
            output_objects=output_objects,
            output_systems=output_systems,
            input_states=input_states,
            output_states=output_states,
            fillable_categories=fillable_categories,
            heatsource_categories=heatsource_categories,
            timesteps=timesteps,
        )

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
    """
    Cooking rule when output is a system.
    Example: beef + tomato + chicken_stock -> stew, heat source is stove, fillable is stockpot.
    """

    @classmethod
    def add_recipe(
        cls,
        name,
        input_objects,
        input_systems,
        output_objects,
        output_systems,
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
            input_objects (dict): Maps object categories to number of instances required for the recipe
            input_systems (list): List of system names required for the recipe
            output_objects (dict): Maps object categories to number of instances to be spawned in the container when the recipe executes
            output_systems (list): List of system names to be spawned in the container when the recipe executes. Currently the length is 1.
            input_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system", "binary_object"] to a list of states that must be satisfied for the recipe to execute
            output_states (None or defaultdict(lambda: defaultdict(list))): Maps object categories to
                ["unary", "bianry_system"] to a list of states that should be set after the output objects are spawned
            fillable_categories (None or set of str): If specified, set of fillable categories which are allowed
                for this recipe. If None, any fillable is allowed
            heatsource_categories (None or set of str): If specified, set of heatsource categories which are allowed
                for this recipe. If None, any heatsource is allowed
            timesteps (None or int): Number of subsequent heating steps required for the recipe to execute. If None,
                it will be set to be 1, i.e.: instantaneous execution
        """
        assert len(output_objects) == 0, f"No output objects can be specified for {cls.__name__}, recipe: {name}!"
        super().add_recipe(
            name=name,
            input_objects=input_objects,
            input_systems=input_systems,
            output_objects=output_objects,
            output_systems=output_systems,
            input_states=input_states,
            output_states=output_states,
            fillable_categories=fillable_categories,
            heatsource_categories=heatsource_categories,
            timesteps=timesteps,
        )

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
        recipe_fpath = os.path.join(
            os.path.dirname(bddl.__file__), "generated_data", "transition_map", "tm_jsons", json_file
        )
        if not os.path.exists(recipe_fpath):
            log.warning(f"Cannot find recipe file at {recipe_fpath}. Skipping importing recipes.")

        with open(recipe_fpath, "r") as f:
            rule_recipes = json.load(f)

        for rule_name in rule_names:
            rule = REGISTERED_RULES[rule_name]
            if rule == WasherRule:
                rule.register_cleaning_conditions(translate_bddl_washer_rule_to_og_washer_rule(rule_recipes))
            elif issubclass(rule, RecipeRule):
                log.info(f"Adding recipes of rule {rule_name}...")
                for recipe in rule_recipes:
                    if "rule_name" in recipe:
                        recipe["name"] = recipe.pop("rule_name")
                    if "container" in recipe:
                        recipe["fillable_synsets"] = set(recipe.pop("container").keys())
                    if "heat_source" in recipe:
                        recipe["heatsource_synsets"] = set(recipe.pop("heat_source").keys())
                    if "machine" in recipe:
                        recipe["fillable_synsets"] = set(recipe.pop("machine").keys())

                    # Route the recipe to the correct rule: CookingObjectRule or CookingSystemRule
                    satisfied = True
                    og_recipe = translate_bddl_recipe_to_og_recipe(**recipe)
                    has_output_system = len(og_recipe["output_systems"]) > 0
                    if (rule == CookingObjectRule and has_output_system) or (
                        rule == CookingSystemRule and not has_output_system
                    ):
                        satisfied = False
                    if satisfied:
                        rule.add_recipe(**og_recipe)
                log.info(f"All recipes of rule {rule_name} imported successfully.")


import_recipes()
