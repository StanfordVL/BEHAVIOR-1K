import operator
from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict
import numpy as np
from copy import copy
import itertools
import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.systems import get_system, is_system_active, PhysicalParticleSystem, REGISTERED_SYSTEMS
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.object_states import *
from omnigibson.utils.python_utils import Registerable, classproperty, subclass_factory
from omnigibson.utils.registry_utils import Registry
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import disclaimer, create_module_logger
from omnigibson.utils.usd_utils import RigidContactAPI

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Default melting temperature
m.MELTING_TEMPERATURE = 100.0

# Where to place objects far out of the scene
m.OBJECT_GRAVEYARD_POS = (100.0, 100.0, 100.0)

# Default "trash" system if an invalid mixing rule transition occurs
m.DEFAULT_GARBAGE_SYSTEM = "sludge"

# Tuple of attributes of objects created in transitions.
# `states` field is dict mapping object state class to arguments to pass to setter for that class
_attrs_fields = ["category", "model", "name", "scale", "obj", "pos", "orn", "bb_pos", "bb_orn", "states"]
ObjectAttrs = namedtuple(
    "ObjectAttrs", _attrs_fields, defaults=(None,) * len(_attrs_fields))

# Tuple of lists of objects to be added or removed returned from transitions.
TransitionResults = namedtuple(
    "TransitionResults", ["add", "remove"], defaults=([], []))

# Global dicts that will contain mappings
REGISTERED_RULES = dict()


class TransitionRuleAPI:
    """
    Monolithic class containing methods to check and execute arbitrary discrete state transitions within the simulator
    """
    # Maps BaseTransitionRule instances to list of valid object candidates combinations to check for transitions
    ACTIVE_RULES = dict()

    # Maps BaseObject instances to dictionary mapping object states to arguments to set for that state when the
    # object is initialized
    _INIT_STATES = dict()

    @classmethod
    def get_rule_candidates(cls, rule, objects):
        """
        Computes valid input object candidates for transition rule @rule, if any exist

        Args:
            rule (BaseTransitionRule): Transition rule whose candidates should be computed
            objects (list of BaseObject): List of objects that will be used to compute object candidates

        Returns:
            None or dict: None if no valid candidates are found, otherwise mapping, with keys "individual", and "group",
                where each respective key maps to a nested dictionary mapping filter name to list of object(s)
                satisfying that filter
        """
        individual_candidates, group_candidates = rule.get_object_candidates(objects=objects)
        n_individual_filters_satisfied, n_group_filters_satisfied = len(individual_candidates), len(group_candidates)
        # Skip if the rule requirements are not met
        if (rule.requires_individual_filters and n_individual_filters_satisfied != len(rule.individual_filters)) \
                or (rule.requires_group_filters and n_group_filters_satisfied != len(rule.group_filters)):
            return

        return dict(individual=individual_candidates, group=group_candidates)

    @classmethod
    def prune_active_rules(cls, objects):
        """
        Prunes the active transition rules, removing any whose filter requirements are not satisifed by object set
        @objects. Useful when the current object set changes, e.g.: an object is removed from the simulator

        Args:
            objects (list of BaseObject): List of objects that will be used to infer which transition rules are
                currently active
        """
        # Need explicit list to iterate over because refresh_rules mutates the ACTIVE_RULES dict in place
        cls.refresh_rules(rules=list(cls.ACTIVE_RULES.keys()), objects=objects)

    @classmethod
    def refresh_all_rules(cls, objects):
        """
        Refreshes all registered rules given the current set of objects @objects

        Args:
            objects (list of BaseObject): List of objects that will be used to infer which transition rules are
                currently active
        """
        global RULES_REGISTRY

        # Clear all active rules
        cls.ACTIVE_RULES = dict()

        # Refresh all registered rules
        cls.refresh_rules(rules=RULES_REGISTRY.objects, objects=objects)

    @classmethod
    def refresh_rules(cls, rules, objects):
        """
        Refreshes the specified transition rules @rules based on current set of objects @objects. This will prune
        any pre-existing rules in cls.ACTIVE_RULES if no valid candidates are found, or add / update the entry if
        valid candidates are found

        Args:
            rules (list of BaseTransitionRule): List of transition rules whose candidate lists should be refreshed
            objects (list of BaseObject): List of objects that will be used to infer which transition rules are
                currently active
        """
        for rule in rules:
            # Check if rule is still valid, if so, update its entry
            candidates = cls.get_rule_candidates(rule=rule, objects=objects)

            # Update candidates if valid, otherwise pop the entry if it exists in cls.ACTIVE_RULES
            if candidates is not None:
                # We have a valid rule which should be active, so grab and initialize all of its conditions
                # NOTE: The rule may ALREADY exist in ACTIVE_RULES, but we still need to update its candidates because
                # the relevant candidate set / information for the rule + its conditions may have changed given the
                # new set of objects
                rule.refresh()
                conditions = []
                combined_candidates = dict(**candidates["individual"], **candidates["group"])
                for condition in rule.conditions:
                    condition.refresh(combined_candidates)
                    conditions.append(condition)
                cls.ACTIVE_RULES[rule] = {
                    "conditions": conditions,
                    "candidates": candidates,
                }
            elif rule in cls.ACTIVE_RULES:
                cls.ACTIVE_RULES.pop(rule)

    @classmethod
    def step(cls):
        """
        Steps all active transition rules, checking if any are satisfied, and if so, executing their transitions
        """
        # First apply any transition object init states from before, and then clear the dictionary
        for obj, states_info in cls._INIT_STATES.items():
            for state, args in states_info.items():
                obj.states[state].set_value(*args)
        cls._INIT_STATES = dict()

        # Iterate over all active rules and process the rule for every valid object candidate combination
        # Cast to list before iterating since ACTIVE_RULES may get updated mid-iteration
        added_obj_attrs = []
        removed_objs = []
        for rule in tuple(cls.ACTIVE_RULES.keys()):
            # Compile candidates (shallow) copy that will be mutated in place by conditions
            conditions, candidates = cls.ACTIVE_RULES[rule]["conditions"], cls.ACTIVE_RULES[rule]["candidates"]
            combined_candidates = dict(**candidates["individual"], **candidates["group"])
            should_transition = True
            for condition in conditions:
                if not condition(object_candidates=combined_candidates):
                    # Condition was not met, so break early
                    should_transition = False
                    break
            if should_transition:
                # Compile all valid transitions, and take the transition
                pruned_individual_candidates = {k: combined_candidates[k] for k in candidates["individual"].keys()}
                group_objects = {k: combined_candidates[k] for k in candidates["group"].keys()}
                for obj_tuple in itertools.product(*list(pruned_individual_candidates.values())):
                    individual_objects = {fname: obj for fname, obj in zip(pruned_individual_candidates.keys(), obj_tuple)}
                    # Take the transition
                    output = rule.transition(individual_objects=individual_objects, group_objects=group_objects)
                    # Store objects to be added / removed
                    added_obj_attrs += output.add
                    removed_objs += output.remove

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
                # TODO: Ideally we want to remove objects, but because of Omniverse's bug on GPU physics, we simply move
                # the objects into a graveyard for now
                removed_obj.set_position(np.array(m.OBJECT_GRAVEYARD_POS) + np.ones(3) * i)
                # og.sim.remove_object(removed_obj)

        # Then add new objects
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
            if added_obj_attr.states is not None:
                cls._INIT_STATES[new_obj] = added_obj_attr.states

    @classmethod
    def clear(cls):
        """
        Clears any internal state when the simulator is restarted (e.g.: when a new stage is opened)
        """
        global RULES_REGISTRY

        # Clear internal dictionaries
        cls.ACTIVE_RULES = dict()
        cls._INIT_STATES = dict()


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
        Refreshes any internal state for this rule condition, given set of input object candidates
        @individual_candidates and @group_candidates

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
        # List of rigid body idxs in the global contact matrix corresponding to filter 2
        self._filter_2_idxs = None

    def refresh(self, object_candidates):
        # Register idx mappings
        self._filter_1_idxs = {obj: [RigidContactAPI.get_body_idx(link.prim_path) for link in obj.links.values()]
                            for obj in object_candidates[self._filter_1_name]}
        self._filter_2_idxs = [RigidContactAPI.get_body_idx(link.prim_path)
                               for obj in object_candidates[self._filter_2_name] for link in obj.links.values()]

    def __call__(self, object_candidates):
        # Get all impulses
        impulses = RigidContactAPI.get_all_impulses()

        # Keep any object that has non-zero impulses between itself and any of the @filter_2_name's objects
        objs = []
        for obj in object_candidates[self._filter_1_name]:
            if np.any(impulses[self._filter_1_idxs[obj]][:, self._filter_2_idxs]):
                objs.append(obj)

        # Update candidates
        object_candidates[self._filter_1_name] = objs

        # If objs is empty, return False, otherwise, True
        return len(objs) > 0


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


class ChangeCondition(RuleCondition):
    """
    Rule condition that checks all objects from @filter_name whether the sum of the prior conditions has changed
    from the previous timestep
    """
    def __init__(
            self,
            filter_name,
    ):
        """
        Args:
            filter_name (str): Name of the filter whose object candidates will be pruned based on whether or not any
                prior conditions have changed or not
        """
        self._filter_name = filter_name
        self._last_valid_candidates = None

    def refresh(self, object_candidates):
        # Initialize last valid candidates
        self._last_valid_candidates = set(object_candidates[self._filter_name])

    def __call__(self, object_candidates):
        # Iterate over all current candidates -- if there's a mismatch in last valid candidates and current, then
        # we store it, otherwise, we don't
        objs = [obj for obj in object_candidates[self._filter_name] if obj not in self._last_valid_candidates]
        object_candidates[self._filter_name] = objs
        self._last_valid_candidates = set(objs)

        # Valid if any object conditions have changed
        return len(objs) > 0


class OrCondition(RuleCondition):
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


class BaseTransitionRule(Registerable):
    """
    Defines a set of categories of objects and how to transition their states.
    """

    def __init_subclass__(cls, **kwargs):
        # Call super first
        super().__init_subclass__(**kwargs)

        # Register this system, and
        # make sure at least one set of filters is specified -- in general, there should never be a rule
        # where no filter is specified
        # Only run this check for actual rules that are being registered
        if cls.__name__ not in cls._do_not_register_classes:
            global RULES_REGISTRY
            RULES_REGISTRY.add(obj=cls)
            assert len(cls.individual_filters) > 0 or len(cls.group_filters) > 0, \
                "At least one of individual_filters or group_filters must be specified!"

    @classproperty
    def required_systems(cls):
        """
        Particle systems that this transition rule cares about. Should be specified by subclass.

        Returns:
            list of str: Particle system names which must be active in order for the transition rule to occur
        """
        return []

    @classproperty
    def individual_filters(cls):
        """
        Individual object filters that this transition rule cares about.
        For each name, filter key-value pair, the global transition rule step will produce tuples of valid
        filtered objects such that the cross product over all individual filter outputs occur.
        For example, if the individual filters are:

            {"apple": CategoryFilter("apple"), "knife": CategoryFilter("knife")},

        the transition rule step will produce all 2-tuples of valid (apple, knife) combinations:

            {"apple": apple_i, "knife": knife_j}

        based on the current instances of each object type in the scene and pass them to @self.condition as the
        @individual_objects entry.
        If empty, then no filter will be applied

        Returns:
            dict: Maps filter name to filter for inferring valid individual object candidates for this transition rule
        """
        # Default is empty dictionary
        return dict()

    @classproperty
    def group_filters(cls):
        """
        Group object filters that this filter cares about. For each name, filter
        key-value pair, the global transition rule step will produce a single dictionary of valid filtered
        objects.
        For example, if the group filters are:

            {"apple": CategoryFilter("apple"), "knife": CategoryFilter("knife")},

        the transition rule step will produce the following dictionary:

            {"apple": [apple0, apple1, ...], "knife": [knife0, knife1, ...]}

        based on the current instances of each object type in the scene and pass them to @self.condition
        as the @group_objects entry.
        If empty, then no filter will be applied

        Returns:
            dict: Maps filter name to filter for inferring valid group object candidates for this transition rule
        """
        # Default is empty dictionary
        return dict()

    @classproperty
    def conditions(cls):
        """
        Rule conditions for this transition rule. These conditions are used to prune object candidates at runtime,
        to determine whether a transition rule should occur at the given timestep

        Returns:
            list of RuleCondition: Condition(s) to enforce to determine whether a transition rule should occur
        """
        raise NotImplementedError

    @classproperty
    def requires_individual_filters(cls):
        """
        Returns:
            bool: Whether this transition rule requires any specific filters
        """
        return len(cls.individual_filters) > 0

    @classproperty
    def requires_group_filters(cls):
        """
        Returns:
            bool: Whether this transition rule requires any group filters
        """
        return len(cls.group_filters) > 0

    @classmethod
    def get_object_candidates(cls, objects):
        """
        Given the set of objects @objects, compute the valid object candidate combinations that may be valid for
        this TransitionRule

        Args:
            objects (list of BaseObject): Objects to filter for valid transition rule candidates

        Returns:
            2-tuple:
                - defaultdict: Maps individual filter to valid object(s) that satisfy that filter
                - defaultdict: Maps group filter to valid object(s) that satisfy that group filter
        """
        # Iterate over all objects and add to dictionary if valid
        individual_obj_dict = defaultdict(list)
        group_obj_dict = defaultdict(list)

        # Only compile candidates if all active system requirements are met
        if np.all([is_system_active(system_name=name) for name in cls.required_systems]):
            individual_filters, group_filters = cls.individual_filters, cls.group_filters
            for obj in objects:
                for fname, f in individual_filters.items():
                    if f(obj):
                        individual_obj_dict[fname].append(obj)
                for fname, f in group_filters.items():
                    if f(obj):
                        group_obj_dict[fname].append(obj)

        return individual_obj_dict, group_obj_dict

    @classmethod
    def refresh(self):
        """
        Refresh any internal state for this rule
        """
        # No-op by default
        pass

    @classmethod
    def transition(cls, individual_objects, group_objects):
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
        raise NotImplementedError()

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
    def individual_filters(cls):
        # TODO: Remove hacky filter once half category is updated properly
        return {"sliceable": AndFilter(filters=[AbilityFilter("sliceable"), NotFilter(NameFilter(name="half"))])}

    @classproperty
    def group_filters(cls):
        return {"slicer": AbilityFilter("slicer")}

    @classproperty
    def conditions(cls):
        # sliceables should be touching any slicer
        return [TouchingAnyCondition(filter_1_name="sliceable", filter_2_name="slicer")]

    @classmethod
    def transition(cls, individual_objects, group_objects):
        sliceable_obj = individual_objects["sliceable"]
        # Object parts offset annotation are w.r.t the base link of the whole object.
        pos, orn = sliceable_obj.get_position_orientation()

        t_results = TransitionResults()

        # Load object parts.
        if sliceable_obj.bddl_object_scope is not None:
            sliced_obj_id = int(sliceable_obj.bddl_object_scope.split("_")[-1])
            sliced_obj_scope_prefix = "_".join(sliceable_obj.bddl_object_scope.split("_")[:-1])
        for i, part in enumerate(sliceable_obj.metadata["object_parts"].values()):
            # List of dicts gets replaced by {'0':dict, '1':dict, ...}

            # Get bounding box info
            part_bb_pos = np.array(part["bb_pos"])
            part_bb_orn = np.array(part["bb_orn"])

            # Determine the relative scale to apply to the object part from the original object
            # Note that proper (rotated) scaling can only be applied when the relative orientation of
            # the object part is a multiple of 90 degrees wrt the parent object, so we assert that here
            # Check by making sure the quaternion is some permutation of +/- (1, 0, 0, 0),
            # +/- (0.707, 0.707, 0, 0), or +/- (0.5, 0.5, 0.5, 0.5)
            # Because orientations are all normalized (same L2-norm), every orientation should have a unique L1-norm
            # So we check the L1-norm of the absolute value of the orientation as a proxy for verifying these values
            assert np.any(np.isclose(np.abs(part_bb_orn).sum(), np.array([1.0, 1.414, 2.0]), atol=1e-3)), \
                "Sliceable objects should only have relative object part orientations that are factors of 90 degrees!"

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
                bddl_object_scope=None if sliceable_obj.bddl_object_scope is None else f"half_{sliced_obj_scope_prefix}_{2 * sliced_obj_id - i}",
            )

            # Add the new object to the results.
            new_obj_attrs = ObjectAttrs(
                obj=part_obj,
                bb_pos=part_bb_pos,
                bb_orn=part_bb_orn,
            )
            t_results.add.append(new_obj_attrs)

        # Delete original object from stage.
        t_results.remove.append(sliceable_obj)

        return t_results


class DicingRule(BaseTransitionRule):
    """
    Transition rule to apply to diceable / slicer object pairs.
    """
    @classproperty
    def individual_filters(cls):
        return {"diceable": AbilityFilter("diceable")}

    @classproperty
    def group_filters(cls):
        return {"slicer": AbilityFilter("slicer")}

    @classproperty
    def conditions(cls):
        # sliceables should be touching any slicer
        return [TouchingAnyCondition(filter_1_name="diceable", filter_2_name="slicer")]

    @classmethod
    def transition(cls, individual_objects, group_objects):
        t_results = TransitionResults()

        diceable_obj = individual_objects["diceable"]
        system = get_system(f"diced_{diceable_obj.category}")
        system.generate_particles_from_link(diceable_obj, diceable_obj.root_link, use_visual_meshes=False)

        # Delete original object from stage.
        t_results.remove.append(diceable_obj)

        return t_results


class CookingPhysicalParticleRule(BaseTransitionRule):
    """
    Transition rule to apply to "cook" physicl particles
    """
    @classproperty
    def group_filters(cls):
        # We want to track all possible fillable heatable objects
        return {"fillable": AndFilter(filters=(AbilityFilter("fillable"), AbilityFilter("heatable")))}

    @classproperty
    def conditions(cls):
        # Only heated objects are valid
        return [StateCondition(filter_name="fillable", state=Heated, val=True, op=operator.eq)]

    @classmethod
    def transition(cls, individual_objects, group_objects):
        t_results = TransitionResults()
        fillable_objs = group_objects["fillable"]

        # Iterate over all active physical particle systems, and for any non-cooked particles inside,
        # convert into cooked particles
        for name, system in PhysicalParticleSystem.get_active_systems().items():
            # Skip any systems that are already cooked
            if "cooked" in name:
                continue

            # TODO: Remove this assert once we have a more standardized method of globally R/W particle positions
            assert len(system.particle_instancers) == 1, \
                f"PhysicalParticleSystem {system.name} should only have one instancer!"

            # Iterate over all fillables -- a given particle should become hot if it is contained in any of the
            # fillable objects
            in_volume = np.zeros(system.n_particles).astype(bool)
            for fillable_obj in fillable_objs:
                in_volume |= fillable_obj.states[ContainedParticles][2]

            # If any are in volume, convert particles
            in_volume_idx = np.where(in_volume)[0]
            if len(in_volume_idx) > 0:
                cooked_system = get_system(f"cooked_{system.name}")
                particle_positions = fillable_obj.states[ContainedParticles][1]
                system.default_particle_instancer.remove_particles(idxs=in_volume_idx)
                cooked_system.default_particle_instancer.add_particles(positions=particle_positions[in_volume_idx])

        return t_results


class MeltingRule(BaseTransitionRule):
    """
    Transition rule to apply to meltable objects to simulate melting
    """
    @classproperty
    def individual_filters(cls):
        # We want to find all meltable objects
        return {"meltable": AbilityFilter("meltable")}

    @classproperty
    def conditions(cls):
        # Only heated objects are valid
        return [StateCondition(filter_name="meltable", state=Temperature, val=m.MELTING_TEMPERATURE, op=operator.ge)]

    @classmethod
    def transition(cls, individual_objects, group_objects):
        t_results = TransitionResults()

        # Convert the meltable object into its melted substance
        meltable_obj = individual_objects["meltable"]
        system = get_system(f"melted_{meltable_obj.category}")
        system.generate_particles_from_link(meltable_obj, meltable_obj.root_link, use_visual_meshes=False)

        # Delete original object from stage.
        t_results.remove.append(meltable_obj)

        return t_results


class MixingRule(BaseTransitionRule):
    """
    Transition rule to apply to mixing objects together
    """
    # Maps recipe name to recipe information
    # NOTE: By defining this class variable directly here, all subclasses SHARE this same dictionary!
    # This is intentional because there may be multiple valid mixing methods,
    # and all of them should share the same global recipe pool
    _RECIPES = dict()

    # Maps active recipe name to recipe information
    _ACTIVE_RECIPES = None

    # Maps object category name to indices in the flattened object array for efficient computation
    _CATEGORY_IDXS = None

    # Flattened array of all simulator objects, sorted by category
    _OBJECTS = None

    @classmethod
    def add_recipe(cls, name, input_objects, input_systems, output_system):
        """
        Adds a recipe to this mixing rule to check against. This defines a valid mapping of inputs that will transform
        into the outputs

        Args:
            name (str): Name of the recipe
            input_objects (dict): Maps object category to number of instances required for the recipe
            input_systems (list of str): Required system names for the recipe
            output_system (str): Output system name that will replace all contained objects if the recipe is executed
        """
        # Store information for this recipe
        cls._RECIPES[name] = {
            "input_objects": input_objects,
            "input_systems": input_systems,
            "output_system": output_system,
        }

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
            if not container.states[Contains].get_value(system=get_system(system_name=system_name)):
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
        relevant_systems = set(recipe["input_systems"])
        for system in PhysicalParticleSystem.get_active_systems():
            if system.name not in relevant_systems and container.states[Contains].get_value(system=system):
                return False
        return True

    @classmethod
    def _validate_recipe_objects_are_contained(cls, recipe, in_volume):
        """
        Validates whether @recipe's input_objects are all contained in the container represented by @in_volume

        Args:
            recipe (dict): Recipe whose objects should be checked
            in_volume (n-array): (N,) flat boolean array corresponding to whether every object from
                cls._OBJECTS is inside the corresponding container

        Returns:
            bool: True if all the input object quantities are contained
        """
        for obj_category, obj_quantity in recipe["input_objects"].items():
            if np.sum(in_volume[cls._CATEGORY_IDXS[obj_category]]) < obj_quantity:
                return False
        return True

    @classmethod
    def _validate_nonrecipe_objects_not_contained(cls, recipe, in_volume):
        """
        Validates whether all objects not relevant to @recipe are not contained in the container
        represented by @in_volume

        Args:
            recipe (dict): Recipe whose systems should be checked
            in_volume (n-array): (N,) flat boolean array corresponding to whether every object from
                cls._OBJECTS is inside the corresponding container

        Returns:
            bool: True if none of the non-relevant objects are contained
        """
        idxs = np.concatenate(cls._CATEGORY_IDXS[obj_category] for obj_category in recipe["input_objects"].keys())
        return not np.any(np.delete(in_volume, idxs))

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
    def refresh(cls):
        # Cache active recipes given the current set of objects
        cls._ACTIVE_RECIPES = dict()
        cls._CATEGORY_IDXS = dict()
        cls._OBJECTS = []

        # Prune any recipes whose objects / system requirements are not met by the current set of objects / systems
        objects_by_category = og.sim.scene.object_registry.get_dict("category")

        for name, recipe in cls._RECIPES.items():
            # Check valid active systems
            if not cls._validate_recipe_systems_exist(recipe=recipe):
                continue

            # Check valid object quantities
            if not cls._validate_recipe_objects_exist(recipe=recipe):
                continue

            # All pre-requisites met, add to active recipes
            cls._ACTIVE_RECIPES[name] = recipe

        # Finally, compute relevant objects and category mapping based on relevant categories
        i = 0
        for category, objects in objects_by_category.items():
            cls._CATEGORY_IDXS[category] = i + np.arange(len(objects))
            cls._OBJECTS += list(objects)
            i += len(objects)

        # Wrap relevant objects as numpy array so we can index into it efficiently
        cls._OBJECTS = np.array(cls._OBJECTS)

    @classproperty
    def group_filters(cls):
        # Fillable object required
        return {"container": AbilityFilter(ability="fillable")}

    @classmethod
    def transition(cls, individual_objects, group_objects):
        t_results = TransitionResults()

        # Compute all relevant object AABB positions
        obj_positions = np.array([obj.aabb_center for obj in cls._OBJECTS])

        # Iterate over all fillable objects, to execute recipes for each one
        for container in individual_objects["container"]:
            # Compute in volume for all relevant object positions
            contained_particles_state = container.states[ContainedParticles]
            in_volume = contained_particles_state.check_in_volume(obj_positions)

            # Check every recipe to find if any is valid
            for name, recipe in cls._ACTIVE_RECIPES.items():
                # Verify all required systems are contained in the container
                if not cls._validate_recipe_systems_are_contained(recipe=recipe, container=container):
                    continue

                # Verify all required object quantities are contained in the container
                if not cls._validate_recipe_objects_are_contained(recipe=recipe, in_volume=in_volume):
                    continue

                # Verify no non-relevant system is contained
                if not cls._validate_nonrecipe_systems_not_contained(recipe=recipe, container=container):
                    continue

                # Verify no non-relevant object is contained
                if not cls._validate_nonrecipe_objects_not_contained(recipe=recipe, in_volume=in_volume):
                    continue

                # Otherwise, all conditions met, we found a valid recipe and so we execute and terminate early
                og.log.info(f"Executing recipe: {name} in container {container.name}!")

                # Compute total volume of all contained items
                volume = 0

                # Remove all recipe system particles contained in the container
                for system_name in recipe["input_systems"]:
                    system = get_system(system_name=system_name)
                    volume += contained_particles_state.get_value()[0] * np.pi * (system.particle_radius ** 3) * 4 / 3
                    container.states[Filled].set_value(system, False)

                # Remove all recipe objects
                objs_to_remove = np.concatenate([
                    cls._OBJECTS[np.where(in_volume[cls._CATEGORY_IDXS[obj_category]])[0]]
                    for obj_category in recipe["input_objects"].keys()
                ]).tolist()
                volume += sum(obj.volume for obj in objs_to_remove)
                t_results.remove += objs_to_remove

                # Spawn in new fluid
                out_system = get_system(recipe["output_system"])
                out_system.generate_particles_from_link(
                    obj=container,
                    link=contained_particles_state.link,
                    mesh_name_prefixes="container",
                    max_samples=volume // (np.pi * (out_system.particle_radius ** 3) * 4 / 3),
                )

                # Terminate early
                return t_results

            # Otherwise, if we didn't find a valid recipe, we execute a garbage transition instead

            # Remove all systems inside the container
            for system in PhysicalParticleSystem.get_active_systems():
                if container.states[Contains].get_value(system):
                    container.states[Filled].set_value(system, False)

            # Remove all objects inside the container
            objs_to_remove = cls._OBJECTS[np.where(in_volume)[0]].tolist()
            t_results.remove += objs_to_remove

            # Spawn in garbage fluid
            container.states[Filled].set_value(get_system(system_name=m.DEFAULT_GARBAGE_SYSTEM), True)

            return t_results

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("MixingRule")
        return classes


class BlenderRule(MixingRule):
    """
    Transition mixing rule that leverages "blender" ability objects, which require toggledOn in order to trigger
    the recipe event
    """
    @classproperty
    def group_filters(cls):
        # Modify the container filter to include "blender" ability as well
        group_filters = super().group_filters
        group_filters["container"] = AndFilter(filters=[group_filters["container"], AbilityFilter(ability="blender")])
        return group_filters

    @classproperty
    def conditions(cls):
        # Container must be toggledOn, and should only be triggered once
        return [
            StateCondition(filter_name="container", state=ToggledOn, val=True, op=operator.eq),
            ChangeCondition(filter_name="container"),
        ]


class MixingWandRule(MixingRule):
    """
    Transition mixing rule that leverages "mixing_wand" ability objects, which require touching between a mixing wand
    and a container in order to trigger the recipe event
    """

    @classproperty
    def group_filters(cls):
        # Add mixing wand group filter as well
        group_filters = super().group_filters
        group_filters["mixing_wand"] = AbilityFilter(ability="mixing_wand")
        return group_filters

    @classproperty
    def conditions(cls):
        # Mixing wand must be touching the container, and should only be triggered once
        return [
            TouchingAnyCondition(filter_1_name="container", filter_2_name="mixing_wand"),
            ChangeCondition(filter_name="container"),
        ]


# Create strawberry smoothie blender rule
BlenderRule.add_recipe(
    name="strawberry_smoothie_recipe",
    input_objects={"strawberry": 5, "ice_cube": 5},
    input_systems=["milk"],
    output_system="strawberry_smoothie"
)
