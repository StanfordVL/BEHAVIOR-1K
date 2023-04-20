from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict
import numpy as np
import itertools
import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.systems import get_system, is_system_active, PhysicalParticleSystem
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.object_states import *
from omnigibson.utils.python_utils import Registerable, classproperty, subclass_factory
from omnigibson.utils.registry_utils import Registry
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import disclaimer

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Where to place objects far out of the scene
m.OBJECT_GRAVEYARD_POS = (100.0, 100.0, 100.0)

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
            None or list of dict: If any valid candidates are found, returns list of unique candidate combinations,
                where each entry is a dictionary of "individual_objects", "group_objects". Otherwise, returns None if
                no valid candidates are found
        """
        individual_objects, group_objects = rule.get_object_candidates(objects=objects)
        n_individual_filters_satisfied, n_group_filters_satisfied = len(individual_objects), len(group_objects)
        # Skip if the rule requirements are not met
        if (rule.requires_individual_filters and n_individual_filters_satisfied != len(rule.individual_filters)) \
                or (rule.requires_group_filters and n_group_filters_satisfied != len(rule.group_filters)):
            return

        # Compile candidates
        # Note: Even if there are no valid individual combos (i.e.: rule does NOT require individual filters), this
        # will still trigger a single pass with individual objects being an empty dictionary, which is what we want
        # since presumably the group objects are used for this rule)
        candidates = []
        for obj_tuple in itertools.product(*list(individual_objects.values())):
            individual_objs = {fname: obj for fname, obj in zip(individual_objects.keys(), obj_tuple)}
            candidates.append({"individual_objects": individual_objs, "group_objects": group_objects})

        return candidates

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
                cls.ACTIVE_RULES[rule] = candidates
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
        added_obj_attrs = []
        removed_objs = []
        for rule, candidates in cls.ACTIVE_RULES.items():
            for candidate in candidates:
                transition_output = rule.process(**candidate)
                if transition_output is not None:
                    # Transition occurred, store objects to be added / removed
                    added_obj_attrs += transition_output.add
                    removed_objs += transition_output.remove

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
            # self.remove_object(removed_obj)
            removed_obj.set_position(np.array(m.OBJECT_GRAVEYARD_POS) + np.ones(3) * i)

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

        # Iterate over all registered rules and also clear their states
        for rule in RULES_REGISTRY.objects:
            rule.clear()


class BaseFilter(metaclass=ABCMeta):
    """
    Defines a filter to apply for inferring which objects are valid candidates for checking a transition rule's
    condition requirements.

    NOTE: These filters should describe STATIC properties about an object -- i.e.: properties that should NOT change
    at runtime, once imported
    """
    # Class global variable for maintaining cached state
    # Maps tuple of unique filter inputs to cached output value (T / F)
    state = None

    @classmethod
    def __new__(cls, *args, **kwargs):
        """
        Initializes the cached state for this filter if it doesn't already exist
        """
        if cls.state is None:
            cls.state = dict()

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
    def process(cls, individual_objects, group_objects):
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
            - None or TransitionResults: Output from @self.transition (None if it was never executed --i.e.:
                condition was never met)
        """
        should_transition = cls.condition(individual_objects=individual_objects, group_objects=group_objects)
        return cls.transition(individual_objects=individual_objects, group_objects=group_objects) \
            if should_transition else None

    @classmethod
    def condition(cls, individual_objects, group_objects):
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
        raise NotImplementedError()

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

    @classmethod
    def clear(cls):
        """
        Clears any internal state when the simulator is restarted (e.g.: a new scene is opened)
        """
        pass

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
        # Define an individual filter dictionary so we can track all valid combos of slicer - sliceable
        return {ability: AbilityFilter(ability) for ability in ("sliceable", "slicer")}

    @classmethod
    def condition(cls, individual_objects, group_objects):
        slicer_obj, sliced_obj = individual_objects["slicer"], individual_objects["sliceable"]
        if not slicer_obj.states[Touching].get_value(sliced_obj):
            return False

        # TODO: How to handle case when multiple slicers are touching the same sliceable object at the same exact time?
        return True

    @classmethod
    def transition(cls, individual_objects, group_objects):
        slicer_obj, sliced_obj = individual_objects["slicer"], individual_objects["sliceable"]
        # Object parts offset annotation are w.r.t the base link of the whole object.
        pos, orn = sliced_obj.get_position_orientation()

        t_results = TransitionResults()

        # Load object parts.
        if sliced_obj.bddl_object_scope is not None:
            sliced_obj_id = int(sliced_obj.bddl_object_scope.split("_")[-1])
            sliced_obj_scope_prefix = "_".join(sliced_obj.bddl_object_scope.split("_")[:-1])
        for i, part in enumerate(sliced_obj.metadata["object_parts"].values()):
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
            scale = np.abs(T.quat2mat(part_bb_orn) @ sliced_obj.scale)

            # Calculate global part bounding box pose.
            part_bb_pos = pos + T.quat2mat(orn) @ (part_bb_pos * scale)
            part_bb_orn = T.quat_multiply(orn, part_bb_orn)
            part_obj_name = f"{sliced_obj.name}_part_{i}"
            part_obj = DatasetObject(
                prim_path=f"/World/{part_obj_name}",
                name=part_obj_name,
                category=part["category"],
                model=part["model"],
                bounding_box=part["bb_size"] * scale,   # equiv. to scale=(part["bb_size"] / self.native_bbox) * (scale)
                bddl_object_scope=None if sliced_obj.bddl_object_scope is None else f"half_{sliced_obj_scope_prefix}_{2 * sliced_obj_id - i}",
            )

            # Add the new object to the results.
            new_obj_attrs = ObjectAttrs(
                obj=part_obj,
                bb_pos=part_bb_pos,
                bb_orn=part_bb_orn,
            )
            t_results.add.append(new_obj_attrs)

        # Delete original object from stage.
        t_results.remove.append(sliced_obj)

        return t_results


class DicingRule(BaseTransitionRule):
    """
    Transition rule to apply to diceable / slicer object pairs.
    """
    @classproperty
    def individual_filters(cls):
        # Define an individual filter dictionary so we can track all valid combos of slicer - sliceable
        return {ability: AbilityFilter(ability) for ability in ("diceable", "slicer")}

    @classmethod
    def condition(cls, individual_objects, group_objects):
        slicer_obj, diced_obj = individual_objects["slicer"], individual_objects["diceable"]

        # Return True if the slicer object is touching the diced object
        return slicer_obj.states[Touching].get_value(diced_obj)

    @classmethod
    def transition(cls, individual_objects, group_objects):
        t_results = TransitionResults()

        slicer_obj, diced_obj = individual_objects["slicer"], individual_objects["diceable"]
        system = get_system(f"diced_{diced_obj.category}")
        system.generate_particles_from_link(diced_obj, diced_obj.root_link, use_visual_meshes=False)

        # Delete original object from stage.
        t_results.remove.append(diced_obj)

        return t_results


class CookingPhysicalParticleRule(BaseTransitionRule):
    """
    Transition rule to apply to "cook" physicl particles
    """
    @classproperty
    def individual_filters(cls):
        # We want to track all possible fillable heatable objects
        return {"fillable": AndFilter(filters=(AbilityFilter("fillable"), AbilityFilter("heatable")))}

    @classmethod
    def condition(cls, individual_objects, group_objects):
        fillable_obj = individual_objects["fillable"]
        # If not heated, immediately return False
        if not fillable_obj.states[Heated].get_value():
            return False

        # Otherwise, return True
        return True

    @classmethod
    def transition(cls, individual_objects, group_objects):
        t_results = TransitionResults()
        fillable_obj = individual_objects["fillable"]
        contained_particles_state = fillable_obj.states[ContainedParticles]

        # Iterate over all active physical particle systems, and for any non-cooked particles inside,
        # convert into cooked particles
        for name, system in PhysicalParticleSystem.get_active_systems().items():
            # Skip any systems that are already cooked or do not contain any particles from this system
            if "cooked" in name or not fillable_obj.states[Contains].get_value(system=system):
                continue
            # TODO: Remove this assert once we have a more standardized method of globally R/W particle positions
            assert len(system.particle_instancers) == 1, \
                f"PhysicalParticleSystem {system.name} should only have one instancer!"
            # Replace all particles inside the container with their cooked versions
            cooked_system = get_system(f"cooked_{system.name}")
            _, positions, in_volume = contained_particles_state.get_value()
            in_volume_idx = np.where(in_volume)[0]
            system.default_particle_instancer.remove_particles(idxs=in_volume_idx)
            cooked_system.default_particle_instancer.add_particles(positions=positions[in_volume_idx])

        return t_results


class MeltingRule(BaseTransitionRule):
    """
    Transition rule to apply to meltable objects to simulate melting
    """
    @classproperty
    def individual_filters(cls):
        # We want to find all meltable objects
        return {"meltable": AbilityFilter("meltable")}

    @classmethod
    def condition(cls, individual_objects, group_objects):
        # Return True if the melter object's temperature is above its melting threshold
        melter_obj = individual_objects["meltable"]
        return melter_obj.states[Temperature].get_value() > m.MELTING_TEMPERATURE

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


class BlenderRuleTemplate(BaseTransitionRule):

    # Keep track of blender volume checker functions
    # Note that ALL blender rule subclasses will share this dictionary -- this is intentional so that each blender's
    # volume will only need to be computed exactly once, across all blender rules!!
    _CHECK_IN_VOLUME = dict()

    @classmethod
    def create(
        cls,
        name,
        output_system,
        particle_requirements=None,
        object_requirements=None,
        **kwargs,
    ):
        """
        Utility function to programmaticlaly generate monolithic blender rule classes

        Args:
            name (str): Name of the rule to generate (should include "Rule" at the end)
            output_system (str): Name of the physical particle to generate once all the input ingredients are blended
            particle_requirements (None or dict): If specified, should map physical particle system name to the minimum
                number of physical particles required in order to successfully blend
            object_requirements (None or dict): If specified, should map object category names to minimum number of that
                type of object required in order to successfully blend
            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @classproperty or @classmethod decorators!)

        Returns:
            BlenderRuleTemplate: Generated blender rule class
        """
        @classproperty
        def cp_output_system(cls):
            return output_system
        kwargs["output_system"] = cp_output_system

        # Override the particle requirements if specified
        if particle_requirements is not None:
            @classproperty
            def cp_particle_requirements(cls):
                return particle_requirements
            kwargs["particle_requirements"] = cp_particle_requirements

        # Override the object requirements and group filters class properties based on the requested object
        # requirements if requested
        if object_requirements is not None:
            @classproperty
            def cp_object_requirements(cls):
                return object_requirements
            kwargs["object_requirements"] = cp_object_requirements

            @classproperty
            def cp_group_filters(cls):
                return {category: CategoryFilter(category) for category in object_requirements.keys()}
            kwargs["group_filters"] = cp_group_filters

        # Create and return the class
        return subclass_factory(name=name, base_classes=cls, **kwargs)

    @classproperty
    def output_system(cls):
        """
        Returns:
            str: Corresponding output system for this specific blender rule
        """
        raise NotImplementedError()

    @classproperty
    def particle_requirements(cls):
        """
        Returns:
            dict: Maps physical particle system name to the minimum number of physical particles required in order to
                successfully blend
        """
        # Default is empty dictionary
        return dict()

    @classproperty
    def object_requirements(cls):
        """
        Returns:
            dict: Maps object category names to minimum number of that type of object required in order to
                successfully blend
        """
        # Default is empty dictionary
        return dict()

    @classproperty
    def individual_filters(cls):
        # We want to filter for any object that is included in @obj_requirements and also separately for blender
        return {"blender": CategoryFilter("blender")}

    @classmethod
    def condition(cls, individual_objects, group_objects):
        blender = individual_objects["blender"]

        # Immediately terminate if the blender isn't toggled on
        if not blender.states[ToggledOn].get_value():
            return False

        # If this blender doesn't exist in our volume checker, we add it
        if blender.name not in cls._CHECK_IN_VOLUME:
            cls._CHECK_IN_VOLUME[blender.name] = lambda pos: blender.states[Filled].check_in_volume(pos.reshape(-1, 3))

        # Check to see which objects are inside the blender container
        for obj_category, objs in group_objects.items():
            inside_objs = []
            for obj in objs:
                if obj.states[Inside].get_value(blender):
                    inside_objs.append(obj)
            # Make sure the number of objects inside meets the required threshold, else we trigger a failure
            if len(inside_objs) < cls.object_requirements[obj_category]:
                return False
            # We mutate the group_objects in place so that we only keep the ones inside the blender
            group_objects[obj_category] = inside_objs

        # Check whether we have sufficient physical particles as well
        for system_name, n_min_particles in cls.particle_requirements.items():
            if not is_system_active(system_name):
                return False

            system = get_system(system_name)
            if system.n_particles == 0:
                return False

            particle_positions = np.concatenate([inst.particle_positions for inst in system.particle_instancers.values()], axis=0)
            n_particles_in_volume = np.sum(cls._CHECK_IN_VOLUME[blender.name](particle_positions))
            if n_particles_in_volume < n_min_particles:
                return False

        # Our condition is whether we have sufficient ingredients or not
        return True

    @classmethod
    def transition(cls, individual_objects, group_objects):
        t_results = TransitionResults()
        blender = individual_objects["blender"]
        # For every object in group_objects, we remove them from the simulator
        for i, (obj_category, objs) in enumerate(group_objects.items()):
            for j, obj in enumerate(objs):
                t_results.remove.append(obj)

        # Remove all physical particles that are inside the blender
        for system_name in cls.particle_requirements.keys():
            system = get_system(system_name)
            # No need to check for whether particle instancers exist because they must due to @self.condition passing!
            for inst in system.particle_instancers.values():
                indices = cls._CHECK_IN_VOLUME[blender.name](inst.particle_positions).nonzero()[0]
                inst.remove_particles(idxs=indices)

        # Spawn in blended physical particles!
        blender.states[Filled].set_value(get_system(cls.output_system), True)

        return t_results

    @classmethod
    def clear(cls):
        # Clear blender volume checkers
        cls._CHECK_IN_VOLUME = dict()

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BlenderRuleTemplate")
        return classes


# Create strawberry smoothie blender rule
StrawberrySmoothieRule = BlenderRuleTemplate.create(
    name="StrawberrySmoothieRule",
    output_system="strawberry_smoothie",
    particle_requirements={"milk": 10},
    obj_requirements={"strawberry": 5, "ice_cube": 5},
)
