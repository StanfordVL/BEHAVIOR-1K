import json
import bddl
import os
import numpy as np
import networkx as nx
from collections import defaultdict
from bddl.activity import (
    get_goal_conditions,
    get_ground_goal_state_options,
    get_initial_conditions,
)
from bddl.backend_abc import BDDLBackend
from bddl.condition_evaluation import Negation
from bddl.logic_base import BinaryAtomicFormula, UnaryAtomicFormula, AtomicFormula
from bddl.object_taxonomy import ObjectTaxonomy
import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.utils.constants import PrimType
from omnigibson.utils.asset_utils import get_all_object_categories, get_all_object_category_models_with_abilities
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.python_utils import Wrapper
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.robots import BaseRobot
from omnigibson import object_states
from omnigibson.object_states.factory import _KINEMATIC_STATE_SET
from omnigibson.systems.system_base import is_system_active, get_system

# Create module logger
log = create_module_logger(module_name=__name__)


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.MIN_DYNAMIC_SCALE = 0.5
m.DYNAMIC_SCALE_INCREMENT = 0.1


class UnsampleablePredicate:
    def _sample(self, *args, **kwargs):
        raise NotImplementedError()


class ObjectStateInsourcePredicate(UnsampleablePredicate, BinaryAtomicFormula):
    def _evaluate(self, entity, **kwargs):
        # Always returns True
        return True


class ObjectStateFuturePredicate(UnsampleablePredicate, UnaryAtomicFormula):
    STATE_NAME = "future"

    def _evaluate(self, entity, **kwargs):
        return not entity.exists


class ObjectStateRealPredicate(UnsampleablePredicate, UnaryAtomicFormula):
    STATE_NAME = "real"

    def _evaluate(self, entity, **kwargs):
        return entity.exists


class ObjectStateUnaryPredicate(UnaryAtomicFormula):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, entity, **kwargs):
        return entity.get_state(self.STATE_CLASS, **kwargs)

    def _sample(self, entity, binary_state, **kwargs):
        return entity.set_state(self.STATE_CLASS, binary_state, **kwargs)


class ObjectStateBinaryPredicate(BinaryAtomicFormula):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, entity1, entity2, **kwargs):
        return entity1.get_state(self.STATE_CLASS, entity2.wrapped_obj, **kwargs) if entity2.exists else False

    def _sample(self, entity1, entity2, binary_state, **kwargs):
        return entity1.set_state(self.STATE_CLASS, entity2.wrapped_obj, binary_state, **kwargs) if entity2.exists else None


def get_unary_predicate_for_state(state_class, state_name):
    return type(
        state_class.__name__ + "StateUnaryPredicate",
        (ObjectStateUnaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )


def get_binary_predicate_for_state(state_class, state_name):
    return type(
        state_class.__name__ + "StateBinaryPredicate",
        (ObjectStateBinaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )


def is_substance_synset(synset):
    return "substance" in OBJECT_TAXONOMY.get_abilities(synset)


def process_single_condition(condition):
    """
    Processes a single BDDL condition

    Args:
        condition (Condition): Condition to process

    Returns:
        2-tuple:
            - Expression: Condition's expression
            - bool: Whether this evaluated condition is positive or negative
    """
    if not isinstance(condition.children[0], Negation) and not isinstance(condition.children[0], AtomicFormula):
        log.debug(("Skipping over sampling of predicate that is not a negation or an atomic formula"))
        return None, None

    if isinstance(condition.children[0], Negation):
        condition = condition.children[0].children[0]
        positive = False
    else:
        condition = condition.children[0]
        positive = True

    return condition, positive


# TODO: Add remaining predicates.
SUPPORTED_PREDICATES = {
    "inside": get_binary_predicate_for_state(object_states.Inside, "inside"),
    "nextto": get_binary_predicate_for_state(object_states.NextTo, "nextto"),
    "ontop": get_binary_predicate_for_state(object_states.OnTop, "ontop"),
    "under": get_binary_predicate_for_state(object_states.Under, "under"),
    "touching": get_binary_predicate_for_state(object_states.Touching, "touching"),
    "covered": get_binary_predicate_for_state(object_states.Covered, "covered"),
    "contains": get_binary_predicate_for_state(object_states.Contains, "contains"),
    "saturated": get_binary_predicate_for_state(object_states.Saturated, "saturated"),
    "filled": get_binary_predicate_for_state(object_states.Filled, "filled"),
    "cooked": get_unary_predicate_for_state(object_states.Cooked, "cooked"),
    "burnt": get_unary_predicate_for_state(object_states.Burnt, "burnt"),
    "frozen": get_unary_predicate_for_state(object_states.Frozen, "frozen"),
    "hot": get_unary_predicate_for_state(object_states.Heated, "hot"),
    "open": get_unary_predicate_for_state(object_states.Open, "open"),
    "toggled_on": get_unary_predicate_for_state(object_states.ToggledOn, "toggled_on"),
    "attached": get_binary_predicate_for_state(object_states.AttachedTo, "attached"),
    "overlaid": get_binary_predicate_for_state(object_states.Overlaid, "overlaid"),
    "folded": get_unary_predicate_for_state(object_states.Folded, "folded"),
    "unfolded": get_unary_predicate_for_state(object_states.Unfolded, "unfolded"),
    "draped": get_binary_predicate_for_state(object_states.Draped, "draped"),
    "future": ObjectStateFuturePredicate,
    "real": ObjectStateRealPredicate,
    "insource": ObjectStateInsourcePredicate,
}

KINEMATIC_STATES_BDDL = frozenset([state.__name__.lower() for state in _KINEMATIC_STATE_SET])


# BEHAVIOR-related
OBJECT_TAXONOMY = ObjectTaxonomy()
# TODO (Josiah): Remove floor synset once we have new bddl release
FLOOR_SYNSET = "floor.n.01"
BEHAVIOR_ACTIVITIES = sorted(os.listdir(os.path.join(os.path.dirname(bddl.__file__), "activity_definitions")))


class OmniGibsonBDDLBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        return SUPPORTED_PREDICATES[predicate_name]


class BDDLEntity(Wrapper):
    """
    Thin wrapper class that wraps an object or system if it exists, or nothing if it does not exist. Will
    dynamically reference an object / system as they become real in the sim
    """
    def __init__(
        self,
        bddl_inst,
        entity=None,
    ):
        """
        Args:
            bddl_inst (str): BDDL synset instance of the entity, e.g.: "almond.n.01_1"
            entity (None or DatasetObject or BaseSystem): If specified, the BDDL entity to wrap. If not
                specified, will initially wrap nothing, but may dynamically reference an actual object or system
                if it exists in the future
        """
        # Store synset and other info, and pass entity internally
        self.bddl_inst = bddl_inst
        self.synset = "_".join(self.bddl_inst.split("_")[:-1])
        self.is_system = is_substance_synset(self.synset)

        # Infer the correct category to assign
        self.og_categories = OBJECT_TAXONOMY.get_subtree_categories(self.synset)

        super().__init__(obj=entity)

    @property
    def name(self):
        """
        Returns:
            None or str: Name of this entity, if it exists, else None
        """
        if self.exists:
            return self.og_categories[0] if self.is_system else self.wrapped_obj.name
        else:
            return None

    @property
    def exists(self):
        """
        Checks whether the entity referenced by @synset exists

        Returns:
            bool: Whether the entity referenced by @synset exists
        """
        return self.wrapped_obj is not None

    def set_entity(self, entity):
        """
        Sets the internal entity, overriding any if it already exists

        Args:
            entity (BaseSystem or BaseObject): Entity to set internally
        """
        self.wrapped_obj = entity

    def clear_entity(self):
        """
        Clears the internal entity, if any
        """
        self.wrapped_obj = None

    def get_state(self, state, *args, **kwargs):
        """
        Helper function to grab wrapped entity's state @state

        Args:
            state (BaseObjectState): State whose get_value() should be called
            *args (tuple): Any arguments to pass to getter, in order
            **kwargs (dict): Any keyword arguments to pass to getter, in order

        Returns:
            any: Returned value(s) from @state if self.wrapped_obj exists (i.e.: not None), else False
        """
        return self.wrapped_obj.states[state].get_value(*args, **kwargs) if self.exists else False

    def set_state(self, state, *args, **kwargs):
        """
        Helper function to set wrapped entity's state @state. Note: Should only be called if the entity exists!

        Args:
            state (BaseObjectState): State whose set_value() should be called
            *args (tuple): Any arguments to pass to getter, in order
            **kwargs (dict): Any keyword arguments to pass to getter, in order

        Returns:
            any: Returned value(s) from @state if self.wrapped_obj exists (i.e.: not None)
        """
        assert self.exists, \
            f"Cannot call set_state() for BDDLEntity {self.synset} when the entity does not exist!"
        return self.wrapped_obj.states[state].set_value(*args, **kwargs)


class BDDLSampler:
    def __init__(
        self,
        env,
        activity_conditions,
        object_scope,
        backend,
        debug=False,
    ):
        # Store internal variables from inputs
        self._env = env
        self._scene_model = self._env.scene.scene_model
        self._agent = self._env.robots[0]
        if debug:
            gm.DEBUG = True
        self._backend = backend
        self._activity_conditions = activity_conditions
        self._object_scope = object_scope
        self._object_instance_to_synset = {
            obj_inst: obj_cat
            for obj_cat in self._activity_conditions.parsed_objects
            for obj_inst in self._activity_conditions.parsed_objects[obj_cat]
        }
        self._substance_instances = {obj_inst for obj_inst in self._object_scope.keys() if
                                     is_substance_synset(self._object_instance_to_synset[obj_inst])}

        # Initialize other variables that will be filled in later
        self._room_type_to_object_instance = None           # dict
        self._inroom_object_instances = None        # set of str
        self._object_sampling_orders = None                 # dict mapping str to list of str
        self._sampled_objects = None                        # set of BaseObject
        self._future_obj_instances = None                   # set of str
        self._inroom_object_conditions = None       # list of (condition, positive) tuple
        self._inroom_object_scope_filtered_initial = None   # dict mapping str to BDDLEntity

    def sample(self, validate_goal=False):
        """
        Run sampling for this BEHAVIOR task

        Args:
            validate_goal (bool): Whether the goal should be validated or not

        Returns:
            2-tuple:
                - bool: Whether sampling was successful or not
                - None or str: None if successful, otherwise the associated error message
        """
        log.info("Sampling task...")
        # Reject scenes with missing non-sampleable objects
        # Populate object_scope with sampleable objects and the robot
        accept_scene, feedback = self._prepare_scene_for_sampling()
        if not accept_scene:
            return accept_scene, feedback
        # Sample objects to satisfy initial conditions
        accept_scene, feedback = self._sample_all_conditions(validate_goal=validate_goal)
        if not accept_scene:
            return accept_scene, feedback

        log.info("Sampling succeeded!")

        return True, None

    def _sample_all_conditions(self, validate_goal=False):
        """
        Run sampling for this BEHAVIOR task

        Args:
            validate_goal (bool): Whether the goal should be validated or not

        Returns:
            2-tuple:
                - bool: Whether sampling was successful or not
                - None or str: None if successful, otherwise the associated error message
        """
        # Auto-initialize all sampleable objects
        with og.sim.playing():
            self._env.scene.reset()

            error_msg = self._sample_initial_conditions()
            if error_msg:
                log.error(error_msg)
                return False, error_msg

            if validate_goal:
                error_msg = self._sample_goal_conditions()
                if error_msg:
                    log.error(error_msg)
                    return False, error_msg

            error_msg = self._sample_initial_conditions_final()
            if error_msg:
                log.error(error_msg)
                return False, error_msg

            self._env.scene.update_initial_state()

        return True, None

    def _prepare_scene_for_sampling(self):
        """
        Runs sanity checks for the current scene for the given BEHAVIOR task

        Returns:
            2-tuple:
                - bool: Whether the generated scene activity should be accepted or not
                - dict: Any feedback from the sampling / initialization process
        """
        error_msg = self._parse_inroom_object_room_assignment()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._build_sampling_order()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._build_inroom_object_scope()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._import_sampleable_objects()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        self._object_scope["agent.n.01_1"] = BDDLEntity(bddl_inst="agent.n.01_1", entity=self._agent)

        return True, None

    def _parse_inroom_object_room_assignment(self):
        """
        Infers which rooms each object is assigned to
        """
        self._room_type_to_object_instance = dict()
        self._inroom_object_instances = set()
        for cond in self._activity_conditions.parsed_initial_conditions:
            if cond[0] == "inroom":
                obj_inst, room_type = cond[1], cond[2]
                obj_synset = self._object_instance_to_synset[obj_inst]
                abilities = OBJECT_TAXONOMY.get_abilities(obj_synset)
                if "sceneObject" not in abilities:
                    # Invalid room assignment
                    return f"You have assigned room type for [{obj_synset}], but [{obj_synset}] is sampleable. " \
                           f"Only non-sampleable (scene) objects can have room assignment."
                if room_type not in og.sim.scene.seg_map.room_sem_name_to_ins_name:
                    # Missing room type
                    return f"Room type [{room_type}] missing in scene [{og.sim.scene.scene_model}]."
                if room_type not in self._room_type_to_object_instance:
                    self._room_type_to_object_instance[room_type] = []
                self._room_type_to_object_instance[room_type].append(obj_inst)

                if obj_inst in self._inroom_object_instances:
                    # Duplicate room assignment
                    return f"Object [{obj_inst}] has more than one room assignment"

                self._inroom_object_instances.add(obj_inst)

        for obj_synset in self._activity_conditions.parsed_objects:
            abilities = OBJECT_TAXONOMY.get_abilities(obj_synset)
            if "sceneObject" not in abilities:
                continue
            for obj_inst in self._activity_conditions.parsed_objects[obj_synset]:
                if obj_inst not in self._inroom_object_instances:
                    # Missing room assignment
                    return f"All non-sampleable (scene) objects should have room assignment. [{obj_inst}] does not have one."

    def _build_sampling_order(self):
        """
        Sampling orders is a list of lists: [[batch_1_inst_1, ... batch_1_inst_N], [batch_2_inst_1, batch_2_inst_M], ...]
        Sampling should happen for batch 1 first, then batch 2, so on and so forth
        Example: OnTop(plate, table) should belong to batch 1, and OnTop(apple, plate) should belong to batch 2
        """
        unsampleable_conditions = []
        sampling_groups = {group: [] for group in ("kinematic", "particle", "unary")}
        self._object_sampling_conditions = {group: [] for group in ("kinematic", "particle", "unary")}
        self._object_sampling_orders = {group: [] for group in ("kinematic", "particle", "unary")}
        self._inroom_object_conditions = []

        # First, sort initial conditions into kinematic, particle and unary groups
        # bddl.condition_evaluation.HEAD, each with one child.
        # This child is either a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate or
        # a Negation of a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate
        for condition in get_initial_conditions(self._activity_conditions, self._backend, self._object_scope):
            condition, positive = process_single_condition(condition)
            if condition is None:
                continue

            # Sampled conditions must always be positive
            # Non-positive (e.g.: NOT onTop) is not restrictive enough for sampling
            if condition.STATE_NAME in KINEMATIC_STATES_BDDL and not positive:
                return "Initial condition has negative kinematic conditions: {}".format(condition.body)

            # Store any unsampleable conditions separately
            if isinstance(condition, UnsampleablePredicate):
                unsampleable_conditions.append(condition)
                continue

            # Infer the group the condition and its object instances belong to
            # (a) Kinematic (binary) conditions, where (ent0, ent1) are both objects
            # (b) Particle (binary) conditions, where (ent0, ent1) are (object, substance)
            # (d) Unary conditions, where (ent0,) is an object
            # Binary conditions have length 2: (ent0, ent1)
            if len(condition.body) == 2:
                group = "particle" if condition.body[1] in self._substance_instances else "kinematic"
            else:
                assert len(condition.body) == 1, \
                    f"Got invalid parsed initial condition; body length should either be 2 or 1. " \
                    f"Got body: {condition.body} for condition: {condition}"
                group = "unary"
            sampling_groups[group].append(condition.body)
            self._object_sampling_conditions[group].append((condition, positive))

            # If the condition involves any non-sampleable object (e.g.: furniture), it's a non-sampleable condition
            # This means that there's no ordering constraint in terms of sampling, because we know the, e.g., furniture
            # object already exists in the scene and is placed, so these specific conditions can be sampled without
            # any dependencies
            if len(self._inroom_object_instances.intersection(set(condition.body))) > 0:
                self._inroom_object_conditions.append((condition, positive))

        # Now, sort each group, ignoring the futures (since they don't get sampled)
        # First handle kinematics, then particles, then unary

        # Start with the non-sampleable objects as the first sampled set, then infer recursively
        cur_batch = self._inroom_object_instances
        while len(cur_batch) > 0:
            next_batch = set()
            for cur_batch_inst in cur_batch:
                inst_batch = set()
                for condition, _ in self._object_sampling_conditions["kinematic"]:
                    if condition.body[1] == cur_batch_inst:
                        inst_batch.add(condition.body[0])
                        next_batch.add(condition.body[0])
                if len(inst_batch) > 0:
                    self._object_sampling_orders["kinematic"].append(inst_batch)
            cur_batch = next_batch

        # Now parse particles -- simply unordered, since particle systems shouldn't impact each other
        self._object_sampling_orders["particle"].append({cond[0] for cond in sampling_groups["particle"]})
        sampled_particle_entities = {cond[1] for cond in sampling_groups["particle"]}

        # Finally, parse unaries -- this is simply unordered, since it is assumed that unary predicates do not
        # affect each other
        self._object_sampling_orders["unary"].append({cond[0] for cond in sampling_groups["unary"]})

        # Aggregate future objects and any unsampleable obj instances
        # Unsampleable obj instances are strictly a superset of future obj instances
        unsampleable_obj_instances = {cond.body[-1] for cond in unsampleable_conditions}
        self._future_obj_instances = {cond.body[0] for cond in unsampleable_conditions if isinstance(cond, ObjectStateFuturePredicate)}

        nonparticle_entities = set(self._object_scope.keys()) - self._substance_instances

        # Sanity check kinematic objects -- any non-system must be kinematically sampled
        remaining_kinematic_entities = nonparticle_entities - unsampleable_obj_instances - \
            self._inroom_object_instances - set.union(*(self._object_sampling_orders["kinematic"] + [set()]))
        if len(remaining_kinematic_entities) != 0:
            return f"Some objects do not have any kinematic condition defined for them in the initial conditions: " \
                   f"{', '.join(remaining_kinematic_entities)}"

        # Sanity check particle systems -- any non-future system must be sampled as part of particle groups
        remaining_particle_entities = self._substance_instances - unsampleable_obj_instances - sampled_particle_entities
        if len(remaining_particle_entities) != 0:
            return f"Some systems do not have any particle condition defined for them in the initial conditions: " \
                   f"{', '.join(remaining_particle_entities)}"

    def _build_inroom_object_scope(self):
        """
        Store simulator object options for non-sampleable objects in self.inroom_object_scope
        {
            "living_room": {
                "table1": {
                    "living_room_0": [URDFObject, URDFObject, URDFObject],
                    "living_room_1": [URDFObject]
                },
                "table2": {
                    "living_room_0": [URDFObject, URDFObject],
                    "living_room_1": [URDFObject, URDFObject]
                },
                "chair1": {
                    "living_room_0": [URDFObject],
                    "living_room_1": [URDFObject]
                },
            }
        }
        """
        room_type_to_scene_objs = {}
        for room_type in self._room_type_to_object_instance:
            room_type_to_scene_objs[room_type] = {}
            for obj_inst in self._room_type_to_object_instance[room_type]:
                room_type_to_scene_objs[room_type][obj_inst] = {}
                obj_synset = self._object_instance_to_synset[obj_inst]

                # We allow burners to be used as if they are stoves
                categories = OBJECT_TAXONOMY.get_subtree_categories(obj_synset)
                abilities = OBJECT_TAXONOMY.get_abilities(obj_synset)

                # Grab all models that fully support all abilities for the corresponding category
                valid_models = {cat: set(get_all_object_category_models_with_abilities(cat, abilities))
                                for cat in categories}

                for room_inst in og.sim.scene.seg_map.room_sem_name_to_ins_name[room_type]:
                    # A list of scene objects that satisfy the requested categories
                    room_objs = og.sim.scene.object_registry("in_rooms", room_inst, default_val=[])
                    scene_objs = [obj for obj in room_objs if obj.category in categories and obj.model in valid_models[obj.category]]

                    if len(scene_objs) != 0:
                        room_type_to_scene_objs[room_type][obj_inst][room_inst] = scene_objs

        error_msg = self._consolidate_room_instance(room_type_to_scene_objs, "initial_pre-sampling")
        if error_msg:
            return error_msg
        self._inroom_object_scope = room_type_to_scene_objs

    def _filter_object_scope(self, input_object_scope, conditions, condition_type):
        """
        Filters the object scope based on given @input_object_scope, @conditions, and @condition_type

        Args:
            input_object_scope (dict):
            conditions (list): List of conditions to filter scope with, where each list entry is
                a tuple of (condition, positive), where @positive is True if the condition has a positive
                evaluation.
            condition_type (str): What type of condition to sample, e.g., "initial"

        Returns:
            2-tuple:

                - dict: Filtered object scope
                - list of str: The name of children object(s) that have the highest proportion of kinematic sampling
                    failures
        """
        filtered_object_scope = {}
        # Maps child obj name (SCOPE name) to parent obj name (OBJECT name) to T / F,
        # ie: if the kinematic relationship was sampled successfully
        problematic_objs = defaultdict(dict)
        for room_type in input_object_scope:
            filtered_object_scope[room_type] = {}
            for scene_obj in input_object_scope[room_type]:
                filtered_object_scope[room_type][scene_obj] = {}
                for room_inst in input_object_scope[room_type][scene_obj]:
                    # These are a list of candidate simulator objects that need sampling test
                    for obj in input_object_scope[room_type][scene_obj][room_inst]:
                        # Temporarily set object_scope to point to this candidate object
                        self._object_scope[scene_obj] = BDDLEntity(bddl_inst=scene_obj, entity=obj)

                        success = True
                        # If this candidate object is not involved in any conditions,
                        # success will be True by default and this object will qualify
                        parent_obj_name = obj.name
                        conditions_to_sample = []
                        for condition, positive in conditions:
                            # Sample positive kinematic conditions that involve this candidate object
                            if condition.STATE_NAME in KINEMATIC_STATES_BDDL and positive and scene_obj in condition.body:
                                child_scope_name = condition.body[0]
                                entity = self._object_scope[child_scope_name]
                                conditions_to_sample.append((condition, positive, entity, child_scope_name))

                        # Sort children based on their AABB so the larger objects are sampled first
                        conditions_to_sample = reversed(sorted(conditions_to_sample, key=lambda x: np.product(x[2].aabb_extent)))

                        # Sample!
                        for condition, positive, entity, child_scope_name in conditions_to_sample:
                            success = condition.sample(binary_state=positive)
                            log_msg = " ".join(
                                [
                                    f"{condition_type} kinematic condition sampling",
                                    room_type,
                                    scene_obj,
                                    room_inst,
                                    parent_obj_name,
                                    condition.STATE_NAME,
                                    str(condition.body),
                                    str(success),
                                ]
                            )
                            log.info(log_msg)

                            # Record the result for the child object
                            assert parent_obj_name not in problematic_objs[child_scope_name], \
                                f"Multiple kinematic relationships attempted for pair {condition.body}"
                            problematic_objs[child_scope_name][parent_obj_name] = success
                            # If any condition fails for this candidate object, skip
                            if not success:
                                break

                        # If this candidate object fails, move on to the next candidate object
                        if not success:
                            continue

                        if room_inst not in filtered_object_scope[room_type][scene_obj]:
                            filtered_object_scope[room_type][scene_obj][room_inst] = []
                        filtered_object_scope[room_type][scene_obj][room_inst].append(obj)

        # Compute most problematic objects
        problematic_objs_by_proportion = defaultdict(list)
        for child_scope_name, parent_obj_names in problematic_objs.items():
            problematic_objs_by_proportion[np.mean(list(parent_obj_names.values()))].append(child_scope_name)

        return filtered_object_scope, problematic_objs_by_proportion[min(problematic_objs_by_proportion.keys())]

    def _consolidate_room_instance(self, filtered_object_scope, condition_type):
        """
        Consolidates room instances

        Args:
            filtered_object_scope (dict): Filtered object scope
            condition_type (str): What type of condition to sample, e.g., "initial"

        Returns:
            None or str: Error message, if any
        """
        for room_type in filtered_object_scope:
            # For each room_type, filter in room_inst that has successful
            # sampling options for all obj_inst in this room_type
            room_inst_satisfied = set.intersection(
                *[
                    set(filtered_object_scope[room_type][obj_inst].keys())
                    for obj_inst in filtered_object_scope[room_type]
                ]
            )

            if len(room_inst_satisfied) == 0:
                error_msg = "{}: Room type [{}] of scene [{}] do not contain or cannot sample all the objects needed.\nThe following are the possible room instances for each object, the intersection of which is an empty set.\n".format(
                    condition_type, room_type, self._scene_model
                )
                for obj_inst in filtered_object_scope[room_type]:
                    error_msg += (
                        "{}: ".format(obj_inst) + ", ".join(filtered_object_scope[room_type][obj_inst].keys()) + "\n"
                    )

                return error_msg

            for obj_inst in filtered_object_scope[room_type]:
                filtered_object_scope[room_type][obj_inst] = {
                    key: val
                    for key, val in filtered_object_scope[room_type][obj_inst].items()
                    if key in room_inst_satisfied
                }

    def _import_sampleable_objects(self):
        """
        Import all objects that can be sampled

        Args:
            env (Environment): Current active environment instance
        """
        assert og.sim.is_stopped(), "Simulator should be stopped when importing sampleable objects"

        # Move the robot object frame to a far away location, similar to other newly imported objects below
        self._agent.set_position_orientation([300, 300, 300], [0, 0, 0, 1])

        self._sampled_objects = set()
        num_new_obj = 0
        # Only populate self.object_scope for sampleable objects
        available_categories = set(get_all_object_categories())
        for obj_synset in self._activity_conditions.parsed_objects:
            # Don't populate agent
            if obj_synset == "agent.n.01":
                continue
            # Don't populate synsets that can't be sampled
            abilities = OBJECT_TAXONOMY.get_abilities(obj_synset)
            if "sceneObject" in abilities:
                continue

            # Populate based on whether it's a substance or not
            if is_substance_synset(obj_synset):
                assert len(self._activity_conditions.parsed_objects[obj_synset]) == 1, "Systems are singletons"
                obj_inst = self._activity_conditions.parsed_objects[obj_synset][0]
                system_name = OBJECT_TAXONOMY.get_subtree_categories(obj_synset)[0]
                self._object_scope[obj_inst] = BDDLEntity(
                    bddl_inst=obj_inst,
                    entity=None if obj_inst in self._future_obj_instances else get_system(system_name),
                )
            else:
                valid_categories = set(OBJECT_TAXONOMY.get_subtree_categories(obj_synset))
                categories = list(valid_categories.intersection(available_categories))
                if len(categories) == 0:
                    return f"None of the following categories could be found in the dataset for synset {obj_synset}: " \
                           f"{valid_categories}"

                for obj_inst in self._activity_conditions.parsed_objects[obj_synset]:
                    # Don't explicitly sample if future
                    if obj_inst in self._future_obj_instances:
                        self._object_scope[obj_inst] = BDDLEntity(bddl_inst=obj_inst)
                        continue

                    category = np.random.choice(categories)

                    # Get all available models that support all of its synset abilities
                    model_choices = get_all_object_category_models_with_abilities(
                        category=category,
                        abilities=OBJECT_TAXONOMY.get_abilities(OBJECT_TAXONOMY.get_synset_from_category(category)),
                    )
                    if len(model_choices) == 0:
                        return f"Missing valid object models for category: {category}"

                    # Randomly select an object model
                    model = np.random.choice(model_choices)

                    # create the object
                    simulator_obj = DatasetObject(
                        name=f"{category}_{len(og.sim.scene.objects)}",
                        category=category,
                        model=model,
                        fit_avg_dim_volume=True,
                        prim_type=PrimType.CLOTH if "cloth" in OBJECT_TAXONOMY.get_abilities(obj_synset) else PrimType.RIGID,
                    )
                    num_new_obj += 1

                    # Load the object into the simulator
                    assert og.sim.scene.loaded, "Scene is not loaded"
                    og.sim.import_object(simulator_obj)

                    # Set these objects to be far-away locations
                    simulator_obj.set_position(np.array([100.0, 100.0, -100.0]) + np.ones(3) * num_new_obj * 5.0)

                    self._sampled_objects.add(simulator_obj)
                    self._object_scope[obj_inst] = BDDLEntity(bddl_inst=obj_inst, entity=simulator_obj)

        og.sim.play()
        og.sim.stop()

    def _sample_initial_conditions(self):
        """
        Sample initial conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        error_msg, self._inroom_object_scope_filtered_initial = self._sample_conditions(
            self._inroom_object_scope, self._inroom_object_conditions, "initial"
        )
        return error_msg

    def _sample_goal_conditions(self):
        """
        Sample goal conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        activity_goal_conditions = get_goal_conditions(self._activity_conditions, self._backend, self._object_scope)
        ground_goal_state_options = get_ground_goal_state_options(self._activity_conditions, self._backend, self._object_scope, activity_goal_conditions)
        np.random.shuffle(ground_goal_state_options)
        log.debug(("number of ground_goal_state_options", len(ground_goal_state_options)))
        num_goal_condition_set_to_test = 10

        goal_condition_success = False
        # Try to fulfill different set of ground goal conditions (maximum num_goal_condition_set_to_test)
        for goal_condition_set in ground_goal_state_options[:num_goal_condition_set_to_test]:
            goal_condition_processed = []
            for condition in goal_condition_set:
                condition, positive = process_single_condition(condition)
                if condition is None:
                    continue
                goal_condition_processed.append((condition, positive))

            error_msg, _ = self._sample_conditions(
                self._inroom_object_scope_filtered_initial, goal_condition_processed, "goal"
            )
            if not error_msg:
                # if one set of goal conditions (and initial conditions) are satisfied, sampling is successful
                goal_condition_success = True
                break

        if not goal_condition_success:
            return error_msg

    def _sample_initial_conditions_final(self):
        """
        Sample final initial conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        # Sample kinematics first, then particle states, then unary states
        state = og.sim.dump_state(serialized=False)
        for group in ("kinematic", "particle", "unary"):
            log.info(f"Sampling {group} states...")
            if len(self._object_sampling_orders[group]) > 0:
                for cur_batch in self._object_sampling_orders[group]:
                    conditions_to_sample = []
                    for condition, positive in self._object_sampling_conditions[group]:
                        # Sample conditions that involve the current batch of objects
                        child_scope_name = condition.body[0]
                        if child_scope_name in cur_batch:
                            entity = self._object_scope[child_scope_name]
                            conditions_to_sample.append((condition, positive, entity, child_scope_name))

                    # If we're sampling kinematics, sort children based on their AABB, so that the larger objects
                    # are sampled first
                    if group == "kinematic":
                        conditions_to_sample = reversed(sorted(conditions_to_sample, key=lambda x: np.product(x[2].aabb_extent)))

                    # Sample!
                    for condition, positive, entity, child_scope_name in conditions_to_sample:
                        success = False
                        while True:
                            num_trials = 1
                            for _ in range(num_trials):
                                success = condition.sample(binary_state=positive)
                                if success:
                                    # Update state
                                    state = og.sim.dump_state(serialized=False)
                                    break
                            if success:
                                # Can terminate immediately
                                break

                            # Can't re-sample non-kinematics or rescale cloth or agent, so in
                            # those cases terminate immediately
                            if group != "kinematic" or "agent" in child_scope_name or entity.prim_type == PrimType.CLOTH:
                                break

                            # If any scales are equal or less than the lower threshold, terminate immediately
                            new_scale = entity.scale - m.DYNAMIC_SCALE_INCREMENT
                            if np.any(new_scale < m.MIN_DYNAMIC_SCALE):
                                break

                            # Re-scale and re-attempt
                            # Re-scaling is not respected unless sim cycle occurs
                            og.sim.stop()
                            entity.scale = new_scale
                            log.info(f"Kinematic sampling {condition.STATE_NAME} {condition.body} failed, rescaling obj: {child_scope_name} to {entity.scale}")
                            og.sim.play()
                            og.sim.load_state(state, serialized=False)
                            og.sim.step_physics()
                        if not success:
                            return f"Sampleable object conditions failed: {condition.STATE_NAME} {condition.body}"

        # One more sim step to make sure the object states are propagated correctly
        # E.g. after sampling Filled.set_value(True), Filled.get_value() will become True only after one step
        og.sim.step()

    def _sample_conditions(self, input_object_scope, conditions, condition_type):
        """
        Sample conditions

        Args:
            input_object_scope (dict):
            conditions (list): List of conditions to filter scope with, where each list entry is
                a tuple of (condition, positive), where @positive is True if the condition has a positive
                evaluation.
            condition_type (str): What type of condition to sample, e.g., "initial"

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        error_msg, problematic_objs = "", []
        while not np.any([np.any(self._object_scope[obj_inst].scale < m.MIN_DYNAMIC_SCALE) for obj_inst in problematic_objs]):
            filtered_object_scope, problematic_objs = self._filter_object_scope(input_object_scope, conditions, condition_type)
            error_msg = self._consolidate_room_instance(filtered_object_scope, condition_type)
            if error_msg is None:
                break
            # Re-scaling is not respected unless sim cycle occurs
            og.sim.stop()
            for obj_inst in problematic_objs:
                obj = self._object_scope[obj_inst]
                # Can't rescale cloth or agent, so play again and then terminate immediately if found
                if "agent" in obj_inst or obj.prim_type == PrimType.CLOTH:
                    og.sim.play()
                    return error_msg, None
                assert np.all(obj.scale > m.DYNAMIC_SCALE_INCREMENT)
                obj.scale -= m.DYNAMIC_SCALE_INCREMENT
            og.sim.play()

        if error_msg:
            return error_msg, None
        return self._maximum_bipartite_matching(filtered_object_scope, condition_type), filtered_object_scope

    def _maximum_bipartite_matching(self, filtered_object_scope, condition_type):
        """
        Matches objects from @filtered_object_scope to specific room instances it can be
        sampled from

        Args:
            filtered_object_scope (dict): Filtered object scope
            condition_type (str): What type of condition to sample, e.g., "initial"

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        # For each room instance, perform maximum bipartite matching between object instance in scope to simulator objects
        # Left nodes: a list of object instance in scope
        # Right nodes: a list of simulator objects
        # Edges: if the simulator object can support the sampling requirement of ths object instance
        for room_type in filtered_object_scope:
            # The same room instances will be shared across all scene obj in a given room type
            some_obj = list(filtered_object_scope[room_type].keys())[0]
            room_insts = list(filtered_object_scope[room_type][some_obj].keys())
            success = False
            # Loop through each room instance
            for room_inst in room_insts:
                graph = nx.Graph()
                # For this given room instance, gether mapping from obj instance to a list of simulator obj
                obj_inst_to_obj_per_room_inst = {}
                for obj_inst in filtered_object_scope[room_type]:
                    obj_inst_to_obj_per_room_inst[obj_inst] = filtered_object_scope[room_type][obj_inst][room_inst]
                top_nodes = []
                log_msg = "MBM for room instance [{}]".format(room_inst)
                log.debug((log_msg))
                for obj_inst in obj_inst_to_obj_per_room_inst:
                    for obj in obj_inst_to_obj_per_room_inst[obj_inst]:
                        # Create an edge between obj instance and each of the simulator obj that supports sampling
                        graph.add_edge(obj_inst, obj)
                        log_msg = "Adding edge: {} <-> {}".format(obj_inst, obj.name)
                        log.debug((log_msg))
                        top_nodes.append(obj_inst)
                # Need to provide top_nodes that contain all nodes in one bipartite node set
                # The matches will have two items for each match (e.g. A -> B, B -> A)
                matches = nx.bipartite.maximum_matching(graph, top_nodes=top_nodes)
                if len(matches) == 2 * len(obj_inst_to_obj_per_room_inst):
                    log.debug(("Object scope finalized:"))
                    for obj_inst, obj in matches.items():
                        if obj_inst in obj_inst_to_obj_per_room_inst:
                            self._object_scope[obj_inst] = BDDLEntity(bddl_inst=obj_inst, entity=obj)
                            log.debug((obj_inst, obj.name))
                    success = True
                    break
            if not success:
                return "{}: Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n".format(
                    condition_type, room_type, self._scene_model
                )
