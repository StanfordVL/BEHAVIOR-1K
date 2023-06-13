import json
import bddl
import os
import numpy as np
import networkx as nx
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
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_object_models_of_category
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.python_utils import Wrapper
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.robots import BaseRobot
from omnigibson import object_states
from omnigibson.object_states.factory import _KINEMATIC_STATE_SET
from omnigibson.systems.system_base import is_system_active, get_system

# Create module logger
log = create_module_logger(module_name=__name__)


class ObjectStateFuturePredicate(UnaryAtomicFormula):
    STATE_NAME = "future"

    def _evaluate(self, entity, **kwargs):
        return not entity.exists()

    def _sample(self, entity, **kwargs):
        raise NotImplementedError()


class ObjectStateRealPredicate(UnaryAtomicFormula):
    STATE_NAME = "real"

    def _evaluate(self, entity, **kwargs):
        return entity.exists()

    def _sample(self, entity, **kwargs):
        raise NotImplementedError()


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
        return entity1.get_state(self.STATE_CLASS, entity2.wrapped_obj, **kwargs) if entity2.exists() else None

    def _sample(self, entity1, entity2, binary_state, **kwargs):
        return entity1.set_state(self.STATE_CLASS, entity2.wrapped_obj, binary_state, **kwargs) if entity2.exists() else None


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
    "open": get_unary_predicate_for_state(object_states.Open, "open"),
    "toggled_on": get_unary_predicate_for_state(object_states.ToggledOn, "toggled_on"),
    "frozen": get_unary_predicate_for_state(object_states.Frozen, "frozen"),
    "future": ObjectStateFuturePredicate,
    "real": ObjectStateRealPredicate,
}

KINEMATIC_STATES_BDDL = frozenset([state.__name__.lower() for state in _KINEMATIC_STATE_SET])

# Load substance and object mapping
with open(f"{bddl.__path__[0]}/../substance_synset_mapping.json", "r") as f:
    SUBSTANCE_SYNSET_MAPPING = json.load(f)


# BEHAVIOR-related
OBJECT_TAXONOMY = ObjectTaxonomy() #hierarchy_type="b1k")
# TODO (Josiah): Remove floor synset once we have new bddl release
FLOOR_SYNSET = "floor.n.01"
with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
    BEHAVIOR_ACTIVITIES = {line.strip() for line in f.readlines()}
NON_SAMPLEABLE_SYNSETS = set()
non_sampleable_category_txt = os.path.join(gm.DATASET_PATH, "metadata/non_sampleable_categories.txt")
if os.path.isfile(non_sampleable_category_txt):
    with open(non_sampleable_category_txt) as f:
        NON_SAMPLEABLE_SYNSETS = set([FLOOR_SYNSET] + [line.strip() for line in f.readlines()])


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
        self.is_system = self.synset in SUBSTANCE_SYNSET_MAPPING

        # Infer the correct category to assign, special casing agents
        if entity is not None and isinstance(entity, BaseRobot):
            self.og_categories = ["agent"]
        else:
            self.og_categories = [SUBSTANCE_SYNSET_MAPPING[self.synset]] if self.is_system else \
                OBJECT_TAXONOMY.get_subtree_igibson_categories(self.synset)

        super().__init__(obj=entity)

    def _find_valid_object(self):
        """
        Internal helper function to find the first valid simulator object whose category is one of @self.og_categories,
        and has not been mapped to a BDDLEntity yet

        Returns:
            None or DatasetObject: If found, the valid object matching a category from @self.og_categories and not
                mapped
        """
        for category in self.og_categories:
            for obj in og.sim.scene.object_registry("category", category, default_val=[]):
                if isinstance(obj, DatasetObject) and obj.bddl_object_scope is None:
                    # Found valid one, return it
                    return obj

    def exists(self):
        """
        Checks whether the entity referenced by @synset exists. Note: this dynamically mutates self.wrapped_obj, and
        potentially removes it or adds a reference if the entity no longer / now exists.

        Returns:
            bool: Whether the entity referenced by @synset exists
        """
        if self.wrapped_obj is None:
            # If system, check to see if active or not and grab it if so
            if self.is_system:
                if is_system_active(self.og_categories[0]):
                    self.wrapped_obj = get_system(self.og_categories[0])
            # Otherwise, is object, check to see if any valid one exists and grab it if so
            else:
                found_obj = self._find_valid_object()
                if found_obj is not None:
                    found_obj.bddl_object_scope = self.bddl_inst
                    self.wrapped_obj = found_obj
        else:
            # Check to see if entity no longer exists
            if self.is_system:
                if not is_system_active(self.og_categories[0]):
                    self.wrapped_obj = None
            # Otherwise, is object, check to see if there are no valid ones
            else:
                if og.sim.scene.object_registry("name", self.wrapped_obj.name) is None:
                    self.wrapped_obj = None

        return self.wrapped_obj is not None

    def get_state(self, state, *args, **kwargs):
        """
        Helper function to grab wrapped entity's state @state

        Args:
            state (BaseObjectState): State whose get_value() should be called
            *args (tuple): Any arguments to pass to getter, in order
            **kwargs (dict): Any keyword arguments to pass to getter, in order

        Returns:
            None or any: Returned value(s) from @state if self.wrapped_obj exists (i.e.: not None), else None
        """
        return self.wrapped_obj.states[state].get_value(*args, **kwargs) if self.exists() else None

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
        assert self.exists(), \
            f"Cannot call set_state() for BDDLEntity {self.synset} when the entity does not exist!"
        return self.wrapped_obj.states[state].set_value(*args, **kwargs)

    def __getattr__(self, attr):
        # Sanity check to make sure wrapped obj is not None -- if so, raise error
        assert self.wrapped_obj is not None, f"Cannot access attribute {attr}, since no valid entity currently " \
                                             f"wrapped for BDDLEntity synset {self.synset}!"

        # Call super
        return super().__getattr__(attr=attr)


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
        self._object_instance_to_category = {
            obj_inst: obj_cat
            for obj_cat in self._activity_conditions.parsed_objects
            for obj_inst in self._activity_conditions.parsed_objects[obj_cat]
        }
        self._substance_instances = {obj_inst for obj_inst in self._object_scope.keys() if
                                     self._object_instance_to_category[obj_inst] in SUBSTANCE_SYNSET_MAPPING}

        # Initialize other variables that will be filled in later
        self._room_type_to_object_instance = None           # dict
        self._non_sampleable_object_instances = None        # set of str
        self._object_sampling_orders = None                 # dict mapping str to list of str
        self._sampled_objects = None                        # set of BaseObject
        self._future_obj_instances = None                   # set of str
        self._non_sampleable_object_conditions = None       # list of (condition, positive) tuple
        self._non_sampleable_object_scope_filtered_initial = None   # dict mapping str to BDDLEntity

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
        # Reject scenes with missing non-sampleable objects
        # Populate object_scope with sampleable objects and the robot
        accept_scene, feedback = self._prepare_scene_for_sampling()
        if not accept_scene:
            return accept_scene, feedback
        # Sample objects to satisfy initial conditions
        accept_scene, feedback = self._sample_all_conditions(validate_goal=validate_goal)
        if not accept_scene:
            return accept_scene, feedback

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
        error_msg = self._parse_non_sampleable_object_room_assignment()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._build_sampling_order()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._build_non_sampleable_object_scope()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._import_sampleable_objects()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        self._object_scope["agent.n.01_1"] = BDDLEntity(object_scope="agent.n.01_1", entity=self._agent)

        return True, None

    def _parse_non_sampleable_object_room_assignment(self):
        """
        Infers which rooms each object is assigned to
        """
        self._room_type_to_object_instance = dict()
        self._non_sampleable_object_instances = set()
        for cond in self._activity_conditions.parsed_initial_conditions:
            if cond[0] == "inroom":
                obj_inst, room_type = cond[1], cond[2]
                obj_cat = self._object_instance_to_category[obj_inst]
                if obj_cat not in NON_SAMPLEABLE_SYNSETS:
                    # Invalid room assignment
                    return "You have assigned room type for [{}], but [{}] is sampleable. Only non-sampleable objects can have room assignment.".format(
                        obj_cat, obj_cat
                    )
                if room_type not in og.sim.scene.seg_map.room_sem_name_to_ins_name:
                    # Missing room type
                    return "Room type [{}] missing in scene [{}].".format(room_type, og.sim.scene.scene_model)
                if room_type not in self._room_type_to_object_instance:
                    self._room_type_to_object_instance[room_type] = []
                self._room_type_to_object_instance[room_type].append(obj_inst)

                if obj_inst in self._non_sampleable_object_instances:
                    # Duplicate room assignment
                    return "Object [{}] has more than one room assignment".format(obj_inst)

                self._non_sampleable_object_instances.add(obj_inst)

        for obj_cat in self._activity_conditions.parsed_objects:
            if obj_cat not in NON_SAMPLEABLE_SYNSETS:
                continue
            for obj_inst in self._activity_conditions.parsed_objects[obj_cat]:
                if obj_inst not in self._non_sampleable_object_instances:
                    # Missing room assignment
                    return "All non-sampleable objects should have room assignment. [{}] does not have one.".format(
                        obj_inst
                    )

    def _build_sampling_order(self):
        """
        Sampling orders is a list of lists: [[batch_1_inst_1, ... batch_1_inst_N], [batch_2_inst_1, batch_2_inst_M], ...]
        Sampling should happen for batch 1 first, then batch 2, so on and so forth
        Example: OnTop(plate, table) should belong to batch 1, and OnTop(apple, plate) should belong to batch 2
        """
        sampling_groups = {group: [] for group in ("kinematic", "particle", "future", "unary")}
        self._object_sampling_conditions = {group: [] for group in ("kinematic", "particle", "future", "unary")}
        self._object_sampling_orders = {group: [] for group in ("kinematic", "particle", "unary")}
        self._non_sampleable_object_conditions = []

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

            # Infer the group the condition and its object instances belong to
            # (a) Kinematic (binary) conditions, where (ent0, ent1) are both objects
            # (b) Particle (binary) conditions, where (ent0, ent1) are (object, substance)
            # (c) Future conditions, where (ent0,) can be either an objects or substance
            # (d) Unary conditions, where (ent0,) is an object
            # Binary conditions have length 2: (ent0, ent1)
            if len(condition.body) == 2:
                group = "particle" if condition.body[1] in self._substance_instances else "kinematic"
            else:
                assert len(condition.body) == 1, \
                    f"Got invalid parsed initial condition; body length should either be 2 or 1. " \
                    f"Got body: {condition.body} for condition: {condition}"
                group = "future" if isinstance(condition, ObjectStateFuturePredicate) else "unary"
            sampling_groups[group].append(condition.body)
            self._object_sampling_conditions[group].append((condition, positive))

            # If the condition involves any non-sampleable object (e.g.: furniture), it's a non-sampleable condition
            # This means that there's no ordering constraint in terms of sampling, because we know the, e.g., furniture
            # object already exists in the scene and is placed, so these specific conditions can be sampled without
            # any dependencies
            if len(self._non_sampleable_object_instances.intersection(set(condition.body))) > 0:
                self._non_sampleable_object_conditions.append((condition, positive))

        # Now, sort each group, ignoring the futures (since they don't get sampled)
        # First handle kinematics, then particles, then unary

        # Start with the non-sampleable objects as the first sampled set, then infer recursively
        cur_batch = self._non_sampleable_object_instances
        while len(cur_batch) > 0:
            next_batch = set()
            for condition, _ in self._object_sampling_conditions["kinematic"]:
                if condition.body[1] in cur_batch:
                    next_batch.add(condition.body[0])
            cur_batch = next_batch
            self._object_sampling_orders["kinematic"].append(cur_batch)
        # pop final value since it's an empty set
        self._object_sampling_orders["kinematic"].pop(-1)

        # Now parse particles -- this time, in reverse order, starting from the final kinematic group
        obj_insts_to_system = {cond[0]: cond[1] for cond in sampling_groups["particle"]}
        sampled_particle_entities = set()
        for batch in reversed(self._object_sampling_orders["kinematic"] + [self._non_sampleable_object_instances]):
            cur_batch = set()
            for obj_inst in batch:
                if obj_inst in obj_insts_to_system:
                    sampled_particle_entities.add(obj_insts_to_system.pop(obj_inst))
                    cur_batch.add(obj_inst)
            self._object_sampling_orders["particle"].append(cur_batch)
        # Finally, make group with all remaining object instances requiring particle sampling
        self._object_sampling_orders["particle"].append(set(obj_insts_to_system.keys()))

        # Finally, parse unaries -- this is simply unordered, since it is assumed that unary predicates do not
        # affect each other
        self._object_sampling_orders["unary"] = {cond[0] for cond in sampling_groups["unary"]}

        # Aggregate future objects
        self._future_obj_instances = {cond[0] for cond in sampling_groups["future"]}
        nonparticle_entities = set(self._object_scope.keys()) - self._substance_instances

        # Sanity check kinematic objects -- any non-system must be kinematically sampled
        remaining_kinematic_entities = nonparticle_entities - self._future_obj_instances - \
            self._non_sampleable_object_instances - set.union(*(self._object_sampling_orders["kinematic"] + [set()]))
        if len(remaining_kinematic_entities) != 0:
            return f"Some objects do not have any kinematic condition defined for them in the initial conditions: " \
                   f"{', '.join(remaining_kinematic_entities)}"

        # Sanity check particle systems -- any non-future system must be sample as part of particle groups
        remaining_particle_entities = self._substance_instances - self._future_obj_instances - sampled_particle_entities
        if len(remaining_particle_entities) != 0:
            return f"Some systems do not have any particle condition defined for them in the initial conditions: " \
                   f"{', '.join(remaining_particle_entities)}"

    def _build_non_sampleable_object_scope(self):
        """
        Store simulator object options for non-sampleable objects in self.non_sampleable_object_scope
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
                obj_cat = self._object_instance_to_category[obj_inst]

                # We allow burners to be used as if they are stoves
                categories = OBJECT_TAXONOMY.get_subtree_igibson_categories(obj_cat)
                if obj_cat == "stove.n.01":
                    categories += OBJECT_TAXONOMY.get_subtree_igibson_categories("burner.n.02")

                for room_inst in og.sim.scene.seg_map.room_sem_name_to_ins_name[room_type]:
                    # A list of scene objects that satisfy the requested categories
                    room_objs = og.sim.scene.object_registry("in_rooms", room_inst, default_val=[])
                    scene_objs = [obj for obj in room_objs if obj.category in categories]

                    if len(scene_objs) != 0:
                        room_type_to_scene_objs[room_type][obj_inst][room_inst] = scene_objs

        error_msg = self._consolidate_room_instance(room_type_to_scene_objs, "initial_pre-sampling")
        if error_msg:
            return error_msg
        self._non_sampleable_object_scope = room_type_to_scene_objs

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
            dict: Filtered object scope
        """
        filtered_object_scope = {}
        for room_type in input_object_scope:
            filtered_object_scope[room_type] = {}
            for scene_obj in input_object_scope[room_type]:
                filtered_object_scope[room_type][scene_obj] = {}
                for room_inst in input_object_scope[room_type][scene_obj]:
                    # These are a list of candidate simulator objects that need sampling test
                    for obj in input_object_scope[room_type][scene_obj][room_inst]:
                        # Temporarily set object_scope to point to this candidate object
                        self._object_scope[scene_obj] = BDDLEntity(object_scope=scene_obj, entity=obj)

                        success = True
                        # If this candidate object is not involved in any conditions,
                        # success will be True by default and this object will qualify
                        for condition, positive in conditions:
                            # Sample positive kinematic conditions that involve this candidate object
                            if condition.STATE_NAME in KINEMATIC_STATES_BDDL and positive and scene_obj in condition.body:

                                success = condition.sample(binary_state=positive)
                                log_msg = " ".join(
                                    [
                                        "{} kinematic condition sampling".format(condition_type),
                                        room_type,
                                        scene_obj,
                                        room_inst,
                                        obj.name,
                                        condition.STATE_NAME,
                                        str(condition.body),
                                        str(success),
                                    ]
                                )
                                log.info(log_msg)

                                # If any condition fails for this candidate object, skip
                                if not success:
                                    break

                        # If this candidate object fails, move on to the next candidate object
                        if not success:
                            continue

                        if room_inst not in filtered_object_scope[room_type][scene_obj]:
                            filtered_object_scope[room_type][scene_obj][room_inst] = []
                        filtered_object_scope[room_type][scene_obj][room_inst].append(obj)

        return filtered_object_scope

    def _consolidate_room_instance(self, filtered_object_scope, condition_type):
        """
        Consolidates room instances

        Args:
            filtered_object_scope (dict): Filtered object scope
            condition_type (str): What type of condition to sample, e.g., "initial"
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
        for obj_cat in self._activity_conditions.parsed_objects:
            # Don't populate agent
            if obj_cat == "agent.n.01":
                continue
            # Don't populate synsets that can't be sampled
            if obj_cat in NON_SAMPLEABLE_SYNSETS:
                continue

            # Populate based on whether it's a substance or not
            if obj_cat in SUBSTANCE_SYNSET_MAPPING:
                assert len(self._activity_conditions.parsed_objects[obj_cat]) == 1, "Systems are singletons"
                obj_inst = self._activity_conditions.parsed_objects[obj_cat][0]
                self._object_scope[obj_inst] = BDDLEntity(
                    object_scope=obj_inst,
                    entity=None if obj_inst in self._future_obj_instances else get_system(SUBSTANCE_SYNSET_MAPPING[obj_cat]),
                )
            else:
                is_sliceable = OBJECT_TAXONOMY.has_ability(obj_cat, "sliceable")
                categories = OBJECT_TAXONOMY.get_subtree_igibson_categories(obj_cat)

                # TODO: temporary hack
                remove_categories = [
                    "pop_case",  # too large
                    "jewel",  # too small
                    "ring",  # too small
                ]
                for remove_category in remove_categories:
                    if remove_category in categories:
                        categories.remove(remove_category)

                for obj_inst in self._activity_conditions.parsed_objects[obj_cat]:
                    # Don't explicitly sample if future
                    if obj_inst in self._future_obj_instances:
                        self._object_scope[obj_inst] = BDDLEntity(object_scope=obj_inst)
                        continue

                    category = np.random.choice(categories)
                    # for sliceable objects, only get the whole objects
                    try:
                        model_choices = get_object_models_of_category(
                            category, filter_method="sliceable_whole" if is_sliceable else None
                        )
                    except:
                        return f"Missing object category: {category}"

                    if len(model_choices) == 0:
                        # restore back to the play state
                        return f"Missing valid object models for category: {category}"

                    # TODO: This no longer works because model ID changes in the new asset
                    # Filter object models if the object category is openable
                    # synset = OBJECT_TAXONOMY.get_class_name_from_igibson_category(category)
                    # if OBJECT_TAXONOMY.has_ability(synset, "openable"):
                    #     # Always use the articulated version of a certain object if its category is openable
                    #     # E.g. backpack, jar, etc
                    #     model_choices = [m for m in model_choices if "articulated_" in m]
                    #     if len(model_choices) == 0:
                    #         return "{} is Openable, but does not have articulated models.".format(category)

                    # Randomly select an object model
                    model = np.random.choice(model_choices)

                    # TODO: temporary hack no longer works because model ID changes in the new asset
                    # for "collecting aluminum cans", we need pop cans (not bottles)
                    # if category == "pop" and self.activity_name in ["collecting_aluminum_cans"]:
                    #     model = np.random.choice([str(i) for i in range(40, 46)])
                    # if category == "spoon" and self.activity_name in ["polishing_silver"]:
                    #     model = np.random.choice([str(i) for i in [2, 5, 6]])

                    # create the object
                    simulator_obj = DatasetObject(
                        name=f"{category}_{len(og.sim.scene.objects)}",
                        category=category,
                        model=model,
                        fit_avg_dim_volume=True,
                    )
                    num_new_obj += 1

                    # Load the object into the simulator
                    assert og.sim.scene.loaded, "Scene is not loaded"
                    og.sim.import_object(simulator_obj)

                    # Set these objects to be far-away locations
                    simulator_obj.set_position(np.array([100.0 + num_new_obj - 1, 100.0, -100.0]))

                    self._sampled_objects.add(simulator_obj)
                    self._object_scope[obj_inst] = BDDLEntity(object_scope=obj_inst, entity=simulator_obj)

    def _sample_initial_conditions(self):
        """
        Sample initial conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        error_msg, self._non_sampleable_object_scope_filtered_initial = self._sample_conditions(
            self._non_sampleable_object_scope, self._non_sampleable_object_conditions, "initial"
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
                self._non_sampleable_object_scope_filtered_initial, goal_condition_processed, "goal"
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
        for group in ("kinematic", "particle", "unary"):
            log.debug(f"Sampling {group} states...")
            if len(self._object_sampling_orders[group]) > 0:
                # # Pop non-sampleable objects
                # self._object_sampling_orders["kinematic"].pop(0)
                for cur_batch in self._object_sampling_orders[group]:
                    for condition, positive in self._object_sampling_conditions[group]:
                        # Sample conditions that involve the current batch of objects
                        if condition.body[0] in cur_batch:
                            num_trials = 10
                            for _ in range(num_trials):
                                success = condition.sample(binary_state=positive)
                                if success:
                                    break
                            if not success:
                                return "Sampleable object conditions failed: {} {}".format(
                                    condition.STATE_NAME, condition.body
                                )

        # Update all the objects' bddl object scopes
        for obj_scope, entity in self._object_scope.items():
            if entity.exists() and isinstance(entity.wrapped_obj, DatasetObject):
                entity.bddl_object_scope = obj_scope

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
        filtered_object_scope = self._filter_object_scope(input_object_scope, conditions, condition_type)
        error_msg = self._consolidate_room_instance(filtered_object_scope, condition_type)
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
                            self._object_scope[obj_inst] = BDDLEntity(object_scope=obj_inst, entity=obj)
                            log.debug((obj_inst, obj.name))
                    success = True
                    break
            if not success:
                return "{}: Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n".format(
                    condition_type, room_type, self._scene_model
                )
