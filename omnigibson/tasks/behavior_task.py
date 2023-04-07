import os
import numpy as np

import networkx as nx

import bddl
from bddl.activity import (
    Conditions,
    evaluate_goal_conditions,
    get_goal_conditions,
    get_ground_goal_state_options,
    get_initial_conditions,
    get_natural_goal_conditions,
    get_object_scope,
)
from bddl.condition_evaluation import Negation
from bddl.logic_base import AtomicFormula
from bddl.object_taxonomy import ObjectTaxonomy

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Pose
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.tasks.bddl_backend import OmniGibsonBDDLBackend
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.asset_utils import get_og_avg_category_specs, get_og_model_path, get_object_models_of_category
from omnigibson.utils.constants import (
    NON_SAMPLEABLE_OBJECTS,
    KINEMATICS_STATES,
    MACRO_PARTICLE_SYNSETS,
    WATER_SYNSETS,
    SYSTEM_SYNSETS_TO_SYSTEM_NAMES,
)
import omnigibson.utils.transform_utils as T
from omnigibson.utils.python_utils import classproperty, assert_valid_key
from omnigibson.systems import get_system
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class BehaviorTask(BaseTask):
    """
    Task for BEHAVIOR

    Args:
        activity_name (None or str): Name of the Behavior Task to instantiate
        activity_definition_id (int): Specification to load for the desired task. For a given Behavior Task, multiple task
            specifications can be used (i.e.: differing goal conditions, or "ways" to complete a given task). This
            ID determines which specification to use
        activity_instance_id (int): Specific pre-configured instance of a scene to load for this BehaviorTask. This
            will be used only if @online_object_sampling is False.
        predefined_problem (None or str): If specified, specifies the raw string definition of the Behavior Task to
            load. This will automatically override @activity_name and @activity_definition_id.
        online_object_sampling (bool): whether to sample object locations online at runtime or not
        debug_object_sampling (None or str): if specified, should be the object name to debug for placement functionality
        highlight_task_relevant_objects (bool): whether to overlay task-relevant objects in the scene with a colored mask
        termination_config (None or dict): Keyword-mapped configuration to use to generate termination conditions. This
            should be specific to the task class. Default is None, which corresponds to a default config being usd.
            Note that any keyword required by a specific task class but not specified in the config will automatically
            be filled in with the default config. See cls.default_termination_config for default values used
        reward_config (None or dict): Keyword-mapped configuration to use to generate reward functions. This should be
            specific to the task class. Default is None, which corresponds to a default config being usd. Note that
            any keyword required by a specific task class but not specified in the config will automatically be filled
            in with the default config. See cls.default_reward_config for default values used
    """
    def __init__(
            self,
            activity_name=None,
            activity_definition_id=0,
            activity_instance_id=0,
            predefined_problem=None,
            online_object_sampling=False,
            debug_object_sampling=None,
            highlight_task_relevant_objects=False,
            termination_config=None,
            reward_config=None,
    ):
        # Make sure task name is valid
        with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
            all_activities = {line.strip() for line in f.readlines()}
            assert_valid_key(key=activity_name, valid_keys=all_activities, name="Behavior Task")

        # Initialize relevant variables

        # BDDL
        self.backend = OmniGibsonBDDLBackend()

        # Activity info
        self.activity_name = None
        self.activity_definition_id = activity_definition_id
        self.activity_instance_id = activity_instance_id
        self.activity_conditions = None
        self.activity_initial_conditions = None
        self.activity_goal_conditions = None
        self.activity_natural_language_goal_conditions = None
        self.ground_goal_state_options = None
        self.instruction_order = None
        self.feedback = None
        self.scene_model = None

        # Object info
        self.object_taxonomy = ObjectTaxonomy()
        self.debug_object_sampling = debug_object_sampling
        self.online_object_sampling = online_object_sampling
        self.highlight_task_relevant_objs = highlight_task_relevant_objects
        self.object_scope = None
        self.object_instance_to_category = None
        self.room_type_to_object_instance = None
        self.non_sampleable_object_instances = None
        self.non_sampleable_object_scope = None
        self.non_sampleable_object_conditions = None
        self.non_sampleable_object_scope_filtered_initial = None
        self.object_sampling_orders = None
        self.sampled_objects = None
        self.sampleable_object_conditions = None

        # Logic-tracking info
        self.currently_viewed_index = None
        self.currently_viewed_instruction = None

        # Load the initial behavior configuration
        self.update_activity(activity_name=activity_name, activity_definition_id=activity_definition_id, predefined_problem=predefined_problem)

        # Run super init
        super().__init__(termination_config=termination_config, reward_config=reward_config)

    def _create_termination_conditions(self):
        # Initialize termination conditions dict and fill in with Timeout and PredicateGoal
        terminations = dict()

        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        terminations["predicate"] = PredicateGoal(goal_fcn=lambda: self.activity_goal_conditions)

        return terminations

    def _create_reward_functions(self):
        # Initialize reward functions dict and fill in with Potential reward
        rewards = dict()

        rewards["potential"] = PotentialReward(
            potential_fcn=self.get_potential,
            r_potential=self._reward_config["r_potential"],
        )

        return rewards

    def _load(self, env):
        # Get the name of the scene
        self.scene_model = og.sim.scene.scene_model

        # Initialize the current activity
        success, self.feedback = self.initialize_activity(env=env)
        assert success, f"Failed to initialize Behavior Activity. Feedback:\n{self.feedback}"

        # Highlight any task relevant objects if requested
        if self.highlight_task_relevant_objs:
            for obj_name, obj in self.object_scope.items():
                if isinstance(obj, BaseRobot):
                    continue
                obj.highlighted = True

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    def update_activity(self, activity_name, activity_definition_id, predefined_problem=None):
        """
        Update the active Behavior activity being deployed

        Args:
            activity_name (None or str): Name of the Behavior Task to instantiate
            activity_definition_id (int): Specification to load for the desired task. For a given Behavior Task, multiple task
                specifications can be used (i.e.: differing goal conditions, or "ways" to complete a given task). This
                ID determines which specification to use
            predefined_problem (None or str): If specified, specifies the raw string definition of the Behavior Task to
                load. This will automatically override @activity_name and @activity_definition_id.
        """
        # Update internal variables based on values

        # Activity info
        self.activity_name = activity_name
        self.activity_definition_id = activity_definition_id
        self.activity_conditions = Conditions(
            activity_name,
            activity_definition_id,
            simulator_name="igibson",       # TODO: Update!
            predefined_problem=predefined_problem,
        )

        # Object info
        self.object_scope = get_object_scope(self.activity_conditions)
        self.object_instance_to_category = {
            obj_inst: obj_cat
            for obj_cat in self.activity_conditions.parsed_objects
            for obj_inst in self.activity_conditions.parsed_objects[obj_cat]
        }

        # Generate initial and goal conditions
        self.activity_initial_conditions = get_initial_conditions(self.activity_conditions, self.backend, self.object_scope)
        self.activity_goal_conditions = get_goal_conditions(self.activity_conditions, self.backend, self.object_scope)
        self.ground_goal_state_options = get_ground_goal_state_options(
            self.activity_conditions, self.backend, self.object_scope, self.activity_goal_conditions
        )

        # Demo attributes
        self.instruction_order = np.arange(len(self.activity_conditions.parsed_goal_conditions))
        np.random.shuffle(self.instruction_order)
        self.currently_viewed_index = 0
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]
        self.activity_natural_language_goal_conditions = get_natural_goal_conditions(self.activity_conditions)

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        Args:
            env (Environment): Current active environment instance

        Returns:
            float: Computed potential
        """
        # Evaluate the first ground goal state option as the potential
        _, satisfied_predicates = evaluate_goal_conditions(self.ground_goal_state_options[0])
        success_score = len(satisfied_predicates["satisfied"]) / (
            len(satisfied_predicates["satisfied"]) + len(satisfied_predicates["unsatisfied"])
        )
        return -success_score

    def initialize_activity(self, env):
        """
        Initializes the desired activity in the current environment @env

        Args:
            env (Environment): Current active environment instance

        Returns:
            2-tuple:
                - bool: Whether the generated scene activity should be accepted or not
                - dict: Any feedback from the sampling / initialization process
        """
        accept_scene = True
        feedback = None

        if self.online_object_sampling:
            # Reject scenes with missing non-sampleable objects
            # Populate object_scope with sampleable objects and the robot
            accept_scene, feedback = self.check_scene(env)
            if not accept_scene:
                return accept_scene, feedback
            # Sample objects to satisfy initial conditions
            accept_scene, feedback = self.sample(env)
            if not accept_scene:
                return accept_scene, feedback
        else:
            # Load existing scene cache and assign object scope accordingly
            self.assign_object_scope_with_cache(env)

        # Generate goal condition with the fully populated self.object_scope
        self.activity_goal_conditions = get_goal_conditions(self.activity_conditions, self.backend, self.object_scope)
        self.ground_goal_state_options = get_ground_goal_state_options(
            self.activity_conditions, self.backend, self.object_scope, self.activity_goal_conditions
        )
        return accept_scene, feedback

    def parse_non_sampleable_object_room_assignment(self, env):
        """
        Infers which rooms each object is assigned to

        Args:
            env (Environment): Current active environment instance
        """
        self.room_type_to_object_instance = dict()
        self.non_sampleable_object_instances = set()
        for cond in self.activity_conditions.parsed_initial_conditions:
            if cond[0] == "inroom":
                obj_inst, room_type = cond[1], cond[2]
                obj_cat = self.object_instance_to_category[obj_inst]
                if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                    # Invalid room assignment
                    return "You have assigned room type for [{}], but [{}] is sampleable. Only non-sampleable objects can have room assignment.".format(
                        obj_cat, obj_cat
                    )
                if room_type not in og.sim.scene.seg_map.room_sem_name_to_ins_name:
                    # Missing room type
                    return "Room type [{}] missing in scene [{}].".format(room_type, og.sim.scene.scene_model)
                if room_type not in self.room_type_to_object_instance:
                    self.room_type_to_object_instance[room_type] = []
                self.room_type_to_object_instance[room_type].append(obj_inst)

                if obj_inst in self.non_sampleable_object_instances:
                    # Duplicate room assignment
                    return "Object [{}] has more than one room assignment".format(obj_inst)

                self.non_sampleable_object_instances.add(obj_inst)

        for obj_cat in self.activity_conditions.parsed_objects:
            if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                continue
            for obj_inst in self.activity_conditions.parsed_objects[obj_cat]:
                if obj_inst not in self.non_sampleable_object_instances:
                    # Missing room assignment
                    return "All non-sampleable objects should have room assignment. [{}] does not have one.".format(
                        obj_inst
                    )

    def build_sampling_order(self, env):
        """
        Sampling orders is a list of lists: [[batch_1_inst_1, ... batch_1_inst_N], [batch_2_inst_1, batch_2_inst_M], ...]
        Sampling should happen for batch 1 first, then batch 2, so on and so forth
        Example: OnTop(plate, table) should belong to batch 1, and OnTop(apple, plate) should belong to batch 2

        Args:
            env (Environment): Current active environment instance
        """
        self.object_sampling_orders = []
        cur_batch = self.non_sampleable_object_instances
        while len(cur_batch) > 0:
            self.object_sampling_orders.append(cur_batch)
            next_batch = set()
            for cond in self.activity_conditions.parsed_initial_conditions:
                if len(cond) == 3 and cond[2] in cur_batch:
                    next_batch.add(cond[1])
            cur_batch = next_batch

        if len(self.object_sampling_orders) > 0:
            remaining_objs = self.object_scope.keys() - set.union(*self.object_sampling_orders)
        else:
            remaining_objs = self.object_scope.keys()

        # Macro particles and water don't need initial conditions
        remaining_objs = {obj_inst for obj_inst in remaining_objs
                          if self.object_instance_to_category[obj_inst] not in MACRO_PARTICLE_SYNSETS.union(WATER_SYNSETS)}
        if len(remaining_objs) != 0:
            return "Some objects do not have any kinematic condition defined for them in the initial conditions: {}".format(
                ", ".join(remaining_objs)
            )

    def build_non_sampleable_object_scope(self, env):
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

        Args:
            env (Environment): Current active environment instance
        """
        room_type_to_scene_objs = {}
        for room_type in self.room_type_to_object_instance:
            room_type_to_scene_objs[room_type] = {}
            for obj_inst in self.room_type_to_object_instance[room_type]:
                room_type_to_scene_objs[room_type][obj_inst] = {}
                obj_cat = self.object_instance_to_category[obj_inst]

                # We allow burners to be used as if they are stoves
                categories = self.object_taxonomy.get_subtree_igibson_categories(obj_cat)
                if obj_cat == "stove.n.01":
                    categories += self.object_taxonomy.get_subtree_igibson_categories("burner.n.02")

                for room_inst in og.sim.scene.seg_map.room_sem_name_to_ins_name[room_type]:
                    # A list of scene objects that satisfy the requested categories
                    room_objs = og.sim.scene.object_registry("in_rooms", room_inst, default_val=[])
                    scene_objs = [obj for obj in room_objs if obj.category in categories]

                    if len(scene_objs) != 0:
                        room_type_to_scene_objs[room_type][obj_inst][room_inst] = scene_objs

        error_msg = self.consolidate_room_instance(room_type_to_scene_objs, "initial_pre-sampling")
        if error_msg:
            return error_msg
        self.non_sampleable_object_scope = room_type_to_scene_objs

    def import_sampleable_objects(self, env):
        """
        Import all objects that can be sampled

        Args:
            env (Environment): Current active environment instance
        """
        assert og.sim.is_stopped(), "Simulator should be stopped when importing sampleable objects"

        # Move the robot object frame to a far away location, similar to other newly imported objects below
        env.robots[0].set_position_orientation([300, 300, 300], [0, 0, 0, 1])

        self.sampled_objects = set()
        num_new_obj = 0
        # Only populate self.object_scope for sampleable objects
        avg_category_spec = get_og_avg_category_specs()
        for obj_cat in self.activity_conditions.parsed_objects:
            if obj_cat == "agent.n.01":
                continue
            if obj_cat in NON_SAMPLEABLE_OBJECTS:
                continue
            if obj_cat in SYSTEM_SYNSETS_TO_SYSTEM_NAMES:
                assert len(self.activity_conditions.parsed_objects[obj_cat]) == 1, "Systems are singletons"
                obj_inst = self.activity_conditions.parsed_objects[obj_cat][0]
                self.object_scope[obj_inst] = get_system(SYSTEM_SYNSETS_TO_SYSTEM_NAMES[obj_cat])
                continue

            is_sliceable = self.object_taxonomy.has_ability(obj_cat, "sliceable")
            categories = self.object_taxonomy.get_subtree_igibson_categories(obj_cat)

            # TODO: temporary hack
            remove_categories = [
                "pop_case",  # too large
                "jewel",  # too small
                "ring",  # too small
            ]
            for remove_category in remove_categories:
                if remove_category in categories:
                    categories.remove(remove_category)

            for obj_inst in self.activity_conditions.parsed_objects[obj_cat]:
                category = np.random.choice(categories)
                # for sliceable objects, only get the whole objects
                try:
                    model_choices = get_object_models_of_category(
                        category, filter_method="sliceable_whole" if is_sliceable else None
                    )
                except:
                    og.sim.play()
                    return f"Missing object category: {category}"

                if len(model_choices) == 0:
                    # restore back to the play state
                    og.sim.play()
                    return f"Missing valid object models for category: {category}"

                # TODO: This no longer works because model ID changes in the new asset
                # Filter object models if the object category is openable
                # synset = self.object_taxonomy.get_class_name_from_igibson_category(category)
                # if self.object_taxonomy.has_ability(synset, "openable"):
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

                model_path = get_og_model_path(category, model)
                usd_path = os.path.join(model_path, "usd", f"{model}.usd")
                obj_name = "{}_{}".format(category, len(og.sim.scene.objects))

                # create the object
                simulator_obj = DatasetObject(
                    prim_path=f"/World/{obj_name}",
                    usd_path=usd_path,
                    name=obj_name,
                    category=category,
                    fit_avg_dim_volume=True,
                )
                num_new_obj += 1

                # Load the object into the simulator
                assert og.sim.scene.loaded, "Scene is not loaded"
                og.sim.import_object(simulator_obj)

                # Set these objects to be far-away locations
                simulator_obj.set_position(np.array([100.0 + num_new_obj - 1, 100.0, -100.0]))

                self.sampled_objects.add(simulator_obj)
                self.object_scope[obj_inst] = simulator_obj

    def check_scene(self, env):
        """
        Runs sanity checks for the current scene for the given BEHAVIOR task

        Args:
            env (Environment): Current active environment instance

        Returns:
            2-tuple:
                - bool: Whether the generated scene activity should be accepted or not
                - dict: Any feedback from the sampling / initialization process
        """
        error_msg = self.parse_non_sampleable_object_room_assignment(env)
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self.build_sampling_order(env)
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self.build_non_sampleable_object_scope(env)
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self.import_sampleable_objects(env)
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        self.object_scope["agent.n.01_1"] = self.get_agent(env)

        return True, None

    def get_agent(self, env):
        """
        Grab the 0th agent from @env

        Args:
            env (Environment): Current active environment instance

        Returns:
            BaseRobot: The 0th robot from the environment instance
        """
        # We assume the relevant agent is the first agent in the scene
        return env.robots[0]

    def assign_object_scope_with_cache(self, env):
        """
        Assigns objects within the current object scope

        Args:
            env (Environment): Current active environment instance
        """
        # Assign object_scope based on a cached scene
        for obj_inst in self.object_scope:
            matched_sim_obj = None
            # If the object scope points to the agent
            if obj_inst == "agent.n.01_1":
                matched_sim_obj = self.get_agent(env)
            # If the object scope points to a system
            elif self.object_instance_to_category[obj_inst] in SYSTEM_SYNSETS_TO_SYSTEM_NAMES:
                matched_sim_obj = get_system(SYSTEM_SYNSETS_TO_SYSTEM_NAMES[self.object_instance_to_category[obj_inst]])
            else:
                log.debug(f"checking objects...")
                for sim_obj in og.sim.scene.objects:
                    log.debug(f"checking bddl obj scope for obj: {sim_obj.name}")
                    if hasattr(sim_obj, "bddl_object_scope") and sim_obj.bddl_object_scope == obj_inst:
                        matched_sim_obj = sim_obj
                        break
            assert matched_sim_obj is not None, obj_inst
            self.object_scope[obj_inst] = matched_sim_obj

    def process_single_condition(self, condition):
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

    def group_initial_conditions(self):
        """
        We group initial conditions by first splitting the desired task-relevant objects into non-sampleable objects
        and sampleable objects.

        Non-sampleable objects are objects that should ALREADY be in the scene, and should NOT be generated / sampled
        on the fly

        Sampleable objects are objects that need to be additionally imported into the scene.

        Returns:
            None or str: None if successful, otherwise failure string
        """
        self.non_sampleable_object_conditions = []
        self.sampleable_object_conditions = []

        # TODO: currently we assume self.initial_conditions is a list of
        # bddl.condition_evaluation.HEAD, each with one child.
        # This child is either a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate or
        # a Negation of a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate
        for condition in self.activity_initial_conditions:
            condition, positive = self.process_single_condition(condition)
            if condition is None:
                continue

            # Sampled conditions must always be positive
            # Non-positive (e.g.: NOT onTop) is not restrictive enough for sampling
            if condition.STATE_NAME in KINEMATICS_STATES and not positive:
                return "Initial condition has negative kinematic conditions: {}".format(condition.body)

            condition_body = set(condition.body)

            # If the condition involves any non-sampleable object (e.g.: furniture), it's a non-sampleable condition
            # This means that there's no ordering constraint in terms of sampling, because we know the, e.g., furniture
            # object already exists in the scene and is placed, so these specific conditions can be sampled without
            # any dependencies
            if len(self.non_sampleable_object_instances.intersection(condition_body)) > 0:
                self.non_sampleable_object_conditions.append((condition, positive))
            else:
                # There are dependencies that must be taken into account
                self.sampleable_object_conditions.append((condition, positive))

    def filter_object_scope(self, input_object_scope, conditions, condition_type):
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
                        self.object_scope[scene_obj] = obj

                        success = True
                        # If this candidate object is not involved in any conditions,
                        # success will be True by default and this object will qualify
                        for condition, positive in conditions:
                            # Sample positive kinematic conditions that involve this candidate object
                            if condition.STATE_NAME in KINEMATICS_STATES and positive and scene_obj in condition.body:
                                # Use pybullet GUI for debugging
                                if self.debug_object_sampling is not None and self.debug_object_sampling == condition.body[0]:
                                    gm.DEBUG = True

                                success = condition.sample(binary_state=positive)
                                log_msg = " ".join(
                                    [
                                        "{} condition sampling".format(condition_type),
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

    def consolidate_room_instance(self, filtered_object_scope, condition_type):
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
                    condition_type, room_type, self.scene_model
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

    def maximum_bipartite_matching(self, filtered_object_scope, condition_type):
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
                            self.object_scope[obj_inst] = obj
                            log.debug((obj_inst, obj.name))
                    success = True
                    break
            if not success:
                return "{}: Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n".format(
                    condition_type, room_type, self.scene_model
                )

    def sample_conditions(self, input_object_scope, conditions, condition_type):
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
        filtered_object_scope = self.filter_object_scope(input_object_scope, conditions, condition_type)
        error_msg = self.consolidate_room_instance(filtered_object_scope, condition_type)
        if error_msg:
            return error_msg, None
        return self.maximum_bipartite_matching(filtered_object_scope, condition_type), filtered_object_scope

    def sample_initial_conditions(self):
        """
        Sample initial conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        error_msg, self.non_sampleable_object_scope_filtered_initial = self.sample_conditions(
            self.non_sampleable_object_scope, self.non_sampleable_object_conditions, "initial"
        )
        return error_msg

    def sample_goal_conditions(self):
        """
        Sample goal conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        np.random.shuffle(self.ground_goal_state_options)
        log.debug(("number of ground_goal_state_options", len(self.ground_goal_state_options)))
        num_goal_condition_set_to_test = 10

        goal_condition_success = False
        # Try to fulfill different set of ground goal conditions (maximum num_goal_condition_set_to_test)
        for goal_condition_set in self.ground_goal_state_options[:num_goal_condition_set_to_test]:
            goal_condition_processed = []
            for condition in goal_condition_set:
                condition, positive = self.process_single_condition(condition)
                if condition is None:
                    continue
                goal_condition_processed.append((condition, positive))

            error_msg, _ = self.sample_conditions(
                self.non_sampleable_object_scope_filtered_initial, goal_condition_processed, "goal"
            )
            if not error_msg:
                # if one set of goal conditions (and initial conditions) are satisfied, sampling is successful
                goal_condition_success = True
                break

        if not goal_condition_success:
            return error_msg

    def sample_initial_conditions_final(self):
        """
        Sample final initial conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        # Do the final round of sampling with object scope fixed
        for condition, positive in self.non_sampleable_object_conditions:
            num_trials = 10
            for _ in range(num_trials):
                success = condition.sample(binary_state=positive)
                if success:
                    break
            if not success:
                error_msg = "Non-sampleable object conditions failed even after successful matching: {}".format(
                    condition.body
                )
                return error_msg

        if len(self.object_sampling_orders) > 0:
            # Pop non-sampleable objects
            self.object_sampling_orders.pop(0)
            for cur_batch in self.object_sampling_orders:
                # First sample non-sliced conditions
                for condition, positive in self.sampleable_object_conditions:
                    if condition.STATE_NAME == "sliced":
                        continue
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

                # Then sample sliced conditions
                for condition, positive in self.sampleable_object_conditions:
                    if condition.STATE_NAME != "sliced":
                        continue
                    # Sample conditions that involve the current batch of objects
                    if condition.body[0] in cur_batch:
                        success = condition.sample(binary_state=positive)
                        if not success:
                            return "Sampleable object conditions failed: {}".format(condition.body)

        # One more sim step to make sure the object states are propagated correctly
        # E.g. after sampling Filled.set_value(True), Filled.get_value() will become True only after one step
        og.sim.step()

    def sample(self, env, validate_goal=False):
        """
        Run sampling for this BEHAVIOR task

        Args:
            env (Environment): Current active environment instance
            validate_goal (bool): Whether the goal should be validated or not

        Returns:
            2-tuple:
                - bool: Whether sampling was successful or not
                - None or str: None if successful, otherwise the associated error message
        """
        # Auto-initialize all sampleable objects
        og.sim.play()
        env.scene.reset()

        error_msg = self.group_initial_conditions()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self.sample_initial_conditions()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        if validate_goal:
            error_msg = self.sample_goal_conditions()
            if error_msg:
                log.error(error_msg)
                return False, error_msg

        error_msg = self.sample_initial_conditions_final()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        env.scene.update_initial_state()
        og.sim.stop()

        return True, None

    def _get_obs(self, env):
        low_dim_obs = dict()
        low_dim_obs["robot_pos"] = np.array(env.robots[0].get_position())
        low_dim_obs["robot_ori_cos"] = np.cos(env.robots[0].get_rpy())
        low_dim_obs["robot_ori_sin"] = np.sin(env.robots[0].get_rpy())

        # Batch rpy calculations for much better efficiency
        objs_rpy = T.quat2euler(np.array([v.states[Pose].get_value()[1] for v in self.object_scope.values()]))

        i = 0
        for idx, v in enumerate(self.object_scope.values()):
            # TODO: May need to update checking here to USDObject? Or even baseobject?
            if isinstance(v, DatasetObject):
                low_dim_obs[f"obj_{i}_valid"] = np.array([1.0])
                low_dim_obs[f"obj_{i}_pos"] = v.states[Pose].get_value()[0]
                low_dim_obs[f"obj_{i}_ori_cos"] = np.cos(objs_rpy[idx])
                low_dim_obs[f"obj_{i}_ori_sin"] = np.sin(objs_rpy[idx])
                for arm in env.robots[0].arm_names:
                    grasping_object = env.robots[0].is_grasping(arm=arm, candidate_obj=v)
                    low_dim_obs[f"obj_{i}_pos_in_gripper_{arm}"] = np.array([float(grasping_object)])
                i += 1

        return low_dim_obs, dict()

    def _step_termination(self, env, action, info=None):
        # Run super first
        done, info = super()._step_termination(env=env, action=action, info=info)

        # Add additional info
        info["goal_status"] = self._termination_conditions["predicate"].goal_status

        return done, info

    def show_instruction(self):
        """
        Get current instruction for user

        Returns:
            3-tuple:
                - str: Current goal condition in natural language
                - 3-tuple: (R,G,B) color to assign to text
                - list of BaseObject: Relevant objects for the current instruction
        """
        satisfied = self.currently_viewed_instruction in self._termination_conditions["predicate"].goal_status["satisfied"]
        natural_language_condition = self.activity_natural_language_goal_conditions[self.currently_viewed_instruction]
        objects = self.activity_goal_conditions[self.currently_viewed_instruction].get_relevant_objects()
        text_color = (
            [83.0 / 255.0, 176.0 / 255.0, 72.0 / 255.0] if satisfied else [255.0 / 255.0, 51.0 / 255.0, 51.0 / 255.0]
        )

        return natural_language_condition, text_color, objects

    def iterate_instruction(self):
        """
        Increment the instruction
        """
        self.currently_viewed_index = (self.currently_viewed_index + 1) % len(self.activity_conditions.parsed_goal_conditions)
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]

    @property
    def name(self):
        """
        Returns:
            str: Name of this task. Defaults to class name
        """
        name_base = super().name

        # Add activity name, def id, and inst id
        return f"{name_base}_{self.activity_name}_{self.activity_definition_id}_{self.activity_instance_id}"

    @classproperty
    def valid_scene_types(cls):
        # Must be an interactive traversable scene
        return {InteractiveTraversableScene}

    @classproperty
    def default_termination_config(cls):
        return {
            "max_steps": 500,
        }

    @classproperty
    def default_reward_config(cls):
        return {
            "r_potential": 1.0,
        }
