import datetime
import logging
from collections import OrderedDict
import os

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

import igibson
# from igibson.external.pybullet_tools.utils import *
from igibson.object_states.on_floor import RoomFloor
from igibson.objects.usd_object import URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.robots.robot_base import BaseRobot
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.simulator import Simulator
from igibson.tasks.bddl_backend import IGibsonBDDLBackend
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.predicate_goal import PredicateGoal
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_model_path, get_object_models_of_category
from igibson.utils.checkpoint_utils import load_checkpoint
from igibson.utils.constants import (
    AGENT_POSE_DIM,
    FLOOR_SYNSET,
    MAX_TASK_RELEVANT_OBJS,
    NON_SAMPLEABLE_OBJECTS,
    TASK_RELEVANT_OBJS_OBS_DIM,
)
from igibson.utils.ig_logging import IGLogWriter
from igibson.utils.python_utils import classproperty, assert_valid_key

KINEMATICS_STATES = frozenset({"inside", "ontop", "under", "onfloor"})


# TODO! WIP
class BehaviorTask(BaseTask):
    """
    Task for BEHAVIOR

    Args:

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
            task_name=None,
            task_id=0,
            predefined_problem=None,
            load_clutter=True,
            online_sampling=False,
            termination_config=None,
            reward_config=None,
    ):
        # Make sure task name is valid
        with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
            all_activities = {line.strip() for line in f.readlines()}
            assert_valid_key(key=task_name, valid_keys=all_activities, name="behavior task")

        super(BehaviorTask, self).__init__(env)
        self.scene = env.scene
        self.termination_conditions = [
            Timeout(self.config),
            PredicateGoal(self.config),
        ]
        self.reward_functions = [
            PotentialReward(self.config),
        ]

        self.object_taxonomy = ObjectTaxonomy()
        self.backend = IGibsonBDDLBackend()
        self.update_problem(
            behavior_activity=self.config.get("task", None),
            activity_definition=self.config.get("task_id", 0),
            predefined_problem=self.config.get("predefined_problem", None),
        )
        self.load_clutter = self.config.get("load_clutter", True)
        self.debug_obj_inst = self.config.get("debug_obj_inst", None)
        self.online_sampling = self.config.get("online_sampling", False)
        self.reset_checkpoint_idx = self.config.get("reset_checkpoint_idx", -1)
        self.reset_checkpoint_dir = self.config.get("reset_checkpoint_dir", None)
        self.task_obs_dim = MAX_TASK_RELEVANT_OBJS * TASK_RELEVANT_OBJS_OBS_DIM + AGENT_POSE_DIM

        self.initialized, self.feedback = self.initialize(env)
        self.state_history = {}
        self.initial_state = self.save_scene(env)
        if self.config.get("should_highlight_task_relevant_objs", True):
            self.highlight_task_relevant_objs(env)

        self.episode_save_dir = self.config.get("episode_save_dir", None)
        if self.episode_save_dir is not None:
            os.makedirs(self.episode_save_dir, exist_ok=True)
        self.log_writer = None

    def highlight_task_relevant_objs(self, env):
        for obj_name, obj in self.object_scope.items():
            if isinstance(obj, BaseRobot) or isinstance(obj, RoomFloor):
                continue
            obj.highlight()

    def update_problem(self, behavior_activity, activity_definition, predefined_problem=None):
        self.behavior_activity = behavior_activity
        self.activity_definition = activity_definition
        self.conds = Conditions(
            behavior_activity, activity_definition, simulator_name="igibson", predefined_problem=predefined_problem
        )
        self.object_scope = get_object_scope(self.conds)
        self.obj_inst_to_obj_cat = {
            obj_inst: obj_cat
            for obj_cat in self.conds.parsed_objects
            for obj_inst in self.conds.parsed_objects[obj_cat]
        }

        # Generate initial and goal conditions
        self.initial_conditions = get_initial_conditions(self.conds, self.backend, self.object_scope)
        self.goal_conditions = get_goal_conditions(self.conds, self.backend, self.object_scope)
        self.ground_goal_state_options = get_ground_goal_state_options(
            self.conds, self.backend, self.object_scope, self.goal_conditions
        )

        # Demo attributes
        self.instruction_order = np.arange(len(self.conds.parsed_goal_conditions))
        np.random.shuffle(self.instruction_order)
        self.currently_viewed_index = 0
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]
        self.current_success = False
        self.current_goal_status = {"satisfied": [], "unsatisfied": []}
        self.natural_language_goal_conditions = get_natural_goal_conditions(self.conds)

    def get_potential(self, env):
        # Evaluate the first ground goal state option as the potential
        _, satisfied_predicates = evaluate_goal_conditions(self.ground_goal_state_options[0])
        success_score = len(satisfied_predicates["satisfied"]) / (
            len(satisfied_predicates["satisfied"]) + len(satisfied_predicates["unsatisfied"])
        )
        return -success_score

    def save_scene(self, env):
        scene_tree, snapshot_id = self.scene.save(pybullet_save_state=True)
        self.state_history[snapshot_id] = scene_tree
        return snapshot_id

    def reset_scene(self, env):
        if self.reset_checkpoint_dir is not None and self.reset_checkpoint_idx != -1:
            load_checkpoint(env.simulator, self.reset_checkpoint_dir, self.reset_checkpoint_idx)
        else:
            self.scene.restore(scene_tree=self.state_history[self.initial_state], pybullet_state_id=self.initial_state)

    def reset_agent(self, env):
        return

    def reset_variables(self, env):
        if self.log_writer is not None:
            self.log_writer.end_log_session()
            del self.log_writer
            self.log_writer = None

        if self.episode_save_dir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            task = self.config["task"]
            task_id = self.config["task_id"]
            scene = self.config["scene_id"]
            vr_log_path = os.path.join(
                self.episode_save_dir,
                "{}_{}_{}_{}_{}.hdf5".format(
                    self.behavior_activity,
                    self.activity_definition,
                    env.scene.scene_id,
                    timestamp,
                    env.current_episode,
                ),
            )
            self.log_writer = IGLogWriter(
                env.simulator,
                frames_before_write=200,
                log_filepath=vr_log_path,
                task=self,
                store_vr=False,
                vr_robot=env.robots[0],
                filter_objects=True,
            )
            self.log_writer.set_up_data_storage()

    def step(self, env):
        if self.log_writer is not None:
            self.log_writer.process_frame()

    def initialize(self, env):
        accept_scene = True
        feedback = None

        if self.online_sampling:
            # Reject scenes with missing non-sampleable objects
            # Populate object_scope with sampleable objects and the robot
            accept_scene, feedback = self.check_scene(env)
            if not accept_scene:
                return accept_scene, feedback
            # Sample objects to satisfy initial conditions
            accept_scene, feedback = self.sample(env)
            if not accept_scene:
                return accept_scene, feedback

            if self.load_clutter:
                # Add clutter objects into the scenes
                self.clutter_scene(env)
        else:
            # Load existing scene cache and assign object scope accordingly
            self.assign_object_scope_with_cache(env)

        # Generate goal condition with the fully populated self.object_scope
        self.goal_conditions = get_goal_conditions(self.conds, self.backend, self.object_scope)
        self.ground_goal_state_options = get_ground_goal_state_options(
            self.conds, self.backend, self.object_scope, self.goal_conditions
        )
        return accept_scene, feedback

    def parse_non_sampleable_object_room_assignment(self):
        self.room_type_to_obj_inst = {}
        self.non_sampleable_object_inst = set()
        for cond in self.conds.parsed_initial_conditions:
            if cond[0] == "inroom":
                obj_inst, room_type = cond[1], cond[2]
                obj_cat = self.obj_inst_to_obj_cat[obj_inst]
                if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                    # Invalid room assignment
                    return "You have assigned room type for [{}], but [{}] is sampleable. Only non-sampleable objects can have room assignment.".format(
                        obj_cat, obj_cat
                    )
                if room_type not in self.scene.room_sem_name_to_ins_name:
                    # Missing room type
                    return "Room type [{}] missing in scene [{}].".format(room_type, self.scene.scene_id)
                if room_type not in self.room_type_to_obj_inst:
                    self.room_type_to_obj_inst[room_type] = []
                self.room_type_to_obj_inst[room_type].append(obj_inst)

                if obj_inst in self.non_sampleable_object_inst:
                    # Duplicate room assignment
                    return "Object [{}] has more than one room assignment".format(obj_inst)

                self.non_sampleable_object_inst.add(obj_inst)

        for obj_cat in self.conds.parsed_objects:
            if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                continue
            for obj_inst in self.conds.parsed_objects[obj_cat]:
                if obj_inst not in self.non_sampleable_object_inst:
                    # Missing room assignment
                    return "All non-sampleable objects should have room assignment. [{}] does not have one.".format(
                        obj_inst
                    )

    def build_sampling_order(self):
        """
        Sampling orders is a list of lists: [[batch_1_inst_1, ... batch_1_inst_N], [batch_2_inst_1, batch_2_inst_M], ...]
        Sampling should happen for batch 1 first, then batch 2, so on and so forth
        Example: OnTop(plate, table) should belong to batch 1, and OnTop(apple, plate) should belong to batch 2
        """
        self.sampling_orders = []
        cur_batch = self.non_sampleable_object_inst
        while len(cur_batch) > 0:
            self.sampling_orders.append(cur_batch)
            next_batch = set()
            for cond in self.conds.parsed_initial_conditions:
                if len(cond) == 3 and cond[2] in cur_batch:
                    next_batch.add(cond[1])
            cur_batch = next_batch

        if len(self.sampling_orders) > 0:
            remaining_objs = self.object_scope.keys() - set.union(*self.sampling_orders)
        else:
            remaining_objs = self.object_scope.keys()

        if len(remaining_objs) != 0:
            return "Some objects do not have any kinematic condition defined for them in the initial conditions: {}".format(
                ", ".join(remaining_objs)
            )

    def build_non_sampleable_object_scope(self):
        """
        Store simulator object options for non-sampleable objects in self.non_sampleable_object_scope
        {
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
        """
        room_type_to_scene_objs = {}
        for room_type in self.room_type_to_obj_inst:
            room_type_to_scene_objs[room_type] = {}
            for obj_inst in self.room_type_to_obj_inst[room_type]:
                room_type_to_scene_objs[room_type][obj_inst] = {}
                obj_cat = self.obj_inst_to_obj_cat[obj_inst]

                # We allow burners to be used as if they are stoves
                categories = self.object_taxonomy.get_subtree_igibson_categories(obj_cat)
                if obj_cat == "stove.n.01":
                    categories += self.object_taxonomy.get_subtree_igibson_categories("burner.n.02")

                for room_inst in self.scene.room_sem_name_to_ins_name[room_type]:
                    if obj_cat == FLOOR_SYNSET:
                        # TODO: remove after split floors
                        # Create a RoomFloor for each room instance
                        # This object is NOT imported by the simulator
                        room_floor = RoomFloor(
                            category="room_floor",
                            name="room_floor_{}".format(room_inst),
                            scene=self.scene,
                            room_instance=room_inst,
                            floor_obj=self.scene.objects_by_name["floors"],
                        )
                        scene_objs = [room_floor]
                    else:
                        room_objs = []
                        if room_inst in self.scene.objects_by_room:
                            room_objs = self.scene.objects_by_room[room_inst]
                        # A list of scene objects that satisfy the requested categories
                        scene_objs = [obj for obj in room_objs if obj.category in categories]

                    if len(scene_objs) != 0:
                        room_type_to_scene_objs[room_type][obj_inst][room_inst] = scene_objs

        error_msg = self.consolidate_room_instance(room_type_to_scene_objs, "initial_pre-sampling")
        if error_msg:
            return error_msg
        self.non_sampleable_object_scope = room_type_to_scene_objs

    def import_sampleable_objects(self, env):
        self.newly_added_objects = set()
        num_new_obj = 0
        # Only populate self.object_scope for sampleable objects
        avg_category_spec = get_ig_avg_category_specs()
        for obj_cat in self.conds.parsed_objects:
            if obj_cat == "agent.n.01":
                continue
            if obj_cat in NON_SAMPLEABLE_OBJECTS:
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

            for obj_inst in self.conds.parsed_objects[obj_cat]:
                category = np.random.choice(categories)
                # for sliceable objects, only get the whole objects
                model_choices = get_object_models_of_category(
                    category, filter_method="sliceable_whole" if is_sliceable else None
                )

                # Filter object models if the object category is openable
                synset = self.object_taxonomy.get_class_name_from_igibson_category(category)
                if self.object_taxonomy.has_ability(synset, "openable"):
                    # Always use the articulated version of a certain object if its category is openable
                    # E.g. backpack, jar, etc
                    model_choices = [m for m in model_choices if "articulated_" in m]
                    if len(model_choices) == 0:
                        return "{} is Openable, but does not have articulated models.".format(category)

                # Randomly select an object model
                model = np.random.choice(model_choices)

                # TODO: temporary hack
                # for "collecting aluminum cans", we need pop cans (not bottles)
                if category == "pop" and self.behavior_activity in ["collecting_aluminum_cans"]:
                    model = np.random.choice([str(i) for i in range(40, 46)])
                if category == "spoon" and self.behavior_activity in ["polishing_silver"]:
                    model = np.random.choice([str(i) for i in [2, 5, 6]])

                model_path = get_ig_model_path(category, model)
                filename = os.path.join(model_path, model + ".urdf")
                obj_name = "{}_{}".format(category, len(self.scene.objects_by_name))

                new_urdf_objs = []

                # create the object
                simulator_obj = URDFObject(
                    filename,
                    name=obj_name,
                    category=category,
                    model_path=model_path,
                    avg_obj_dims=avg_category_spec.get(category),
                    fit_avg_dim_volume=True,
                    texture_randomization=False,
                    overwrite_inertial=True,
                )
                new_urdf_objs.append(simulator_obj)
                num_new_obj += 1

                if is_sliceable:
                    whole_object = simulator_obj
                    object_parts = []
                    assert "object_parts" in simulator_obj.metadata, "object_parts not found in metadata: [{}]".format(
                        model_path
                    )

                    for i, part in enumerate(simulator_obj.metadata["object_parts"]):
                        category = part["category"]
                        model = part["model"]
                        # Scale the offset accordingly
                        part_pos = part["pos"] * whole_object.scale
                        part_orn = part["orn"]
                        model_path = get_ig_model_path(category, model)
                        filename = os.path.join(model_path, model + ".urdf")
                        obj_name = whole_object.name + "_part_{}".format(i)
                        # create the object parts (half objects) and set the initial position to be far-away
                        simulator_obj_part = URDFObject(
                            filename,
                            name=obj_name,
                            category=category,
                            model_path=model_path,
                            avg_obj_dims=avg_category_spec.get(category),
                            fit_avg_dim_volume=False,
                            scale=whole_object.scale,
                            texture_randomization=False,
                            overwrite_inertial=True,
                        )
                        new_urdf_objs.append(simulator_obj_part)
                        num_new_obj += 1
                        object_parts.append((simulator_obj_part, (part_pos, part_orn)))

                    assert len(object_parts) > 0
                    grouped_obj_parts = ObjectGrouper(object_parts)
                    simulator_obj = ObjectMultiplexer(
                        whole_object.name + "_multiplexer", [whole_object, grouped_obj_parts], 0
                    )

                # Load the object into the simulator
                assert self.scene.loaded, "Scene is not loaded"
                env.simulator.import_object(simulator_obj)

                # Set these objects to be far-away locations
                for i, new_urdf_obj in enumerate(new_urdf_objs):
                    new_urdf_obj.set_position([100 + num_new_obj - len(new_urdf_objs) + i, 100, -100])

                self.newly_added_objects.add(simulator_obj)
                self.object_scope[obj_inst] = simulator_obj

    def check_scene(self, env):
        error_msg = self.parse_non_sampleable_object_room_assignment()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        error_msg = self.build_sampling_order()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        error_msg = self.build_non_sampleable_object_scope()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        error_msg = self.import_sampleable_objects(env)
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        self.object_scope["agent.n.01_1"] = self.get_agent(env)

        return True, None

    def get_agent(self, env):
        return env.robots[0]

    def assign_object_scope_with_cache(self, env):
        # Assign object_scope based on a cached scene
        for obj_inst in self.object_scope:
            matched_sim_obj = None
            # TODO: remove after split floors
            if "floor.n.01" in obj_inst:
                floor_obj = self.scene.objects_by_name["floors"]
                bddl_object_scope = floor_obj.bddl_object_scope.split(",")
                bddl_object_scope = {item.split(":")[0]: item.split(":")[1] for item in bddl_object_scope}
                assert obj_inst in bddl_object_scope
                room_inst = bddl_object_scope[obj_inst].replace("room_floor_", "")
                matched_sim_obj = RoomFloor(
                    category="room_floor",
                    name=bddl_object_scope[obj_inst],
                    scene=self.scene,
                    room_instance=room_inst,
                    floor_obj=floor_obj,
                )
            elif obj_inst == "agent.n.01_1":
                matched_sim_obj = self.get_agent(env)
            else:
                for _, sim_obj in self.scene.objects_by_name.items():
                    if hasattr(sim_obj, "bddl_object_scope") and sim_obj.bddl_object_scope == obj_inst:
                        matched_sim_obj = sim_obj
                        break
            assert matched_sim_obj is not None, obj_inst
            self.object_scope[obj_inst] = matched_sim_obj

    def process_single_condition(self, condition):
        if not isinstance(condition.children[0], Negation) and not isinstance(condition.children[0], AtomicFormula):
            logging.warning(("Skipping over sampling of predicate that is not a negation or an atomic formula"))
            return None, None

        if isinstance(condition.children[0], Negation):
            condition = condition.children[0].children[0]
            positive = False
        else:
            condition = condition.children[0]
            positive = True

        return condition, positive

    def group_initial_conditions(self):
        self.non_sampleable_obj_conditions = []
        self.sampleable_obj_conditions = []

        # TODO: currently we assume self.initial_conditions is a list of
        # bddl.condition_evaluation.HEAD, each with one child.
        # This chid is either a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate or
        # a Negation of a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate
        for condition in self.initial_conditions:
            condition, positive = self.process_single_condition(condition)
            if condition is None:
                continue

            if condition.STATE_NAME in KINEMATICS_STATES and not positive:
                return "Initial condition has negative kinematic conditions: {}".format(condition.body)

            condition_body = set(condition.body)
            if len(self.non_sampleable_object_inst.intersection(condition_body)) > 0:
                self.non_sampleable_obj_conditions.append((condition, positive))
            else:
                self.sampleable_obj_conditions.append((condition, positive))

    def filter_object_scope(self, input_object_scope, conditions, condition_type):
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
                                if self.debug_obj_inst is not None and self.debug_obj_inst == condition.body[0]:
                                    igibson.debug_sampling = True
                                    obj_pos = obj.get_position()
                                    # Set the camera to have a bird's eye view of the sampling process
                                    p.resetDebugVisualizerCamera(
                                        cameraDistance=3.0,
                                        cameraYaw=0,
                                        cameraPitch=-89.99999,
                                        cameraTargetPosition=obj_pos,
                                    )

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
                                logging.warning(log_msg)

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
                    condition_type, room_type, self.scene.scene_id
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
                logging.warning((log_msg))
                for obj_inst in obj_inst_to_obj_per_room_inst:
                    for obj in obj_inst_to_obj_per_room_inst[obj_inst]:
                        # Create an edge between obj instance and each of the simulator obj that supports sampling
                        graph.add_edge(obj_inst, obj)
                        log_msg = "Adding edge: {} <-> {}".format(obj_inst, obj.name)
                        logging.warning((log_msg))
                        top_nodes.append(obj_inst)
                # Need to provide top_nodes that contain all nodes in one bipartite node set
                # The matches will have two items for each match (e.g. A -> B, B -> A)
                matches = nx.bipartite.maximum_matching(graph, top_nodes=top_nodes)
                if len(matches) == 2 * len(obj_inst_to_obj_per_room_inst):
                    logging.warning(("Object scope finalized:"))
                    for obj_inst, obj in matches.items():
                        if obj_inst in obj_inst_to_obj_per_room_inst:
                            self.object_scope[obj_inst] = obj
                            logging.warning((obj_inst, obj.name))
                    success = True
                    break
            if not success:
                return "{}: Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n".format(
                    condition_type, room_type, self.scene.scene_id
                )

    def sample_conditions(self, input_object_scope, conditions, condition_type):
        filtered_object_scope = self.filter_object_scope(input_object_scope, conditions, condition_type)
        error_msg = self.consolidate_room_instance(filtered_object_scope, condition_type)
        if error_msg:
            return error_msg, None
        return self.maximum_bipartite_matching(filtered_object_scope, condition_type), filtered_object_scope

    def sample_initial_conditions(self):
        error_msg, self.non_sampleable_object_scope_filtered_initial = self.sample_conditions(
            self.non_sampleable_object_scope, self.non_sampleable_obj_conditions, "initial"
        )
        return error_msg

    def sample_goal_conditions(self):
        np.random.shuffle(self.ground_goal_state_options)
        logging.warning(("number of ground_goal_state_options", len(self.ground_goal_state_options)))
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
        # Do the final round of sampling with object scope fixed
        for condition, positive in self.non_sampleable_obj_conditions:
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

        # Use ray casting for ontop and inside sampling for sampleable objects
        for condition, positive in self.sampleable_obj_conditions:
            if condition.STATE_NAME in ["inside", "ontop"]:
                condition.kwargs["use_ray_casting_method"] = True

        if len(self.sampling_orders) > 0:
            # Pop non-sampleable objects
            self.sampling_orders.pop(0)
            for cur_batch in self.sampling_orders:
                # First sample non-sliced conditions
                for condition, positive in self.sampleable_obj_conditions:
                    if condition.STATE_NAME == "sliced":
                        continue
                    # Sample conditions that involve the current batch of objects
                    if condition.body[0] in cur_batch:
                        num_trials = 100
                        for _ in range(num_trials):
                            success = condition.sample(binary_state=positive)
                            if success:
                                break
                        if not success:
                            return "Sampleable object conditions failed: {} {}".format(
                                condition.STATE_NAME, condition.body
                            )

                # Then sample non-sliced conditions
                for condition, positive in self.sampleable_obj_conditions:
                    if condition.STATE_NAME != "sliced":
                        continue
                    # Sample conditions that involve the current batch of objects
                    if condition.body[0] in cur_batch:
                        success = condition.sample(binary_state=positive)
                        if not success:
                            return "Sampleable object conditions failed: {}".format(condition.body)

    def sample(self, env, validate_goal=True):
        # Before sampling, set robot to be far away
        env.robots[0].set_position_orientation([300, 300, 300], [0, 0, 0, 1])
        error_msg = self.group_initial_conditions()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        error_msg = self.sample_initial_conditions()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        if validate_goal:
            error_msg = self.sample_goal_conditions()
            if error_msg:
                logging.warning(error_msg)
                return False, error_msg

        error_msg = self.sample_initial_conditions_final()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        return True, None

    def import_non_colliding_objects(self, env, clutter_scene, existing_objects=[], min_distance=0.5):
        """
        Loads objects into the scene such that they don't collide with existing objects.

        :param env: iGibsonEnv
        :param clutter_scene: A clutter scene to load new clutter objects from
        :param existing_objects: A list of objects that needs to be kept min_distance away when loading the new objects
        :param min_distance: A minimum distance to require for objects to load
        """
        state_id = p.saveState()
        objects_to_add = []

        for obj_name, obj in clutter_scene.objects_by_name.items():
            # Do not allow duplicate object categories
            if obj.category in env.simulator.scene.objects_by_category:
                continue

            bbox_center_pos, bbox_center_orn = clutter_scene.object_states[obj_name]["bbox_center_pose"]
            rotated_offset = p.multiplyTransforms([0, 0, 0], bbox_center_orn, obj.scaled_bbxc_in_blf, [0, 0, 0, 1])[0]
            base_link_pos, base_link_orn = bbox_center_pos + rotated_offset, bbox_center_orn

            add = True
            body_ids = []
            for i, urdf_path in enumerate(obj.urdf_paths):
                # Load the object only into pybullet
                body_id = p.loadURDF(obj.urdf_paths[i])
                body_ids.append(body_id)
                sub_urdf_pos, sub_urdf_orn = p.multiplyTransforms(
                    base_link_pos, base_link_orn, *obj.local_transforms[i]
                )
                # Convert to CoM frame
                dynamics_info = p.getDynamicsInfo(body_id, -1)
                inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
                sub_urdf_pos, sub_urdf_orn = p.multiplyTransforms(
                    sub_urdf_pos, sub_urdf_orn, inertial_pos, inertial_orn
                )
                sub_urdf_pos = np.array(sub_urdf_pos)
                sub_urdf_pos[2] += 0.01  # slighly above to not touch furniture
                p.resetBasePositionAndOrientation(body_id, sub_urdf_pos, sub_urdf_orn)
                for existing_object in existing_objects:
                    # If a sliced obj is an existing_object, get_position will not work
                    if isinstance(existing_object, ObjectMultiplexer) and isinstance(
                        existing_object.current_selection(), ObjectGrouper
                    ):
                        obj_pos = np.array([obj.get_position() for obj in existing_object.objects]).mean(axis=0)
                    else:
                        obj_pos = existing_object.get_position()
                    distance = np.linalg.norm(np.array(sub_urdf_pos) - np.array(obj_pos))
                    if distance < min_distance:
                        add = False
                        break
                if not add:
                    break

            # Filter based on collisions with any existing object
            if add:
                p.stepSimulation()
                for body_id in body_ids:
                    if len(p.getContactPoints(body_id)) > 0:
                        add = False
                        break

            if add:
                objects_to_add.append(obj)

            # Always remove all the body ids for this object
            for body_id in body_ids:
                p.removeBody(body_id)

            restoreState(state_id)

        p.removeState(state_id)

        # Actually load the object into the simulator
        for obj in objects_to_add:
            env.simulator.import_object(obj)

        # Restore clutter objects to their correct poses
        clutter_scene.restore_object_states(clutter_scene.object_states)

    def clutter_scene(self, env):
        """
        Load clutter objects into the scene from a random clutter scene
        """
        scene_id = self.scene.scene_id
        clutter_ids = [""] + list(range(2, 5))
        clutter_id = np.random.choice(clutter_ids)
        clutter_scene = InteractiveIndoorScene(scene_id, "{}_clutter{}".format(scene_id, clutter_id))
        existing_objects = [value for key, value in self.object_scope.items() if "floor.n.01" not in key]
        self.import_non_colliding_objects(
            env=env, clutter_scene=clutter_scene, existing_objects=existing_objects, min_distance=0.5
        )

    def get_task_obs(self, env):
        state = OrderedDict()
        task_obs = np.zeros((self.task_obs_dim))
        state["robot_pos"] = np.array(env.robots[0].get_position())
        state["robot_orn"] = np.array(env.robots[0].get_rpy())

        i = 0
        for _, v in self.object_scope.items():
            if isinstance(v, URDFObject):
                state["obj_{}_valid".format(i)] = 1.0
                state["obj_{}_pos".format(i)] = np.array(v.get_position())
                state["obj_{}_orn".format(i)] = np.array(p.getEulerFromQuaternion(v.get_orientation()))
                grasping_objects = env.robots[0].is_grasping(v.get_body_ids())
                for grasp_idx, grasping in enumerate(grasping_objects):
                    state["obj_{}_pos_in_gripper_{}".format(i, grasp_idx)] = float(grasping)
                i += 1

        state_list = []
        for k, v in state.items():
            if isinstance(v, list):
                state_list.extend(v)
            elif isinstance(v, tuple):
                state_list.extend(list(v))
            elif isinstance(v, np.ndarray):
                state_list.extend(list(v))
            elif isinstance(v, (float, int)):
                state_list.append(v)
            else:
                raise ValueError("cannot serialize task obs")

        assert len(state_list) <= len(task_obs)
        task_obs[: len(state_list)] = state_list

        return task_obs

    def check_success(self):
        self.current_success, self.current_goal_status = evaluate_goal_conditions(self.goal_conditions)
        return self.current_success, self.current_goal_status

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(BehaviorTask, self).get_termination(env, collision_links, action, info)
        info["goal_status"] = self.current_goal_status
        return done, info

    def show_instruction(self):
        satisfied = self.currently_viewed_instruction in self.current_goal_status["satisfied"]
        natural_language_condition = self.natural_language_goal_conditions[self.currently_viewed_instruction]
        objects = self.goal_conditions[self.currently_viewed_instruction].get_relevant_objects()
        text_color = (
            [83.0 / 255.0, 176.0 / 255.0, 72.0 / 255.0] if satisfied else [255.0 / 255.0, 51.0 / 255.0, 51.0 / 255.0]
        )

        return natural_language_condition, text_color, objects

    def iterate_instruction(self):
        self.currently_viewed_index = (self.currently_viewed_index + 1) % len(self.conds.parsed_goal_conditions)
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]

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
