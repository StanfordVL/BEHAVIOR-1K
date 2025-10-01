import os
import json
from pathlib import Path
import random

import torch as th
from bddl.activity import (
    Conditions,
    evaluate_goal_conditions,
    get_goal_conditions,
    get_ground_goal_state_options,
    get_initial_conditions,
    get_natural_goal_conditions,
    get_natural_initial_conditions,
    get_object_scope,
)

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.object_states import Pose
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.scenes.scene_base import Scene
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.bddl_utils import (
    BEHAVIOR_ACTIVITIES,
    BDDLEntity,
    BDDLSampler,
    OmniGibsonBDDLBackend,
    get_processed_bddl,
)
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.config_utils import TorchEncoder
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
        use_presampled_robot_pose (bool): Whether to use presampled robot poses from scene metadata
        randomize_presampled_pose (bool): If True, randomly selects from available presampled poses. If False, always
            uses the first pose. Only applies when use_presampled_robot_pose is True. Default is False.
        sampling_whitelist (None or dict): If specified, should map synset name (e.g.: "table.n.01" to a dictionary
            mapping category name (e.g.: "breakfast_table") to a list of valid models to be sampled from
            that category. During sampling, if a given synset is found in this whitelist, only the specified
            models will be used as options
        sampling_blacklist (None or dict): If specified, should map synset name (e.g.: "table.n.01" to a dictionary
            mapping category name (e.g.: "breakfast_table") to a list of invalid models that should not be sampled from
            that category. During sampling, if a given synset is found in this blacklist, all specified
            models will not be used as options
        highlight_task_relevant_objects (bool): whether to overlay task-relevant objects in the scene with a colored mask
        termination_config (None or dict): Keyword-mapped configuration to use to generate termination conditions. This
            should be specific to the task class. Default is None, which corresponds to a default config being usd.
            Note that any keyword required by a specific task class but not specified in the config will automatically
            be filled in with the default config. See cls.default_termination_config for default values used
        reward_config (None or dict): Keyword-mapped configuration to use to generate reward functions. This should be
            specific to the task class. Default is None, which corresponds to a default config being usd. Note that
            any keyword required by a specific task class but not specified in the config will automatically be filled
            in with the default config. See cls.default_reward_config for default values used
        include_obs (bool): Whether to include observations or not for this task
    """

    def __init__(
        self,
        activity_name=None,
        activity_definition_id=0,
        activity_instance_id=0,
        predefined_problem=None,
        online_object_sampling=False,
        use_presampled_robot_pose=True,
        randomize_presampled_pose=False,
        sampling_whitelist=None,
        sampling_blacklist=None,
        highlight_task_relevant_objects=False,
        termination_config=None,
        reward_config=None,
        include_obs=True,
    ):
        # Make sure object states are enabled
        assert gm.ENABLE_OBJECT_STATES, "Must set gm.ENABLE_OBJECT_STATES=True in order to use BehaviorTask!"

        # Make sure task name is valid if not specifying a predefined problem
        if predefined_problem is None:
            assert (
                activity_name is not None
            ), "Activity name must be specified if no predefined_problem is specified for BehaviorTask!"
            assert_valid_key(key=activity_name, valid_keys=BEHAVIOR_ACTIVITIES, name="Behavior Task")
        else:
            # Infer activity name
            activity_name = predefined_problem.split("problem ")[-1].split("-")[0]

        # Make sure to not use presampled robot pose if we're using online object sampling
        assert not (
            online_object_sampling and use_presampled_robot_pose
        ), "Cannot use presampled robot pose if online_object_sampling is True!"

        # Initialize relevant variables

        # BDDL
        self.backend = OmniGibsonBDDLBackend()

        # Activity info
        self.activity_name = activity_name
        self.activity_definition_id = activity_definition_id
        self.activity_instance_id = activity_instance_id
        self.predefined_problem = predefined_problem
        self.activity_conditions = None
        self.activity_initial_conditions = None
        self.activity_goal_conditions = None
        self.ground_goal_state_options = None
        self.feedback = None  # None or str
        self.sampler = None  # BDDLSampler

        # Scene info
        self.scene_name = None

        # Object info
        self.online_object_sampling = online_object_sampling  # bool
        self.use_presampled_robot_pose = use_presampled_robot_pose
        self.randomize_presampled_pose = randomize_presampled_pose
        self.sampling_whitelist = sampling_whitelist  # Maps str to str to list
        self.sampling_blacklist = sampling_blacklist  # Maps str to str to list
        self.highlight_task_relevant_objs = highlight_task_relevant_objects  # bool
        self.object_scope = None  # Maps str to BDDLEntity
        self.object_instance_to_category = None  # Maps str to str
        self.future_obj_instances = None  # set of str

        # Info for demonstration collection
        self.instruction_order = None  # th.tensor of int
        self.currently_viewed_index = None  # int
        self.currently_viewed_instruction = None  # tuple of str
        self.activity_natural_language_initial_conditions = None  # str
        self.activity_natural_language_goal_conditions = None  # str

        # Run super init
        super().__init__(termination_config=termination_config, reward_config=reward_config, include_obs=include_obs)

    @classmethod
    def get_cached_activity_scene_filename(
        cls, scene_model, activity_name, activity_definition_id, activity_instance_id
    ):
        """
        Helper method to programmatically construct the scene filename for a given pre-cached task configuration

        Args:
            scene_model (str): Name of the scene (e.g.: Rs_int)
            activity_name (str): Name of the task activity (e.g.: putting_away_halloween_decorations)
            activity_definition_id (int): ID of the task definition
            activity_instance_id (int): ID of the task instance

        Returns:
            str: Filename which, if exists, should include the cached activity scene
        """
        return f"{scene_model}_task_{activity_name}_{activity_definition_id}_{activity_instance_id}_template"

    @classmethod
    def verify_scene_and_task_config(cls, scene_cfg, task_cfg):
        # Run super first
        super().verify_scene_and_task_config(scene_cfg=scene_cfg, task_cfg=task_cfg)

        # Possibly modify the scene to load if we're using online_object_sampling
        scene_instance, scene_file = scene_cfg["scene_instance"], scene_cfg["scene_file"]
        activity_name = (
            task_cfg["predefined_problem"].split("problem ")[-1].split("-")[0]
            if task_cfg.get("predefined_problem", None) is not None
            else task_cfg["activity_name"]
        )
        if scene_file is None and scene_instance is None and not task_cfg["online_object_sampling"]:
            scene_instance = cls.get_cached_activity_scene_filename(
                scene_model=scene_cfg.get("scene_model", "Scene"),
                activity_name=activity_name,
                activity_definition_id=task_cfg.get("activity_definition_id", 0),
                activity_instance_id=task_cfg.get("activity_instance_id", 0),
            )
            # Update the value in the scene config
            scene_cfg["scene_instance"] = scene_instance

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
        # Load the initial behavior configuration
        self.update_activity(
            env=env,
            activity_name=self.activity_name,
            activity_definition_id=self.activity_definition_id,
            predefined_problem=self.predefined_problem,
        )

        # Initialize the current activity
        success, self.feedback = self.initialize_activity(env=env)
        # assert success, f"Failed to initialize Behavior Activity. Feedback:\n{self.feedback}"

        # Store the scene name
        self.scene_name = env.scene.scene_model if isinstance(env.scene, TraversableScene) else None

        # Highlight any task relevant objects if requested
        if self.highlight_task_relevant_objs:
            for entity in self.object_scope.values():
                if entity.synset == "agent":
                    continue
                if not entity.is_system and entity.exists:
                    entity.highlighted = True

        # Add callbacks to handle internal processing when new systems / objects are added / removed to the scene
        callback_name = f"{self.activity_name}_refresh"
        og.sim.add_callback_on_add_obj(name=callback_name, callback=self._update_bddl_scope_from_added_obj)
        og.sim.add_callback_on_remove_obj(name=callback_name, callback=self._update_bddl_scope_from_removed_obj)

        og.sim.add_callback_on_system_init(name=callback_name, callback=self._update_bddl_scope_from_system_init)
        og.sim.add_callback_on_system_clear(name=callback_name, callback=self._update_bddl_scope_from_system_clear)

    def reset(self, env):
        super().reset(env)

        # Use presampled robot pose if specified (only available for officially supported mobile manipulators)
        if self.use_presampled_robot_pose:
            robot = self.get_agent(env)
            presampled_poses = env.scene.get_task_metadata(key="robot_poses")
            assert (
                robot.model_name in presampled_poses
            ), f"{robot.model_name} presampled pose is not found in task metadata; please set use_presampled_robot_pose to False in task config"

            # Select pose based on randomize_presampled_pose flag
            available_poses = presampled_poses[robot.model_name]
            if self.randomize_presampled_pose:
                robot_pose = random.choice(available_poses)
            else:
                robot_pose = available_poses[0]  # Use first presampled pose

            robot.set_position_orientation(robot_pose["position"], robot_pose["orientation"])

        # Force wake objects
        for obj in self.object_scope.values():
            if obj.exists and isinstance(obj, DatasetObject):
                obj.wake()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    def update_activity(self, env, activity_name, activity_definition_id, predefined_problem=None):
        """
        Update the active Behavior activity being deployed

        Args:
            env (og.Environment): OmniGibson active environment
            activity_name (None or str): Name of the Behavior Task to instantiate
            activity_definition_id (int): Specification to load for the desired task. For a given Behavior Task, multiple task
                specifications can be used (i.e.: differing goal conditions, or "ways" to complete a given task). This
                ID determines which specification to use
            predefined_problem (None or str): If specified, specifies the raw string definition of the Behavior Task to
                load. This will automatically override @activity_name and @activity_definition_id.
        """
        # We parse the raw BDDL to be compatible with OmniGibson
        # This requires converting wildcard-denoted synset instances into explicit synsets
        # that are compatible with all valid scene objects in the current OG scene
        if predefined_problem is None:
            # Process the task
            predefined_problem = get_processed_bddl(
                activity_name,
                activity_definition_id,
                scene=env.scene,
            )

        # Activity info
        self.activity_name = activity_name
        self.activity_definition_id = activity_definition_id
        self.activity_conditions = Conditions(
            activity_name,
            activity_definition_id,
            simulator_name="omnigibson",
            predefined_problem=predefined_problem,
        )

        # Get scope, making sure agent is the first entry
        self.object_scope = {"agent.n.01_1": None}
        self.object_scope.update(get_object_scope(self.activity_conditions))

        # Object info
        self.object_instance_to_category = {
            obj_inst: obj_cat
            for obj_cat in self.activity_conditions.parsed_objects
            for obj_inst in self.activity_conditions.parsed_objects[obj_cat]
        }

        # Generate initial and goal conditions
        self.activity_initial_conditions = get_initial_conditions(
            self.activity_conditions, self.backend, self.object_scope
        )
        self.activity_goal_conditions = get_goal_conditions(self.activity_conditions, self.backend, self.object_scope)
        self.ground_goal_state_options = get_ground_goal_state_options(
            self.activity_conditions, self.backend, self.object_scope, self.activity_goal_conditions
        )

        # Demo attributes
        self.instruction_order = th.arange(len(self.activity_conditions.parsed_goal_conditions))
        self.instruction_order = self.instruction_order[th.randperm(self.instruction_order.size(0))]

        self.currently_viewed_index = 0
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]
        self.activity_natural_language_initial_conditions = get_natural_initial_conditions(self.activity_conditions)
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

        # Generate sampler
        self.sampler = BDDLSampler(
            env=env,
            activity_conditions=self.activity_conditions,
            object_scope=self.object_scope,
            backend=self.backend,
        )

        # Compose future objects
        self.future_obj_instances = {
            init_cond.body[1] for init_cond in self.activity_initial_conditions if init_cond.body[0] == "future"
        }

        if self.online_object_sampling:
            # Sample online
            accept_scene, feedback = self.sampler.sample(
                sampling_whitelist=self.sampling_whitelist,
                sampling_blacklist=self.sampling_blacklist,
            )
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
        # Load task metadata
        inst_to_name = env.scene.get_task_metadata(key="inst_to_name")

        # Assign object_scope based on a cached scene
        for obj_inst in self.object_scope:
            if obj_inst in self.future_obj_instances:
                entity = None
            else:
                assert obj_inst in inst_to_name, (
                    f"BDDL object instance {obj_inst} should exist in cached metadata "
                    f"from loaded scene, but could not be found!"
                )
                name = inst_to_name[obj_inst]
                is_system = name in env.scene.available_systems.keys()
                # TODO: If we load a robot with a different set of configs, we will not be able to match with the
                # original object_scope. This is a temporary fix to handle this case. A proper fix involves
                # storing the robot (potentially only base pose) in the task metadata instead of as a regular object
                if "agent.n." in obj_inst:
                    idx = int(obj_inst.split("_")[-1].lstrip("0")) - 1
                    entity = env.robots[idx]
                else:
                    entity = env.scene.get_system(name) if is_system else env.scene.object_registry("name", name)
            self.object_scope[obj_inst] = BDDLEntity(
                bddl_inst=obj_inst,
                entity=entity,
            )

        # Write back to task metadata
        self.update_bddl_scope_metadata(env)

    def update_bddl_scope_metadata(self, env):
        """
        Updates the task metadata with the current instance-to-name mapping for all existing entities.

        Args:
            env (Environment): The environment containing the scene to update
        """
        env.scene.write_task_metadata(
            key="inst_to_name", data={inst: entity.name for inst, entity in self.object_scope.items() if entity.exists}
        )

    def _get_obs(self, env):
        low_dim_obs = dict()

        # Batch rpy calculations for much better efficiency
        objs_exist = {obj: obj.exists for obj in self.object_scope.values() if not obj.is_system}
        objs_rpy = T.quat2euler(
            th.stack(
                [
                    obj.states[Pose].get_value()[1] if obj_exist else th.tensor([0, 0, 0, 1.0])
                    for obj, obj_exist in objs_exist.items()
                ]
            )
        )
        objs_rpy_cos = th.cos(objs_rpy)
        objs_rpy_sin = th.sin(objs_rpy)

        # Always add agent info first
        agent = self.get_agent(env=env)

        for (obj, obj_exist), obj_rpy, obj_rpy_cos, obj_rpy_sin in zip(
            objs_exist.items(), objs_rpy, objs_rpy_cos, objs_rpy_sin
        ):
            # TODO: May need to update checking here to USDObject? Or even baseobject?
            # TODO: How to handle systems as part of obs?
            if obj_exist:
                low_dim_obs[f"{obj.bddl_inst}_real"] = th.tensor([1.0])
                low_dim_obs[f"{obj.bddl_inst}_pos"] = obj.states[Pose].get_value()[0]
                low_dim_obs[f"{obj.bddl_inst}_ori_cos"] = obj_rpy_cos
                low_dim_obs[f"{obj.bddl_inst}_ori_sin"] = obj_rpy_sin
                if obj.name != agent.name:
                    for arm in agent.arm_names:
                        grasping_object = agent.is_grasping(arm=arm, candidate_obj=obj.wrapped_obj)
                        low_dim_obs[f"{obj.bddl_inst}_in_gripper_{arm}"] = th.tensor([float(grasping_object)])
            else:
                low_dim_obs[f"{obj.bddl_inst}_real"] = th.zeros(1)
                low_dim_obs[f"{obj.bddl_inst}_pos"] = th.zeros(3)
                low_dim_obs[f"{obj.bddl_inst}_ori_cos"] = th.zeros(3)
                low_dim_obs[f"{obj.bddl_inst}_ori_sin"] = th.zeros(3)
                for arm in agent.arm_names:
                    low_dim_obs[f"{obj.bddl_inst}_in_gripper_{arm}"] = th.zeros(1)

        return low_dim_obs, dict()

    def _step_termination(self, env, action, info=None):
        # Run super first
        done, info = super()._step_termination(env=env, action=action, info=info)

        # Add additional info
        info["goal_status"] = self._termination_conditions["predicate"].goal_status

        return done, info

    def _update_bddl_scope_from_added_obj(self, obj):
        """
        Internal callback function to be called when new objects are added to the simulator to potentially update internal
        bddl object scope

        Args:
            obj (BaseObject): Newly imported object
        """
        # Iterate over all entities, and if they don't exist, check if any category matches @obj's category, and set it
        # if it does, and immediately return
        for inst, entity in self.object_scope.items():
            if not entity.exists and not entity.is_system and obj.category in set(entity.og_categories):
                entity.set_entity(entity=obj)
                return

    def _update_bddl_scope_from_removed_obj(self, obj):
        """
        Internal callback function to be called when sim._pre_remove_object() is called to potentially update internal
        bddl object scope

        Args:
            obj (BaseObject): Newly removed object
        """
        # Iterate over all entities, and if they exist, check if any name matches @obj's name, and remove it
        # if it does, and immediately return
        for entity in self.object_scope.values():
            if entity.exists and not entity.is_system and obj.name == entity.name:
                entity.clear_entity()
                return

    def _update_bddl_scope_from_system_init(self, system):
        """
        Internal callback function to be called when system.initialize() is called to potentially update internal
        bddl object scope

        Args:
            system (BaseSystem): Newly initialized system
        """
        # Iterate over all entities, and potentially match the system to the scope
        for inst, entity in self.object_scope.items():
            if not entity.exists and entity.is_system and entity.og_categories[0] == system.name:
                entity.set_entity(entity=system)
                return

    def _update_bddl_scope_from_system_clear(self, system):
        """
        Internal callback function to be called when system.clear() is called to potentially update internal
        bddl object scope

        Args:
            system (BaseSystem): Newly cleared system
        """
        # Iterate over all entities, and potentially remove the matched system from the scope
        for inst, entity in self.object_scope.items():
            if entity.exists and entity.is_system and system.name == entity.name:
                entity.clear_entity()
                return

    def show_instruction(self):
        """
        Get current instruction for user

        Returns:
            3-tuple:
                - str: Current goal condition in natural language
                - 3-tuple: (R,G,B) color to assign to text
                - list of BaseObject: Relevant objects for the current instruction
        """
        satisfied = (
            self.currently_viewed_instruction in self._termination_conditions["predicate"].goal_status["satisfied"]
        )
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
        self.currently_viewed_index = (self.currently_viewed_index + 1) % len(
            self.activity_conditions.parsed_goal_conditions
        )
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]

    def save_task(self, env, save_dir=None, override=False, task_relevant_only=False, suffix=None):
        """
        Writes the current scene configuration to a .json file

        Args:
            env (og.Environment): OmniGibson active environment
            save_dir (None or str): If specified, absolute fpath to the desired directory to write the .json. Default is
                {gm.DATA_PATH}/behavior-1k-assets/scenes/<SCENE_MODEL>/json/...>
            override (bool): Whether to override any files already found at the path to write the task .json
            task_relevant_only (bool): Whether to only save the task relevant object scope states. If True, will only
                call dump_state() on all the BDDL instances in self.object_scope, else will save the entire sim state
                via env.scene.save()
            suffix (None or str): If specified, suffix to add onto the end of the scene filename that will be saved
        """
        save_dir = (
            os.path.join(get_dataset_path("behavior-1k-assets"), "scenes", self.scene_name, "json")
            if save_dir is None
            else save_dir
        )
        assert self.scene_name is not None, "Scene name must be set in order to save task"
        fname = self.get_cached_activity_scene_filename(
            scene_model=self.scene_name,
            activity_name=self.activity_name,
            activity_definition_id=self.activity_definition_id,
            activity_instance_id=self.activity_instance_id,
        )
        path = os.path.join(save_dir, f"{fname}.json")
        if task_relevant_only:
            path = path.replace(".json", "-tro_state.json")
        if suffix is not None:
            path = path.replace(".json", f"-{suffix}.json")
        if os.path.exists(path) and not override:
            log.warning(f"Scene json already exists at {path}. Use override=True to force writing of new json.")
            return

        # Save based on whether we're only storing task-relevant object scope states or not
        if task_relevant_only:
            task_relevant_state_dict = {
                bddl_name: bddl_inst.dump_state(serialized=False)
                for bddl_name, bddl_inst in env.task.object_scope.items()
                if bddl_inst.exists
            }
            Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
            with open(path, "w+") as f:
                json.dump(task_relevant_state_dict, f, cls=TorchEncoder, indent=4)
        else:
            # Update task metadata and save
            self.update_bddl_scope_metadata(env)
            env.scene.save(json_path=path)

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
        # Any scene can be used
        return {Scene}

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
