import os
import json
from pathlib import Path
import random

import torch as th

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
    get_behavior_activities,
    BDDLSampler,
    get_knowledge_base,
    is_system_bddl_inst,
    og_categories_from_bddl_inst,
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

        assert activity_name is not None, "Activity name must be specified for BehaviorTask!"
        assert_valid_key(key=activity_name, valid_keys=get_behavior_activities(), name="Behavior Task")

        # Make sure to not use presampled robot pose if we're using online object sampling
        assert not (
            online_object_sampling and use_presampled_robot_pose
        ), "Cannot use presampled robot pose if online_object_sampling is True!"

        # Activity info
        self.activity_name = activity_name
        self.activity_definition_id = activity_definition_id
        self.activity_instance_id = activity_instance_id
        self.compiled_task = None
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
        self.object_scope = None  # Maps str to sim object (BaseObject/BaseSystem) or None
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
        activity_name = task_cfg["activity_name"]
        if scene_file is None and scene_instance is None and not task_cfg["online_object_sampling"]:
            scene_instance = cls.get_cached_activity_scene_filename(
                scene_model=scene_cfg.get("scene_model", "Scene"),
                activity_name=activity_name,
                activity_definition_id=task_cfg.get("activity_definition_id", 0),
                activity_instance_id=task_cfg.get("activity_instance_id", 0),
            )
            # Update the value in the scene config
            scene_cfg["scene_instance"] = scene_instance

    def _evaluate_predicate(self, env_idx, predicate_name, *entities):
        from omnigibson.utils.bddl_utils import evaluate_bddl_predicate

        return evaluate_bddl_predicate(predicate_name, *[self.object_scope[env_idx][ent] for ent in entities])

    def _create_termination_conditions(self):
        # Initialize termination conditions dict and fill in with Timeout and PredicateGoal
        terminations = dict()

        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        # PredicateGoal calls check_goal_fn(env_idx); thread env_idx through to the predicate evaluator
        # so the (singular) compiled task evaluates against the right env's object_scope binding.
        # TODO(vector): This needs to be extensively tested.
        terminations["predicate"] = PredicateGoal(
            check_goal_fn=lambda env_idx: self.compiled_task.check_goal(
                lambda predicate_name, *entities: self._evaluate_predicate(env_idx, predicate_name, *entities)
            ),
        )

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
        # Store env reference for use in callbacks
        self._env = env

        # Load the initial behavior configuration
        self.update_activity(
            env=env,
            activity_name=self.activity_name,
            activity_definition_id=self.activity_definition_id,
        )

        # Initialize the current activity
        success, self.feedback = self.initialize_activity(env=env)
        # assert success, f"Failed to initialize Behavior Activity. Feedback:\n{self.feedback}"

        # Store the scene name
        self.scene_name = env.scenes[0].scene_model if isinstance(env.scenes[0], TraversableScene) else None

        # Highlight any task relevant objects if requested
        if self.highlight_task_relevant_objs:
            for env_idx in range(env.num_envs):
                for inst, entity in self.object_scope[env_idx].items():
                    if "agent.n." in inst:
                        continue
                    if not is_system_bddl_inst(inst) and entity is not None:
                        entity.highlighted = True

        # Add callbacks to handle internal processing when new systems / objects are added / removed to the scene
        callback_name = f"{self.activity_name}_refresh"
        og.sim.add_callback_on_add_obj(name=callback_name, callback=self._update_bddl_scope_from_added_obj)
        og.sim.add_callback_on_remove_obj(name=callback_name, callback=self._update_bddl_scope_from_removed_obj)

        og.sim.add_callback_on_system_init(name=callback_name, callback=self._update_bddl_scope_from_system_init)
        og.sim.add_callback_on_system_clear(name=callback_name, callback=self._update_bddl_scope_from_system_clear)

    def reset(self, env, env_indices=None):
        super().reset(env, env_indices=env_indices)

        if env_indices is None:
            env_indices = th.arange(self._num_envs)

        # Use presampled robot pose if specified (only available for officially supported mobile manipulators)
        if self.use_presampled_robot_pose:
            for env_idx in env_indices:
                robot = self.get_agent(env, env_idx)
                presampled_poses = env.scenes[env_idx].get_task_metadata(key="robot_poses")
                # make all lowercase
                presampled_poses = {k.lower(): v for k, v in presampled_poses.items()}
                # use generic "robot" key if it exists, otherwise look for model-specific key
                if "robot" in presampled_poses:
                    available_poses = presampled_poses["robot"]
                elif robot.model.lower() in presampled_poses:
                    print("No generic presampled robot pose found, using robot-specific pose.")
                    available_poses = presampled_poses[robot.model.lower()]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose found for {robot.model}!")
                if self.randomize_presampled_pose:
                    robot_pose = random.choice(available_poses)
                else:
                    robot_pose = available_poses[0]  # Use first presampled pose

                robot.set_position_orientation(robot_pose["position"], robot_pose["orientation"])

        # Force wake objects
        for env_idx in env_indices:
            for obj in self.object_scope[env_idx].values():
                if obj is not None and isinstance(obj, DatasetObject):
                    obj.wake()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    @staticmethod
    def _build_scene_layout_from_rooms(scene, room_instances):
        """Build a scene layout dict for BDDL wildcard expansion from specific room instances.

        Args:
            scene: The scene object.
            room_instances: Dict mapping room_type -> room_instance_name for the
                specific room instances that were selected during object scope assignment.

        Returns:
            dict: Maps room_type -> {category: count} for the selected room instances.
        """
        from collections import Counter

        layout = {}
        for room_type, room_inst in room_instances.items():
            objs = scene.object_registry("in_rooms", room_inst, default_val=[])
            counts = Counter(obj.category for obj in objs)
            layout[room_type] = dict(counts)
        return layout

    def update_activity(self, env, activity_name, activity_definition_id):
        """
        Update the active Behavior activity being deployed.

        Parses the base (non-wildcard) scope from the task definition. Full
        compilation is deferred to initialize_activity(), after object scope
        selection determines which specific room instances will be used for
        wildcard expansion.

        Args:
            env (og.Environment): OmniGibson active environment
            activity_name (None or str): Name of the Behavior Task to instantiate
            activity_definition_id (int): Specification to load for the desired task
        """
        # Activity info
        self.activity_name = activity_name
        self.activity_definition_id = activity_definition_id
        self._task_def = get_knowledge_base().get_task(f"{activity_name}-{activity_definition_id}")

        # Parse base scope (strips wildcards if any, giving us the non-wildcard instances)
        self._base_conditions, base_scope, self._base_inroom_assignments = self._task_def.parse_base_scope()
        # compiled_task and downstream conditions describe symbolic task content shared across
        # all envs (every env is a clone of the same scene with identical room layout), so they
        # are populated as a single instance during initialize_activity().
        self.compiled_task = None
        self.activity_initial_conditions = None
        self.activity_goal_conditions = None
        self.ground_goal_state_options = None
        # Remembers which room dict was used to compile self.compiled_task — checked against
        # every env's room layout in _compile_with_rooms() to detect any drift from the
        # "all envs use the same rooms" assumption.
        self._compiled_rooms = None

        # Set up base object scope per-env (agent first, then base instances)
        self.object_scope = [None] * env.num_envs
        for env_idx in range(env.num_envs):
            self.object_scope[env_idx] = {"agent.n.01_1": None}
            self.object_scope[env_idx].update({name: None for name in base_scope})

        # Object instance to category mapping (base only for now)
        self.object_instance_to_category = {
            obj_inst: obj_cat
            for obj_cat in self._base_conditions.parsed_objects
            for obj_inst in self._base_conditions.parsed_objects[obj_cat]
        }

    def _finalize_compiled_task(self):
        """Populate symbolic attributes from self.compiled_task.

        Runs exactly once after the (singular) compiled task is built. All fields set
        here are symbolic / scene-independent and shared across envs.
        """
        compiled = self.compiled_task

        # Object info — derived from the compiled (wildcard-expanded) task definition
        self.object_instance_to_category = {
            obj_inst: obj_cat for obj_cat in compiled.parsed_objects for obj_inst in compiled.parsed_objects[obj_cat]
        }

        # Conditions
        self.activity_initial_conditions = compiled.initial_conditions
        self.activity_goal_conditions = compiled.goal_conditions
        self.ground_goal_state_options = compiled.ground_goal_state_options

        # Demo attributes
        self.instruction_order = th.arange(len(compiled.conditions.parsed_goal_conditions))
        self.instruction_order = self.instruction_order[th.randperm(self.instruction_order.size(0))]
        self.currently_viewed_index = 0
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]
        self.activity_natural_language_initial_conditions = compiled.natural_language_initial_conditions
        self.activity_natural_language_goal_conditions = compiled.natural_language_goal_conditions

    def _finalize_object_scope(self, env_idx):
        """Rebuild self.object_scope[env_idx] from self.compiled_task.object_scope.

        Called per env after compile. Preserves any objects already bound to base-scope
        instances so we don't lose assignments made before wildcard expansion.

        Args:
            env_idx (int): Index of the env whose scope should be rebuilt.
        """
        existing = dict(self.object_scope[env_idx])
        self.object_scope[env_idx] = {"agent.n.01_1": existing.get("agent.n.01_1")}
        for name in self.compiled_task.object_scope:
            self.object_scope[env_idx][name] = existing.get(name)

    def _determine_room_instances(self, env, env_idx):
        """Determine which specific room instances to use based on assigned objects in @env_idx's scene.

        For each room type in the task's inroom assignments, finds which room
        instance the assigned object is actually in. All objects assigned to the
        same room type should be in the same room instance (ensured by the
        sampler's room consolidation logic).

        Args:
            env: The environment with the active scene.
            env_idx (int): Index of the env / scene.

        Returns:
            dict: Maps room_type (str) -> room_instance_name (str).
        """
        room_instances = {}
        for obj_inst, room_type in self._base_inroom_assignments.items():
            entity = self.object_scope[env_idx].get(obj_inst)
            if entity is None:
                continue
            # Find which room instance this object is in
            if hasattr(entity, "in_rooms") and entity.in_rooms:
                for room_inst in entity.in_rooms:
                    inst_room_type = room_inst.rsplit("_", 1)[0]  # Get room type by removing instance number suffix
                    if inst_room_type == room_type:
                        if room_type in room_instances and room_instances[room_type] != room_inst:
                            log.warning(
                                f"Multiple room instances for room type '{room_type}': "
                                f"{room_instances[room_type]} vs {room_inst}. Using {room_instances[room_type]}."
                            )
                        else:
                            room_instances[room_type] = room_inst
                        break
        return room_instances

    def _compile_with_rooms(self, env, env_idx):
        """Compile the wildcard task for using the specific room instances from its assigned objects.

        After object scope has been assigned (via cache or sampling), this
        determines which room instances are being used, counts objects in those
        rooms, and compiles the task with proper wildcard expansion.

        Args:
            env: The environment with the active scene.
            env_idx (int): Index of the env / scene to align scope for.
        """
        room_instances = self._determine_room_instances(env, env_idx)

        if self.compiled_task is None:
            scene_layout = self._build_scene_layout_from_rooms(env.scenes[env_idx], room_instances)
            self.compiled_task = self._task_def.compile(scene_layout=scene_layout)
            self._compiled_rooms = room_instances
            self._finalize_compiled_task()
        else:
            assert room_instances == self._compiled_rooms, (
                f"Room layout mismatch between envs: compiled with {self._compiled_rooms} "
                f"but env {env_idx} reports {room_instances}. Single-compile assumes uniform room layout."
            )

        # Per-env scope: rebuild against the compiled task's scope, preserving prior assignments
        old_scope = dict(self.object_scope[env_idx])
        self._finalize_object_scope(env_idx)
        for inst, entity in old_scope.items():
            if inst in self.object_scope[env_idx]:
                self.object_scope[env_idx][inst] = entity

    def get_potential(self, env, env_idx):
        # Bind env_idx into the predicate evaluator so check_goal sees an arity-2 callback
        _, satisfied_predicates = self.compiled_task.check_goal(
            lambda predicate_name, *entities: self._evaluate_predicate(env_idx, predicate_name, *entities)
        )
        success_score = len(satisfied_predicates["satisfied"]) / (
            len(satisfied_predicates["satisfied"]) + len(satisfied_predicates["unsatisfied"])
        )
        return -success_score

    def initialize_activity(self, env):
        """
        Initializes the desired activity in the current environment @env.

        The flow is:
        1. Select objects for each env's base (non-wildcard) scope via sampling or cache.
        2. Determine which room instances those objects are in (the first env's choice is
           canonical; other envs are asserted to match).
        3. Compile the task once with the correct scene layout (expanding any wildcards).
        4. Assign any wildcard-expanded instances per env.

        Args:
            env (Environment): Current active environment instance

        Returns:
            2-tuple:
                - bool: Whether the generated scene activity should be accepted or not
                - list[dict]: Per-env feedback from the sampling / initialization process
        """

        # self.sampler is a single instance bound to env 0:
        # - Online mode: actively used (num_envs guaranteed to be 1 by the assert below).
        # - Cache mode: created so downstream tooling (scripts/sampling/multiply_b1k_tasks.py)
        #   can poke its internals to re-sample. Not exercised during init.
        self.sampler = BDDLSampler(
            env=env,
            env_idx=0,
            activity_conditions=self._base_conditions,
            object_scope=self.object_scope[0],
        )

        if self.online_object_sampling:
            assert env.num_envs == 1, "Online sampling mode only works with num_envs==1"
            env_idx = 0

            # Phase 1: assign objects using only parsed conditions (no compilation needed)
            accept, fb = self.sampler.assign_objects(
                sampling_whitelist=self.sampling_whitelist,
                sampling_blacklist=self.sampling_blacklist,
            )
            if not accept:
                return accept, [fb]

            # Compile with the correct rooms now that objects are assigned
            self._compile_with_rooms(env, env_idx)

            # Phase 2: sample states using compiled conditions
            accept, fb = self.sampler.sample_states(self.compiled_task)
            if not accept:
                return accept, [fb]

            # Assign any wildcard-expanded instances to remaining scene objects
            self._assign_wildcard_instances(env, env_idx)

            return True, [fb]

        # Cache mode — num_envs can be >= 1
        feedback = [None] * env.num_envs

        # Derive future instances from parsed conditions for cache assignment
        self.future_obj_instances = {
            cond[1] for cond in self._base_conditions.parsed_initial_conditions if cond[0] == "future"
        }

        # Assign base scope objects from cache (non-strict: skip instances
        # not in cache, e.g. wildcard instances that don't exist yet)
        for env_idx in range(env.num_envs):
            self.assign_object_scope_with_cache(env, env_idx)
            self._compile_with_rooms(env, env_idx)

        # Refine future instances using the now-compiled task's (singular) initial conditions
        self.future_obj_instances = {
            init_cond.body[1] for init_cond in self.activity_initial_conditions if init_cond.body[0] == "future"
        }

        # Second pass per env: re-assign from cache (scope now includes wildcard instances)
        # and fill in any remaining wildcard slots from scene objects.
        # TODO @wensi-ai: Check object scope again to see if any wildcard objects are recorded.
        # 2026+ tasks do this, 2025 ones don't.
        for env_idx in range(env.num_envs):
            # Use non-strict so that wildcard-expanded instances absent from cache are handled by
            # _assign_wildcard_instances below rather than raising an assertion error.
            self.assign_object_scope_with_cache(env, env_idx)
            # TODO @wensi-ai: Assign objects to remaining wildcard objects. This is a no-op for 2026+ tasks.
            self._assign_wildcard_instances(env, env_idx)
            # assert that everything in the object scope that's not a future object is not None
            for inst, entity in self.object_scope[env_idx].items():
                if inst not in self.future_obj_instances and entity is None:
                    raise ValueError(
                        f"Object instance '{inst}' (env_idx={env_idx}) was not assigned an entity during cache assignment!"
                    )

        return True, feedback

    def _assign_wildcard_instances(self, env, env_idx):
        """Assign wildcard-expanded instances to scene objects in the selected rooms.

        After wildcard compilation, new instances exist in the scope that need
        to be matched to objects in the scene that weren't part of the base scope.

        Args:
            env: The environment with the active scene.
            env_idx (int): Index of the env / scene to assign for.
        """
        for inst in self.object_scope[env_idx]:
            if self.object_scope[env_idx][inst] is not None:
                continue
            if "agent.n." in inst:
                continue
            # Try to find a matching object in this env's scene
            categories = set(og_categories_from_bddl_inst(inst))
            for obj in env.scenes[env_idx].objects:
                # Check category match and that obj isn't already assigned
                if obj.category in categories and obj not in self.object_scope[env_idx].values():
                    self.object_scope[env_idx][inst] = obj
                    break

    def get_agent(self, env, env_idx=0):
        """
        Grab the 0th agent from @env for a specific scene

        Args:
            env (Environment): Current active environment instance
            env_idx (int): Index of the environment/scene

        Returns:
            BaseRobot: The 0th robot from the specified scene
        """
        # We assume the relevant agent is the first agent in the scene
        return env.scenes[env_idx].robots[0]

    def assign_object_scope_with_cache(self, env, env_idx):
        """
        Assigns objects within the current object scope from cached scene metadata.

        Args:
            env (Environment): Current active environment instance
            env_idx (int): Index of the environment/scene
        """
        scene = env.scenes[env_idx]

        # Load task metadata
        inst_to_name = scene.get_task_metadata(key="inst_to_name")

        # Assign object_scope based on a cached scene
        for obj_inst in self.object_scope[env_idx]:
            if obj_inst in self.future_obj_instances:
                entity = None
            elif obj_inst not in inst_to_name:
                # Skip instances not found (e.g., future objects
                # when future_obj_instances isn't fully populated yet)
                continue
            else:
                name = inst_to_name[obj_inst]
                is_system = name in scene.available_systems.keys()
                # TODO: If we load a robot with a different set of configs, we will not be able to match with the
                # original object_scope. This is a temporary fix to handle this case. A proper fix involves
                # storing the robot (potentially only base pose) in the task metadata instead of as a regular object
                if "agent.n." in obj_inst:
                    idx = int(obj_inst.split("_")[-1].lstrip("0")) - 1
                    entity = scene.robots[idx]
                else:
                    entity = scene.get_system(name) if is_system else scene.object_registry("name", name)
            self.object_scope[env_idx][obj_inst] = entity

    def update_bddl_scope_metadata(self, env, env_idx):
        """
        Updates the task metadata with the current instance-to-name mapping for all existing entities.

        Args:
            env (Environment): The environment containing the scene to update
            env_idx (int): Index of the environment/scene
        """

        def _get_name(inst, entity):
            if is_system_bddl_inst(inst):
                return og_categories_from_bddl_inst(inst)[0]
            return entity.name

        env.scenes[env_idx].write_task_metadata(
            key="inst_to_name",
            data={
                inst: _get_name(inst, entity)
                for inst, entity in self.object_scope[env_idx].items()
                if entity is not None
            },
        )

    def _get_obs(self, env, env_idx):
        low_dim_obs = dict()

        # Collect non-system instances with existence status, drawn from THIS env's scope only.
        obj_entries = [
            (inst, obj, obj is not None)
            for inst, obj in self.object_scope[env_idx].items()
            if not is_system_bddl_inst(inst)
        ]

        # Batch rpy calculations for much better efficiency
        objs_rpy = T.quat2euler(
            th.stack(
                [
                    obj.states[Pose].get_value()[1] if obj_exist else th.tensor([0, 0, 0, 1.0])
                    for _, obj, obj_exist in obj_entries
                ]
            )
        )
        objs_rpy_cos = th.cos(objs_rpy)
        objs_rpy_sin = th.sin(objs_rpy)

        # Always add agent info first
        agent = self.get_agent(env=env, env_idx=env_idx)

        for (inst, obj, obj_exist), obj_rpy, obj_rpy_cos, obj_rpy_sin in zip(
            obj_entries, objs_rpy, objs_rpy_cos, objs_rpy_sin
        ):
            if obj_exist:
                low_dim_obs[f"{inst}_real"] = th.tensor([1.0])
                low_dim_obs[f"{inst}_pos"] = obj.states[Pose].get_value()[0]
                low_dim_obs[f"{inst}_ori_cos"] = obj_rpy_cos
                low_dim_obs[f"{inst}_ori_sin"] = obj_rpy_sin
                if obj.name != agent.name:
                    for arm in agent.arm_names:
                        grasping_object = agent.is_grasping(arm=arm, candidate_obj=obj)
                        low_dim_obs[f"{inst}_in_gripper_{arm}"] = th.tensor([float(grasping_object)])
            else:
                low_dim_obs[f"{inst}_real"] = th.zeros(1)
                low_dim_obs[f"{inst}_pos"] = th.zeros(3)
                low_dim_obs[f"{inst}_ori_cos"] = th.zeros(3)
                low_dim_obs[f"{inst}_ori_sin"] = th.zeros(3)
                for arm in agent.arm_names:
                    low_dim_obs[f"{inst}_in_gripper_{arm}"] = th.zeros(1)

        return low_dim_obs, dict()

    def _step_termination(self, env, action, infos=None):
        # Run super first
        dones, infos = super()._step_termination(env=env, action=action, infos=infos)

        # Add additional info per env
        goal_status = self._termination_conditions["predicate"].goal_status
        for env_idx in range(self._num_envs):
            infos[env_idx]["goal_status"] = goal_status[env_idx]

        return dones, infos

    def _update_bddl_scope_from_added_obj(self, obj):
        """
        Internal callback function to be called when new objects are added to the simulator to potentially update internal
        bddl object scope

        Args:
            obj (USDObject): Newly imported object
        """
        # Each object belongs to exactly one scene. Find which env's scene owns this object
        # and update only that env's scope.
        for env_idx in range(len(self.object_scope)):
            if obj.scene is not None and obj.scene is not self._env.scenes[env_idx]:
                continue
            for inst, entity in self.object_scope[env_idx].items():
                if (
                    entity is None
                    and not is_system_bddl_inst(inst)
                    and obj.category in set(og_categories_from_bddl_inst(inst))
                ):
                    self.object_scope[env_idx][inst] = obj
                    return

    def _update_bddl_scope_from_removed_obj(self, obj):
        """
        Internal callback function to be called when sim._pre_remove_object() is called to potentially update internal
        bddl object scope

        Args:
            obj (USDObject): Newly removed object
        """
        for env_idx in range(len(self.object_scope)):
            if obj.scene is not None and obj.scene is not self._env.scenes[env_idx]:
                continue
            for inst, entity in self.object_scope[env_idx].items():
                if entity is not None and not is_system_bddl_inst(inst) and obj.name == entity.name:
                    self.object_scope[env_idx][inst] = None
                    return

    def _update_bddl_scope_from_system_init(self, system):
        """
        Internal callback function to be called when system.initialize() is called to potentially update internal
        bddl object scope

        Args:
            system (BaseSystem): Newly initialized system
        """
        for env_idx in range(len(self.object_scope)):
            if system.scene is not None and system.scene is not self._env.scenes[env_idx]:
                continue
            for inst, entity in self.object_scope[env_idx].items():
                if (
                    entity is None
                    and is_system_bddl_inst(inst)
                    and og_categories_from_bddl_inst(inst)[0] == system.name
                ):
                    self.object_scope[env_idx][inst] = system
                    return

    def _update_bddl_scope_from_system_clear(self, system):
        """
        Internal callback function to be called when system.clear() is called to potentially update internal
        bddl object scope

        Args:
            system (BaseSystem): Newly cleared system
        """
        for env_idx in range(len(self.object_scope)):
            if system.scene is not None and system.scene is not self._env.scenes[env_idx]:
                continue
            for inst, entity in self.object_scope[env_idx].items():
                if entity is not None and is_system_bddl_inst(inst) and system.name == entity.name:
                    self.object_scope[env_idx][inst] = None
                    return

    def show_instruction(self, env_idx=0):
        """
        Get current instruction for user

        Args:
            env_idx (int): Index of the environment/scene. Default is 0.

        Returns:
            3-tuple:
                - str: Current goal condition in natural language
                - 3-tuple: (R,G,B) color to assign to text
                - list of USDObject: Relevant objects for the current instruction
        """
        satisfied = (
            self.currently_viewed_instruction
            in self._termination_conditions["predicate"].goal_status[env_idx]["satisfied"]
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
            self.compiled_task.conditions.parsed_goal_conditions
        )
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]

    def save_task(self, env, save_dir=None, override=False, task_relevant_only=False, suffix=None, env_idx=0):
        """
        Writes the current scene configuration to a .json file

        Args:
            env (og.Environment): OmniGibson active environment
            save_dir (None or str): If specified, absolute fpath to the desired directory to write the .json. Default is
                {gm.DATA_PATH}/2025-challenge-task-instances/scenes/<SCENE_MODEL>/json/...>
            override (bool): Whether to override any files already found at the path to write the task .json
            task_relevant_only (bool): Whether to only save the task relevant object scope states. If True, will only
                call dump_state() on all the BDDL instances in self.object_scope, else will save the entire sim state
                via env.scene.save()
            suffix (None or str): If specified, suffix to add onto the end of the scene filename that will be saved
        """
        save_dir = (
            os.path.join(get_dataset_path("2025-challenge-task-instances"), "scenes", self.scene_name, "json")
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
                bddl_name: bddl_obj.dump_state(serialized=False)
                for bddl_name, bddl_obj in env.task.object_scope[env_idx].items()
                if bddl_obj is not None and "agent" not in bddl_name
            }
            Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
            with open(path, "w+") as f:
                json.dump(task_relevant_state_dict, f, cls=TorchEncoder, indent=4)
        else:
            # Update task metadata and save
            self.update_bddl_scope_metadata(env, env_idx)
            env.scenes[env_idx].save(json_path=path)

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
