import numpy as np
import os
from bddl.activity import (
    Conditions,
    evaluate_goal_conditions,
    get_goal_conditions,
    get_ground_goal_state_options,
    get_initial_conditions,
    get_natural_goal_conditions,
    get_object_scope,
)

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Pose
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.systems.system_base import get_system, add_callback_on_system_init, add_callback_on_system_clear, \
    REGISTERED_SYSTEMS
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.utils.bddl_utils import OmniGibsonBDDLBackend, BDDLEntity, BEHAVIOR_ACTIVITIES, BDDLSampler
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout
import omnigibson.utils.transform_utils as T
from omnigibson.utils.python_utils import classproperty, assert_valid_key
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
        debug_object_sampling (bool): whether to debug placement functionality
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
            debug_object_sampling=False,
            highlight_task_relevant_objects=False,
            termination_config=None,
            reward_config=None,
    ):
        # Make sure object states are enabled
        assert gm.ENABLE_OBJECT_STATES, "Must set gm.ENABLE_OBJECT_STATES=True in order to use BehaviorTask!"

        # Make sure task name is valid if not specifying a predefined problem
        if predefined_problem is None:
            assert activity_name is not None, \
                "Activity name must be specified if no predefined_problem is specified for BehaviorTask!"
            assert_valid_key(key=activity_name, valid_keys=BEHAVIOR_ACTIVITIES, name="Behavior Task")
        else:
            # Infer activity name
            activity_name = predefined_problem.split("problem ")[-1].split("-")[0]

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
        self.ground_goal_state_options = None
        self.feedback = None                                                    # None or str
        self.sampler = None                                                     # BDDLSampler

        # Object info
        self.debug_object_sampling = debug_object_sampling                      # bool
        self.online_object_sampling = online_object_sampling                    # bool
        self.highlight_task_relevant_objs = highlight_task_relevant_objects     # bool
        self.object_scope = None                                                # Maps str to BDDLEntity
        self.object_instance_to_category = None                                 # Maps str to str
        self.future_obj_instances = None                                        # set of str

        # Info for demonstration collection
        self.instruction_order = None                                           # np.array of int
        self.currently_viewed_index = None                                      # int
        self.currently_viewed_instruction = None                                # tuple of str
        self.activity_natural_language_goal_conditions = None                   # str

        # Load the initial behavior configuration
        self.update_activity(activity_name=activity_name, activity_definition_id=activity_definition_id, predefined_problem=predefined_problem)

        # Run super init
        super().__init__(termination_config=termination_config, reward_config=reward_config)

    @classmethod
    def get_cached_activity_scene_filename(cls, scene_model, activity_name, activity_definition_id, activity_instance_id):
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
        activity_name = task_cfg["predefined_problem"].split("problem ")[-1].split("-")[0] if \
            task_cfg.get("predefined_problem", None) is not None else task_cfg["activity_name"]
        if scene_file is None and scene_instance is None and not task_cfg["online_object_sampling"]:
            scene_instance = cls.get_cached_activity_scene_filename(
                scene_model=scene_cfg["scene_model"],
                activity_name=activity_name,
                activity_definition_id=task_cfg.get("activity_definition_id", 0),
                activity_instance_id=task_cfg.get("activity_instance_id", 0),
            )
            # Update the value in the scene config
            scene_cfg["scene_instance"] = scene_instance

    def write_task_metadata(self):
        # Store mapping from entity name to its corresponding BDDL instance name
        metadata = dict(
            inst_to_name={inst: entity.name for inst, entity in self.object_scope.items() if entity.exists},
        )

        # Write to sim
        og.sim.write_metadata(key="task", data=metadata)

    def load_task_metadata(self):
        # Load from sim
        return og.sim.get_metadata(key="task")

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
        # Initialize the current activity
        success, self.feedback = self.initialize_activity(env=env)
        # assert success, f"Failed to initialize Behavior Activity. Feedback:\n{self.feedback}"

        # Highlight any task relevant objects if requested
        if self.highlight_task_relevant_objs:
            for entity in self.object_scope.values():
                if entity.synset == "agent":
                    continue
                if not entity.is_system and entity.exists:
                    entity.highlighted = True

        # Add callbacks to handle internal processing when new systems / objects are added / removed to the scene
        callback_name = f"{self.activity_name}_refresh"
        og.sim.add_callback_on_import_obj(name=callback_name, callback=self._update_bddl_scope_from_added_obj)
        og.sim.add_callback_on_remove_obj(name=callback_name, callback=self._update_bddl_scope_from_removed_obj)
        add_callback_on_system_init(name=callback_name, callback=self._update_bddl_scope_from_system_init)
        add_callback_on_system_clear(name=callback_name, callback=self._update_bddl_scope_from_system_clear)

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

        # Generate sampler
        self.sampler = BDDLSampler(
            env=env,
            activity_conditions=self.activity_conditions,
            object_scope=self.object_scope,
            backend=self.backend,
            debug=self.debug_object_sampling,
        )

        # Compose future objects
        self.future_obj_instances = \
            {init_cond.body[1] for init_cond in self.activity_initial_conditions if init_cond.body[0] == "future"}

        if self.online_object_sampling:
            # Sample online
            accept_scene, feedback = self.sampler.sample()
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
        inst_to_name = self.load_task_metadata()["inst_to_name"]

        # Assign object_scope based on a cached scene
        for obj_inst in self.object_scope:
            if obj_inst in self.future_obj_instances:
                entity = None
            else:
                assert obj_inst in inst_to_name, f"BDDL object instance {obj_inst} should exist in cached metadata " \
                                                 f"from loaded scene, but could not be found!"
                name = inst_to_name[obj_inst]
                is_system = name in REGISTERED_SYSTEMS
                entity = get_system(name) if is_system else og.sim.scene.object_registry("name", name)
            self.object_scope[obj_inst] = BDDLEntity(
                bddl_inst=obj_inst,
                entity=entity,
            )

    def _get_obs(self, env):
        low_dim_obs = dict()

        # Batch rpy calculations for much better efficiency
        objs_exist = {obj: obj.exists for obj in self.object_scope.values() if not obj.is_system}
        objs_rpy = T.quat2euler(np.array([obj.states[Pose].get_value()[1] if obj_exist else np.array([0, 0, 0, 1.0])
                                          for obj, obj_exist in objs_exist.items()]))
        objs_rpy_cos = np.cos(objs_rpy)
        objs_rpy_sin = np.sin(objs_rpy)

        # Always add agent info first
        agent = self.get_agent(env=env)

        for (obj, obj_exist), obj_rpy, obj_rpy_cos, obj_rpy_sin in zip(objs_exist.items(), objs_rpy, objs_rpy_cos, objs_rpy_sin):

            # TODO: May need to update checking here to USDObject? Or even baseobject?
            # TODO: How to handle systems as part of obs?
            if obj_exist:
                low_dim_obs[f"{obj.bddl_inst}_real"] = np.array([1.0])
                low_dim_obs[f"{obj.bddl_inst}_pos"] = obj.states[Pose].get_value()[0]
                low_dim_obs[f"{obj.bddl_inst}_ori_cos"] = obj_rpy_cos
                low_dim_obs[f"{obj.bddl_inst}_ori_sin"] = obj_rpy_sin
                if obj.name != agent.name:
                    for arm in agent.arm_names:
                        grasping_object = agent.is_grasping(arm=arm, candidate_obj=obj.wrapped_obj)
                        low_dim_obs[f"{obj.bddl_inst}_in_gripper_{arm}"] = np.array([float(grasping_object)])
            else:
                low_dim_obs[f"{obj.bddl_inst}_real"] = np.zeros(1)
                low_dim_obs[f"{obj.bddl_inst}_pos"] = np.zeros(3)
                low_dim_obs[f"{obj.bddl_inst}_ori_cos"] = np.zeros(3)
                low_dim_obs[f"{obj.bddl_inst}_ori_sin"] = np.zeros(3)
                for arm in agent.arm_names:
                    low_dim_obs[f"{obj.bddl_inst}_in_gripper_{arm}"] = np.zeros(1)

        return low_dim_obs, dict()

    def _step_termination(self, env, action, info=None):
        # Run super first
        done, info = super()._step_termination(env=env, action=action, info=info)

        # Add additional info
        info["goal_status"] = self._termination_conditions["predicate"].goal_status

        return done, info

    def _update_bddl_scope_from_added_obj(self, obj):
        """
        Internal callback function to be called when sim.import_object() is called to potentially update internal
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
        Internal callback function to be called when sim.remove_object() is called to potentially update internal
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

    def save_task(self, path=None, override=False):
        """
        Writes the current scene configuration to a .json file

        Args:
            path (None or str): If specified, absolute fpath to the desired path to write the .json. Default is
                <gm.DATASET_PATH/scenes/<SCENE_MODEL>/json/...>
            override (bool): Whether to override any files already found at the path to write the task .json
        """
        if path is None:
            fname = self.get_cached_activity_scene_filename(
                scene_model=og.sim.scene.scene_model,
                activity_name=self.activity_name,
                activity_definition_id=self.activity_definition_id,
                activity_instance_id=self.activity_instance_id,
            )
            path = os.path.join(gm.DATASET_PATH, "scenes", og.sim.scene.scene_model, "json", f"{fname}.json")

        if os.path.exists(path) and not override:
            log.warning(f"Scene json already exists at {path}. Use override=True to force writing of new json.")
            return
        # Write metadata and then save
        self.write_task_metadata()
        og.sim.save(json_path=path)

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
