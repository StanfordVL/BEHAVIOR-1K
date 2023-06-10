import numpy as np
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
from omnigibson.object_states import Pose
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.systems.system_base import get_system
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.utils.bddl_utils import OmniGibsonBDDLBackend, SUBSTANCE_SYNSET_MAPPING, BDDLEntity, \
    BEHAVIOR_ACTIVITIES, BDDLSampler
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
        # Make sure task name is valid if not specifying a predefined problem
        if predefined_problem is None:
            assert activity_name is not None, \
                "Activity name must be specified if no predefined_problem is specified for BehaviorTask!"
            assert_valid_key(key=activity_name, valid_keys=BEHAVIOR_ACTIVITIES, name="Behavior Task")

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
        self.agent_obj_idx = None                                               # int

        # Info for demonstration collection
        self.instruction_order = None                                           # np.array of int
        self.currently_viewed_index = None                                      # int
        self.currently_viewed_instruction = None                                # tuple of str
        self.activity_natural_language_goal_conditions = None                   # str

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
        # Initialize the current activity
        success, self.feedback = self.initialize_activity(env=env)
        # assert success, f"Failed to initialize Behavior Activity. Feedback:\n{self.feedback}"

        # Highlight any task relevant objects if requested
        if self.highlight_task_relevant_objs:
            for entity in self.object_scope.values():
                if entity.synset == "agent":
                    continue
                if not entity.is_system and entity.exists():
                    entity.highlighted = True

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

        # Store index of agent within object scope
        agent = self.get_agent(env=env)
        idx = 0
        for entity in self.object_scope.values():
            if not entity.is_system:
                if entity.wrapped_obj == agent:
                    self.agent_obj_idx = idx
                    break
                else:
                    idx += 1
        assert self.agent_obj_idx is not None, "Could not find valid index for agent within object scope!"

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
        # Assign object_scope based on a cached scene
        for obj_inst in self.object_scope:
            entity = None
            # If the object scope points to the agent
            if obj_inst == "agent.n.01_1":
                entity = BDDLEntity(object_scope=obj_inst, entity=self.get_agent(env))
            # If object is a future
            elif obj_inst in self.future_obj_instances:
                entity = BDDLEntity(object_scope=obj_inst)
            # If the object scope points to an (active) system
            elif self.object_instance_to_category[obj_inst] in SUBSTANCE_SYNSET_MAPPING:
                system_name = SUBSTANCE_SYNSET_MAPPING[self.object_instance_to_category[obj_inst]]
                entity = BDDLEntity(object_scope=obj_inst, entity=get_system(system_name))
            else:
                log.debug(f"checking objects...")
                for sim_obj in og.sim.scene.objects:
                    log.debug(f"checking bddl obj scope for obj: {sim_obj.name}")
                    if hasattr(sim_obj, "bddl_object_scope") and sim_obj.bddl_object_scope == obj_inst:
                        entity = BDDLEntity(object_scope=obj_inst, entity=sim_obj)
                        break
            assert entity is not None, f"Could not assign valid BDDLEntity for {obj_inst}!"
            self.object_scope[obj_inst] = entity

    def _get_obs(self, env):
        low_dim_obs = dict()

        # Batch rpy calculations for much better efficiency
        objs_exist = {obj: obj.exists() for obj in self.object_scope.values() if not obj.is_system}
        objs_rpy = T.quat2euler(np.array([obj.states[Pose].get_value()[1] if obj_exist else np.array([0, 0, 0, 1.0])
                                          for obj, obj_exist in objs_exist.items()]))

        # Always add agent info first
        agent = self.get_agent(env=env)
        low_dim_obs["robot_pos"] = agent.states[Pose].get_value()[1]
        low_dim_obs["robot_ori_cos"] = np.cos(objs_rpy[self.agent_obj_idx])
        low_dim_obs["robot_ori_sin"] = np.sin(objs_rpy[self.agent_obj_idx])

        for idx, ((obj, obj_exist), obj_rpy) in enumerate(zip(objs_exist.items(), objs_rpy)):
            # Skip if agent idx -- note: doesn't increment object!
            if idx == self.agent_obj_idx:
                continue

            # TODO: May need to update checking here to USDObject? Or even baseobject?
            # TODO: How to handle systems as part of obs?
            if obj_exist:
                low_dim_obs[f"{obj.object_scope}_real"] = np.array([1.0])
                low_dim_obs[f"{obj.object_scope}_pos"] = obj.states[Pose].get_value()[0]
                low_dim_obs[f"{obj.object_scope}_ori_cos"] = np.cos(obj_rpy)
                low_dim_obs[f"{obj.object_scope}_ori_sin"] = np.sin(obj_rpy)
                for arm in agent.arm_names:
                    grasping_object = agent.is_grasping(arm=arm, candidate_obj=obj.wrapped_obj)
                    low_dim_obs[f"{obj.object_scope}_in_gripper_{arm}"] = np.array([float(grasping_object)])
            else:
                low_dim_obs[f"{obj.object_scope}_real"] = np.zeros(1)
                low_dim_obs[f"{obj.object_scope}_pos"] = np.zeros(3)
                low_dim_obs[f"{obj.object_scope}_ori_cos"] = np.zeros(3)
                low_dim_obs[f"{obj.object_scope}_ori_sin"] = np.zeros(3)
                for arm in agent.arm_names:
                    low_dim_obs[f"{obj.object_scope}_in_gripper_{arm}"] = np.zeros(1)

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
