import argparse
import logging
import os
import time
from collections import OrderedDict

import gym
import numpy as np

from transforms3d.euler import euler2quat

import igibson.macros as m
from igibson import object_states
from igibson.envs.env_base import BaseEnv
from igibson.robots.robot_base import BaseRobot
from igibson.tasks import REGISTERED_TASKS
from igibson.scenes import REGISTERED_SCENES
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.utils import quatToXYZW
from igibson.utils.python_utils import assert_valid_key, create_class_from_registry_and_config


# How many predefined randomized scene object configurations we have per scene
N_PREDEFINED_OBJ_RANDOMIZATIONS = 10


class iGibsonEnv(BaseEnv):
    """
    iGibson Environment (OpenAI Gym interface).
    """

    def __init__(
        self,
        configs,
        scene_model=None,
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        rendering_settings=None,
        vr=False,
        vr_settings=None,
        device_idx=0,
        automatic_reset=False,
    ):
        """
        :param configs (str or list of str): config_file path(s). If multiple configs are specified, they will
            be merged sequentially in the order specified. This allows procedural generation of a "full" config from
            small sub-configs.
        :param scene_model: override scene_id in config file
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        vr (bool): Whether we're using VR or not
        :param vr_settings: vr_settings to override the default one
        :param device_idx: which GPU to run the simulation and rendering on
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        # Store settings and other initialized values
        self._automatic_reset = automatic_reset
        self._predefined_object_randomization_idx = 0
        self._n_predefined_object_randomizations = N_PREDEFINED_OBJ_RANDOMIZATIONS

        # Initialize other placeholders that will be filled in later
        self._ignore_robot_object_collisions = None         # This will be a list of len(scene.robots) containing sets of objects to ignore collisions for
        self._ignore_robot_self_collisions = None           # This will be a list of len(scene.robots) containing sets of RigidPrims (robot's links) to ignore collisions for
        self._initial_pos_z_offset = None                   # how high to offset object placement to account for one action step of dropping
        self._texture_randomization_freq = None
        self._object_randomization_freq = None
        self._task = None
        self._current_episode = 0

        # Variables reset at the beginning of each episode
        self._current_collisions = None
        self._current_step = 0
        self._collision_step = 0

        # Run super
        super(iGibsonEnv, self).__init__(
            configs=configs,
            scene_model=scene_model,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            rendering_settings=rendering_settings,
            vr=vr,
            vr_settings=vr_settings,
            device_idx=device_idx,
        )

    def add_ignore_robot_object_collision(self, robot_idn, obj):
        """
        Add a new robot-object pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            obj (BaseObject): Which object to ignore a collision for
        """
        self._ignore_robot_object_collisions[robot_idn].add(obj)

    def add_ignore_robot_self_collision(self, robot_idn, link):
        """
        Add a new robot-link self pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            link (RigidPrim): Which robot link to ignore a collision for
        """
        self._ignore_robot_self_collisions[robot_idn].add(link)

    def _load_variables(self):
        # Store additional variables after config has been loaded fully
        self._initial_pos_z_offset = self.env_config["initial_pos_z_offset"]
        self._texture_randomization_freq = self.env_config["texture_randomization_freq"]
        self._object_randomization_freq = self.env_config["object_randomization_freq"]

        # Set other values
        self._ignore_robot_object_collisions = [[] for _ in self.robots_config]
        self._ignore_robot_self_collisions = [[] for _ in self.robots_config]

        # Reset bookkeeping variables
        self._reset_variables()
        self._current_episode = 0           # Manually set this to 0 since resetting actually increments this

        # - Potentially overwrite the USD entry for the scene if none is specified and we're online sampling -

        # Make sure the requested scene is valid
        scene_type = self.scene_config["type"]
        assert_valid_key(key=scene_type, valid_keys=REGISTERED_SCENES, name="scene type")

        # If we're using a BehaviorTask, we may load a pre-cached scene configuration
        if self.task_config["type"] == "BehaviorTask":
            usd_file = self.scene_config["usd_file"]
            if usd_file is None and not self.task_config["online_object_sampling"]:
                usd_file = "{}_task_{}_{}_{}_fixed_furniture_template".format(
                    self.scene_config["scene_model"],
                    self.task_config["activity_name"],
                    self.task_config["activity_definition_id"],
                    self.task_config["activity_instance_id"],
                )
            # Update the value in the scene config
            self.scene_config["usd_file"] = usd_file

        # - Additionally run some sanity checks on these values -

        # Check to make sure our z offset is valid -- check that the distance travelled over 1 action timestep is
        # less than the offset we set (dist = 0.5 * gravity * (t^2))
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self._initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

    def _load_task(self):
        """
        Load task
        """
        # Sanity check task to make sure it's valid
        task_type = self.task_config["type"]
        assert_valid_key(key=task_type, valid_keys=REGISTERED_TASKS, name="task type")

        # Grab the kwargs relevant for the specific task and create the task
        self._task = create_class_from_registry_and_config(
            cls_name=self.task_config["type"],
            cls_registry=REGISTERED_TASKS,
            cfg=self.task_config,
            cls_type_descriptor="task",
        )

        # Load task
        # TODO: Does this need to occur somewhere else?
        self._task.load(env=self)

    def _load_observation_space(self):
        # Grab robot(s) and task obs spaces
        obs_space = OrderedDict()

        for robot in self.robots:
            # Load the observation space for the robot
            obs_space[robot.name] = robot.load_observation_space()

        # Also load the task obs space
        obs_space["task"] = self._task.load_observation_space()
        return obs_space

    def _load_action_space(self):
        """
        Load action space for each robot
        """
        self.action_space = gym.spaces.Dict({robot.name: robot.action_space for robot in self.robots})

    def load(self):
        """
        Load environment.
        """
        # Do any normal loading first from the super() call
        super().load()

        # Load the task
        self._load_task()

        # Start the simulation, then reset the environment
        self.simulator.play()
        self.reset()

        # Update the initial scene state
        self.scene.update_initial_state()

        # Load the obs / action spaces
        self.load_observation_space()
        self._load_action_space()

    def reload_model_object_randomization(self, predefined_object_randomization_idx=None):
        """
        Reload the same model, with either @object_randomization_idx seed or the next object randomization random seed.

        Args:
            predefined_object_randomization_idx (None or int): If set, specifies the specific pre-defined object
                randomization instance to use. Otherwise, the current seed will be incremented by 1 and used.
        """
        assert self._object_randomization_freq is not None, \
            "object randomization must be active to reload environment with object randomization!"
        self._predefined_object_randomization_idx = predefined_object_randomization_idx if \
            predefined_object_randomization_idx is not None else \
            (self._predefined_object_randomization_idx + 1) % self._n_predefined_object_randomizations
        self.load()

    def get_obs(self):
        # Grab obs from super call
        obs = super().get_obs()

        # Add task obs
        obs["task"] = self._task.get_obs(env=self)

        return obs

    # TODO!! Wait until omni dev team has an answer
    def update_collisions(self, filtered=True):
        """
        Grab collisions that occurred during the most recent physics timestep

        Args:
            filtered (bool): Whether to filter the raw collisions or not based on
                self._ignore_robot_[object/self]_collisions

        Returns:
            set of 2-tuple: Unique collision pairs occurring in the simulation at the current timestep, represented
                by their prim_paths
        """
        # Grab collisions based on the status of our contact reporting macro flag
        if m.ENABLE_GLOBAL_CONTACT_REPORTING:
            collisions = {(c.body0, c.body1)
                          for obj_group in (self.scene.objects, self.robots)
                          for obj in obj_group
                          for c in obj.contact_list()}
        elif m.ENABLE_ROBOT_CONTACT_REPORTING:
            collisions = {(c.body0, c.body1) for robot in self.robots for c in robot.contact_list()}
        else:
            collisions = set()
        self._current_collisions = self._filter_collisions(collisions) if filtered else collisions
        return self._current_collisions

    def _filter_collisions(self, collisions):
        """
        Filter out collisions that should be ignored, based on self._ignore_robot_[object/self]_collisions

        Args:
            collisions (set of 2-tuple): Unique collision pairs occurring in the simulation at the current timestep,
                represented by their prim_paths

        Returns:
            set of 2-tuple: Filtered collision pairs occurring in the simulation at the current timestep,
                represented by their prim_paths
        """
        # Iterate over all robots
        new_collisions = set()
        for robot, filtered_links, filtered_objs in zip(
                self.robots, self._ignore_robot_self_collisions, self._ignore_robot_object_collisions
        ):
            # Grab all link prim paths owned by the robot
            robot_prims = {link.prim_path for link in robot.links.values()}
            # Loop over all filtered links and compose them
            filtered_link_prims = {link.prim_path for link in filtered_links}
            # Loop over all filtered objects and compose them
            filtered_obj_prims = {link.prim_path for obj in filtered_objs for link in obj.links.values()}
            # Get union of robot and obj prims
            filtered_robot_obj_prims = robot_prims.union(filtered_obj_prims)
            # Iterate over all collision pairs
            for col_pair in collisions:
                # First check for self_collision -- we check by first subtracting all filtered links from the col_pair
                # set and making sure the length is < 2, and then subtracting all robot_prims and making sure that the
                # length is 0. If both conditions are met, then we know this is a filtered collision!
                col_pair_set = set(col_pair)
                if len(col_pair_set - filtered_link_prims) < 2 and len(col_pair_set - robot_prims) == 0:
                    # Filter this collision
                    continue
                # Check for object filtering -- we check by first subtracting all robot links from the col_pair
                # set and making sure the length is < 2, and then subtracting all filtered_obj_prims and making sure
                # that the length is < 2. If both conditions are met, this means that this was a collision between
                # the robot and a filtered object, so we know this is a filtered collision!
                elif len(col_pair_set - robot_prims) < 2 and len(col_pair_set - filtered_obj_prims) < 2:
                    # Filter this collision
                    continue
                else:
                    # Add this collision
                    new_collisions.add(col_pair)

        return new_collisions

    def _populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        Args:
            info (dict): Information dictionary to populate

        Returns:
            dict: Information dictionary with added info
        """
        info["episode_length"] = self._current_step
        info["collision_step"] = self._collision_step

    def step(self, action=None):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: gym.spaces.Dict, dict, np.array, or None, robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        # Apply actions if specified
        if action is not None:
            # If the action is not a dictionary, convert into a dictionary
            if not isinstance(action, dict) and not isinstance(action, gym.spaces.Dict):
                action_dict = OrderedDict()
                idx = 0
                for robot in self.robots:
                    action_dim = robot.action_dim
                    action_dict[robot.name] = action[idx: idx + action_dim]
                    idx += action_dim
            else:
                # Our inputted action is the action dictionary
                action_dict = action

            # Iterate over all robots and apply actions
            for robot in self.robots:
                robot.apply_action(action_dict[robot.name])

        # Run simulation step
        self._simulator_step()

        # Grab collisions and store internally
        collisions = self.update_collisions(filtered=True)

        # Update the collision count
        self._collision_step += int(len(collisions) > 0)

        # Grab observations
        obs = self.get_obs()

        # Grab reward, done, and info, and populate with internal info
        reward, done, info = self.task.step(self, action)
        self._populate_info(info)

        if done and self._automatic_reset:
            # Add lost observation to our information dict, and reset
            info["last_observation"] = obs
            obs = self.reset()

        # Increment step
        self._current_step += 1


        return obs, reward, done, info

    def check_collision(self, obj):
        """
        Check whether the given object @obj has collision after one simulator step

        Args:
            obj (EntityPrim): Object to check for collision

        Returns:
            bool: Whether the object @obj is in collision or not
        """
        # Run simulator step and update contacts
        self._simulator_step()
        collisions = self.update_collisions(filtered=True)

        # Grab all link prim paths owned by the object
        link_paths = {link.prim_path for link in obj.links.values()}

        # Loop over all current collisions and check for any matches
        in_collision = False
        for col_pair in collisions:
            if len(set(col_pair) - link_paths) < 2:
                in_collision = True
                break

        # Only going into this if it is for logging --> efficiency
        if logging.root.level <= logging.DEBUG:
            for item in collisions:
                logging.debug("linkA:{}, linkB:{}".format(item[0], item[1]))

        return in_collision

    # TODO!
    def set_pos_orn_with_z_offset(self, obj, pos, ori=None, offset=None):
        """
        Reset position and orientation for the object @obj with optional offset @offset.

        Args:
            obj (BaseObject): Object to place in the environment
            pos (3-array): Global (x,y,z) location to place the object
            ori (None or 3-array): Optional (r,p,y) orientation when placing the robot. If None, a random orientation
                about the z-axis will be sampled
            offset (None or float): Optional z-offset to place object with. If None, default self._initial_pos_z_offset
                will be used
        """
        ori = np.array([0, 0, np.random.uniform(0, np.pi * 2)]) if ori is None else ori
        offset = self._initial_pos_z_offset if offset is None else offset

        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*ori), "wxyz"))
        # get the AABB in this orientation
        # lower, _ = obj.states[object_states.AABB].get_value() # TODO!
        lower = np.array([0, 0, pos[2]])
        # Get the stable Z
        stable_z = pos[2] + (pos[2] - lower[2])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position(np.array([pos[0], pos[1], stable_z + offset]))

    def test_valid_position(self, obj, pos, ori=None):
        """
        Test if the object can be placed with no collision.

        Args:
            obj (BaseObject): Object to place in the environment
            pos (3-array): Global (x,y,z) location to place the object
            ori (None or 3-array): Optional (r,p,y) orientation when placing the robot. If None, a random orientation
                about the z-axis will be sampled

        Returns:
            bool: Whether the placed object position is valid
        """
        # Set the position of the object
        self.set_pos_orn_with_z_offset(obj=obj, pos=pos, ori=ori)

        # If we're placing a robot, make sure it's reset and not moving
        if isinstance(obj, BaseRobot):
            obj.reset()
            obj.keep_still()

        # Valid if there are no collisions
        return not self.check_collision(obj=obj)

    def land(self, obj, pos, ori):
        """
        Land the object at the specified position @pos, given a valid position and orientation.

        Args:
            obj (BaseObject): Object to place in the environment
            pos (3-array): Global (x,y,z) location to place the object
            ori (None or 3-array): Optional (r,p,y) orientation when placing the robot. If None, a random orientation
                about the z-axis will be sampled
        """
        # Set the position of the object
        self.set_pos_orn_with_z_offset(obj=obj, pos=pos, ori=ori)

        # If we're placing a robot, make sure it's reset and not moving
        is_robot = isinstance(obj, BaseRobot)
        if is_robot:
            obj.reset()
            obj.keep_still()

        # Check to make sure we landed successfully
        # land for maximum 1 second, should fall down ~5 meters
        land_success = False
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            # Run a sim step and see if we have any contacts
            self._simulator_step()
            self.update_collisions(filtered=False)
            land_success = self.check_collision(obj=obj)
            if land_success:
                # Once we're successful, we can break immediately
                print(f"Landed robot successfully!")
                break

        # Print out warning in case we failed to land the object successfully
        if not land_success:
            logging.warning("Object failed to land.")

        # Make sure robot isn't moving at the end if we're a robot
        if is_robot:
            obj.reset()
            obj.keep_still()

    def _reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self._current_episode += 1
        self._current_collisions = None
        self._current_step = 0
        self._collision_step = 0

    def randomize_domain(self):
        """
        Randomize domain.
        Object randomization loads new object models with the same poses.
        Texture randomization loads new materials and textures for the same object models.
        """
        if self._object_randomization_freq is not None:
            if self._current_episode % self._object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self._texture_randomization_freq is not None:
            if self._current_episode % self._texture_randomization_freq == 0:
                self._simulator.scene.randomize_texture()

    def reset(self):
        """
        Reset episode.
        """
        # Stop and restart the simulation
        self._simulator.stop()
        self._simulator.play()

        # Do any domain randomization
        self.randomize_domain()

        # # Move all robots away from the scene since the task will place the robots anyways
        # for robot in self.robots:
        #     robot.set_position(np.array([100.0, 100.0, 100.0]))

        # Reset the task
        self.task.reset(self)

        # Reset internal variables
        self._reset_variables()

        # Run a single simulator step to make sure we can grab updated observations
        self._simulator_step()

        # Grab and return observations
        obs = self.get_obs()

        if self.observation_space is not None and not self.observation_space.contains(obs):
            print("Error: Observation space does not match returned observations!")
            # Print out all observations for all robots and task
            for robot in self.robots:
                for key, value in self.observation_space[robot.name].items():
                    print(key, value.dtype, value.shape)
                    print('obs', obs['robot0'][key].dtype, obs[robot.name][key].shape)
            for key, value in self.observation_space['task'].items():
                print(key, value.dtype, value.shape)
                print('obs', obs['task'][key].dtype, obs['task'][key].shape)

        return obs

    @property
    def episode_steps(self):
        """
        Returns:
            int: Current number of steps in episode
        """
        return self._current_step

    @property
    def episode_collisions(self):
        """
        Returns:
            int: Total number of collisions in current episode
        """
        return self._collision_step

    @property
    def current_collisions(self):
        """
        Returns:
            set of 2-tuple: Cached collision pairs from the last time self.update_collisions() was called
        """
        return self._current_collisions

    @property
    def task(self):
        """
        Returns:
            BaseTask: Active task instance
        """
        return self._task

    @property
    def task_config(self):
        """
        Returns:
            dict: Task-specific configuration kwargs
        """
        return self.config["task"]

    @property
    def default_config(self):
        # Call super first to grab initial pre-populated config
        cfg = super().default_config

        # Add additional values
        cfg["env"]["initial_pos_z_offset"] = 0.1
        cfg["env"]["texture_randomization_freq"] = None
        cfg["env"]["object_randomization_freq"] = None
        cfg["scene"]["usd_file"] = None

        # Add task kwargs
        cfg["task"] = {
            "type": "DummyTask",

            # If we're using a BehaviorTask
            "activity_definition_id": 0,
            "activity_instance_id": 0,
        }

        return cfg

# TODO!
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default="headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = iGibsonEnv(config_file=args.config, mode=args.mode, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)

    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        for _ in range(100):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print("reward", reward)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.episode_steps, time.time() - start))
    env.close()
