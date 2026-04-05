import h5py
import json
import logging
import torch as th
from typing import Any

import omnigibson as og
from omnigibson.controllers.controller_base import ControlType
from omnigibson.envs.env_base import Environment
from omnigibson.envs.env_wrapper import EnvironmentWrapper, create_wrapper
from omnigibson.macros import gm, macros
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
from omnigibson.utils.data_utils import merge_scene_files
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)
log.setLevel(logging.INFO)


class DataWrapper(EnvironmentWrapper):
    """
    An OmniGibson environment wrapper for writing data to a dataset file.
    """

    def __init__(
        self,
        env: Environment,
        output_path: str,
        overwrite: bool = True,
        only_successes: bool = True,
        flush_every_n_traj: int = 10,
    ) -> None:
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): path to store data file
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file

        Note:
            ``self._fps`` is only initialized from ``env.env_config["rendering_frequency"]`` if it does not
            already exist. This allows subclasses (e.g. playback wrappers) to set an FPS sourced from recorded
            dataset metadata before calling ``super().__init__``.
        """
        # Make sure the wrapped environment inherits correct omnigibson format
        assert isinstance(
            env, (Environment, EnvironmentWrapper)
        ), "Expected wrapped @env to be a subclass of OmniGibson's Environment class or EnvironmentWrapper!"

        # Only one scene is supported for now
        assert len(og.sim.scenes) == 1, "Only one scene is currently supported for DataWrapper env!"

        self.traj_count = 0
        self.step_count = 0
        self.only_successes = only_successes
        self.flush_every_n_traj = flush_every_n_traj
        self.current_obs = None
        self.current_traj_history = []

        if not hasattr(self, "_fps"):
            self._fps = int(env.env_config["rendering_frequency"])

        # Create dataset
        self.create_dataset(output_path, env, overwrite=overwrite)

        # Run super
        super().__init__(env=env)

    def create_dataset(self, output_path: str, env: Environment, overwrite: bool = True) -> None:
        """
        Creates a dataset at @output_path, possibly overwriting it if @overwrite is set

        Args:
            output_path (str): path to store data. May be either directory or filepath depending on the
                dataset type
            env (Environment): The wrapped environment
            overwrite (bool): Whether to overwrite any pre-existing data or not
        """
        raise NotImplementedError

    @property
    def fps(self) -> int:
        """
        Returns:
            int: Frames per second used by this wrapper.
        """
        return self._fps

    def step(self, action: th.Tensor | dict, n_render_iterations: int = 1) -> tuple[dict, float, bool, bool, dict]:
        """
        Run the environment step() function and collect data

        Args:
            action (th.Tensor | dict): action to take in environment
            n_render_iterations (int): Number of rendering iterations to use before returning observations

        Returns:
            5-tuple:
                - dict: state, i.e. next observation
                - float: reward, i.e. reward at this current timestep
                - bool: terminated, i.e. whether this episode ended due to a failure or success
                - bool: truncated, i.e. whether this episode ended due to a time limit etc.
                - dict: info, i.e. dictionary with any useful information
        """
        # Make sure actions are always flattened numpy arrays
        if isinstance(action, dict):
            action = th.cat([act for act in action.values()])

        next_obs, reward, terminated, truncated, info = self.env.step(action, n_render_iterations=n_render_iterations)
        self.step_count += 1

        self._record_step_trajectory(action, next_obs, reward, terminated, truncated, info)

        return next_obs, reward, terminated, truncated, info

    def _record_step_trajectory(
        self, action: th.Tensor, obs: dict, reward: float, terminated: bool, truncated: bool, info: dict
    ) -> None:
        """
        Record the current step data to the trajectory history

        Args:
            action (th.Tensor): action deployed resulting in @obs
            obs (dict): state, i.e. observation
            reward (float): reward, i.e. reward at this current timestep
            terminated (bool): terminated, i.e. whether this episode ended due to a failure or success
            truncated (bool): truncated, i.e. whether this episode ended due to a time limit etc.
            info (dict): info, i.e. dictionary with any useful information
        """
        # Aggregate step data
        step_data = self._parse_step_data(action, obs, reward, terminated, truncated, info)

        # Update obs and traj history
        self.current_traj_history.append(step_data)
        self.current_obs = obs

    def _parse_step_data(
        self, action: th.Tensor, obs: dict, reward: float, terminated: bool, truncated: bool, info: dict
    ) -> dict:
        """
        Parse the output from the internal self.env.step() call and write relevant data to record to a dictionary

        Args:
            action (th.Tensor): action deployed resulting in @obs
            obs (dict): state, i.e. observation
            reward (float): reward, i.e. reward at this current timestep
            terminated (bool): terminated, i.e. whether this episode ended due to a failure or success
            truncated (bool): truncated, i.e. whether this episode ended due to a time limit etc.
            info (dict): info, i.e. dictionary with any useful information

        Returns:
            dict: Keyword-mapped data that should be recorded in the dataset.
        """
        raise NotImplementedError()

    def _process_obs(self, obs: dict, info: dict) -> dict:
        """
        Pre-process the raw observation data from the environment into the desired format for storing in the dataset.
        """
        # Default is no-op
        return obs

    def reset(self) -> tuple[dict, dict]:
        """
        Run the environment reset() function and flush data

        Returns:
            2-tuple:
                - dict: Environment observation space after reset occurs
                - dict: Information related to observation metadata
        """
        if len(self.current_traj_history) > 0:
            self.flush_current_traj()

        self.current_obs, info = self.env.reset()
        # Store initial obs as the first entry in the trajectory
        self.current_traj_history.append({"obs": self._process_obs(self.current_obs, info)})

        return self.current_obs, info

    def process_traj_to_dataset(self, traj_data: list[dict]) -> Any:
        """
        Process the given trajectory data and write it to the dataset.
        This is called at the end of every episode for any trajectories that should be saved,
        and is where the logic for how trajectory data should be stored in the dataset should be implemented.

        Args:
            traj_data (list of dict): Trajectory data, where each entry is a keyword-mapped set of data for a single
                sim step
        """
        raise NotImplementedError()

    @property
    def should_save_current_episode(self) -> bool:
        """
        Returns:
            bool: Whether the current episode should be saved or discarded
        """
        # Only save successful demos and if actually recording,
        # or there's only one observation in the trajectory (i.e. the initial obs after reset)
        return (self.env.task.success or not self.only_successes) and not (
            len(self.current_traj_history) == 1 and set(self.current_traj_history[0].keys()) == {"obs"}
        )

    def flush_current_traj(self) -> None:
        """
        Flush current trajectory data
        """
        # Only save successful demos and if actually recording
        if self.should_save_current_episode:
            self.process_traj_to_dataset(self.current_traj_history)
            self.traj_count += 1

            # Potentially write to disk
            if self.traj_count % self.flush_every_n_traj == 0:
                self.flush_current_file()
        else:
            # Remove this demo
            self.step_count -= len(self.current_traj_history)

        # Clear trajectory and transition buffers
        self.current_traj_history = []

    def flush_current_file(self) -> None:
        raise NotImplementedError

    def save_data(self) -> None:
        """
        Flushes any remaining data and saves the dataset to disk
        """
        if len(self.current_traj_history) > 0:
            self.flush_current_traj()

        self.close_dataset()

    def close_dataset(self) -> None:
        """
        Closes the active dataset, if open
        """
        raise NotImplementedError


class DataPlaybackWrapper(DataWrapper):
    """
    An OmniGibson environment wrapper for playing back data and collecting observations.

    NOTE: This assumes a HDF5CollectionWrapper environment has been used to collect data!
    """

    @classmethod
    def create_from_hdf5(
        cls,
        input_path: str,
        output_path: str,
        robot_obs_modalities: tuple[str, ...] = tuple(),
        robot_proprio_keys: list[str] | None = None,
        robot_sensor_config: dict[str, Any] | None = None,
        external_sensors_config: list[dict[str, Any]] | None = None,
        include_sensor_names: list[str] | None = None,
        exclude_sensor_names: list[str] | None = None,
        n_render_iterations: int = 1,
        overwrite: bool = True,
        only_successes: bool = False,
        flush_every_n_traj: int = 10,
        include_env_wrapper: bool = False,
        additional_wrapper_configs: list[dict[str, Any]] | None = None,
        full_scene_file: str | None = None,
        include_task: bool = True,
        include_task_obs: bool = True,
        include_robot_control: bool = True,
        include_contacts: bool = True,
        load_room_instances: list[str] | None = None,
        **kwargs,
    ) -> "DataPlaybackWrapper":
        """
        Create a DataPlaybackWrapper environment instance form the recorded demonstration info
        from @hdf5_path, and aggregate observation_modalities @obs during playback

        Args:
            input_path (str): Absolute path to the input hdf5 file containing the relevant collected data to playback
            output_path (str): Absolute path to the output hdf5 file that will contain the recorded observations from
                the replayed data
            robot_obs_modalities (list): Robot observation modalities to use. This list is directly passed into
                the robot_cfg (`obs_modalities` kwarg) when spawning the robot
            robot_proprio_keys (None or list of str): If specified, a list of proprioception keys to use for the robot.
            robot_sensor_config (None or dict): If specified, the sensor configuration to use for the robot. See the
                example sensor_config in fetch_behavior.yaml env config. This can be used to specify relevant sensor
                params, such as image_height and image_width
            external_sensors_config (None or list): If specified, external sensor(s) to use. This will override the
                external_sensors kwarg in the env config when the environment is loaded. Each entry should be a
                dictionary specifying an individual external sensor's relevant parameters. See the example
                external_sensors key in fetch_behavior.yaml env config. This can be used to specify additional sensors
                to collect observations during playback.
            include_sensor_names (None or list of str): If specified, substring(s) to check for in all raw sensor prim
                paths found on the robot. A sensor must include one of the specified substrings in order to be included
                in this robot's set of sensors during playback
            exclude_sensor_names (None or list of str): If specified, substring(s) to check against in all raw sensor
                prim paths found on the robot. A sensor must not include any of the specified substrings in order to
                be included in this robot's set of sensors during playback
            n_render_iterations (int): Number of rendering iterations to use when loading each stored frame from the
                recorded data. This is needed because the omniverse real-time raytracing always lags behind the
                underlying physical state by a few frames, and additionally produces transient visual artifacts when
                the physical state changes. Increasing this number will improve the rendered quality at the expense of
                speed.
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file
            include_env_wrapper (bool): Whether to include environment wrapper stored in the underlying env config
            additional_wrapper_configs (None or list of dict): If specified, list of wrapper config(s) specifying
                environment wrappers to wrap the internal environment class in
            full_scene_file (None or str): If specified, the full scene file to use for playback. During data collection
                the scene file stored may be partial, and will be used to fill in the missing scene objects from the
                full scene file.
            include_task (bool): Whether to include the original task or not. If False, will use a DummyTask instead
            include_task_obs (bool): Whether to include task observations or not. If False, will not include task obs
            include_robot_control (bool): Whether or not to include robot control. If False, will disable all joint control.
            include_contacts (bool): Whether or not to include (enable) contacts in the sim. If False, will set all
                objects to be visual_only
            load_room_instances (None or list of str): If specified, list of room instance names to load during
                playback
            kwargs (dict): Any remaining keyword arguments to pass into class constructor

        Returns:
            DataPlaybackWrapper: Generated playback environment
        """
        # Read from the HDF5 file
        f = h5py.File(input_path, "r")
        config = json.loads(f["data"].attrs["config"])

        # Hot swap in additional info for playing back data

        if include_contacts:
            # Minimize physics leakage during playback (we need to take an env step when loading state)
            config["env"]["action_frequency"] = 1000.0
            config["env"]["rendering_frequency"] = 1000.0
            config["env"]["physics_frequency"] = 1000.0
        else:
            # Since we are setting all objects to be visual-only, preserve frequencies from the input dataset config
            # Simulator-level visual-only set to True
            gm.VISUAL_ONLY = True

        # Make sure obs space is flattened for recording
        config["env"]["flatten_obs_space"] = True

        # Set the scene file either to the one stored in the hdf5 or the hot swap scene file
        config["scene"]["scene_file"] = json.loads(f["data"].attrs["scene_file"])
        if full_scene_file:
            with open(full_scene_file, "r") as json_file:
                full_scene_json = json.load(json_file)
            config["scene"]["scene_file"] = merge_scene_files(
                scene_a=full_scene_json, scene_b=config["scene"]["scene_file"], keep_robot_from="b"
            )
            # Overwrite rooms type to avoid loading room types from the hdf5 file
            config["scene"]["load_room_types"] = None
            config["scene"]["load_room_instances"] = load_room_instances
        else:
            config["scene"]["scene_file"] = json.loads(f["data"].attrs["scene_file"])

        # Use dummy task if not loading task
        if not include_task:
            config["task"] = {"type": "DummyTask"}

        # Maybe include task observations
        config["task"]["include_obs"] = include_task_obs

        # Set scene file and disable online object sampling if BehaviorTask is being used
        if config["task"]["type"] == "BehaviorTask":
            config["task"]["online_object_sampling"] = False
            # Don't use presampled robot pose
            config["task"]["use_presampled_robot_pose"] = False

        # Because we're loading directly from the cached scene file, we need to disable any additional objects that are being added since
        # they will already be cached in the original scene file
        config["objects"] = []

        # Set observation modalities and update sensor config
        for robot_cfg in config["robots"]:
            robot_cfg["obs_modalities"] = list(robot_obs_modalities)
            robot_cfg["include_sensor_names"] = include_sensor_names
            robot_cfg["exclude_sensor_names"] = exclude_sensor_names
            if robot_proprio_keys is not None:
                robot_cfg["proprio_obs"] = robot_proprio_keys
            if robot_sensor_config is not None:
                robot_cfg["sensor_config"] = robot_sensor_config
                # Extract modalities from sensor_config and add to obs_modalities
                for sensor_cfg in robot_cfg["sensor_config"].values():
                    if "modalities" in sensor_cfg:
                        modalities = sensor_cfg["modalities"]
                        if isinstance(modalities, list):
                            robot_cfg["obs_modalities"].extend(modalities)
                        else:
                            robot_cfg["obs_modalities"].append(modalities)
        if external_sensors_config is not None:
            config["env"]["external_sensors"] = external_sensors_config

        # Load env
        env = Environment(configs=config)

        # Update robot sensor / proprio configuration
        if robot_proprio_keys is not None:
            for robot in env.robots:
                robot._proprio_obs = list(robot_proprio_keys)
        if robot_sensor_config is not None:
            for robot in env.robots:
                for sensor in robot.sensors.values():
                    sensor_cls_name = sensor.__class__.__name__
                    sensor_kwargs = robot_sensor_config.get(sensor_cls_name, dict()).get("sensor_kwargs", dict())
                    for kwarg, value in sensor_kwargs.items():
                        setattr(sensor, kwarg, value)
            env.load_observation_space()

        # Optionally include the desired environment wrapper specified in the config
        if include_env_wrapper:
            env = create_wrapper(env=env)

        if additional_wrapper_configs is not None:
            for wrapper_cfg in additional_wrapper_configs:
                env = create_wrapper(env=env, wrapper_cfg=wrapper_cfg)

        # Wrap and return env
        return cls(
            env=env,
            input_path=input_path,
            output_path=output_path,
            n_render_iterations=n_render_iterations,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
            **kwargs,
        )

    def __init__(
        self,
        env: Environment,
        input_path: str,
        output_path: str,
        n_render_iterations: int = 1,
        overwrite: bool = True,
        only_successes: bool = False,
        flush_every_n_traj: int = 10,
        full_scene_file: str | None = None,
        load_room_instances: list[str] | None = None,
        include_robot_control: bool = True,
        include_contacts: bool = True,
        **kwargs,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            input_path (str): path to input hdf5 collected data file
            output_path (str): path to store output hdf5 data file
            n_render_iterations (int): Number of rendering iterations to use when loading each stored frame from the
                recorded data
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file across episodes
            full_scene_file (None or str): If specified, the full scene file to use for playback. During data collection,
                the scene file stored may be partial, and this will be used to fill in the missing scene objects from the
                full scene file.
            load_room_instances (None or list[str]): If specified, the room instances to load for playback.
            include_robot_control (bool): Whether or not to include robot control. If False, will disable all joint control.
            include_contacts (bool): Whether or not to include (enable) contacts in the sim. If False, will set all objects to be visual_only
            kwargs (dict): Arguments to pass to super class
        """
        # Make sure transition rules are DISABLED for playback since we manually propagate transitions
        assert not gm.ENABLE_TRANSITION_RULES, "Transition rules must be disabled for DataPlaybackWrapper env!"

        # Stabilize skipped objects
        # we can do this here because we know that whatever's skipped during load state must have been asleep during data collection
        # which means they're not moving and we can safely keep them still
        with macros.unlocked():
            macros.utils.registry_utils.STABILIZE_SKIPPED_OBJECTS = True

        # Store scene file so we can restore the data upon each episode reset
        self.input_hdf5 = h5py.File(input_path, "r")
        input_config = json.loads(self.input_hdf5["data"].attrs["config"])
        self._fps = int(
            input_config.get("env", dict()).get("rendering_frequency", env.env_config["rendering_frequency"])
        )
        self.scene_file = json.loads(self.input_hdf5["data"].attrs["scene_file"])
        assert not (
            load_room_instances and not full_scene_file
        ), "Full scene file must be specified in order to load room instances"
        if full_scene_file:
            with open(full_scene_file, "r") as json_file:
                full_scene_json = json.load(json_file)
            self.scene_file = merge_scene_files(scene_a=full_scene_json, scene_b=self.scene_file, keep_robot_from="b")
            if load_room_instances is not None and full_scene_file is not None:
                # we loaded more room than the stored scene file, but still not the full scene
                # we need to save the current scene file here to avoid errors
                self.scene_file = env.scene.save(as_dict=True)

        # Store additional variables
        self.current_traj_history = []
        self.n_render_iterations = n_render_iterations
        self.current_episode_step_count = 0
        self.include_robot_control = include_robot_control
        self.include_contacts = include_contacts

        # Run super
        super().__init__(
            env=env,
            output_path=output_path,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            **kwargs,
        )

    def _parse_step_data(
        self,
        action: th.Tensor,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> dict:
        # Store action, obs, reward, terminated, truncated, info
        step_data = dict()
        step_data["obs"] = self._process_obs(obs, info)
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated
        return step_data

    def playback_episode(self, episode_id: int, record_data: bool = True) -> None:
        """
        Playback episode @episode_id, and optionally record observation data if @record is True

        Args:
            episode_id (int): Episode to playback. This should be a valid demo ID number from the inputted collected
                data hdf5 file
            record_data (bool): Whether to record data during playback or not
        """
        data_grp = self.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        # Grab episode data
        # Skip early if found malformed data
        try:
            transitions = json.loads(traj_grp.attrs["transitions"])
            traj_grp = h5py_group_to_torch(traj_grp)
            init_metadata = traj_grp["init_metadata"]
            action = traj_grp["action"]
            state = traj_grp["state"]
            state_size = traj_grp["state_size"]
            reward = traj_grp["reward"]
            terminated = traj_grp["terminated"]
            truncated = traj_grp["truncated"]
        except KeyError as e:
            print(f"Got error when trying to load episode {episode_id}:")
            print(f"Error: {str(e)}")
            return

        # Reset environment and update this to be the new initial state
        self.scene.restore(self.scene_file, update_initial_file=True)

        # Reset object attributes from the stored metadata
        with og.sim.stopped():
            for attr, vals in init_metadata.items():
                assert len(vals) == self.scene.n_objects
            for i, obj in enumerate(self.scene.objects):
                for attr, vals in init_metadata.items():
                    val = vals[i]
                    setattr(obj, attr, val.item() if val.ndim == 0 else val)
        self.reset()

        # If not controlling robots, disable for all robots
        if not self.include_robot_control:
            for robot in self.robots:
                robot.control_enabled = False
                # Set all controllers to effort mode with zero gain, this keeps the robot still
                for controller in robot.controllers.values():
                    for i, dof in enumerate(controller.dof_idx):
                        dof_joint = robot.joints[robot.dof_names_ordered[dof]]
                        dof_joint.set_control_type(
                            control_type=ControlType.EFFORT,
                            kp=None,
                            kd=None,
                        )

        # Restore to initial state
        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)

        # If record, record initial observations
        if record_data:
            # Grab initial observations directly from restored state[0], before any action is applied.
            first_time_load_n_iteration = 10
            for _ in range(self.n_render_iterations + first_time_load_n_iteration):
                og.sim.render()
            self.current_obs, init_info = self.env.get_obs()

            assert len(self.current_traj_history) == 1 and set(self.current_traj_history[-1].keys()) == {
                "obs"
            }, "Expected reset() to have inserted an initial obs-only entry into the trajectory history!"
            self.current_traj_history[-1]["obs"] = self._process_obs(self.current_obs, init_info)

        for i, (a, s, ss, r, te, tr) in enumerate(zip(action, state, state_size, reward, terminated, truncated)):
            # Execute any transitions that should occur at this current step
            if str(i) in transitions:
                cur_transitions = transitions[str(i)]
                scene = og.sim.scenes[0]
                for add_sys_name in cur_transitions["systems"]["add"]:
                    scene.get_system(add_sys_name, force_init=True)
                for remove_sys_name in cur_transitions["systems"]["remove"]:
                    scene.clear_system(remove_sys_name)
                for remove_obj_name in cur_transitions["objects"]["remove"]:
                    obj = scene.object_registry("name", remove_obj_name)
                    scene.remove_object(obj)
                for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                    obj = create_object_from_init_info(add_obj_info)
                    scene.add_object(obj)
                    obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
                # Step physics to initialize any new objects
                og.sim.step()

            # Restore the sim state, and take a very small step with the action to make sure physics are
            # properly propagated after the sim state update
            og.sim.load_state(s[: int(ss)], serialized=True)
            if not self.include_contacts:
                # When all objects/systems are visual-only, keep them still on every step
                for obj in self.scene.objects:
                    obj.keep_still()
                for system in self.scene.systems:
                    # TODO: Implement keep_still for other systems
                    if isinstance(system, MacroPhysicalParticleSystem):
                        system.set_particles_velocities(
                            lin_vels=th.zeros((system.n_particles, 3)), ang_vels=th.zeros((system.n_particles, 3))
                        )
            self.current_obs, _, _, _, info = self.env.step(action=a, n_render_iterations=self.n_render_iterations)

            # If recording, record data
            if record_data:
                step_data = self._parse_step_data(
                    action=a,
                    obs=self.current_obs,
                    reward=r,
                    terminated=te,
                    truncated=tr,
                    info=info,
                )
                # append to current trajectory history
                self.current_traj_history.append(step_data)

            self.current_episode_step_count += 1
            self.step_count += 1

        if record_data:
            self.flush_current_traj()

    def playback_dataset(self, record_data: bool = False) -> None:
        """
        Playback all episodes from the input HDF5 file, and optionally record observation data if @record is True

        Args:
            record_data (bool): Whether to record data during playback or not
        """
        for episode_id in range(self.input_hdf5["data"].attrs["n_episodes"]):
            self.playback_episode(
                episode_id=episode_id,
                record_data=record_data,
            )
