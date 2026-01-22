import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import logging

import h5py
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs.env_wrapper import EnvironmentWrapper, create_wrapper
from omnigibson.macros import gm, macros
from omnigibson.objects.object_base import BaseObject
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
import omnigibson.utils.transform_utils as T
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.data_utils import merge_scene_files
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch, assert_valid_key
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.controllers.controller_base import ControlType
from omnigibson.utils.backend_utils import _compute_backend as cb

from lerobot.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
    CODEBASE_VERSION as LEROBOT_CODEBASE_VERSION,
)
from omnigibson.learning.utils.lerobot_utils import OmniGibsonLeRobotV2Dataset, OmniGibsonLeRobotV3Dataset

import shutil

# Create module logger
log = create_module_logger(module_name=__name__)
log.setLevel(logging.INFO)


try:
    from omnigibson.learning.utils.obs_utils import write_video
except ImportError:
    write_video = None

h5py.get_config().track_order = True


class DataWrapper(EnvironmentWrapper):
    """
    An OmniGibson environment wrapper for writing data to an HDF5 file.
    """

    def __init__(
        self,
        env,
        output_path,
        overwrite=True,
        only_successes=True,
        flush_every_n_traj=10,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): path to store data file
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file
        """
        # Make sure the wrapped environment inherits correct omnigibson format
        assert isinstance(
            env, (og.Environment, EnvironmentWrapper)
        ), "Expected wrapped @env to be a subclass of OmniGibson's Environment class or EnvironmentWrapper!"

        # Only one scene is supported for now
        assert len(og.sim.scenes) == 1, "Only one scene is currently supported for DataWrapper env!"

        self.traj_count = 0
        self.step_count = 0
        self.only_successes = only_successes
        self.flush_every_n_traj = flush_every_n_traj
        self.current_obs = None
        self.current_traj_history = []

        # Create dataset
        self.create_dataset(output_path, env, overwrite=overwrite)

        # Run super
        super().__init__(env=env)

    def create_dataset(self, output_path, env, overwrite=True):
        """
        Creates a dataset at @output_path, possibly overwriting it if @overwrite is set

        Args:
            output_path (str): path to store data. May be either directory or filepath depending on the
                dataset type
            env (Environment): The wrapped environment
            overwrite (bool): Whether to overwrite any pre-existing data or not
        """
        raise NotImplementedError

    def step(self, action, n_render_iterations=1):
        """
        Run the environment step() function and collect data

        Args:
            action (th.Tensor): action to take in environment
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

    def _record_step_trajectory(self, action, obs, reward, terminated, truncated, info):
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

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
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
            dict: Keyword-mapped data that should be recorded in the HDF5
        """
        raise NotImplementedError()

    def reset(self):
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

        return self.current_obs, info

    def observation_spec(self):
        """
        Grab the normal environment observation_spec

        Returns:
            dict: Observations from the environment
        """
        return self.env.observation_spec()

    def process_traj_to_dataset(self, traj_data, nested_keys=("obs",)):
        """
        Processes trajectory data @traj_data and stores them as a new group under @traj_grp_name.

        Args:
            traj_data (list of dict): Trajectory data, where each entry is a keyword-mapped set of data for a single
                sim step
            nested_keys (list of str): Name of key(s) corresponding to nested data in @traj_data. This specific data
                is assumed to be its own keyword-mapped dictionary of numpy array values, and will be parsed
                differently from the rest of the data

        Returns:
            hdf5.Group: Generated hdf5 group storing the recorded trajectory data
        """
        raise NotImplementedError

    @property
    def should_save_current_episode(self):
        """
        Returns:
            bool: Whether the current episode should be saved or discarded
        """
        # Only save successful demos and if actually recording
        return self.env.task.success or not self.only_successes

    def flush_current_traj(self):
        """
        Flush current trajectory data
        """
        # Only save successful demos and if actually recording
        if self.should_save_current_episode:
            self.process_traj_to_dataset(self.current_traj_history, nested_keys=["obs"])
            self.traj_count += 1

            # Potentially write to disk
            if self.traj_count % self.flush_every_n_traj == 0:
                self.flush_current_file()
        else:
            # Remove this demo
            self.step_count -= len(self.current_traj_history)

        # Clear trajectory and transition buffers
        self.current_traj_history = []

    def flush_current_file(self):
        raise NotImplementedError

    def save_data(self):
        """
        Save collected trajectories as a hdf5 file in the robomimic format
        """
        if len(self.current_traj_history) > 0:
            self.flush_current_traj()

        self.close_dataset()

    def close_dataset(self):
        """
        Closes the active dataset, if open
        """
        raise NotImplementedError


class HDF5DataWrapper(DataWrapper):
    """
    Specific data wrapper for writing data to HDF5 format
    """

    def __init__(
        self,
        env,
        output_path,
        overwrite=True,
        only_successes=True,
        flush_every_n_traj=10,
        compression=None,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): path to store hdf5 data file. Should end in .hdf5
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file
            compression (None or dict): If specified, the compression arguments to use for the hdf5 file.
                For more information, check out https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
        """
        self.compression = dict() if compression is None else compression
        self.hdf5_file = None

        # Run super
        super().__init__(
            env=env,
            output_path=output_path,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
        )

    def create_dataset(self, output_path, env, overwrite=True):
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        log.info(f"\nWriting dataset hdf5 to: {output_path}\n")
        self.hdf5_file = h5py.File(output_path, "w" if overwrite else "a")
        if "data" not in set(self.hdf5_file.keys()):
            data_grp = self.hdf5_file.create_group("data")
        else:
            data_grp = self.hdf5_file["data"]

        if overwrite or "config" not in set(data_grp.attrs.keys()):
            if isinstance(env.task, BehaviorTask):
                env.task.update_bddl_scope_metadata(env)
            scene_file = env.scene.save()
            config = deepcopy(env.config)
            self.add_metadata(group=data_grp, name="config", data=config)
            self.add_metadata(group=data_grp, name="scene_file", data=scene_file)

    def update_scene_file(self, scene_file=None):
        """
        Updates the scene file in the HDF5 file

        Args:
            scene_file (str): Path to the scene file to update. If None, will save the current scene file
        """
        scene_file = self.env.scene.save() if scene_file is None else scene_file
        self.add_metadata(group=self.hdf5_file["data"], name="scene_file", data=scene_file)

    @property
    def should_save_current_episode(self):
        """
        Returns:
            bool: Whether the current episode should be saved or discarded
        """
        # Only save successful demos and if actually recording
        success = super().should_save_current_episode
        return success and self.hdf5_file is not None

    def process_traj_to_dataset(self, traj_data, nested_keys=("obs",)):
        traj_grp_name = f"demo_{self.traj_count}"
        traj_grp = self._process_traj_to_hdf5(self.current_traj_history, traj_grp_name, nested_keys=["obs"])
        self._postprocess_traj_group(traj_grp)

    def _process_traj_to_hdf5(self, traj_data, traj_grp_name, nested_keys=("obs",), data_grp=None):
        """
        Processes trajectory data @traj_data and stores them as a new group under @traj_grp_name.

        Args:
            traj_data (list of dict): Trajectory data, where each entry is a keyword-mapped set of data for a single
                sim step
            traj_grp_name (str): Name of the trajectory group to store
            nested_keys (list of str): Name of key(s) corresponding to nested data in @traj_data. This specific data
                is assumed to be its own keyword-mapped dictionary of numpy array values, and will be parsed
                differently from the rest of the data
            data_grp (None or h5py.Group): If specified, the h5py Group under which a new group wtih name
                @traj_grp_name will be created. If None, will default to "data" group

        Returns:
            hdf5.Group: Generated hdf5 group storing the recorded trajectory data
        """
        nested_keys = set(nested_keys)
        data_grp = self.hdf5_file.require_group("data") if data_grp is None else data_grp
        traj_grp = data_grp.create_group(traj_grp_name)
        traj_grp.attrs["num_samples"] = len(traj_data)

        # Create the data dictionary -- this will dynamically add keys as we iterate through our trajectory
        # We need to do this because we're not guaranteed to have a full set of keys at every trajectory step; e.g.
        # if the first step only has state or observations but no actions
        data = defaultdict(list)
        for key in nested_keys:
            data[key] = defaultdict(list)

        for step_data in traj_data:
            for k, v in step_data.items():
                if k in nested_keys:
                    for mod, step_mod_data in v.items():
                        data[k][mod].append(step_mod_data)
                else:
                    data[k].append(v)

        for k, dat in data.items():
            # Skip over all entries that have no data
            if not dat:
                continue

            # Create datasets for all keys with valid data
            if k in nested_keys:
                obs_grp = traj_grp.create_group(k)
                for mod, traj_mod_data in dat.items():
                    obs_grp.create_dataset(mod, data=th.stack(traj_mod_data, dim=0).cpu(), **self.compression)
            else:
                traj_data = th.stack(dat, dim=0) if isinstance(dat[0], th.Tensor) else th.tensor(dat)
                traj_grp.create_dataset(k, data=traj_data, **self.compression)

        return traj_grp

    def _postprocess_traj_group(self, traj_grp):
        """
        Runs any necessary postprocessing on the given trajectory group @traj_grp. This should be an
        in-place operation!

        Args:
            traj_grp (h5py.Group): Trajectory group to postprocess
        """
        # Default is no-op
        pass

    def flush_current_file(self):
        self.hdf5_file.flush()  # Flush data to disk to avoid large memory footprint
        # Retrieve the file descriptor and use os.fsync() to flush to disk
        fd = self.hdf5_file.id.get_vfd_handle()
        os.fsync(fd)
        log.info("Flushing hdf5")

    def add_metadata(self, group, name, data):
        """
        Adds metadata to the current HDF5 file under the @name key under @group

        Args:
            group (hdf5.File or hdf5.Group): HDF5 object to add an attribute to
            name (str): Name to assign to the data
            data (Any): Data to add. Note that this only supports relatively primitive data types --
                if the data is a dictionary it will be converted into a string-json format using TorchEncoder
        """
        group.attrs[name] = json.dumps(data, cls=TorchEncoder) if isinstance(data, dict) else data

    def close_dataset(self):
        """
        Closes the active dataset, if open
        """
        if self.hdf5_file is not None:
            log.info(
                f"\nSaved:\n"
                f"{self.traj_count} trajectories / {self.step_count} total steps\n"
                f"to hdf5: {self.hdf5_file.filename}\n"
            )
            self.hdf5_file["data"].attrs["n_episodes"] = self.traj_count
            self.hdf5_file["data"].attrs["n_steps"] = self.step_count
            self.hdf5_file.close()


class HDF5CollectionWrapper(HDF5DataWrapper):
    """
    An OmniGibson environment wrapper for collecting data in an optimized way.

    NOTE: This does NOT aggregate observations. Please use DataPlaybackWrapper to aggregate an observation
    dataset!
    """

    def __init__(
        self,
        env,
        output_path,
        viewport_camera_path="/World/viewer_camera",
        overwrite=True,
        only_successes=True,
        flush_every_n_traj=10,
        use_vr=False,
        obj_attr_keys=None,
        keep_checkpoint_rollback_data=False,
        enable_dump_filters=True,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): path to store hdf5 data file
            viewport_camera_path (str): prim path to the camera to use when rendering the main viewport during
                data collection
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file
            use_vr (bool): Whether to use VR headset for data collection
            obj_attr_keys (None or list of str): If set, a list of object attributes that should be
                cached at the beginning of every episode, e.g.: "scale", "visible", etc. This is useful
                for domain randomization settings where specific object attributes not directly tied to
                the object's runtime kinematic state are being modified once at the beginning of every episode,
                while the simulation is stopped.
            keep_checkpoint_rollback_data (bool): Whether to record any trajectory data pruned from rolling back to a
                previous checkpoint
            enable_dump_filters (bool): Whether to enable dump filters for optimized data collection. Defaults to True.
        """
        # Store additional variables needed for optimized data collection

        # Denotes the maximum serialized state size for the current episode
        self.max_state_size = 0

        # Dict capturing serialized per-episode initial information (e.g.: scales / visibilities) about every object
        self.obj_attr_keys = [] if obj_attr_keys is None else obj_attr_keys
        self.init_metadata = dict()

        # Maps episode step ID to dictionary of systems and objects that should be added / removed to the simulator at
        # the given simulator step. See add_transition_info() for more info
        self.current_transitions = dict()

        # Cached state to rollback to if requested
        self.checkpoint_states = []
        self.checkpoint_step_idxs = []

        # Info for keeping checkpoint rollback data
        self.checkpoint_rollback_trajs = dict() if keep_checkpoint_rollback_data else None

        self._is_recording = True
        self._filter_current_frame = False
        self.use_vr = use_vr

        # Add callbacks on import / remove objects and systems
        og.sim.add_callback_on_system_init(
            name="data_collection", callback=lambda system: self.add_transition_info(obj=system, add=True)
        )
        og.sim.add_callback_on_system_clear(
            name="data_collection", callback=lambda system: self.add_transition_info(obj=system, add=False)
        )
        og.sim.add_callback_on_add_obj(
            name="data_collection", callback=lambda obj: self.add_transition_info(obj=obj, add=True)
        )
        og.sim.add_callback_on_remove_obj(
            name="data_collection", callback=lambda obj: self.add_transition_info(obj=obj, add=False)
        )

        # Run super
        super().__init__(
            env=env,
            output_path=output_path,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
        )

        # Configure the simulator to optimize for data collection
        self._enable_dump_filters = enable_dump_filters
        if viewport_camera_path:
            self._optimize_sim_for_data_collection(viewport_camera_path=viewport_camera_path)

    def update_checkpoint(self):
        """
        Updates the internal cached checkpoint state to be the current simulation state. If @rollback_to_checkpoint() is
        called, it will rollback to this cached checkpoint state
        """
        # Save the current full state and corresponding step idx
        self.disable_dump_filters()
        self.checkpoint_states.append(self.scene.save(json_path=None, as_dict=True))
        self.checkpoint_step_idxs.append(len(self.current_traj_history))
        if self._enable_dump_filters:
            self.enable_dump_filters()

    def rollback_to_checkpoint(self, index=-1):
        """
        Rolls back the current state to the checkpoint stored in @self.checkpoint_states. If no checkpoint
        is found, this results in reset() being called

        Args:
            index (int): Index of the checkpoint to rollback to. Any checkpoints after this point will be discarded
        """
        if len(self.checkpoint_states) == 0:
            print("No checkpoint found, resetting environment instead!")
            self.reset()

        else:
            # Restore to checkpoint
            self.scene.restore(self.checkpoint_states[index])

            # Configure the simulator to optimize for data collection
            self._optimize_sim_for_data_collection(viewport_camera_path=og.sim.viewer_camera.active_camera_path)

            # Prune all data stored at the current checkpoint step and beyond
            checkpoint_step_idx = self.checkpoint_step_idxs[index]
            n_steps_to_remove = len(self.current_traj_history) - checkpoint_step_idx
            pruned_traj_history = self.current_traj_history[checkpoint_step_idx:]
            self.current_traj_history = self.current_traj_history[:checkpoint_step_idx]
            self.step_count -= n_steps_to_remove

            # Also prune any transition info that occurred after the checkpoint step idx
            pruned_transitions = dict()
            for step in tuple(self.current_transitions.keys()):
                if step >= checkpoint_step_idx:
                    pruned_transitions[step] = self.current_transitions.pop(step)

            # Update environment env step count
            self.env._current_step = checkpoint_step_idx - 1

            # Save checkpoint rollback data if requested
            if self.checkpoint_rollback_trajs is not None:
                step = self.env.episode_steps
                if step not in self.checkpoint_rollback_trajs:
                    self.checkpoint_rollback_trajs[step] = []
                self.checkpoint_rollback_trajs[step].append(
                    {
                        "step_data": pruned_traj_history,
                        "transitions": pruned_transitions,
                    }
                )

            # Prune any values after the checkpoint index
            if index != -1:
                self.checkpoint_states = self.checkpoint_states[: index + 1]
                self.checkpoint_step_idxs = self.checkpoint_step_idxs[: index + 1]

    def _process_traj_to_hdf5(self, traj_data, traj_grp_name, nested_keys=("obs",), data_grp=None):
        # First pad all state values to be the same max (uniform) size
        for step_data in traj_data:
            state = step_data["state"]
            padded_state = th.zeros(self.max_state_size, dtype=th.float32)
            padded_state[: len(state)] = state
            step_data["state"] = padded_state

        # Call super
        traj_grp = super()._process_traj_to_hdf5(traj_data, traj_grp_name, nested_keys, data_grp)

        return traj_grp

    def _postprocess_traj_group(self, traj_grp):
        super()._postprocess_traj_group(traj_grp=traj_grp)

        # Add in transition info
        self.add_metadata(group=traj_grp, name="transitions", data=self.current_transitions)

        # Add initial metadata information
        metadata_grp = traj_grp.create_group("init_metadata")
        for name, data in self.init_metadata.items():
            metadata_grp.create_dataset(name, data=data)

        # Potentially save cached checkpoint rollback data
        if self.checkpoint_rollback_trajs is not None and len(self.checkpoint_rollback_trajs) > 0:
            rollback_grp = traj_grp.create_group("rollbacks")
            for step, rollback_trajs in self.checkpoint_rollback_trajs.items():
                for i, rollback_traj in enumerate(rollback_trajs):
                    rollback_traj_grp = self.process_traj_to_hdf5(
                        traj_data=rollback_traj["step_data"],
                        traj_grp_name=f"step_{step}-{i}",
                        nested_keys=["obs"],
                        data_grp=rollback_grp,
                    )
                    self.add_metadata(group=rollback_traj_grp, name="transitions", data=rollback_traj["transitions"])

    @property
    def is_recording(self):
        return self._is_recording

    @is_recording.setter
    def is_recording(self, value: bool):
        self._is_recording = value

    @property
    def filter_current_frame(self):
        return self._filter_current_frame

    @filter_current_frame.setter
    def filter_current_frame(self, value: bool):
        self._filter_current_frame = value

    def _record_step_trajectory(self, action, obs, reward, terminated, truncated, info):
        if self.is_recording:
            super()._record_step_trajectory(action, obs, reward, terminated, truncated, info)

    def _optimize_sim_for_data_collection(self, viewport_camera_path):
        """
        Configures the simulator to optimize for data collection

        Args:
            viewport_camera_path (str): Prim path to the camera to use for the viewer for data collection
        """
        # Disable all render products to save on speed
        # See https://forums.developer.nvidia.com/t/speeding-up-simulation-2023-1-1/300072/6
        for sensor in VisionSensor.SENSORS.values():
            sensor.render_product.hydra_texture.set_updates_enabled(False)

        # Set the main viewport camera path
        og.sim.viewer_camera.active_camera_path = viewport_camera_path

        # Use asynchronous rendering for faster performance
        # We have to do a super hacky workaround to avoid the GUI freezing, which is
        # toggling these settings to be True -> False -> True
        # Only setting it to True once will actually freeze the GUI for some reason!
        if not gm.HEADLESS:
            # Async rendering does not work in VR mode
            if not self.use_vr:
                lazy.carb.settings.get_settings().set_bool("/app/asyncRendering", True)
                lazy.carb.settings.get_settings().set_bool("/app/asyncRenderingLowLatency", True)
                lazy.carb.settings.get_settings().set_bool("/app/asyncRendering", False)
                lazy.carb.settings.get_settings().set_bool("/app/asyncRenderingLowLatency", False)
                lazy.carb.settings.get_settings().set_bool("/app/asyncRendering", True)
                lazy.carb.settings.get_settings().set_bool("/app/asyncRenderingLowLatency", True)

            # Disable mouse grabbing since we're only using the UI passively
            lazy.carb.settings.get_settings().set_bool("/physics/mouseInteractionEnabled", False)
            lazy.carb.settings.get_settings().set_bool("/physics/mouseGrab", False)
            lazy.carb.settings.get_settings().set_bool("/physics/forceGrab", False)

        # Set the dump filter for better performance
        # TODO: Possibly remove this feature once we have fully tensorized state saving, which may be more efficient
        if self._enable_dump_filters:
            self.enable_dump_filters()

    def enable_dump_filters(self):
        """
        Enables dump filters for optimized per-step state caching
        """
        self.env.scene.object_registry.set_dump_filter(dump_filter=lambda obj: obj.is_active and obj.initialized)

    def disable_dump_filters(self):
        """
        Disables dump filters for full state caching
        """
        self.env.scene.object_registry.set_dump_filter(dump_filter=lambda obj: True)

    def reset(self):
        # Call super first
        init_obs, init_info = super().reset()

        # Make sure all objects are awake to begin to guarantee we save their initial states
        for obj in self.scene.objects:
            obj.wake()

        # Store this initial state as part of the trajectory if recording
        if self.is_recording:
            state = og.sim.dump_state(serialized=True)
            step_data = {
                "state": state,
                "state_size": len(state),
                "filter_mask": 1 if self.filter_current_frame else 0,
            }
            self.current_traj_history.append(step_data)

            # Update max state size
            self.max_state_size = max(self.max_state_size, len(state))

        # Also store initial metadata not recorded in serialized state
        # This is simply serialized
        metadata = {key: [] for key in self.obj_attr_keys}
        for obj in self.scene.objects:
            for key in self.obj_attr_keys:
                metadata[key].append(getattr(obj, key))
        self.init_metadata = {
            key: th.stack(vals, dim=0) if isinstance(vals[0], th.Tensor) else th.tensor(vals, dtype=type(vals[0]))
            for key, vals in metadata.items()
        }

        # Clear checkpoint states
        self.checkpoint_states = []
        self.checkpoint_step_idxs = []
        if self.checkpoint_rollback_trajs is not None:
            self.checkpoint_rollback_trajs = dict()

        return init_obs, init_info

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        # Store dumped state, reward, terminated, truncated
        step_data = dict()
        state = og.sim.dump_state(serialized=True)
        step_data["action"] = action
        step_data["state"] = state
        step_data["state_size"] = len(state)
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated
        step_data["filter_mask"] = 1 if self.filter_current_frame else 0

        # Update max state size
        self.max_state_size = max(self.max_state_size, len(state))

        return step_data

    def process_traj_to_hdf5(self, traj_data, traj_grp_name, nested_keys=("obs",), data_grp=None):
        # First pad all state values to be the same max (uniform) size
        for step_data in traj_data:
            state = step_data["state"]
            padded_state = th.zeros(self.max_state_size, dtype=th.float32)
            padded_state[: len(state)] = state
            step_data["state"] = padded_state

        # Call super
        traj_grp = super().process_traj_to_hdf5(traj_data, traj_grp_name, nested_keys, data_grp)

        return traj_grp

    def flush_current_traj(self):
        # Call super first
        super().flush_current_traj()

        # Clear transition buffer and max state size
        self.max_state_size = 0
        self.current_transitions = dict()

    @property
    def should_save_current_episode(self):
        # In addition to default conditions, we only save the current episode if we are actually recording
        return super().should_save_current_episode and self.is_recording

    def add_transition_info(self, obj, add=True):
        """
        Adds transition info to the current sim step for specific object @obj.

        Args:
            obj (BaseObject or BaseSystem): Object / system whose information should be stored
            add (bool): If True, assumes the object is being imported. Else, assumes the object is being removed
        """
        # If we're at the current checkpoint idx, this means that we JUST created a checkpoint and we're still at
        # the same sim step.
        # This is dangerous because it means that a transition is happening that will NOT be tracked properly
        # if we rollback the state -- i.e.: the state will be rolled back to just BEFORE this transition was executed,
        # and will therefore not be tracked properly in subsequent states during playback. So we assert that the current
        # idx is NOT the current checkpoint idx
        if len(self.checkpoint_step_idxs) > 0:
            assert (
                self.checkpoint_step_idxs[-1] - 1 != self.env.episode_steps
            ), "A checkpoint was just updated. Any subsequent transitions at this immediate timestep will not be replayed properly!"

        if self.env.episode_steps not in self.current_transitions:
            self.current_transitions[self.env.episode_steps] = {
                "systems": {"add": [], "remove": []},
                "objects": {"add": [], "remove": []},
            }

        # Add info based on type -- only need to store name unless we're an object being added
        info = obj.get_init_info() if isinstance(obj, BaseObject) and add else obj.name
        dic_key = "objects" if isinstance(obj, BaseObject) else "systems"
        val_key = "add" if add else "remove"
        self.current_transitions[self.env.episode_steps][dic_key][val_key].append(info)


class DataPlaybackWrapper(DataWrapper):
    """
    An OmniGibson environment wrapper for playing back data and collecting observations.

    NOTE: This assumes a DataCollectionWrapper environment has been used to collect data!
    """

    @classmethod
    def create_from_hdf5(
        cls,
        input_path,
        output_path,
        robot_obs_modalities=tuple(),
        robot_grasping_mode=None,
        robot_proprio_keys=None,
        robot_sensor_config=None,
        external_sensors_config=None,
        include_sensor_names=None,
        exclude_sensor_names=None,
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        flush_every_n_steps=0,
        include_env_wrapper=False,
        additional_wrapper_configs=None,
        full_scene_file=None,
        include_task=True,
        include_task_obs=True,
        include_robot_control=True,
        include_contacts=True,
        load_room_instances=None,
        **kwargs,
    ):
        """
        Create a DataPlaybackWrapper environment instance form the recorded demonstration info
        from @hdf5_path, and aggregate observation_modalities @obs during playback

        Args:
            input_path (str): Absolute path to the input hdf5 file containing the relevant collected data to playback
            output_path (str): Absolute path to the output hdf5 file that will contain the recorded observations from
                the replayed data
            robot_grasping_mode (None or str): If specified, the grasping mode to use for the robot. This will override the
                grasping mode specified in the env config when the environment is loaded. Valid modes include: "physical", "assisted", "sticky".
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
            flush_every_n_steps (int): How often to flush (write) current data to file within an episode.
                If this is greater than 0, flush_every_n_traj must be set to 1.
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
        # check flush parameters
        if flush_every_n_steps > 0:
            assert flush_every_n_traj == 1, "flush_every_n_traj must be 1 if flush_every_n_steps is greater than 0"
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
            # Since we are setting all objects to be visual-only, physics will not be propogating
            config["env"]["action_frequency"] = 30.0
            config["env"]["rendering_frequency"] = 30.0
            config["env"]["physics_frequency"] = 120.0
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
            if robot_grasping_mode is not None:
                robot_cfg["grasping_mode"] = robot_grasping_mode
            # For robot obs, make sure sensor modalities key is removed so that the modalities are
            # auto-inferred from the set of robot obs modalities requested
            for sensor_cfg in robot_cfg["sensor_config"].values():
                if "modalities" in sensor_cfg:
                    sensor_cfg.pop("modalities")
        if external_sensors_config is not None:
            config["env"]["external_sensors"] = external_sensors_config

        # Load env
        env = og.Environment(configs=config)

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

        # Override grasping mode after environment creation if specified
        if robot_grasping_mode is not None:
            for robot in env.robots:
                robot._grasping_mode = robot_grasping_mode

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
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
            robot_grasping_mode=robot_grasping_mode,
            **kwargs,
        )

    def __init__(
        self,
        env,
        input_path,
        output_path,
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        flush_every_n_steps=0,
        full_scene_file=None,
        load_room_instances=None,
        include_robot_control=True,
        include_contacts=True,
        robot_grasping_mode=None,
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
            flush_every_n_steps (int): How often to flush (write) current data to file within an episode.
                If this is greater than 0, flush_every_n_traj must be set to 1.
            full_scene_file (None or str): If specified, the full scene file to use for playback. During data collection,
                the scene file stored may be partial, and this will be used to fill in the missing scene objects from the
                full scene file.
            load_room_instances (None or str): If specified, the room instances to load for playback.
            include_robot_control (bool): Whether or not to include robot control. If False, will disable all joint control.
            include_contacts (bool): Whether or not to include (enable) contacts in the sim. If False, will set all objects to be visual_only
            robot_grasping_mode (None or str): If specified, the grasping mode to use for all robots. This will override
                the grasping mode set during robot initialization. Valid modes include: "physical", "assisted", "sticky".
            kwargs (dict): Arguments to pass to super class
        """
        # Override grasping mode for all robots if specified
        if robot_grasping_mode is not None:
            for robot in env.robots:
                robot._grasping_mode = robot_grasping_mode

        # Make sure transition rules are DISABLED for playback since we manually propagate transitions
        assert not gm.ENABLE_TRANSITION_RULES, "Transition rules must be disabled for DataPlaybackWrapper env!"

        # Stabilize skipped objects
        # we can do this here because we know that whatever's skipped during load state must have been asleep during data collection
        # which means they're not moving and we can safely keep them still
        with macros.unlocked():
            macros.utils.registry_utils.STABILIZE_SKIPPED_OBJECTS = True

        # Store scene file so we can restore the data upon each episode reset
        self.input_hdf5 = h5py.File(input_path, "r")
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
        if flush_every_n_steps > 0:
            assert flush_every_n_traj == 1, "flush_every_n_traj must be 1 if flush_every_n_steps is greater than 0"
        self.flush_every_n_steps = flush_every_n_steps
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

    def _process_obs(self, obs, info):
        """
        Modifies @obs inplace for any relevant post-processing

        Args:
            obs (dict): Keyword-mapped relevant observations from the immediate env step
            info (dict): Keyword-mapped relevant information from the immediate env step
        """
        # Default is a no-op
        return obs

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        # Store action, obs, reward, terminated, truncated, info
        step_data = dict()
        step_data["obs"] = self._process_obs(obs=obs, info=info)
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated
        return step_data

    def playback_episode(self, episode_id, record_data=True, video_writers=None):
        """
        Playback episode @episode_id, and optionally record observation data if @record is True

        Args:
            episode_id (int): Episode to playback. This should be a valid demo ID number from the inputted collected
                data hdf5 file
            record_data (bool): Whether to record data during playback or not
            video_writers (Any): Optional video writers to record the playback
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
            # Load filter_mask if available (defaults to all zeros / unfiltered if not present)
            filter_mask = traj_grp.get("filter_mask", th.zeros(len(state), dtype=th.int64))
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
            # We need to step the environment to get the initial observations propagated
            first_time_load_n_iteration = 10
            self.current_obs, _, _, _, init_info = self.env.step(
                action=action[0], n_render_iterations=self.n_render_iterations + first_time_load_n_iteration
            )
            # Only record initial observation if not filtered
            if not filter_mask[0]:
                step_data = {"obs": self._process_obs(obs=self.current_obs, info=init_info)}
                self.current_traj_history.append(step_data)

        for i, (a, s, ss, r, te, tr) in enumerate(
            zip(action, state[1:], state_size[1:], reward, terminated, truncated)
        ):
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

            # If recording, record data (skip if this frame is filtered)
            if record_data and not filter_mask[i + 1]:
                step_data = self._parse_step_data(
                    action=a,
                    obs=self.current_obs,
                    reward=r,
                    terminated=te,
                    truncated=tr,
                    info=info,
                )
                if self.flush_every_n_steps > 0:
                    if i == 0:
                        self.allocate_traj(step_data, episode_id, num_samples=len(action), video_writers=video_writers)
                    if i % self.flush_every_n_steps == 0:
                        self.flush_partial_traj(num_samples=len(action), video_writers=video_writers)
                # append to current trajectory history
                self.current_traj_history.append(step_data)

            self.current_episode_step_count += 1
            self.step_count += 1

        if record_data:
            if self.flush_every_n_steps > 0:
                self.flush_partial_traj(num_samples=len(action), video_writers=video_writers)
            self.flush_current_traj()

    def allocate_traj(
        self,
        step_data,
        episode_id,
        num_samples: int,
        nested_keys=("obs",),
        video_writers=None,
    ):
        """
        Allocate trajectory data space from @step_data given the number of samples @num_samples.

        Args:
            step_data (dict): Keyword-mapped set of data for a single sim step
            episode_id (int): Trajectory episode ID
            num_samples (int): Number of samples in the trajectory
            nested_keys (list of str): Name of key(s) corresponding to nested data in @step_data. This specific data
                is assumed to be its own keyword-mapped dictionary of numpy array values, and will be parsed
                differently from the rest of the data.
            video_writers (None or dict): If specified, a dictionary mapping observation keys to video writers
                for saving video frames during replay
        """
        raise NotImplementedError

    def playback_dataset(self, record_data=False):
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

    def add_to_dataset(self, keys, idxs, data, idx_max=None):
        """
        Adds tensorized data @data to the active dataset with specified nested keys @keys, at indexes @idxs

        Args:
            keys (list of str): Key(s) to write data to. A list should be specified if using nested keys
            idxs (int or list of int): Index(es) within batched tensor to write tensor data to
        """
        raise NotImplementedError

    def flush_partial_traj(self, num_samples: int, video_writers=None):
        """
        Flush the current trajectory data to file.
        If flush_every_n_steps is greater than 0, flush the current trajectory data to file every n steps.
        Args:
            num_samples: (int): The number of samples to flush.
            video_writers: (None or dict): If specified, a dictionary mapping observation keys to video writers
                for saving video frames during replay
        """
        log.info(f"Storing partial trajectory at step {self.current_episode_step_count}...")
        assert self.flush_every_n_steps > 0, "flush_every_n_steps must be greater than 0 to flush partial trajectory"
        data_length_to_flush = len(self.current_traj_history)
        # At step 0, we only have observation data, so observation data will only have one more offset than others
        if self.current_episode_step_count == 0:
            assert data_length_to_flush == 1
            for key, dat in self.current_traj_history[0].items():
                for mod in dat.keys():
                    if video_writers is not None and mod in video_writers.keys():
                        assert (
                            write_video is not None
                        ), "video_writers not imported! Please make sure you have omnigibson setup with eval dependencies!"
                        # write to video
                        write_video(
                            self.current_traj_history[0][key][mod].unsqueeze(0).numpy(),
                            video_writer=video_writers[mod],
                            batch_size=None,
                            mode=mod.split("::")[-1],
                        )
                    else:
                        self.add_to_dataset(keys=[key, mod], idxs=0, data=self.current_traj_history[0][key][mod])
        else:
            for key, dat in self.current_traj_history[0].items():
                if isinstance(dat, dict):
                    for mod in dat.keys():
                        obs_data_length = (
                            data_length_to_flush
                            if self.current_episode_step_count < num_samples
                            else data_length_to_flush - 1
                        )
                        if obs_data_length > 0:
                            data_to_write = th.stack(
                                [self.current_traj_history[i][key][mod] for i in range(obs_data_length)], dim=0
                            )
                            if video_writers is not None and mod in video_writers.keys():
                                assert (
                                    write_video is not None
                                ), "video_writers not imported! Please make sure you have omnigibson setup with eval dependencies!"
                                # write to video
                                write_video(
                                    data_to_write.numpy(),
                                    video_writer=video_writers[mod],
                                    batch_size=None,
                                    mode=mod.split("::")[-1],
                                )
                            else:
                                self.add_to_dataset(
                                    keys=[key, mod],
                                    idxs=list(
                                        range(
                                            self.current_episode_step_count - data_length_to_flush + 1,
                                            self.current_episode_step_count + 1,
                                        )
                                    ),
                                    data=data_to_write,
                                )
                else:
                    self.add_to_dataset(
                        keys=[key],
                        idxs=list(
                            range(
                                self.current_episode_step_count - data_length_to_flush,
                                self.current_episode_step_count,
                            )
                        ),
                        data=th.stack([self.current_traj_history[i][key] for i in range(data_length_to_flush)], dim=0),
                    )
        # Reset the current trajectory history
        self.current_traj_history = []

    def flush_current_traj(self):
        """
        Flush current trajectory data
        For playback, we assume that all data needs to be stored.
        """
        if self.flush_every_n_steps == 0:
            super().flush_current_traj()
        else:
            self.postprocess_current_traj()
            self.flush_current_file()
            # Clear trajectory and transition buffers
            self.traj_count += 1
            self.current_episode_step_count = 0
            self.current_traj_history = []

    def postprocess_current_traj(self):
        """
        Postprocesses the current trajectory data
        """
        raise NotImplementedError


class HDF5PlaybackWrapper(DataPlaybackWrapper, HDF5DataWrapper):
    """
    Playback wrapper for replaying data and writing to an HDF5 file
    """

    def __init__(
        self,
        env,
        input_path,
        output_path,
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        flush_every_n_steps=0,
        full_scene_file=None,
        load_room_instances=None,
        include_robot_control=True,
        include_contacts=True,
        compression=None,
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
            flush_every_n_steps (int): How often to flush (write) current data to file within an episode.
                If this is greater than 0, flush_every_n_traj must be set to 1.
            full_scene_file (None or str): If specified, the full scene file to use for playback. During data collection,
                the scene file stored may be partial, and this will be used to fill in the missing scene objects from the
                full scene file.
            load_room_instances (None or str): If specified, the room instances to load for playback.
            include_robot_control (bool): Whether or not to include robot control. If False, will disable all joint control.
            include_contacts (bool): Whether or not to include (enable) contacts in the sim. If False, will set all objects to be visual_only
            compression (None or dict): If specified, the compression arguments to use for the hdf5 file.
        """
        self.current_traj_grp = None
        self.traj_dsets = dict()

        # Run super
        super().__init__(
            env=env,
            input_path=input_path,
            output_path=output_path,
            n_render_iterations=n_render_iterations,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
            compression=compression,
        )

    def allocate_traj(
        self,
        step_data,
        episode_id,
        num_samples: int,
        nested_keys=("obs",),
        video_writers=None,
    ):
        # Allocate a new dataset group in HDF5
        self.current_traj_grp, self.traj_dsets = self._allocate_traj_to_hdf5(
            step_data,
            f"demo_{episode_id}",
            num_samples=num_samples,
            nested_keys=nested_keys,
            video_writers=video_writers,
        )

    def _allocate_traj_to_hdf5(
        self, step_data, traj_grp_name, num_samples: int, nested_keys=("obs",), data_grp=None, video_writers=None
    ):
        """
        Allocate trajectory data space from @step_data given the number of samples @num_samples.

        Args:
            step_data (dict): Keyword-mapped set of data for a single sim step
            traj_grp_name (str): Name of the trajectory group to store
            num_samples (int): Number of samples in the trajectory
            nested_keys (list of str): Name of key(s) corresponding to nested data in @step_data. This specific data
                is assumed to be its own keyword-mapped dictionary of numpy array values, and will be parsed
                differently from the rest of the data.
            data_grp (None or h5py.Group): If specified, the h5py Group under which a new group wtih name
                @traj_grp_name will be created. If None, will default to "data" group
            video_writers (None or dict): If specified, a dictionary mapping observation keys to video writers
                for saving video frames during replay

        Returns:
            Tuple[h5py.Group, dict(str, hdf5.Dataset)]: Generated hdf5 group and datasets to store the trajectory data in the future
        """
        traj_dsets = dict()
        nested_keys = set(nested_keys)
        for k in nested_keys:
            traj_dsets[k] = dict()
        data_grp = self.hdf5_file.require_group("data") if data_grp is None else data_grp
        traj_grp = data_grp.create_group(traj_grp_name)
        log.info(f"Number of samples: {num_samples}")
        traj_grp.attrs["num_samples"] = num_samples

        for k, dat in step_data.items():
            if k in nested_keys:
                obs_grp = traj_grp.create_group(k)
                for mod, step_mod_data in dat.items():
                    if video_writers is None or mod not in video_writers.keys():
                        traj_dsets[k][mod] = obs_grp.create_dataset(
                            mod,
                            shape=(num_samples, *step_mod_data.shape),
                            dtype=step_mod_data.numpy().dtype,
                            **self.compression,
                            chunks=(1, *step_mod_data.shape),
                            shuffle=True,
                        )
                    else:
                        log.info(f"Skipping storing {mod} in h5, writing to video instead.")
            else:
                traj_dsets[k] = traj_grp.create_dataset(
                    k, shape=(num_samples, *dat.shape), dtype=dat.numpy().dtype, **self.compression, shuffle=True
                )

        return traj_grp, traj_dsets

    def add_to_dataset(self, keys, idxs, data, mod=None, idx_max=None):
        dset = self.traj_dset
        for key in keys:
            dset = dset[key]
        dset[idxs] = data

    def postprocess_current_traj(self):
        self._postprocess_traj_group(self.current_traj_grp)


class LeRobotPlaybackWrapper(DataPlaybackWrapper):
    """
    An OmniGibson environment wrapper for playing back data and collecting observations to be stored in LeRobotV3 format

    NOTE: This assumes a DataCollectionWrapper environment has been used to collect data!
    """

    def __init__(
        self,
        env,
        input_path,
        output_path,
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        flush_every_n_steps=0,
        full_scene_file=None,
        load_room_instances=None,
        include_robot_control=True,
        include_contacts=True,
        robot_grasping_mode=None,
        root_dir=HF_LEROBOT_HOME,
        robot_type=None,
        image_writer_threads=10,
        image_writer_processes=5,
        task_name=None,
        lerobot_version="v3.0",
        use_videos=True,
        include_multi_action_representation=False,
        include_modalities=None,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            input_path (str): path to input hdf5 collected data file
            output_path (str): path to the output lerobot dataset. This value is synonymous with lerobot's
                @repo_id key, and should specify the name of the repo for saving the dataset, e.g. <username>/<dataset_name>
            n_render_iterations (int): Number of rendering iterations to use when loading each stored frame from the
                recorded data
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file across episodes
            flush_every_n_steps (int): How often to flush (write) current data to file within an episode.
                If this is greater than 0, flush_every_n_traj must be set to 1.
            full_scene_file (None or str): If specified, the full scene file to use for playback. During data collection,
                the scene file stored may be partial, and this will be used to fill in the missing scene objects from the
                full scene file.
            load_room_instances (None or str): If specified, the room instances to load for playback.
            include_robot_control (bool): Whether or not to include robot control. If False, will disable all joint control.
            include_contacts (bool): Whether or not to include (enable) contacts in the sim. If False, will set all objects to be visual_only
            robot_grasping_mode (None or str): If specified, the grasping mode to use for all robots. This will override
                the grasping mode set during robot initialization. Valid modes include: "physical", "assisted", "sticky".
            root_dir (str): Root directory to store output dataset files
            robot_type (None or str): Name of the robot within this dataset. If not specified, will be inferred
                from environment
            image_writer_threads (int): How many threads to use for writing images
            image_writer_processes (int): How many processes to use for writing images
            task_name (None or str): If specified, task that will be recorded in LeRobot dataset. If not specified,
                will try to automatically infer if the wrapped environment is a BehaviorTask
            lerobot_version (str): Version of LeRobot to use when saving dataset. This must be aligned with the
                installed lerobot library version
            use_videos (bool): Whether to save high dimensional image data as video or not
            include_multi_action_representation (bool): Whether to include multi action representation in the dataset. This will make the action entry a concatenation of multiple action representations, with
                corresponding annotations written to modalities.json in the meta folder
            include_modalities (None or list of str): If specified, only include these observation modalities in the
                dataset. Valid modalities include: "rgb", "depth_linear", "proprio", "seg_semantic", etc.
                If None, all modalities are included. Example: ["rgb", "depth_linear", "proprio"]
        """
        # Store variables
        # For robot_type, if multiple robots, join their class names with "_"
        if robot_type is None:
            robot_type = "_".join(robot.__class__.__name__.lower() for robot in env.robots)
        self.lerobot_dataset_kwargs = {
            "repo_id": output_path,
            "root": f"{root_dir}/{output_path}",
            "robot_type": robot_type,
            "image_writer_threads": image_writer_threads,
            "image_writer_processes": image_writer_processes,
        }
        self.dataset = None
        self.obs_mapping = None  # Maps OG obs name -> lerobot obs name
        assert_valid_key(lerobot_version, valid_keys={"v2.1", "v3.0"}, name="lerobot version")
        assert lerobot_version == LEROBOT_CODEBASE_VERSION, (
            f"Got mismatch between requested LeRobot version {lerobot_version} and actual version {LEROBOT_CODEBASE_VERSION}!\n\n"
            + "Make sure LeRobot repo is cloned and sourced locally.\n\n"
            + "For v2.1, run `git checkout v0.3.3`\n"
            + "For v3.0, run `git checkout `v0.4.2`\n\n"
        )
        self.lerobot_version = lerobot_version
        self.use_videos = use_videos
        self.last_actions = None
        self.controller_action_start_idxs = None
        self.include_multi_action_representation = include_multi_action_representation
        if self.include_multi_action_representation:
            assert include_robot_control, "Multi action representation requires robot control!"
        
        # Store modality filter - if specified, only include these modalities in the dataset
        self.include_modalities = include_modalities

        # Infer task name
        if task_name is None:
            if isinstance(env.task, BehaviorTask):
                task_name = env.task.activity_name.replace("_", " ")
            else:
                task_name = "Do something"
        self.task_name = task_name

        # Run super
        super().__init__(
            env=env,
            input_path=input_path,
            output_path=output_path,
            n_render_iterations=n_render_iterations,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
            robot_grasping_mode=robot_grasping_mode,
        )

    @classmethod
    def get_lerobot_obs_mapping(cls, env, use_videos=True, include_modalities=None):
        """
        Generate observation mapping from OmniGibson to LeRobot format.
        
        Args:
            env: The environment
            use_videos (bool): Whether to use videos for image data
            include_modalities (None or list of str): If specified, only include these modalities.
                Valid modalities include: "rgb", "depth_linear", "proprio", "seg_semantic", etc.
                If None, all modalities are included.
                
        Returns:
            tuple: (obs_mapping, obs_features) dictionaries
        """
        obs_mapping, obs_features = dict(), dict()
        
        # First pass: collect all proprio keys and compute total shape for multi-robot concatenation
        proprio_keys = []
        total_proprio_dim = 0
        
        for key, gym_shape in env.observation_space.items():
            modality = key.split("::")[-1]
            
            # Filter by modality if include_modalities is specified
            if include_modalities is not None:
                modality_included = any(inc_mod in modality for inc_mod in include_modalities)
                if not modality_included:
                    continue
            
            if "proprio" in modality or "low_dim" in modality:
                proprio_keys.append(key)
                total_proprio_dim += gym_shape.shape[0]
        
        # Add concatenated proprio feature if any proprio keys exist
        if proprio_keys:
            obs_features["observation.state"] = {
                "dtype": "float32",
                "shape": (total_proprio_dim,),
                "names": (None,),
            }
            # Map all proprio keys to the same observation.state (they will be concatenated)
            for key in proprio_keys:
                obs_mapping[key] = "observation.state"
        
        # Second pass: handle all other modalities
        for key, gym_shape in env.observation_space.items():
            modality = key.split("::")[-1]
            
            # Filter by modality if include_modalities is specified
            if include_modalities is not None:
                modality_included = any(inc_mod in modality for inc_mod in include_modalities)
                if not modality_included:
                    continue
            
            # Skip proprio keys (already handled above)
            if "proprio" in modality or "low_dim" in modality:
                continue
            
            info = dict()
            # Parse the relevant name to assign
            # For multi-robot support, we keep robot prefix in obs_name for disambiguation
            obs_name_strs = key.split("::")[-2].split(":")
            robot_prefix = ""
            if "robot" in obs_name_strs[0]:
                # Keep the robot prefix for multi-robot disambiguation
                robot_prefix = obs_name_strs[0] + "_" if len(env.robots) > 1 else ""
                # Remove the prefix (e.g. "robot_xxxxxx") and keep the remainder
                obs_name_strs = obs_name_strs[1:]
            # Join with "_" and make lowercase to make final name
            obs_name = "_".join(obs_name_strs).lower()
            if obs_name == "":
                obs_name = "robot"
            # Add robot prefix for multi-robot environments
            obs_name = robot_prefix.lower() + obs_name
            if "rgb" in modality:
                info["dtype"] = "video" if use_videos else "image"
                info["shape"] = gym_shape.shape[:-1] + (3,)
                info["names"] = ["height", "width", "channel"]
            elif "depth" in modality:
                info["dtype"] = "video" if use_videos else "image"
                info["shape"] = gym_shape.shape + (1,)
                info["names"] = ["height", "width"]

                # We also add relative camera transforms (wrt robot egocentric frame) in case we
                # want to convert depth to point clouds
                # So we add an extra entry here
                tf_name = f"observation.robot2cam_pose.{obs_name}"
                if tf_name not in obs_features:
                    obs_features[tf_name] = {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": None,
                    }
            elif "seg_semantic" in modality:
                info["dtype"] = "int32"
                info["shape"] = gym_shape.shape
                info["names"] = ["height", "width"]
            else:
                raise ValueError(f"Got LeRobot-incompatible observation modality: {modality}")

            # Add this key to features, and store the obs name mapping
            lerobot_obs_name = f"observation.{modality}.{obs_name}"
            obs_features[lerobot_obs_name] = info
            obs_mapping[key] = lerobot_obs_name

        return obs_mapping, obs_features

    @classmethod
    def og_to_lerobot_obs(cls, env, obs_flat, obs_mapping, include_modalities=None):
        """
        Convert OmniGibson observations to LeRobot format.
        
        Args:
            env: The environment
            obs_flat (dict): Flattened observations from the environment
            obs_mapping (dict): Mapping from OG observation names to LeRobot names
            include_modalities (None or list of str): If specified, only include these modalities.
                
        Returns:
            dict: Observations in LeRobot format
        """
        # Add tfs to flattened obs for each robot
        # For multi-robot support, we compute relative poses wrt the first robot's frame
        # (or each robot's own frame for its own sensors)
        primary_robot_tf_inv = T.pose_inv(T.pose2mat(env.robots[0].get_position_orientation()))
        
        # Add external sensor poses relative to primary robot
        if env.external_sensors is not None:
            for name, sensor in env.external_sensors.items():
                obs_flat[f"{name}::rel_pose"] = th.cat(
                    T.mat2pose(primary_robot_tf_inv @ T.pose2mat(sensor.get_position_orientation()))
                )
        
        # Add each robot's sensor poses relative to that robot
        for robot in env.robots:
            robot_tf_inv = T.pose_inv(T.pose2mat(robot.get_position_orientation()))
            for name, sensor in robot.sensors.items():
                obs_flat[f"{name}::rel_pose"] = th.cat(
                    T.mat2pose(robot_tf_inv @ T.pose2mat(sensor.get_position_orientation()))
                )

        # Compose lerobot format obs
        frame = dict()
        n_robots = len(env.robots)
        
        # Collect proprio observations for concatenation (preserving order from observation_space)
        proprio_obs_list = []
        
        for name in env.observation_space.keys():
            # Skip if not in obs_mapping (filtered out by include_modalities)
            if name not in obs_mapping:
                continue
                
            obs = obs_flat[name]
            # Prune alpha channel if keeping RGB
            if "rgb" in name:
                obs = obs[..., :-1]
            elif "depth" in name:
                # Add channel dim at the end
                obs = obs.unsqueeze(-1)
                # If we haven't already added the sensor pose obs, do so now
                obs_name_strs = name.split("::")[-2].split(":")
                robot_prefix = ""
                if "robot" in obs_name_strs[0]:
                    # Keep the robot prefix for multi-robot disambiguation
                    robot_prefix = obs_name_strs[0] + "_" if n_robots > 1 else ""
                    # Remove the prefix (e.g. "robot_xxxxxx") and keep the remainder
                    obs_name_strs = obs_name_strs[1:]
                # Join with "_" and make lowercase to make final name
                obs_name = robot_prefix.lower() + "_".join(obs_name_strs).lower()
                tf_name = f"observation.robot2cam_pose.{obs_name}"
                if tf_name not in frame:
                    sensor_name = name.split("::")[-2]
                    frame[tf_name] = obs_flat[f"{sensor_name}::rel_pose"]
            elif "proprio" in name:
                # Map float64 -> float32 and collect for concatenation
                proprio_obs_list.append(obs.float())
                continue  # Don't add individually, will concatenate below
            # Add the observation to the current frame
            frame[obs_mapping[name]] = obs

        # Concatenate all proprio observations into a single state vector
        if proprio_obs_list:
            frame["observation.state"] = th.cat(proprio_obs_list, dim=0)

        return frame

    def create_dataset(self, output_path, env, overwrite=True):
        # Sanity checks
        assert (
            output_path == self.lerobot_dataset_kwargs["repo_id"]
        ), f"Expected LeRobot repo_id path ({self.lerobot_dataset_kwargs['repo_id']}) to match output_path ({output_path})!"

        abs_output_path = f"{self.lerobot_dataset_kwargs['root']}"

        if os.path.exists(abs_output_path):
            if overwrite:
                # Remove any data from this path
                shutil.rmtree(abs_output_path)
            else:
                raise ValueError(f"Found pre-existing LeRobot dataset at: {abs_output_path}")

        # Support multiple robots
        n_robots = len(env.robots)
        assert n_robots >= 1, "At least one robot must be present in the environment!"

        modality_info = {
            "annotation": {
                "language.language_instruction": {},
                "language.language_instruction_2": {},
                "language.language_instruction_3": {},
            },
        }

        # Add video modality info (filtered by include_modalities if specified)
        video_modality_info = dict()
        for i, (sensor_name, sensor) in enumerate(env.external_sensors.items()):
            if isinstance(sensor, VisionSensor):
                for mod in ["rgb", "depth_linear"]:
                    if mod in sensor.modalities:
                        # Skip if include_modalities is specified and this modality is not included
                        if self.include_modalities is not None:
                            if not any(inc_mod in mod for inc_mod in self.include_modalities):
                                continue
                        mod_name = f"observation.{mod}.{sensor_name}"
                        key = f"exterior_image_{i}_{mod}"
                        video_modality_info[key] = {
                            "type": mod,
                            "original_key": mod_name,
                        }
        
        # Add video modality info for each robot's sensors
        robot_sensor_idx = 0
        for robot_idx, robot in enumerate(env.robots):
            robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
            for i, (sensor_name, sensor) in enumerate(robot.sensors.items()):
                if isinstance(sensor, VisionSensor):
                    for mod in ["rgb", "depth_linear"]:
                        if mod in sensor.modalities:
                            # Skip if include_modalities is specified and this modality is not included
                            if self.include_modalities is not None:
                                if not any(inc_mod in mod for inc_mod in self.include_modalities):
                                    continue
                            remapped_sensor_name = "_".join(sensor_name.split(":")[1:]).lower()
                            # Add robot prefix for multi-robot disambiguation
                            obs_name = f"{robot_prefix.lower()}{remapped_sensor_name}"
                            mod_name = f"observation.{mod}.{obs_name}"
                            key = f"robot{robot_idx}_image_{i}_{mod}" if n_robots > 1 else f"robot_image_{robot_sensor_idx}_{mod}"
                            video_modality_info[key] = {
                                "type": mod,
                                "original_key": mod_name,
                            }
                            robot_sensor_idx += 1
        modality_info["video" if self.use_videos else "image"] = video_modality_info

        # If we are including multi action representation, run some additional sanity checks
        if self.include_multi_action_representation:
            # Get start index for each controller across all robots
            self.controller_action_start_idxs = dict()
            cmd_start_idx = 0
            for robot in env.robots:
                robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
                for name, controller in robot.controllers.items():
                    controller_key = f"{robot_prefix}{name}"
                    self.controller_action_start_idxs[controller_key] = cmd_start_idx
                    cmd_start_idx += controller.command_dim
            
            action_modality_info = dict()
            idx = 0
            from omnigibson.controllers import InverseKinematicsController, MultiFingerGripperController

            for robot in env.robots:
                robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
                for arm in robot.arm_names:
                    arm_name = f"arm_{arm}"
                    arm_controller = robot.controllers[arm_name]
                    assert isinstance(
                        arm_controller, InverseKinematicsController
                    ), "Only IKController supported for multi action representation!"
                    gripper_name = f"gripper_{arm}"
                    gripper_controller = robot.controllers[gripper_name]
                    assert isinstance(
                        gripper_controller, MultiFingerGripperController
                    ), "Only MultiFingerGripperController supported for multi action representation!"
                    assert (
                        gripper_controller.command_dim == 1
                    ), "Only binary gripper commands supported for multi action representation!"
                    assert gripper_controller._mode in {
                        "smooth",
                        "binary",
                    }, "Only smooth or binary gripper commands supported for multi action representation!"
                    assert (
                        gripper_controller._motor_type == "position"
                    ), "Only position motor type supported for multi action representation!"
                    
                    # Add robot prefix for multi-robot disambiguation
                    prefixed_arm_name = f"{robot_prefix}{arm_name}"
                    prefixed_gripper_name = f"{robot_prefix}{gripper_name}"
                    
                    action_modality_info[f"{prefixed_arm_name}_eef_pos"] = {
                        "start": idx,
                        "end": idx + 3,
                    }
                    idx += 3
                    action_modality_info[f"{prefixed_arm_name}_eef_aa"] = {
                        "start": idx,
                        "end": idx + 3,
                    }
                    idx += 3
                    action_modality_info[f"{prefixed_arm_name}_delta_eef_pos"] = {
                        "start": idx,
                        "end": idx + 3,
                    }
                    idx += 3
                    action_modality_info[f"{prefixed_arm_name}_delta_eef_aa"] = {
                        "start": idx,
                        "end": idx + 3,
                    }
                    idx += 3
                    action_modality_info[f"{prefixed_gripper_name}_pos"] = {
                        "start": idx,
                        "end": idx + 1,
                    }
                    idx += 1
                    action_modality_info[f"{prefixed_arm_name}_joint_pos"] = {
                        "start": idx,
                        "end": idx + arm_controller.control_dim,
                    }
                    idx += arm_controller.control_dim
                    action_modality_info[f"{prefixed_arm_name}_delta_joint_pos"] = {
                        "start": idx,
                        "end": idx + arm_controller.control_dim,
                    }
                    idx += arm_controller.control_dim
            modality_info["action"] = action_modality_info
            action_shape = (idx,)
        else:
            # Compute total action shape across all robots
            total_action_dim = sum(env.action_space[robot.name].shape[0] for robot in env.robots)
            action_shape = (total_action_dim,)

        # Extract relevant info from original source env config
        config = json.loads(self.input_hdf5["data"].attrs["config"])

        # Create LeRobot dataset, define features to store
        # Define standard features (RL-related entries, language instructions)
        features = {
            "action": {
                "dtype": "float32",
                "shape": action_shape,  # add all actions here
                "names": ["action"],
            },
            # RL-specific fields
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
            "next.terminated": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["done"],
            },
            "next.truncated": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["done"],
            },
            # Language annotation fields store int64 task indices
            # Mapping stored in meta/tasks.jsonl (managed by LeRobot)
            "annotation.language.language_instruction": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["language_instruction"],
            },
            "annotation.language.language_instruction_2": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["language_instruction_2"],
            },
            "annotation.language.language_instruction_3": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["language_instruction_3"],
            },
        }

        obs_mapping, obs_features = self.get_lerobot_obs_mapping(
            env=env, 
            use_videos=self.use_videos,
            include_modalities=self.include_modalities,
        )
        features.update(obs_features)

        if self.lerobot_version == "v2.1":
            dataset_cls = OmniGibsonLeRobotV2Dataset
        elif self.lerobot_version == "v3.0":
            dataset_cls = OmniGibsonLeRobotV3Dataset
        else:
            raise ValueError(f"Got invalid lerobot version: {self.lerobot_version}")
        self.dataset = dataset_cls.create(
            fps=config["env"]["action_frequency"],
            use_videos=self.use_videos,
            features=features,
            **self.lerobot_dataset_kwargs,
        )
        self.obs_mapping = obs_mapping

        # Store proprio shape mapping for each robot (if proprio is included)
        # For multiple robots, concatenate all proprio observations with continuous indices
        proprio_included = self.include_modalities is None or any("proprio" in mod for mod in self.include_modalities)
        if proprio_included:
            proprio_shape_mapping = dict()
            idx = 0  # Continuous index across all robots
            for robot in env.robots:
                robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
                proprio_dict = robot._get_proprioception_dict()
                for obs in robot.proprio_obs:
                    obs_dim = len(proprio_dict[obs])
                    proprio_key = f"{robot_prefix}{obs}" if n_robots > 1 else obs
                    proprio_shape_mapping[proprio_key] = {
                        "start": idx,
                        "end": idx + obs_dim,
                    }
                    idx += obs_dim
            modality_info["state"] = proprio_shape_mapping

        # Add in camera K matrices
        cam_intrinsics = dict()

        # Render to avoid degen intrinsic matrices
        for _ in range(10):
            og.sim.render()

        for sensor_name, sensor in env.external_sensors.items():
            if isinstance(sensor, VisionSensor):
                K = sensor.intrinsic_matrix.cpu()
                cam_intrinsics[sensor_name] = K.numpy().tolist()
        
        # Add camera intrinsics for each robot's sensors
        for robot in env.robots:
            robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
            for sensor_name, sensor in robot.sensors.items():
                if isinstance(sensor, VisionSensor):
                    # Remove robot naming prefix and add multi-robot prefix if needed
                    remapped_sensor_name = "_".join(sensor_name.split(":")[1:]).lower()
                    cam_key = f"{robot_prefix.lower()}{remapped_sensor_name}"
                    K = sensor.intrinsic_matrix.cpu()
                    cam_intrinsics[cam_key] = K.numpy().tolist()
        self.dataset.set_omnigibson_metadata(key="cam_intrinsics", value=cam_intrinsics)

        # Write modality data
        with open(os.path.join(abs_output_path, "meta", "modality.json"), "w+") as f:
            json.dump(modality_info, f, indent=4)

    def process_traj_to_dataset(self, traj_data, nested_keys=("obs",)):
        # Write to LeRobot dataset
        # The dataset length is (N_steps + 1), since the first entry only includes the env reset observations
        # LeRobot expects (s,a) tuples to be paired with rewards from the next step, so we match the obs with
        # all other entries from the proceeding (i.e.: t+1) step

        for frame_idx, traj_step in enumerate(traj_data):
            if frame_idx == 0:
                assert (
                    len(traj_step.keys()) == 1
                ), f"Expected only one key in 0th traj step, but got: {traj_step.keys()}"
                assert "obs" in traj_step, f"Expected 'obs' key in 0th traj step, but got: {traj_step.keys()}"
                continue

            # Compose frame to add to dataset
            frame = {
                "action": traj_step["action"],
                "next.reward": th.tensor([traj_step["reward"]]),
                "next.terminated": th.tensor([traj_step["terminated"]]),
                "next.truncated": th.tensor([traj_step["truncated"]]),
                # TODO: Make these not hardcoded by default
                "annotation.language.language_instruction": th.zeros(1, dtype=th.int64),
                "annotation.language.language_instruction_2": th.zeros(1, dtype=th.int64),
                "annotation.language.language_instruction_3": th.zeros(1, dtype=th.int64),
                **traj_step["obs"],
            }

            # Modify additional differing kwargs between V2 and V3 datasets
            kwargs = {"frame": frame}
            if isinstance(self.dataset, OmniGibsonLeRobotV2Dataset):
                # Task should be separate
                kwargs["task"] = self.task_name
            else:
                # V3, task should be part of dict
                kwargs["frame"]["task"] = self.task_name

            self.dataset.add_frame(**kwargs)

        self.dataset.save_episode()

    def reset(self):
        # Call super first
        out = super().reset()
        self.last_actions = None
        return out

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        step_data = super()._parse_step_data(action, obs, reward, terminated, truncated, info)

        # Handle multi action representation
        if self.include_multi_action_representation:
            # Our action representation is composed of [eef_pos, eef_aa, delta_eef_pos, delta_eef_aa, gripper_pos, joint_pos, delta_joint_pos] for each arm
            # Support multiple robots
            n_robots = len(self.env.robots)
            action = []
            is_first_action = self.last_actions is None
            if is_first_action:
                self.last_actions = dict()
            
            for robot in self.env.robots:
                robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
                for arm in robot.arm_names:
                    arm_name = f"arm_{arm}"
                    arm_controller = robot.controllers[arm_name]
                    
                    # Key for storing last actions (includes robot prefix for multi-robot)
                    action_key = f"{robot_prefix}{arm_name}"

                    # Add eef action
                    action.append(cb.to_torch(arm_controller.goal["target_pos"]))
                    action.append(T.quat2axisangle(T.mat2quat(cb.to_torch(arm_controller.goal["target_ori_mat"]))))

                    # Add delta eef action
                    if is_first_action:
                        action += [th.zeros(3), th.zeros(3)]  # zero delta commands
                    else:
                        action.append(
                            cb.to_torch(
                                arm_controller.goal["target_pos"] - self.last_actions[action_key]["goal"]["target_pos"]
                            )
                        )
                        action.append(
                            T.quat2axisangle(
                                T.mat2quat(
                                    cb.to_torch(
                                        self.last_actions[action_key]["goal"]["target_ori_mat"].T
                                        @ arm_controller.goal["target_ori_mat"]
                                    )
                                )
                            )
                        )

                    # Add gripper action
                    gripper_name = f"gripper_{arm}"
                    gripper_controller = robot.controllers[gripper_name]
                    controller_key = f"{robot_prefix}{gripper_name}"
                    start_idx = self.controller_action_start_idxs[controller_key]
                    end_idx = start_idx + gripper_controller.command_dim
                    action.append(step_data["action"][start_idx:end_idx])

                    # Add joint action
                    action.append(cb.to_torch(arm_controller.control))

                    # Add delta joint action
                    if is_first_action:
                        action.append(th.zeros(arm_controller.control_dim))
                    else:
                        action.append(cb.to_torch(arm_controller.control - self.last_actions[action_key]["control"]))

                    # Update last actions
                    self.last_actions[action_key] = {
                        "goal": arm_controller.goal,
                        "control": arm_controller.control,
                    }

            step_data["action"] = th.cat(action)

        return step_data

    def _process_obs(self, obs, info):
        # Include camera poses wrt robot frame
        obs = super()._process_obs(obs, info)

        # Convert to lerobot format
        obs = self.og_to_lerobot_obs(
            env=self.env, 
            obs_flat=obs, 
            obs_mapping=self.obs_mapping,
            include_modalities=self.include_modalities,
        )
        return obs

    def postprocess_current_traj(self):
        # Does nothing currently
        pass

    def flush_current_file(self):
        # Does nothing currently
        pass

    def close_dataset(self):
        if isinstance(self.dataset, OmniGibsonLeRobotV2Dataset):
            self.dataset.stop_image_writer()
        else:
            # V3, need to finalize to close all active writers
            self.dataset.finalize()

        # Remove the image directory
        abs_output_path = f"{self.lerobot_dataset_kwargs['root']}"
        images_dir = os.path.join(abs_output_path, "images")
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)

    def allocate_traj(
        self,
        step_data,
        episode_id,
        num_samples: int,
        nested_keys=("obs",),
        video_writers=None,
    ):
        # Does nothing currently
        pass

    def add_to_dataset(self, keys, idxs, data, idx_max=None):
        # This should hopefully never be called from LeRobot wrapper, so just raise Error if so
        raise NotImplementedError("add_to_dataset not implemented for LeRobotPlaybackWrapper!")
