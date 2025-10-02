import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import logging

import h5py
import imageio
import torch as th
import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs.env_wrapper import EnvironmentWrapper, create_wrapper
from omnigibson.macros import gm, macros
from omnigibson.objects.object_base import BaseObject
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
from omnigibson.utils.vision_utils import instance_to_bbox, overlay_bboxes_with_names, overlay_bboxes, overlay_segmentation_mask
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.data_utils import merge_scene_files
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch, assert_valid_key
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.utils.scene_graph_utils import FrameWriter
from omnigibson.controllers.controller_base import ControlType

# Create module logger
log = create_module_logger(module_name=__name__)
log.setLevel(logging.INFO)


try:
    from omnigibson.learning.utils.obs_utils import write_video
except ImportError:
    write_video = None

h5py.get_config().track_order = True

EXTRA_OBJECT_WHITE_LIST = {
    "straight_chair_uofiqj_0",
    "door_bexenl_0",
    "drop_in_sink_lkklqs_0",
    "coffee_table_rlsebe_0",
    "door_bexenl_0",
    "breakfast_table_xftrki_0",
    "garden_chair_lraplz_0",
    "garden_chair_cottya_0",
    "garden_chair_cottya_1",
    "garden_chair_cottya_2",
    "toilet_udiezm_0",
    "door_vudhlc_1",
    "sliding_door_tprpvb_10",
    "bottom_cabinet_rhdbzv_0",
    "door_bexenl_0",
    "drop_in_sink_awvzkn_0"
}


class DataWrapper(EnvironmentWrapper):
    """
    An OmniGibson environment wrapper for writing data to an HDF5 file.
    """

    def __init__(
        self, env, output_path, compression=dict(), overwrite=True, only_successes=True, flush_every_n_traj=10
    ):
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): path to store hdf5 data file
            compression (dict): If specified, the compression arguments to use for the hdf5 file.
                For more information, check out https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
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
        self.compression = compression

        self.current_traj_history = []

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

        # Run super
        super().__init__(env=env)

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

    def process_traj_to_hdf5(self, traj_data, traj_grp_name, nested_keys=("obs",), data_grp=None):
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

    @property
    def should_save_current_episode(self):
        """
        Returns:
            bool: Whether the current episode should be saved or discarded
        """
        # Only save successful demos and if actually recording
        success = self.env.task.success or not self.only_successes
        return success and self.hdf5_file is not None

    def postprocess_traj_group(self, traj_grp):
        """
        Runs any necessary postprocessing on the given trajectory group @traj_grp. This should be an
        in-place operation!

        Args:
            traj_grp (h5py.Group): Trajectory group to postprocess
        """
        # Default is no-op
        pass

    def flush_current_traj(self):
        """
        Flush current trajectory data
        """
        # Only save successful demos and if actually recording
        if self.should_save_current_episode:
            traj_grp_name = f"demo_{self.traj_count}"
            traj_grp = self.process_traj_to_hdf5(self.current_traj_history, traj_grp_name, nested_keys=["obs"])
            self.traj_count += 1
            self.postprocess_traj_group(traj_grp)

            # Potentially write to disk
            if self.traj_count % self.flush_every_n_traj == 0:
                self.flush_current_file()
        else:
            # Remove this demo
            self.step_count -= len(self.current_traj_history)

        # Clear trajectory and transition buffers
        self.current_traj_history = []

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

    def save_data(self):
        """
        Save collected trajectories as a hdf5 file in the robomimic format
        """
        if len(self.current_traj_history) > 0:
            self.flush_current_traj()

        if self.hdf5_file is not None:
            log.info(
                f"\nSaved:\n"
                f"{self.traj_count} trajectories / {self.step_count} total steps\n"
                f"to hdf5: {self.hdf5_file.filename}\n"
            )

            self.hdf5_file["data"].attrs["n_episodes"] = self.traj_count
            self.hdf5_file["data"].attrs["n_steps"] = self.step_count
            self.hdf5_file.close()


class DataCollectionWrapper(DataWrapper):
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

    def postprocess_traj_group(self, traj_grp):
        super().postprocess_traj_group(traj_grp=traj_grp)

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

        # Store this initial state as part of the trajectory
        state = og.sim.dump_state(serialized=True)
        step_data = {
            "state": state,
            "state_size": len(state),
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
        compression=dict(),
        robot_obs_modalities=tuple(),
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
    ):
        """
        Create a DataPlaybackWrapper environment instance form the recorded demonstration info
        from @hdf5_path, and aggregate observation_modalities @obs during playback

        Args:
            input_path (str): Absolute path to the input hdf5 file containing the relevant collected data to playback
            output_path (str): Absolute path to the output hdf5 file that will contain the recorded observations from
                the replayed data
            compression (dict): If specified, the compression arguments to use for the hdf5 file.
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
        if external_sensors_config is not None:
            config["env"]["external_sensors"] = external_sensors_config

        # Load env
        env = og.Environment(configs=config)

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
            compression=compression,
            n_render_iterations=n_render_iterations,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
        )

    def __init__(
        self,
        env,
        input_path,
        output_path,
        compression=dict(),
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        flush_every_n_steps=0,
        full_scene_file=None,
        load_room_instances=None,
        include_robot_control=True,
        include_contacts=True,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            input_path (str): path to input hdf5 collected data file
            output_path (str): path to store output hdf5 data file
            compression (dict): If specified, the compression arguments to use for the hdf5 file.
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
        self.n_render_iterations = n_render_iterations
        if flush_every_n_steps > 0:
            assert flush_every_n_traj == 1, "flush_every_n_traj must be 1 if flush_every_n_steps is greater than 0"
        self.flush_every_n_steps = flush_every_n_steps

        self.current_traj_grp = None
        self.current_episode_step_count = 0
        self.traj_dsets = dict()
        self.include_robot_control = include_robot_control
        self.include_contacts = include_contacts

        # Run super
        super().__init__(
            env=env,
            output_path=output_path,
            compression=compression,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
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

    def playback_episode(self, episode_id, record_data=True, video_writers=None, video_rgb_keys=None, replay_config=None):
        """
        Playback episode @episode_id, and optionally record observation data if @record is True

        Args:
            episode_id (int): Episode to playback. This should be a valid demo ID number from the inputted collected
                data hdf5 file
            record_data (bool): Whether to record data during playback or not
            video_writers (None or list of imageio.Writer): If specified, writer objects that RGB frames will be written to
            video_rgb_keys (None or list of str): If specified, observation keys representing the RGB frames to write to video.
                If @video_writers is specified, this must also be specified!
        """
        # Validate video_writers and video_rgb_keys
        if video_writers is not None:
            assert video_rgb_keys is not None, "If video_writers is specified, video_rgb_keys must also be specified!"
            assert len(video_writers) == len(
                video_rgb_keys
            ), "video_writers and video_rgb_keys must have the same length!"

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

        # # pre-defined colors for robot skin
        # robot_skin_colors = {
        #     "red": th.tensor([1.0, 0.0, 0.0]),
        #     "green": th.tensor([0.0, 1.0, 0.0]), 
        #     "blue": th.tensor([0.0, 0.0, 1.0]),
        #     "white": th.tensor([1.0, 1.0, 1.0]),
        #     "black": th.tensor([0.0, 0.0, 0.0])
        # }

        # skin_color_name = "white"

        # # include robot skin control
        # active_robot = self.env.robots[0]
        # for mat in active_robot.materials:
        #     mat.diffuse_texture = ""
        # for mat in active_robot.materials:
        #     mat.diffuse_color_constant = robot_skin_colors[skin_color_name] # set to white


        # Restore to initial state
        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)

        # If record, record initial observations
        if record_data:
            # We need to step the environment to get the initial observations propagated
            first_time_load_n_iteration = 10
            self.current_obs, _, _, _, init_info = self.env.step(
                action=action[0], n_render_iterations=self.n_render_iterations + first_time_load_n_iteration
            )
            step_data = {"obs": self._process_obs(obs=self.current_obs, info=init_info)}
            self.current_traj_history.append(step_data)

        # import random
        # random.seed(42)
        if replay_config is not None and replay_config["record_visibility"]:
            # get task_relevant_object names
            task = self.env.task
            task_objects = [bddl_obj.wrapped_obj for bddl_obj in task.object_scope.values() 
                            if bddl_obj.wrapped_obj is not None and bddl_obj.exists]
            task_relevant_objects = [obj.name for obj in task_objects if not isinstance(obj, BaseRobot) and 'floors' not in obj.name]

            # get extra object white list
            extra_object_white_list = [obj.name for obj in self.scene.objects if obj.name in EXTRA_OBJECT_WHITE_LIST]

            # get all object names
            task_relevant_objects = task_relevant_objects + extra_object_white_list

        
        for i, (a, s, ss, r, te, tr) in enumerate(
            zip(action, state[1:], state_size[1:], reward, terminated, truncated)
        ):

            # if i % 200 == 0:
            #     # update robot skin color 
            #     active_robot = self.env.robots[0]
            #     last_skin_color_name = skin_color_name
            #     remaining_colors = {k: v for k, v in robot_skin_colors.items() if k != last_skin_color_name}
            #     skin_color_name = random.choice(list(remaining_colors.keys()))
            #     for mat in active_robot.materials:
            #         mat.diffuse_color_constant = remaining_colors[skin_color_name]
            #     for _ in range(100):
            #         og.sim.render()
                

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
                if self.flush_every_n_steps > 0:
                    if i == 0:
                        self.current_traj_grp, self.traj_dsets = self.allocate_traj_to_hdf5(
                            step_data, f"demo_{episode_id}", num_samples=len(action), video_writers=video_writers
                        )
                    if i % self.flush_every_n_steps == 0:
                        self.flush_partial_traj(num_samples=len(action), video_writers=video_writers)
                # append to current trajectory history
                self.current_traj_history.append(step_data)

            if replay_config is not None and replay_config["record_visibility"]:
                object_visibility_in_camera_dict = {}
                seg_camera_keys = [key.replace("rgb", "seg_instance") for key in replay_config["record_rgb_keys"]]
                seg_camera_info_list = info["obs_info"]
                camera_types = seg_camera_info_list.keys()

                bbox_dict = {}
                instance_mapping_dict = {}
                visiblility_matrix_dict = {}

                for sensor in replay_config["sensors"]:
                    for camera_type in camera_types:
                        if camera_type not in sensor:
                            continue

                        # below is the object id to object name mapping
                        current_camera_info = seg_camera_info_list[camera_type][sensor]['seg_instance']
                        all_unique_obj_ids = current_camera_info.keys()
                        # now we get the corresponding seg_camera visibility matrix
                        # we noticed that the sensor name is not the same as the obs stored name
                        # so we need to find the correct
                        found_seg_obs_key = None
                        for key in seg_camera_keys:
                            if sensor in key:
                                found_seg_obs_key = key
                                break
                        assert found_seg_obs_key is not None, f"We cannot find the corresponding seg_camera visibility matrix for {sensor} and {camera_type}"

                        # below is a tensor of shape (H, W), where each element is the object id of the visible object for curerent camera view
                        current_seg_camera_visibility_matrix = self.current_obs[found_seg_obs_key]

                        # for each object, will store a dict like {camera_name: [pixel_num, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, img_H, img_W]}
                        
                        # 1. we first get object id to num_pixels mapping
                        # build a dictionary of {object_id: num_pixels}
                        obj_pixels_dict = {}
                        available_obj_ids = th.unique(current_seg_camera_visibility_matrix)
                        for obj_id in available_obj_ids:
                            obj_pixels_dict[obj_id.item()] = (current_seg_camera_visibility_matrix == obj_id).sum().item()
                        
                        # 2. we then get the bbox for each object
                        bboxes = instance_to_bbox(current_seg_camera_visibility_matrix, current_camera_info, all_unique_obj_ids)
                        bbox_dict[sensor] = bboxes
                        instance_mapping_dict[sensor] = current_camera_info
                        visiblility_matrix_dict[sensor] = current_seg_camera_visibility_matrix


            # If writing to video, write desired frame
            if video_writers is not None:
                for writer, rgb_key in zip(video_writers, video_rgb_keys):
                    assert_valid_key(rgb_key, self.current_obs.keys(), "video_rgb_key")
                    raw_frame = self.current_obs[rgb_key][:, :, :3].numpy()
                    camera_type = rgb_key.split("::")[1]
                    if replay_config is not None and replay_config["record_visibility"] and camera_type in bbox_dict:
                        # Get the unique instance IDs for this camera
                        unique_ids = th.unique(visiblility_matrix_dict[camera_type]).cpu().numpy().tolist()
                        # Ensure the array is contiguous and compatible with OpenCV
                        if not raw_frame.flags.c_contiguous:
                            raw_frame = np.ascontiguousarray(raw_frame)
                        # Use segmentation mask overlay with 50% transparency
                        frame = overlay_segmentation_mask(
                            img=raw_frame,
                            visibility_matrix=visiblility_matrix_dict[camera_type],
                            unique_ins_ids=unique_ids,
                            instance_mapping=instance_mapping_dict[camera_type],
                            task_relevant_objects=task_relevant_objects,
                            in_place=True,
                        )
                    else:
                        frame = raw_frame
                    writer.append_data(frame)
                    del raw_frame

            self.current_episode_step_count += 1
            self.step_count += 1

        if record_data:
            if self.flush_every_n_steps > 0:
                self.flush_partial_traj(num_samples=len(action), video_writers=video_writers)
            self.flush_current_traj()

    def playback_dataset(self, record_data=False, video_writers=None, video_rgb_keys=None):
        """
        Playback all episodes from the input HDF5 file, and optionally record observation data if @record is True

        Args:
            record_data (bool): Whether to record data during playback or not
            video_writers (None or list of imageio.Writer): If specified, writer object that RGB frames will be written to
            video_rgb_keys (None or list of str): If specified, observation key representing the RGB frames to write to video.
                If @video_writer is specified, this must also be specified!
        """
        for episode_id in range(self.input_hdf5["data"].attrs["n_episodes"]):
            self.playback_episode(
                episode_id=episode_id,
                record_data=record_data,
                video_writers=video_writers,
                video_rgb_keys=video_rgb_keys,
            )

    def create_video_writer(self, fpath, fps=30):
        """
        Creates a video writer to write video frames to when playing back the dataset

        Args:
            fpath (str): Absolute path that the generated video writer will write to. Should end in .mp4
            fps (int): Desired frames per second when generating video

        Returns:
            imageio.Writer: Generated video writer
        """
        return imageio.get_writer(fpath, fps=fps)

    def create_frame_writer(self, output_dir, filename_prefix=""):
        """
        Creates a FrameWriter object for saving RGB frames as PNG files.

        Args:
            output_dir (str): Directory where PNG frames will be saved
            filename_prefix (str): Optional prefix for filenames (default: "")

        Returns:
            FrameWriter: Writer object for saving frames
        """
        return FrameWriter(output_dir=output_dir, filename_prefix=filename_prefix)

    def allocate_traj_to_hdf5(
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
                        self.traj_dsets[key][mod][0] = self.current_traj_history[0][key][mod]
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
                                self.traj_dsets[key][mod][
                                    self.current_episode_step_count
                                    - data_length_to_flush
                                    + 1 : self.current_episode_step_count + 1
                                ] = data_to_write
                else:
                    self.traj_dsets[key][
                        self.current_episode_step_count - data_length_to_flush : self.current_episode_step_count
                    ] = th.stack([self.current_traj_history[i][key] for i in range(data_length_to_flush)], dim=0)
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
            self.postprocess_traj_group(self.current_traj_grp)
            self.flush_current_file()
            # Clear trajectory and transition buffers
            self.traj_count += 1
            self.current_episode_step_count = 0
            self.current_traj_history = []


class SceneGraphDataPlaybackWrapper(DataPlaybackWrapper):
    """
    A wrapper that supports the scene graph recording.
    It inherits from the original DataPlaybackWrapper and overrides the `create_from_hdf5` and `playback_episode` method.
    """
    @classmethod
    def create_from_hdf5(
        cls,
        input_path,
        output_path,
        robot_obs_modalities=tuple(),
        robot_sensor_config=None,
        external_sensors_config=None,
        include_sensor_names=None,
        exclude_sensor_names=None,
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        include_env_wrapper=False,
        additional_wrapper_configs=None,
        full_scene_file=None,
        include_task=True,
        include_task_obs=True,
        include_robot_control=True,
        include_contacts=True,
        scene_graph_config=None,
    ):
        """
        Create a DataPlaybackWrapper environment instance form the recorded demonstration info
        from @hdf5_path, and aggregate observation_modalities @obs during playback

        Args:
            input_path (str): Absolute path to the input hdf5 file containing the relevant collected data to playback
            output_path (str): Absolute path to the output hdf5 file that will contain the recorded observations from
                the replayed data
            robot_obs_modalities (list): Robot observation modalities to use. This list is directly passed into
                the robot_cfg (`obs_modalities` kwarg) when spawning the robot
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
            include_robot_control (bool): Whether or not to include robot control. If False, will set all
                robot.control_enabled=False
            include_contacts (bool): Whether or not to include (enable) contacts in the sim. If False, will set all
                objects to be visual_only
            scene_graph_config (None or dict): If specified, the scene graph configuration to use for the replay.
                This can be used to specify relevant scene graph parameters, such as the scene file to use.

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
            # Since we are setting all objects to be visual-only, physics will not be propogating
            config["env"]["action_frequency"] = 30.0
            config["env"]["rendering_frequency"] = 30.0
            config["env"]["physics_frequency"] = 120.0

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
            config["scene"]["load_room_types"] = None
            config["scene"]["load_room_instances"] = None

        # Use dummy task if not loading task
        if not include_task:
            config["task"] = {"type": "DummyTask"}

        # Maybe include task observations
        config["task"]["include_obs"] = include_task_obs

        # Set scene file and disable online object sampling if BehaviorTask is being used
        if config["task"]["type"] == "BehaviorTask":
            config["task"]["online_object_sampling"] = False

        # Because we're loading directly from the cached scene file, we need to disable any additional objects that are being added since
        # they will already be cached in the original scene file
        config["objects"] = []

        # Set observation modalities and update sensor config
        for robot_cfg in config["robots"]:
            robot_cfg["obs_modalities"] = list(robot_obs_modalities)
            robot_cfg["include_sensor_names"] = include_sensor_names
            robot_cfg["exclude_sensor_names"] = exclude_sensor_names

            if robot_sensor_config is not None:
                robot_cfg["sensor_config"] = robot_sensor_config
        if external_sensors_config is not None:
            config["env"]["external_sensors"] = external_sensors_config


        # Add scene graph config
        if scene_graph_config is not None:
            config["scene_graph"] = scene_graph_config
        else:
            # load default scene graph config here
            print("No scene graph config provided, using default config")
            print("We set to None for now")
            scene_graph_config = {
                "robot_names": [robot_cfg["name"] for robot_cfg in config["robots"]],
                "egocentric": False,           # Use world coordinate system for analysis
                "full_obs": True,              # Record all objects, not just in view
                "only_true": True,            # Include all relationship states, including False
                "merge_parallel_edges": True, # Merge parallel edges
                "only_task_relevant_objects": True # Only consider task relevant objects
            }
            config["scene_graph"] = scene_graph_config


        # Load env
        env = og.Environment(configs=config)

        if not include_contacts:
            with og.sim.stopped():
                for obj in env.scene.objects:
                    obj.visual_only = True

        # If not controlling robots, disable for all robots
        # for robot in env.robots:
        #     robot.control_enabled = include_robot_control

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
            include_contacts=include_contacts,
            include_robot_control=include_robot_control,
        )
    
    def playback_episode(
        self, 
        episode_id, 
        record_data=True, 
        video_writers=None, 
        video_rgb_keys=None, 
        frame_writers=None,
        frame_rgb_keys=None,
        start_frame=None, 
        end_frame=None,
        scene_graph_writer=None,
        replay_config=None,
    ):
        """
        Playback episode @episode_id, and optionally record observation data if @record is True

        Args:
            episode_id (int): Episode to playback. This should be a valid demo ID number from the inputted collected
                data hdf5 file
            record_data (bool): Whether to record data during playback or not
            video_writers (None or list of imageio.Writer): If specified, writer objects that RGB frames will be written to
            video_rgb_keys (None or list of str): If specified, observation keys representing the RGB frames to write to video.
                If @video_writers is specified, this must also be specified!
            start_frame (None or int): If specified, the frame to start playback from. If not specified, will start from the first frame
            end_frame (None or int): If specified, the frame to end playback at. If not specified, will end at the last frame
            scene_graph_writer (None or SceneGraphWriter): If specified, a SceneGraphWriter object that will be used to record the scene graph
            frame_writers (None or list of FrameWriter): If specified, FrameWriter objects to save RGB frames to
            frame_rgb_keys (None or list of str): If specified, observation keys representing the RGB frames to save as PNG files.
            replay_config (None or dict): If specified, the replay configuration to use for the replay, eg, whether to record the pixel visibility of objects from certain viewpoints. Must include the following keys:
                "record_visibility": bool, whether to record the pixel visibility of objects from certain viewpoints
                "record_rgb_keys": list of str, the observation keys representing the RGB frames to record
                "sensors": list of str, the viewpoints to record the pixel visibility of objects
        """
        # Validate video_writers and video_rgb_keys
        if video_writers is not None:
            assert video_rgb_keys is not None, "If video_writers is specified, video_rgb_keys must also be specified!"
            assert len(video_writers) == len(
                video_rgb_keys
            ), "video_writers and video_rgb_keys must have the same length!"

        data_grp = self.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        # Get start and end frame
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else len(traj_grp["state"])

        assert start_frame >= 0 and start_frame < end_frame, "start_frame must be less than end_frame and greater than 0!"
        assert end_frame <= len(traj_grp["state"]), "end_frame must be less than the total number of frames!"

        log.info(f"Playing back episode {episode_id} from frame {start_frame} to frame {end_frame}")

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

        # Restore to initial state
        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)
        if scene_graph_writer is not None:
            scene_graph_writer.step(self.env.get_scene_graph(), 0)

        # If record, record initial observations
        if record_data:
            init_obs, _, _, _, init_info = self.env.step(action=action[0], n_render_iterations=self.n_render_iterations)
            step_data = {"obs": self._process_obs(obs=init_obs, info=init_info)}
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
                    # TODO: delete this object from scene graph
                for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                    obj = create_object_from_init_info(add_obj_info)
                    scene.add_object(obj)
                    obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
                # Step physics to initialize any new objects
                og.sim.step()

            self.step_count += 1

            # Restore the sim state, and take a very small step with the action to make sure physics are
            # properly propagated after the sim state update
            og.sim.load_state(s[: int(ss)], serialized=True)
            
            if i+1 < start_frame:
                continue
            if i+1 > end_frame:
                break
            
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
                self.current_traj_history.append(step_data)

            # If writing to video, write desired frame
            if video_writers is not None:
                for writer, rgb_key in zip(video_writers, video_rgb_keys):
                    assert_valid_key(rgb_key, self.current_obs.keys(), "video_rgb_key")
                    writer.append_data(self.current_obs[rgb_key][:, :, :3].numpy())
                    
            # If saving frames as PNG, save desired frame
            if frame_writers is not None:
                for frame_writer, frame_rgb_key in zip(frame_writers, frame_rgb_keys):
                    assert_valid_key(frame_rgb_key, self.current_obs.keys(), "frame_rgb_key")
                    frame_data = self.current_obs[frame_rgb_key][:, :, :3].numpy()
                    frame_writer.append_data(frame_data, i+1)
            
            if replay_config is not None and replay_config["record_visibility"]:
                obj_visibility_in_camera_dict = {} # this is a dict of {obj_name:{camera_name: visibility}}

                seg_camera_keys = [key.replace("rgb", "seg_instance") for key in replay_config["record_rgb_keys"]]
                seg_camera_info_list = info["obs_info"]
                camera_types = seg_camera_info_list.keys()

                for sensor in replay_config["sensors"]:
                    for camera_type in camera_types:
                        if camera_type not in sensor:
                            continue

                        # below is the object id to object name mapping
                        current_camera_info = seg_camera_info_list[camera_type][sensor]['seg_instance']
                        all_unique_obj_ids = current_camera_info.keys()
                        # now we get the corresponding seg_camera visibility matrix
                        # we noticed that the sensor name is not the same as the obs stored name
                        # so we need to find the correct
                        found_seg_obs_key = None
                        for key in seg_camera_keys:
                            if sensor in key:
                                found_seg_obs_key = key
                                break
                        assert found_seg_obs_key is not None, f"We cannot find the corresponding seg_camera visibility matrix for {sensor} and {camera_type}"

                        # below is a tensor of shape (H, W), where each element is the object id of the visible object for curerent camera view
                        current_seg_camera_visibility_matrix = self.current_obs[found_seg_obs_key]
                        current_rgb_image = self.current_obs[found_seg_obs_key.replace("seg_instance", "rgb")]

                        # for each object, will store a dict like {camera_name: [pixel_num, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, img_H, img_W]}
                        
                        # 1. we first get object id to num_pixels mapping
                        # build a dictionary of {object_id: num_pixels}
                        obj_pixels_dict = {}
                        available_obj_ids = th.unique(current_seg_camera_visibility_matrix)
                        for obj_id in available_obj_ids:
                            obj_pixels_dict[obj_id.item()] = (current_seg_camera_visibility_matrix == obj_id).sum().item()
                        
                        # 2. we then get the bbox for each object
                        bboxes = instance_to_bbox(current_seg_camera_visibility_matrix, current_camera_info, all_unique_obj_ids)
                        bboxes_dict = {bbox[4]: bbox[:4] for bbox in bboxes}

                        # 3. we then get the delta_e for each object (skipped for now, as it is too slow)
                        # delta_e_dict = calculate_delta_e(current_seg_camera_visibility_matrix, current_rgb_image, current_camera_info, all_unique_obj_ids)

                        # 4. lastly, we get the img_H and img_W
                        img_H = current_rgb_image.shape[0]
                        img_W = current_rgb_image.shape[1]

                        # now we store the above info into the format we want: {obj_name: {sensor: [pixel_num, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, img_H, img_W]}}
                        for obj_id in available_obj_ids:
                            obj_name = current_camera_info[obj_id.item()]
                            if obj_name not in obj_visibility_in_camera_dict:
                                obj_visibility_in_camera_dict[obj_name] = {}
                            obj_visibility_in_camera_dict[obj_name][sensor] = [obj_pixels_dict[obj_id.item()], bboxes_dict[obj_id.item()][0], bboxes_dict[obj_id.item()][1], bboxes_dict[obj_id.item()][2], bboxes_dict[obj_id.item()][3], img_H, img_W]
                        break


            if scene_graph_writer is not None:
                scene_graph_writer.step(self.env.get_scene_graph(), i+1, obj_visibility_in_camera_dict)


        if record_data:
            self.flush_current_traj()

        if scene_graph_writer is not None:
            scene_graph_writer.close()
        
        if frame_writers is not None:
            for frame_writer in frame_writers:
                frame_writer.close()