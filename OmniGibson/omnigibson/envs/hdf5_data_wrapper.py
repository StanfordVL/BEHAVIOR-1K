import h5py
import json
import logging
import os
import torch as th
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs.data_wrapper import DataWrapper, DataPlaybackWrapper
from omnigibson.envs.env_base import Environment
from omnigibson.macros import gm
from omnigibson.objects.usd_object import USDObject
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.utils.asset_utils import get_omnigibson_git_hash
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)
log.setLevel(logging.INFO)


class HDF5DataWrapper(DataWrapper):
    """
    Specific data wrapper for writing data to HDF5 format
    """

    def __init__(
        self,
        env: Environment,
        output_path: str,
        overwrite: bool = True,
        only_successes: bool = True,
        flush_every_n_traj: int = 10,
        compression: dict | None = None,
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
                Example: {"compression": "gzip", "compression_opts": 9} for gzip with level 9 compression
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

    def create_dataset(self, output_path: str, env: Environment, overwrite: bool = True) -> None:
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
            self.add_metadata(group=data_grp, name="omnigibson_git_hash", data=get_omnigibson_git_hash())

    def process_traj_to_dataset(self, traj_data: list[dict]) -> None:
        traj_grp_name = f"demo_{self.traj_count}"
        traj_grp = self._process_traj_to_hdf5(traj_data, traj_grp_name, nested_keys=["obs"])
        self._postprocess_traj_group(traj_grp)

    def _process_traj_to_hdf5(
        self,
        traj_data: list[dict],
        traj_grp_name: str,
        nested_keys: list[str] = ["obs"],
        data_grp: h5py.Group | None = None,
    ) -> h5py.Group:
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
        assert len(traj_data) > 0, "Expected non-empty trajectory data"
        nested_keys = set(nested_keys)
        data_grp = self.hdf5_file.require_group("data") if data_grp is None else data_grp
        traj_grp = data_grp.create_group(traj_grp_name)
        traj_grp.attrs["num_samples"] = len(traj_data) - 1  # account for the initial obs/state from env reset

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
            num_samples = traj_grp.attrs["num_samples"]
            if k in nested_keys:
                obs_grp = traj_grp.create_group(k)
                for mod, traj_mod_data in dat.items():
                    obs_grp.create_dataset(
                        mod, data=th.stack(traj_mod_data, dim=0)[:num_samples].cpu(), **self.compression
                    )
            else:
                traj_data = (
                    (
                        th.stack(dat, dim=0)[:num_samples]
                        if isinstance(dat[0], th.Tensor)
                        else th.tensor(dat)[:num_samples]
                    )
                    .cpu()
                    .contiguous()
                )
                traj_grp.create_dataset(k, data=traj_data, **self.compression)

        return traj_grp

    def _postprocess_traj_group(self, traj_grp: h5py.Group) -> None:
        """
        Runs any necessary postprocessing on the given trajectory group @traj_grp.
        NOTE: This should be an in-place operation!

        Args:
            traj_grp (h5py.Group): Trajectory group to postprocess
        """
        # Default is no-op
        pass

    def flush_current_file(self) -> None:
        self.hdf5_file.flush()  # Flush data to disk to avoid large memory footprint
        # Retrieve the file descriptor and use os.fsync() to flush to disk
        fd = self.hdf5_file.id.get_vfd_handle()
        os.fsync(fd)
        log.info("Flushing hdf5")

    def add_metadata(self, group: h5py.Group, name: str, data: Any) -> None:
        """
        Adds metadata to the current HDF5 file under the @name key under @group

        Args:
            group (hdf5.File or hdf5.Group): HDF5 object to add an attribute to
            name (str): Name to assign to the data
            data (Any): Data to add. Note that this only supports relatively primitive data types --
                if the data is a dictionary it will be converted into a string-json format using TorchEncoder
        """
        group.attrs[name] = json.dumps(data, cls=TorchEncoder) if isinstance(data, dict) else data

    def close_dataset(self) -> None:
        """
        Closes the active dataset, if open
        """
        if self.hdf5_file.id.valid:
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
        env: Environment,
        output_path: str,
        overwrite: bool = True,
        only_successes: bool = True,
        flush_every_n_traj: int = 10,
        compression: dict | None = None,
        viewport_camera_path: str = "/World/viewer_camera",
        use_vr: bool = False,
        obj_attr_keys: list[str] | None = None,
        keep_checkpoint_rollback_data: bool = False,
        enable_dump_filters: bool = True,
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
            compression (None or dict): If specified, the compression arguments to use for the hdf5 file.
                For more information, check out https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
                Example: {"compression": "gzip", "compression_opts": 9} for gzip with level 9 compression
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
            compression=compression,
        )

        # Configure the simulator to optimize for data collection
        self._enable_dump_filters = enable_dump_filters
        if viewport_camera_path:
            self._optimize_sim_for_data_collection(viewport_camera_path=viewport_camera_path)

    def update_checkpoint(self) -> None:
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

    def rollback_to_checkpoint(self, index: int = -1) -> None:
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

    def _process_traj_to_hdf5(
        self,
        traj_data: list[dict],
        traj_grp_name: str,
        nested_keys: list[str] = ["obs"],
        data_grp: h5py.Group | None = None,
    ) -> h5py.Group:
        # First pad all state values to be the same max (uniform) size
        for step_data in traj_data:
            state = step_data["state"]
            padded_state = th.zeros(self.max_state_size, dtype=th.float32)
            padded_state[: len(state)] = state
            step_data["state"] = padded_state

        # Call super
        traj_grp = super()._process_traj_to_hdf5(traj_data, traj_grp_name, nested_keys, data_grp)

        return traj_grp

    def _postprocess_traj_group(self, traj_grp: h5py.Group) -> None:
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
                    rollback_traj_grp = self._process_traj_to_hdf5(
                        traj_data=rollback_traj["step_data"],
                        traj_grp_name=f"step_{step}-{i}",
                        nested_keys=["obs"],
                        data_grp=rollback_grp,
                    )
                    self.add_metadata(group=rollback_traj_grp, name="transitions", data=rollback_traj["transitions"])

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @is_recording.setter
    def is_recording(self, value: bool) -> None:
        self._is_recording = value

    def _record_step_trajectory(
        self,
        action: th.Tensor,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None:
        if self.is_recording:
            super()._record_step_trajectory(action, obs, reward, terminated, truncated, info)

    def _optimize_sim_for_data_collection(self, viewport_camera_path: str) -> None:
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

    def enable_dump_filters(self) -> None:
        """
        Enables dump filters for optimized per-step state caching
        """
        self.env.scene.object_registry.set_dump_filter(dump_filter=lambda obj: obj.is_active and obj.initialized)

    def disable_dump_filters(self) -> None:
        """
        Disables dump filters for full state caching
        """
        self.env.scene.object_registry.set_dump_filter(dump_filter=lambda obj: True)

    def reset(self) -> tuple[dict, dict]:
        # Call super first
        init_obs, init_info = super().reset()

        # Make sure all objects are awake to begin to guarantee we save their initial states
        for obj in self.scene.objects:
            obj.wake()

        # Store this initial state on the first reset entry so obs/state share step 0.
        state = og.sim.dump_state(serialized=True)
        step_data = {
            "state": state,
            "state_size": len(state),
        }
        self.current_traj_history = [step_data]

        # Update max state size
        self.max_state_size = max(self.max_state_size, len(state))

        # Also store initial metadata not recorded in serialized state
        # This is simply serialized
        metadata = {key: [] for key in self.obj_attr_keys}
        for obj in self.scene.objects:
            for key in self.obj_attr_keys:
                metadata[key].append(getattr(obj, key))
        self.init_metadata = {
            key: th.stack(vals, dim=0) if isinstance(vals[0], th.Tensor) else th.tensor(vals)
            for key, vals in metadata.items()
        }

        # Clear checkpoint states
        self.checkpoint_states = []
        self.checkpoint_step_idxs = []
        if self.checkpoint_rollback_trajs is not None:
            self.checkpoint_rollback_trajs = dict()

        return init_obs, init_info

    def _parse_step_data(
        self,
        action: th.Tensor,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> dict:
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

    def _process_traj_to_hdf5(
        self,
        traj_data: list[dict],
        traj_grp_name: str,
        nested_keys: list[str] = ["obs"],
        data_grp: h5py.Group | None = None,
    ) -> h5py.Group:
        # First pad all state values to be the same max (uniform) size
        for step_data in traj_data:
            state = step_data["state"]
            padded_state = th.zeros(self.max_state_size, dtype=th.float32)
            padded_state[: len(state)] = state
            step_data["state"] = padded_state

        # Call super
        traj_grp = super()._process_traj_to_hdf5(traj_data, traj_grp_name, nested_keys, data_grp)

        return traj_grp

    def flush_current_traj(self) -> None:
        # Call super first
        super().flush_current_traj()

        # Clear transition buffer and max state size
        self.max_state_size = 0
        self.current_transitions = dict()

    @property
    def should_save_current_episode(self) -> bool:
        # In addition to default conditions, we only save the current episode if we are actually recording
        return super().should_save_current_episode and self.is_recording

    def add_transition_info(self, obj: Union[USDObject, "MacroPhysicalParticleSystem"], add: bool = True) -> None:
        """
        Adds transition info to the current sim step for specific object @obj.

        Args:
            obj (USDObject or BaseSystem): Object / system whose information should be stored
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
        info = obj.get_init_info() if isinstance(obj, USDObject) and add else obj.name
        dic_key = "objects" if isinstance(obj, USDObject) else "systems"
        val_key = "add" if add else "remove"
        self.current_transitions[self.env.episode_steps][dic_key][val_key].append(info)


class HDF5PlaybackWrapper(DataPlaybackWrapper, HDF5DataWrapper):
    """
    Playback wrapper for replaying data and writing to an HDF5 file
    """

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
        compression: dict | None = None,
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
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
            compression=compression,
        )
