"""
IG logging classes that write/read iGibson data to/from HDF5. These classes
can be used to write regular logs, iGATUS task logs or VR logs to HDF5 for saving and replay.
"""

import copy
import datetime
import time

import h5py
import numpy as np


# TODO!
# from igibson.robots.behavior_robot import HAND_BASE_ROTS
from igibson.utils.git_utils import project_git_info
from igibson.utils.utils import dump_config, parse_str_config
from igibson.utils.vr_utils import VR_BUTTON_COMBO_NUM, VrData, convert_button_data_to_binary


class IGLogWriter(object):
    """Class that handles saving of physics data, VR data, iGATUS task data and user-defined actions.

    Usage:
    1) Before simulation loop
    init -> N x register_action -> set_up_data_storage

    2) During simulation:
    N x save_action (at any point during frame) -> process_frame (at end of frame)

    3) After simulation, before disconnecting from PyBullet sever:
    end_log_session
    """

    def __init__(
        self,
        sim,
        log_filepath,
        frames_before_write=200,
        task=None,
        store_vr=False,
        vr_robot=None,
        filter_objects=True,
        profiling_mode=False,
        log_status=True,
    ):
        """
        Initializes IGLogWriter
        :param sim: Simulator object
        :param frames_before_write: number of frames to accumulate data in program memory before writing to file storage
        :param log_filepath: filepath to save to
        :param task: iGATUS task - will not store task-relevant features if None
        :param store_vr: boolean indicating whether to store VR data
        :param vr_robot: BehaviorRobot object
        :param filter_objects: whether to filter objects
        :param profiling_mode: whether to print out how much time each log-write takes
        :param log_status: whether to log status updates to the console
        """
        self.sim = sim
        # The number of frames to store data on the stack before writing to HDF5.
        # We buffer and flush data like this to cause a small an impact as possible
        # on the VR frame-rate.
        self.frames_before_write = frames_before_write
        self.log_filepath = log_filepath
        self.filter_objects = filter_objects
        self.profiling_mode = profiling_mode
        self.log_status = log_status
        # Reuse online checking calls
        self.task = task
        self.store_vr = store_vr
        self.vr_robot = vr_robot
        self.data_map = None
        if self.task:
            self.obj_body_id_to_name = {}
            for obj_name, obj in self.task.object_scope.items():
                for body_id in obj.get_body_ids():
                    self.obj_body_id_to_name[body_id] = obj_name
            self.obj_body_id_to_name_str = dump_config(self.obj_body_id_to_name)

        if self.task and self.filter_objects:
            self.tracked_objects = {}
            for obj_name, obj in self.task.object_scope.items():
                for body_id in obj.get_body_ids():
                    self.tracked_objects[body_id] = obj
        else:
            self.tracked_objects = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]

        self.joint_map = {bid: p.getNumJoints(bid) for bid in self.tracked_objects}
        # Sentinel that indicates a certain value was not set in the HDF5
        self.default_fill_sentinel = -1.0
        # Numpy dtype common to all values
        self.np_dtype = np.float64
        # Counts number of frames (reset to 0 every self.frames_before_write)
        self.frame_counter = 0
        # Counts number of frames and does not reset
        self.persistent_frame_count = 0
        # Handle of HDF5 file
        self.hf = None
        # Name path data - used to extract data from data map and save to hdf5
        self.name_path_data = []
        self.generate_name_path_data()
        # Create data map
        self.create_data_map()

    def generate_name_path_data(self):
        """Generates lists of name paths for resolution in hdf5 saving.
        Eg. ['vr', 'vr_camera', 'right_eye_view']."""
        self.name_path_data.extend([["frame_data"]])

        for bid in self.tracked_objects:
            obj = str(bid)
            base = ["physics_data", obj]
            for registered_property in ["position", "orientation", "joint_state"]:
                self.name_path_data.append(copy.deepcopy(base) + [registered_property])

        if self.task:
            self.name_path_data.extend([["goal_status", "satisfied"], ["goal_status", "unsatisfied"]])
        if self.store_vr:
            self.name_path_data.extend(
                [
                    ["vr", "vr_camera", "right_eye_view"],
                    ["vr", "vr_camera", "right_eye_proj"],
                    ["vr", "vr_camera", "right_camera_pos"],
                    ["vr", "vr_device_data", "hmd"],
                    ["vr", "vr_device_data", "left_controller"],
                    ["vr", "vr_device_data", "right_controller"],
                    ["vr", "vr_device_data", "vr_position_data"],
                    ["vr", "vr_device_data", "torso_tracker"],
                    ["vr", "vr_button_data", "left_controller"],
                    ["vr", "vr_button_data", "right_controller"],
                    ["vr", "vr_eye_tracking_data"],
                    ["vr", "vr_event_data", "left_controller"],
                    ["vr", "vr_event_data", "right_controller"],
                    ["vr", "vr_event_data", "reset_actions"],
                ]
            )
        if self.vr_robot:
            self.name_path_data.extend([["agent_actions", "vr_robot"]])

    def create_data_map(self):
        """Creates data map of data that will go into HDF5 file. All the data in the
        map is reset after every self.frames_before_write frames, by refresh_data_map."""
        self.data_map = dict()
        self.data_map["action"] = dict()
        self.data_map["frame_data"] = np.full(
            (self.frames_before_write, 4), self.default_fill_sentinel, dtype=self.np_dtype
        )

        if self.task:
            self.task.check_success()
            self.total_goals = len(self.task.current_goal_status["satisfied"]) + len(
                self.task.current_goal_status["unsatisfied"]
            )

            self.data_map["goal_status"] = {
                "satisfied": np.full((self.frames_before_write, self.total_goals), self.default_fill_sentinel),
                "unsatisfied": np.full((self.frames_before_write, self.total_goals), self.default_fill_sentinel),
            }

        self.data_map["physics_data"] = dict()
        for bid in self.tracked_objects:
            obj = str(bid)
            self.data_map["physics_data"][obj] = dict()
            handle = self.data_map["physics_data"][obj]
            handle["position"] = np.full((self.frames_before_write, 3), self.default_fill_sentinel)
            handle["orientation"] = np.full((self.frames_before_write, 4), self.default_fill_sentinel)
            handle["joint_state"] = np.full((self.frames_before_write, self.joint_map[bid]), self.default_fill_sentinel)

        if self.store_vr:
            self.data_map["vr"] = {
                "vr_camera": {
                    "right_eye_view": np.full(
                        (self.frames_before_write, 4, 4), self.default_fill_sentinel, dtype=self.np_dtype
                    ),
                    "right_eye_proj": np.full(
                        (self.frames_before_write, 4, 4), self.default_fill_sentinel, dtype=self.np_dtype
                    ),
                    "right_camera_pos": np.full(
                        (self.frames_before_write, 3), self.default_fill_sentinel, dtype=self.np_dtype
                    ),
                },
                "vr_device_data": {
                    "hmd": np.full((self.frames_before_write, 17), self.default_fill_sentinel, dtype=self.np_dtype),
                    "left_controller": np.full(
                        (self.frames_before_write, 27 if self.vr_robot else 21),
                        self.default_fill_sentinel,
                        dtype=self.np_dtype,
                    ),
                    "right_controller": np.full(
                        (self.frames_before_write, 27 if self.vr_robot else 21),
                        self.default_fill_sentinel,
                        dtype=self.np_dtype,
                    ),
                    "vr_position_data": np.full(
                        (self.frames_before_write, 12 if self.vr_robot else 6),
                        self.default_fill_sentinel,
                        dtype=self.np_dtype,
                    ),
                    "torso_tracker": np.full(
                        (self.frames_before_write, 8), self.default_fill_sentinel, dtype=self.np_dtype
                    ),
                },
                "vr_button_data": {
                    "left_controller": np.full(
                        (self.frames_before_write, 3), self.default_fill_sentinel, dtype=self.np_dtype
                    ),
                    "right_controller": np.full(
                        (self.frames_before_write, 3), self.default_fill_sentinel, dtype=self.np_dtype
                    ),
                },
                "vr_eye_tracking_data": np.full(
                    (self.frames_before_write, 9), self.default_fill_sentinel, dtype=self.np_dtype
                ),
                "vr_event_data": {
                    "left_controller": np.full(
                        (self.frames_before_write, VR_BUTTON_COMBO_NUM), self.default_fill_sentinel, dtype=self.np_dtype
                    ),
                    "right_controller": np.full(
                        (self.frames_before_write, VR_BUTTON_COMBO_NUM), self.default_fill_sentinel, dtype=self.np_dtype
                    ),
                    "reset_actions": np.full(
                        (self.frames_before_write, 2), self.default_fill_sentinel, dtype=self.np_dtype
                    ),
                },
            }

        if self.vr_robot:
            self.data_map["agent_actions"] = {
                "vr_robot": np.full((self.frames_before_write, 28), self.default_fill_sentinel, dtype=self.np_dtype)
            }

    def register_action(self, action_path, action_shape):
        """Registers an action to be saved every frame in the VRLogWriter.

        Args:
            action_path: The /-separated path specifying where to save action data. All entries but the last will be treated
                as group names, and the last entry will the be the dataset. The parent group for all
                actions is called action. Eg. action_path = vr_hand/constraint. This will end up in
                action (group) -> vr_hand (group) -> constraint (dataset) in the saved data.
            action_shape: tuple representing action shape. It is expected that all actions will be numpy arrays. They
                are stacked over time in the first dimension to create a persistent action data store.
        """
        # Extend name path data - this is used for fast saving and lookup later on
        act_path = ["action"]
        path_tokens = action_path.split("/")
        act_path.extend(path_tokens)
        self.name_path_data.append(act_path)

        # Add action to dictionary - create any new dictionaries that don't yet exist
        curr_dict = self.data_map["action"]
        for tok in path_tokens[:-1]:
            if tok in curr_dict.keys():
                curr_dict = curr_dict[tok]
            else:
                curr_dict[tok] = dict()
                curr_dict = curr_dict[tok]

        # Curr_dict refers to the last group - we then add in the dataset of the right shape
        # The action is extended across self.frames_before_write rows
        extended_shape = (self.frames_before_write,) + action_shape
        curr_dict[path_tokens[-1]] = np.full(extended_shape, self.default_fill_sentinel, dtype=self.np_dtype)

    def set_up_data_storage(self):
        """Performs set up of internal data structures needed for storage, once
        VRLogWriter has been initialized and all actions have been registered."""
        # Note: this erases the file contents previously stored as self.log_filepath
        hf = h5py.File(self.log_filepath, "w")
        for name_path in self.name_path_data:
            joined_path = "/".join(name_path)
            curr_data_shape = (0,) + self.get_data_for_name_path(name_path).shape[1:]
            # None as first shape value allows dataset to grow without bound through time
            max_shape = (None,) + curr_data_shape[1:]
            # Create_dataset with a '/'-joined path automatically creates the required groups
            # Important note: we store values with double precision to avoid truncation
            hf.create_dataset(joined_path, curr_data_shape, maxshape=max_shape, dtype=np.float64)

        hf.close()
        # Now open in r+ mode to append to the file
        self.hf = h5py.File(self.log_filepath, "r+")
        self.hf.attrs["/metadata/start_time"] = str(datetime.datetime.now())
        self.hf.attrs["/metadata/physics_timestep"] = self.sim.physics_timestep
        self.hf.attrs["/metadata/render_timestep"] = self.sim.render_timestep
        self.hf.attrs["/metadata/git_info"] = dump_config(project_git_info())

        if self.task:
            self.hf.attrs["/metadata/atus_activity"] = self.task.behavior_activity
            self.hf.attrs["/metadata/filter_objects"] = self.filter_objects
            self.hf.attrs["/metadata/activity_definition"] = self.task.activity_definition
            self.hf.attrs["/metadata/scene_id"] = self.task.scene.scene_id
            self.hf.attrs["/metadata/obj_body_id_to_name"] = self.obj_body_id_to_name_str
            self.hf.attrs["/metadata/urdf_file"] = self.task.scene.fname

        # VR config YML is stored as a string in metadata
        if self.store_vr:
            self.hf.attrs["/metadata/vr_settings"] = self.sim.vr_settings.dump_vr_settings()

    def get_data_for_name_path(self, name_path):
        """Resolves a list of names (group/dataset) into a numpy array.
        eg. [vr, vr_camera, right_eye_view] -> self.data_map['vr']['vr_camera']['right_eye_view']"""
        next_data = self.data_map
        for name in name_path:
            next_data = next_data[name]

        return next_data

    def save_action(self, action_path, action):
        """Saves a single action to the VRLogWriter. It is assumed that this function will
        be called every frame, including the first.

        Args:
            action_path: The /-separated action path that was used to register this action
            action: The action as a numpy array - must have the same shape as the action_shape that
                was registered along with this action path
        """
        full_action_path = "action/" + action_path
        act_data = self.get_data_for_name_path(full_action_path.split("/"))
        act_data[self.frame_counter, ...] = action

    def write_frame_data_to_map(self):
        """Writes frame data to the data map.

        Args:
            s (simulator): used to extract information about VR system
        """
        frame_data = np.array(
            [
                self.persistent_frame_count,
                self.sim.last_physics_timestep,
                self.sim.last_render_timestep,
                self.sim.last_frame_dur,
            ]
        )

        self.data_map["frame_data"][self.frame_counter, ...] = frame_data[:]

    def write_vr_data_to_map(self):
        """Writes all VR data to map. This will write data
        that the user has not even processed in their demos.
        For example, we will store eye tracking data if it is
        valid, even if they do not explicitly use that data
        in their code. This will help us store all the necessary
        data without remembering to call the simulator's data
        extraction functions every time we want to save data.

        Args:
            s (simulator): used to extract information about VR system
        """
        # At end of each frame, renderer has camera information for VR right eye
        self.data_map["vr"]["vr_camera"]["right_eye_view"][self.frame_counter, ...] = self.sim.renderer.V
        self.data_map["vr"]["vr_camera"]["right_eye_proj"][self.frame_counter, ...] = self.sim.renderer.P
        self.data_map["vr"]["vr_camera"]["right_camera_pos"][self.frame_counter, ...] = self.sim.renderer.camera

        if self.vr_robot:
            forces = {
                "left_controller": p.getConstraintState(self.vr_robot._parts["left_hand"].movement_cid),
                "right_controller": p.getConstraintState(self.vr_robot._parts["right_hand"].movement_cid),
            }
        for device in ["hmd", "left_controller", "right_controller"]:
            is_valid, trans, rot = self.sim.get_data_for_vr_device(device)
            right, up, forward = self.sim.get_device_coordinate_system(device)
            if is_valid is not None:
                data_list = [is_valid]
                data_list.extend(list(trans))
                data_list.extend(list(rot))
                data_list.extend(list(right))
                data_list.extend(list(up))
                data_list.extend(list(forward))
                # Add final model rotation for controllers
                if device == "left_controller" or device == "right_controller":
                    if not self.vr_robot:
                        # Store identity quaternion if no agent used
                        data_list.extend([0, 0, 0, 1])
                    else:
                        # Calculate model rotation and store
                        if device == "left_controller":
                            base_rot = HAND_BASE_ROTS["left_hand"]
                        else:
                            base_rot = HAND_BASE_ROTS["right_hand"]
                        controller_rot = rot
                        # Use dummy translation to calculation final rotation
                        final_rot = p.multiplyTransforms([0, 0, 0], controller_rot, [0, 0, 0], base_rot)[1]
                        data_list.extend(final_rot)
                if self.vr_robot and device in forces:
                    data_list.extend(list(forces[device]))
                self.data_map["vr"]["vr_device_data"][device][self.frame_counter, ...] = np.array(data_list)

            if device == "left_controller" or device == "right_controller":
                button_data_list = self.sim.get_button_data_for_controller(device)
                self.data_map["vr"]["vr_button_data"][device][self.frame_counter, ...] = np.array(button_data_list)

        is_valid, torso_trans, torso_rot = self.sim.get_data_for_vr_tracker(self.sim.vr_settings.torso_tracker_serial)
        torso_data_list = [is_valid]
        torso_data_list.extend(list(torso_trans))
        torso_data_list.extend(list(torso_rot))
        self.data_map["vr"]["vr_device_data"]["torso_tracker"][self.frame_counter, ...] = np.array(torso_data_list)

        vr_pos_data = []
        vr_pos_data.extend(list(self.sim.get_vr_pos()))
        vr_pos_data.extend(list(self.sim.get_vr_offset()))
        if self.vr_robot:
            vr_pos_data.extend(p.getConstraintState(self.vr_robot._parts["body"].movement_cid))
        self.data_map["vr"]["vr_device_data"]["vr_position_data"][self.frame_counter, ...] = np.array(vr_pos_data)

        # On systems where eye tracking is not supported, we get dummy data and a guaranteed False validity reading
        is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = self.sim.get_eye_tracking_data()
        if is_valid:
            eye_data_list = [is_valid]
            eye_data_list.extend(origin)
            eye_data_list.extend(dir)
            eye_data_list.append(left_pupil_diameter)
            eye_data_list.append(right_pupil_diameter)
            self.data_map["vr"]["vr_eye_tracking_data"][self.frame_counter, ...] = np.array(eye_data_list)

        controller_events = {"left_controller": [], "right_controller": []}
        for device_id, button_idx, press_id in self.sim.get_vr_events():
            device_name = "left_controller" if device_id == 0 else "right_controller"
            controller_events[device_name].append((button_idx, press_id))
        for controller in controller_events.keys():
            bin_button_data = convert_button_data_to_binary(controller_events[controller])
            self.data_map["vr"]["vr_event_data"][controller][self.frame_counter, ...] = np.array(bin_button_data)

        reset_actions = []
        for controller in controller_events.keys():
            reset_actions.append(self.sim.query_vr_event(controller, "reset_agent"))
        self.data_map["vr"]["vr_event_data"]["reset_actions"][self.frame_counter, ...] = np.array(reset_actions)

    def write_pybullet_data_to_map(self):
        """Write all pybullet data to the class' internal map."""
        if self.task and self.filter_objects:
            for obj_bid, obj in self.tracked_objects.items():
                obj_name = str(obj_bid)
                # TODO: currently we must hack around storing object pose for multiplexed objects
                try:
                    pos, orn = obj.get_position_orientation()
                except ValueError as e:
                    pos, orn = obj.objects[0].get_position_orientation()

                handle = self.data_map["physics_data"][obj_name]
                handle["position"][self.frame_counter] = pos
                handle["orientation"][self.frame_counter] = orn
                handle["joint_state"][self.frame_counter] = np.array(
                    [p.getJointState(obj_bid, n)[0] for n in range(self.joint_map[obj_bid])]
                )
        else:
            for bid in self.tracked_objects:
                obj_name = str(bid)
                pos, orn = p.getBasePositionAndOrientation(bid)
                handle = self.data_map["physics_data"][obj_name]
                handle["position"][self.frame_counter] = pos
                handle["orientation"][self.frame_counter] = orn
                handle["joint_state"][self.frame_counter] = np.array(
                    [p.getJointState(bid, n)[0] for n in range(self.joint_map[bid])]
                )

    def _print_pybullet_data(self):
        """Print pybullet debug data - hidden API since this is used for debugging purposes only."""
        print("----- PyBullet data at the end of frame {} -----".format(self.persistent_frame_count))
        if self.task and self.filter_objects:
            for obj_bid, obj in self.tracked_objects.items():
                pos, orn = obj.get_position_orientation()
                print("{} - pos: {} and orn: {}".format(obj_bid, pos, orn))
        else:
            for bid in self.tracked_objects:
                pos, orn = p.getBasePositionAndOrientation(bid)
                print("{} - pos: {} and orn: {}".format(bid, pos, orn))

    @staticmethod
    def one_hot_encoding(hits, categories):
        if len(hits) > 0:
            hits = np.array(hits)
            one_hot = np.eye(categories)[hits]
            return np.sum(one_hot, axis=0)
        else:
            return np.zeros(categories)

    def write_predicate_data_to_map(self):
        satisfied = self.task.current_goal_status["satisfied"]
        unsatisfied = self.task.current_goal_status["unsatisfied"]

        self.data_map["goal_status"]["unsatisfied"][self.frame_counter] = self.one_hot_encoding(
            unsatisfied, self.total_goals
        )
        self.data_map["goal_status"]["satisfied"][self.frame_counter] = self.one_hot_encoding(
            satisfied, self.total_goals
        )

    def write_agent_data_to_map(self):
        self.data_map["agent_actions"]["vr_robot"][self.frame_counter] = self.vr_robot.dump_action()

    # TIMELINE: Call this at the end of each frame (eg. at end of while loop)
    def process_frame(self):
        """Asks the VRLogger to process frame data. This includes:
        -- updating pybullet data
        -- incrementing frame counter by 1
        """
        if self.store_vr:
            self.write_frame_data_to_map()
            self.write_vr_data_to_map()
        self.write_pybullet_data_to_map()
        if self.task:
            self.write_predicate_data_to_map()
        if self.vr_robot:
            self.write_agent_data_to_map()
        self.frame_counter += 1
        self.persistent_frame_count += 1
        if self.frame_counter >= self.frames_before_write:
            # We have accumulated enough data, which we will write to hdf5
            self.write_to_hdf5()

    def refresh_data_map(self):
        """Resets all values stored in self.data_map to the default sentinel value.
        This function is called after we have written the last self.frames_before_write
        frames to HDF5 and can start inputting new frame data into the data map."""
        for name_path in self.name_path_data:
            np_data = self.get_data_for_name_path(name_path)
            np_data.fill(self.default_fill_sentinel)

    def write_to_hdf5(self):
        """Writes data stored in self.data_map to hdf5.
        The data is saved each time this function is called, so data
        will be saved even if a Ctrl+C event interrupts the program."""
        if self.log_status:
            print("----- Writing log data to hdf5 on frame: {0} -----".format(self.persistent_frame_count))

        start_time = time.time()

        frames_to_write = self.persistent_frame_count - self.hf["frame_data"].shape[0]
        if frames_to_write > 0:
            for name_path in self.name_path_data:
                curr_dset = self.hf["/".join(name_path)]
                # Resize to accommodate new data
                curr_dset.resize(curr_dset.shape[0] + frames_to_write, axis=0)
                # Set the last frames_to_write rows to numpy data from data map
                curr_dset[-frames_to_write:, ...] = self.get_data_for_name_path(name_path)[:frames_to_write, ...]

            self.refresh_data_map()
            self.frame_counter = 0

        delta = time.time() - start_time
        if self.profiling_mode:
            print("Time to write: {0}".format(delta))

    def end_log_session(self):
        """Closes hdf5 log file at end of logging session."""
        # Write the remaining data to hdf
        self.write_to_hdf5()
        if self.log_status:
            print("IG LOGGER INFO: Ending log writing session after {} frames".format(self.persistent_frame_count))
        self.hf.close()


class IGLogReader(object):
    def __init__(self, log_filepath, log_status=True):
        """
        :param log_filepath: path for logging files to be read from
        :param log_status: whether to print status updates to the command line
        """
        self.log_filepath = log_filepath
        self.log_status = log_status
        # Frame counter keeping track of how many frames have been reproduced
        self.frame_counter = -1
        self.hf = h5py.File(self.log_filepath, "r")
        self.pb_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        # Get total frame num (dataset row length) from an arbitary dataset
        self.total_frame_num = self.hf["frame_data"].shape[0]
        # Placeholder VrData object, which will be filled every frame if we are performing action replay
        self.vr_data = VrData()
        if self.log_status:
            print("----- IGLogReader initialized -----")
            print("Preparing to read {0} frames".format(self.total_frame_num))

    @staticmethod
    def get_obj_body_id_to_name(vr_log_path):
        f = h5py.File(vr_log_path, "r")
        return parse_str_config(f.attrs["/metadata/obj_body_id_to_name"])

    @staticmethod
    def has_metadata_attr(vr_log_path, attr_name):
        """
        Checks whether a given HDF5 log has a metadata attribute.
        """
        f = h5py.File(vr_log_path, "r")
        return attr_name in f.attrs

    @staticmethod
    def get_all_metadata_attrs(vr_log_path):
        """
        Returns a list of available metadata attributes
        """
        f = h5py.File(vr_log_path, "r")
        return f.attrs

    @staticmethod
    def read_metadata_attr(vr_log_path, attr_name):
        """
        Reads a metadata attribute from a given HDF5 log path.
        """
        f = h5py.File(vr_log_path, "r")
        if attr_name in f.attrs:
            return f.attrs[attr_name]
        else:
            return None

    def _print_pybullet_data(self):
        """Print pybullet debug data - hidden API since this is used for debugging purposes only."""
        print("----- PyBullet data at the end of frame {} -----".format(self.frame_counter))
        for pb_id in self.pb_ids:
            pos, orn = p.getBasePositionAndOrientation(pb_id)
            print("{} - pos: {} and orn: {}".format(pb_id, pos, orn))

    def set_replay_camera(self, sim):
        """Sets camera based on saved camera matrices. Only valid if VR was used to save a demo.
        :param sim: Simulator object
        """
        sim.renderer.V = self.hf["vr/vr_camera/right_eye_view"][self.frame_counter]
        sim.renderer.P = self.hf["vr/vr_camera/right_eye_proj"][self.frame_counter]
        right_cam_pos = self.hf["vr/vr_camera/right_camera_pos"][self.frame_counter]
        sim.renderer.camera = right_cam_pos
        sim.renderer.set_light_position_direction(
            [right_cam_pos[0], right_cam_pos[1], 10], [right_cam_pos[0], right_cam_pos[1], 0]
        )

    def get_vr_data(self):
        """
        Returns VR for the current frame as a VrData object. This can be indexed
        into to analyze individual values, or can be passed into the BehaviorRobot to drive
        its actions for a single frame.
        """
        # Update VrData with new HF data
        self.vr_data.refresh_action_replay_data(self.hf, self.frame_counter)
        return self.vr_data

    def get_agent_action(self, agent_name):
        """
        Gets action for agent with a specific name.
        """
        agent_action_path = "agent_actions/{}".format(agent_name)
        if agent_action_path not in self.hf:
            raise RuntimeError("Unable to find agent action path: {} in saved HDF5 file".format(agent_action_path))
        return self.hf[agent_action_path][self.frame_counter]

    def read_value(self, value_path):
        """Reads any saved value at value_path for the current frame.

        Args:
            value_path: /-separated string representing the value to fetch. This should be one of the
            values list in the comment at the top of this file.
            Eg. vr/vr_button_data/right_controller
        """
        return self.hf[value_path][self.frame_counter]

    def read_action(self, action_path):
        """Reads the action at action_path for the current frame.

        Args:
            action_path: /-separated string representing the action to fetch. This should match
                an action that was previously registered with the VRLogWriter during data saving
        """
        full_action_path = "action/" + action_path
        return self.hf[full_action_path][self.frame_counter]

    def get_data_left_to_read(self):
        """Returns whether there is still data left to read."""
        self.frame_counter += 1
        if self.frame_counter >= self.total_frame_num:
            return False
        else:
            return True

    def end_log_session(self):
        """Call this once reading has finished to clean up resources used."""
        self.hf.close()

        if self.log_status:
            print("Ending frame reading session after reading {0} frames".format(self.total_frame_num))
            print("----- IGLogReader shutdown -----")
