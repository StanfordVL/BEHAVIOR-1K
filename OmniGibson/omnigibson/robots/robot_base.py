import os
from copy import deepcopy

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.objects.usd_object import USDObject
from omnigibson.sensors import (
    ALL_SENSOR_MODALITIES,
    SENSOR_PRIMS_TO_SENSOR_CLS,
    ScanSensor,
    VisionSensor,
    create_sensor,
)
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.constants import PrimType
from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty, merge_nested_dicts
from omnigibson.utils.usd_utils import ControllableObjectViewAPI, absolute_prim_path_to_scene_relative
from omnigibson.utils.vision_utils import segmentation_to_rgb, change_pcd_frame

# Global dicts that will contain mappings
REGISTERED_ROBOTS = dict()

# Add proprio sensor modality to ALL_SENSOR_MODALITIES
ALL_SENSOR_MODALITIES.add("proprio")

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Name of the category to assign to all robots
m.ROBOT_CATEGORY = "agent"


class BaseRobot(USDObject, ControllableObject, GymObservable):
    """
    Base class for USD-based robot agents.

    This class handles object loading, and provides method interfaces that should be
    implemented by subclassed robots.
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
        load_config=None,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                we will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is ["rgb", "proprio"].
                Valid options are "all", or a list containing any subset of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            include_sensor_names (None or list of str): If specified, substring(s) to check for in all raw sensor prim
                paths found on the robot. A sensor must include one of the specified substrings in order to be included
                in this robot's set of sensors
            exclude_sensor_names (None or list of str): If specified, substring(s) to check against in all raw sensor
                prim paths found on the robot. A sensor must not include any of the specified substrings in order to
                be included in this robot's set of sensors
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store inputs
        self._obs_modalities = (
            obs_modalities
            if obs_modalities == "all"
            else {obs_modalities}
            if isinstance(obs_modalities, str)
            else set(obs_modalities)
        )  # this will get updated later when we fill in our sensors
        self._proprio_obs = self.default_proprio_obs if proprio_obs == "default" else list(proprio_obs)
        self._sensor_config = sensor_config

        # Process abilities
        robot_abilities = {"robot": {}}
        abilities = robot_abilities if abilities is None else robot_abilities.update(abilities)

        # Initialize internal attributes that will be loaded later
        self._include_sensor_names = None if include_sensor_names is None else set(include_sensor_names)
        self._exclude_sensor_names = None if exclude_sensor_names is None else set(exclude_sensor_names)
        self._sensors = None  # e.g.: scan sensor, vision sensor

        # All BaseRobots should have xform properties pre-loaded
        load_config = {} if load_config is None else load_config
        load_config["xform_props_pre_loaded"] = True

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            usd_path=self.usd_path,
            name=name,
            category=m.ROBOT_CATEGORY,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=PrimType.RIGID,
            include_default_states=True,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            **kwargs,
        )

        assert not isinstance(self._load_config["scale"], th.Tensor) or th.all(
            self._load_config["scale"] == self._load_config["scale"][0]
        ), f"Robot scale must be uniform! Got: {self._load_config['scale']}"

    def _post_load(self):
        # Run super post load first
        super()._post_load()

        # Load the sensors
        self._load_sensors()

    def _initialize(self):
        # Run super
        super()._initialize()

        # Initialize all sensors
        for sensor in self._sensors.values():
            sensor.initialize()

        # Load the observation space for this robot
        self.load_observation_space()

        # Validate this robot configuration
        self._validate_configuration()

        self._reset_joint_pos_aabb_extent = self.aabb_extent

    def _load_sensors(self):
        """
        Loads sensor(s) to retrieve observations from this object.
        Stores created sensors as dictionary mapping sensor names to specific sensor
        instances used by this object.
        """
        # Populate sensor config
        self._sensor_config = self._generate_sensor_config(custom_config=self._sensor_config)

        # Search for any sensors this robot might have attached to any of its links
        self._sensors = dict()
        obs_modalities = set()
        for link_name, link in self._links.items():
            # Search through all children prims and see if we find any sensor
            sensor_counts = {p: 0 for p in SENSOR_PRIMS_TO_SENSOR_CLS.keys()}
            for prim in link.prim.GetChildren():
                prim_type = prim.GetPrimTypeInfo().GetTypeName()
                if prim_type in SENSOR_PRIMS_TO_SENSOR_CLS:
                    # Possibly filter out the sensor based on name
                    prim_path = str(prim.GetPrimPath())
                    not_blacklisted = self._exclude_sensor_names is None or not any(
                        name in prim_path for name in self._exclude_sensor_names
                    )
                    whitelisted = self._include_sensor_names is None or any(
                        name in prim_path for name in self._include_sensor_names
                    )
                    # Also make sure that the include / exclude sensor names are mutually exclusive
                    if self._exclude_sensor_names is not None and self._include_sensor_names is not None:
                        assert (
                            len(set(self._exclude_sensor_names).intersection(set(self._include_sensor_names))) == 0
                        ), (
                            f"include_sensor_names and exclude_sensor_names must be mutually exclusive! "
                            f"Got: {self._include_sensor_names} and {self._exclude_sensor_names}"
                        )
                    if not (not_blacklisted and whitelisted):
                        continue

                    # Infer what obs modalities to use for this sensor
                    sensor_cls = SENSOR_PRIMS_TO_SENSOR_CLS[prim_type]
                    sensor_kwargs = self._sensor_config[sensor_cls.__name__]
                    if "modalities" not in sensor_kwargs:
                        sensor_kwargs["modalities"] = (
                            sensor_cls.all_modalities
                            if self._obs_modalities == "all"
                            else sensor_cls.all_modalities.intersection(self._obs_modalities)
                        )
                    # If the modalities list is empty, don't import the sensor.
                    if not sensor_kwargs["modalities"]:
                        continue

                    obs_modalities = obs_modalities.union(sensor_kwargs["modalities"])
                    # Create the sensor and store it internally
                    sensor = create_sensor(
                        sensor_type=prim_type,
                        relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, prim_path),
                        name=f"{self.name}:{link_name}:{prim_type}:{sensor_counts[prim_type]}",
                        **sensor_kwargs,
                    )
                    sensor.load(self.scene)
                    self._sensors[sensor.name] = sensor
                    sensor_counts[prim_type] += 1

        # Since proprioception isn't an actual sensor, we need to possibly manually add it here as well
        if self._obs_modalities == "all" or "proprio" in self._obs_modalities:
            obs_modalities.add("proprio")

        # Update our overall obs modalities
        self._obs_modalities = obs_modalities

    def _generate_sensor_config(self, custom_config=None):
        """
        Generates a fully-populated sensor config, overriding any default values with the corresponding values
        specified in @custom_config

        Args:
            custom_config (None or Dict[str, ...]): nested dictionary mapping sensor class name(s) to specific custom
                sensor configurations for this object. This will override any default values specified by this class

        Returns:
            dict: Fully-populated nested dictionary mapping sensor class name(s) to specific sensor configurations
                for this object
        """
        sensor_config = {} if custom_config is None else deepcopy(custom_config)

        # Merge the sensor dictionaries
        sensor_config = merge_nested_dicts(
            base_dict=self._default_sensor_config,
            extra_dict=sensor_config,
        )

        return sensor_config

    def _validate_configuration(self):
        """
        Run any needed sanity checks to make sure this robot was created correctly.
        """
        pass

    def get_obs(self):
        """
        Grabs all observations from the robot. This is keyword-mapped based on each observation modality
            (e.g.: proprio, rgb, etc.)

        Returns:
            2-tuple:
                dict: Keyword-mapped dictionary mapping observation modality names to
                    observations (usually np arrays)
                dict: Keyword-mapped dictionary mapping observation modality names to
                    additional info
        """
        # Our sensors already know what observation modalities it has, so we simply iterate over all of them
        # and grab their observations, processing them into a flat dict
        obs_dict = dict()
        info_dict = dict()
        for sensor_name, sensor in self._sensors.items():
            obs_dict[sensor_name], info_dict[sensor_name] = sensor.get_obs()
            for key in obs_dict[sensor_name]:
                if "pointcloud" in key:
                    # convert point cloud from world frame to robot base frame
                    obs_dict[sensor_name][key] = change_pcd_frame(
                        pcd=obs_dict[sensor_name][key],
                        rel_pose=th.cat(self.get_position_orientation()),
                    )

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_dict["proprio"], info_dict["proprio"] = self.get_proprioception()

        return obs_dict, info_dict

    def get_proprioception(self):
        """
        Returns:
            n-array: numpy array of all robot-specific proprioceptive observations.
            dict: empty dictionary, a placeholder for additional info
        """
        proprio_dict = self._get_proprioception_dict()
        return th.cat([proprio_dict[obs] for obs in self._proprio_obs]), {}

    def _get_proprioception_dict(self):
        """
        Returns:
            dict: keyword-mapped proprioception observations available for this robot.
                Can be extended by subclasses
        """
        joint_positions = cb.to_torch(
            cb.copy(ControllableObjectViewAPI.get_joint_positions(self.articulation_root_path))
        )
        joint_velocities = cb.to_torch(
            cb.copy(ControllableObjectViewAPI.get_joint_velocities(self.articulation_root_path))
        )
        joint_efforts = cb.to_torch(cb.copy(ControllableObjectViewAPI.get_joint_efforts(self.articulation_root_path)))
        pos, quat = ControllableObjectViewAPI.get_position_orientation(self.articulation_root_path)
        pos, quat = cb.to_torch(cb.copy(pos)), cb.to_torch(cb.copy(quat))
        ori = T.quat2euler(quat)

        ori_2d = T.z_angle_from_quat(quat).unsqueeze(0)  # Convert to 1D tensor

        # Pack everything together
        return dict(
            joint_qpos=joint_positions,
            joint_qpos_sin=th.sin(joint_positions),
            joint_qpos_cos=th.cos(joint_positions),
            joint_qvel=joint_velocities,
            joint_qeffort=joint_efforts,
            robot_pos=pos,
            robot_ori_cos=th.cos(ori),
            robot_ori_sin=th.sin(ori),
            robot_2d_ori=ori_2d,
            robot_2d_ori_cos=th.cos(ori_2d),
            robot_2d_ori_sin=th.sin(ori_2d),
            robot_lin_vel=cb.to_torch(
                cb.copy(ControllableObjectViewAPI.get_linear_velocity(self.articulation_root_path))
            ),
            robot_ang_vel=cb.to_torch(
                cb.copy(ControllableObjectViewAPI.get_angular_velocity(self.articulation_root_path))
            ),
        )

    def _load_observation_space(self):
        # We compile observation spaces from our sensors
        obs_space = dict()

        for sensor_name, sensor in self._sensors.items():
            # Load the sensor observation space
            obs_space[sensor_name] = sensor.load_observation_space()

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_space["proprio"] = self._build_obs_box_space(
                shape=(self.proprioception_dim,), low=-float("inf"), high=float("inf"), dtype=NumpyTypes.FLOAT32
            )

        return obs_space

    def add_obs_modality(self, modality):
        """
        Adds observation modality @modality to this robot. Note: Should be one of omnigibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to add to this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we add it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.add_modality(modality=modality)

    def remove_obs_modality(self, modality):
        """
        Remove observation modality @modality from this robot. Note: Should be one of
        omnigibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to remove from this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we remove it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.remove_modality(modality=modality)

    def visualize_sensors(self):
        """
        Renders this robot's key sensors, visualizing them via matplotlib plots
        """
        frames = dict()
        remaining_obs_modalities = deepcopy(self.obs_modalities)
        for sensor in self.sensors.values():
            obs, _ = sensor.get_obs()
            sensor_frames = []
            if isinstance(sensor, VisionSensor):
                # We check for rgb, depth, normal, seg_instance
                for modality in ["rgb", "depth", "normal", "seg_instance"]:
                    if modality in sensor.modalities:
                        ob = obs[modality]
                        if modality == "rgb":
                            # Ignore alpha channel, map to floats
                            ob = ob[:, :, :3] / 255.0
                        elif modality == "seg_instance":
                            # Map IDs to rgb
                            ob = segmentation_to_rgb(ob, N=256) / 255.0
                        elif modality == "normal":
                            # Re-map to 0 - 1 range
                            ob = (ob + 1.0) / 2.0
                        else:
                            # Depth, nothing to do here
                            pass
                        # Add this observation to our frames and remove the modality
                        sensor_frames.append((modality, ob))
                        remaining_obs_modalities -= {modality}
                    else:
                        # Warn user that we didn't find this modality
                        print(f"Modality {modality} is not active in sensor {sensor.name}, skipping...")
            elif isinstance(sensor, ScanSensor):
                # We check for occupancy_grid
                occupancy_grid = obs.get("occupancy_grid", None)
                if occupancy_grid is not None:
                    sensor_frames.append(("occupancy_grid", occupancy_grid))
                    remaining_obs_modalities -= {"occupancy_grid"}

            # Map the sensor name to the frames for that sensor
            frames[sensor.name] = sensor_frames

        # Warn user that any remaining modalities are not able to be visualized
        if len(remaining_obs_modalities) > 0:
            print(f"Modalities: {remaining_obs_modalities} cannot be visualized, skipping...")

        # Write all the frames to a plot
        import matplotlib.pyplot as plt

        for sensor_name, sensor_frames in frames.items():
            n_sensor_frames = len(sensor_frames)
            if n_sensor_frames > 0:
                fig, axes = plt.subplots(nrows=1, ncols=n_sensor_frames)
                if n_sensor_frames == 1:
                    axes = [axes]
                # Dump frames and set each subtitle
                for i, (modality, frame) in enumerate(sensor_frames):
                    axes[i].imshow(frame)
                    axes[i].set_title(modality)
                    axes[i].set_axis_off()
                # Set title
                fig.suptitle(sensor_name)
                plt.show(block=False)

        # One final plot show so all the figures get rendered
        plt.show()

    def remove(self):
        """
        Do NOT call this function directly to remove a prim - call og.sim.remove_prim(prim) for proper cleanup
        """
        # Remove all sensors
        for sensor in self._sensors.values():
            sensor.remove()

        # Run super
        super().remove()

    @property
    def reset_joint_pos_aabb_extent(self):
        """
        This is the aabb extent of the robot in the robot frame after resetting the joints.
        Returns:
            3-array: Axis-aligned bounding box extent of the robot base
        """
        return self._reset_joint_pos_aabb_extent

    def teleop_data_to_action(self, teleop_action) -> th.Tensor:
        """
        Generate action data from teleoperation action data
        Args:
            teleop_action (TeleopAction): teleoperation action data
        Returns:
            th.tensor: array of action data filled with update value
        """
        return th.zeros(self.action_dim)

    @property
    def sensors(self):
        """
        Returns:
            dict: Keyword-mapped dictionary mapping sensor names to BaseSensor instances owned by this robot
        """
        return self._sensors

    @property
    def obs_modalities(self):
        """
        Returns:
            set of str: Observation modalities used for this robot (e.g.: proprio, rgb, etc.)
        """
        assert self._loaded, "Cannot check observation modalities until we load this robot!"
        return self._obs_modalities

    @property
    def proprioception_dim(self):
        """
        Returns:
            int: Size of self.get_proprioception() vector
        """
        return len(self.get_proprioception()[0])

    @property
    def _default_sensor_config(self):
        """
        Returns:
            dict: default nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. See kwargs from omnigibson/sensors/__init__/create_sensor for more
                details

                Expected structure is as follows:
                    SensorClassName1:
                        modalities: ...
                        enabled: ...
                        noise_type: ...
                        noise_kwargs:
                            ...
                        sensor_kwargs:
                            ...
                    SensorClassName2:
                        modalities: ...
                        enabled: ...
                        noise_type: ...
                        noise_kwargs:
                            ...
                        sensor_kwargs:
                            ...
                    ...
        """
        return {
            "VisionSensor": {
                "enabled": True,
                "noise_type": None,
                "noise_kwargs": None,
                "sensor_kwargs": {
                    "image_height": 128,
                    "image_width": 128,
                },
            },
            "ScanSensor": {
                "enabled": True,
                "noise_type": None,
                "noise_kwargs": None,
                "sensor_kwargs": {
                    # Basic LIDAR kwargs
                    "min_range": 0.05,
                    "max_range": 10.0,
                    "horizontal_fov": 360.0,
                    "vertical_fov": 1.0,
                    "yaw_offset": 0.0,
                    "horizontal_resolution": 1.0,
                    "vertical_resolution": 1.0,
                    "rotation_rate": 0.0,
                    "draw_points": False,
                    "draw_lines": False,
                    # Occupancy Grid kwargs
                    "occupancy_grid_resolution": 128,
                    "occupancy_grid_range": 5.0,
                    "occupancy_grid_inner_radius": 0.5,
                    "occupancy_grid_local_link": None,
                },
            },
        }

    @property
    def default_proprio_obs(self):
        """
        Returns:
            list of str: Default proprioception observations to use
        """
        return []

    @property
    def model_name(self):
        """
        Returns:
            str: name of this robot model. usually corresponds to the class name of a given robot model
        """
        return self.__class__.__name__

    @property
    def usd_path(self):
        # By default, sets the standardized path
        model = self.model_name.lower()
        return os.path.join(get_dataset_path("omnigibson-robot-assets"), f"models/{model}/usd/{model}.usda")

    @property
    def urdf_path(self):
        """
        Returns:
            str: file path to the robot urdf file.
        """
        # By default, sets the standardized path
        model = self.model_name.lower()
        return os.path.join(get_dataset_path("omnigibson-robot-assets"), f"models/{model}/urdf/{model}.urdf")

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseRobot")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global robot registry -- override super registry
        global REGISTERED_ROBOTS
        return REGISTERED_ROBOTS
