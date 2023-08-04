from abc import abstractmethod
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from omnigibson.macros import gm, create_module_macros
from omnigibson.sensors import create_sensor, SENSOR_PRIMS_TO_SENSOR_CLS, ALL_SENSOR_MODALITIES, VisionSensor, ScanSensor
from omnigibson.objects.usd_object import USDObject
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.vision_utils import segmentation_to_rgb
from omnigibson.utils.constants import PrimType
from pxr import PhysxSchema

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
        prim_path=None,
        class_id=None,
        uuid=None,
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        load_config=None,

        # Unique to USDObject hierarchy
        abilities=None,

        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,

        # Unique to this class
        obs_modalities="all",
        proprio_obs="default",

        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            class_id (None or int): What class ID the object should be assigned in semantic segmentation rendering mode.
                If None, the ID will be inferred from this object's category.
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self.default_joint_pos will be used instead.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
                corresponds to all modalities being used.
                Otherwise, valid options should be part of omnigibson.sensors.ALL_SENSOR_MODALITIES.
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store inputs
        self._obs_modalities = obs_modalities if obs_modalities == "all" else \
            {obs_modalities} if isinstance(obs_modalities, str) else set(obs_modalities)              # this will get updated later when we fill in our sensors
        self._proprio_obs = self.default_proprio_obs if proprio_obs == "default" else list(proprio_obs)

        # Process abilities
        robot_abilities = {"robot": {}}
        abilities = robot_abilities if abilities is None else robot_abilities.update(abilities)

        # Initialize internal attributes that will be loaded later
        self._sensors = None                     # e.g.: scan sensor, vision sensor

        # If specified, make sure scale is uniform -- this is because non-uniform scale can result in non-matching
        # collision representations for parts of the robot that were optimized (e.g.: bounding sphere for wheels)
        assert scale is None or isinstance(scale, int) or isinstance(scale, float) or np.all(scale == scale[0]), \
            f"Robot scale must be uniform! Got: {scale}"

        # Run super init
        super().__init__(
            prim_path=prim_path,
            usd_path=self.usd_path,
            name=name,
            category=m.ROBOT_CATEGORY,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=PrimType.RIGID,
            include_default_states=True,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            **kwargs,
        )

    def _post_load(self):
        # Run super post load first
        super()._post_load()

        # Search for any sensors this robot might have attached to any of its links
        self._sensors = dict()
        obs_modalities = set()
        for link_name, link in self._links.items():
            # Search through all children prims and see if we find any sensor
            for prim in link.prim.GetChildren():
                prim_type = prim.GetPrimTypeInfo().GetTypeName()
                if prim_type in SENSOR_PRIMS_TO_SENSOR_CLS:
                    # Infer what obs modalities to use for this sensor
                    sensor_cls = SENSOR_PRIMS_TO_SENSOR_CLS[prim_type]
                    modalities = sensor_cls.all_modalities if self._obs_modalities == "all" else \
                        sensor_cls.all_modalities.intersection(self._obs_modalities)
                    obs_modalities = obs_modalities.union(modalities)
                    # Create the sensor and store it internally
                    sensor = create_sensor(
                        sensor_type=prim_type,
                        prim_path=str(prim.GetPrimPath()),
                        name=f"{self.name}:{link_name}_{prim_type}_sensor",
                        modalities=modalities,
                    )
                    self._sensors[sensor.name] = sensor

        # Since proprioception isn't an actual sensor, we need to possibly manually add it here as well
        if self._obs_modalities == "all" or "proprio" in self._obs_modalities:
            obs_modalities.add("proprio")

        # Update our overall obs modalities
        self._obs_modalities = obs_modalities

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Initialize all sensors
        for sensor in self._sensors.values():
            sensor.initialize()

        # Load the observation space for this robot
        self.load_observation_space()

        # Validate this robot configuration
        self._validate_configuration()

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
            dict: Keyword-mapped dictionary mapping observation modality names to
                observations (usually np arrays)
        """
        # Our sensors already know what observation modalities it has, so we simply iterate over all of them
        # and grab their observations, processing them into a flat dict
        obs_dict = dict()
        for sensor_name, sensor in self._sensors.items():
            sensor_obs = sensor.get_obs()
            for obs_modality, obs in sensor_obs.items():
                obs_dict[f"{sensor_name}_{obs_modality}"] = obs

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_dict["proprio"] = self.get_proprioception()

        return obs_dict

    def get_proprioception(self):
        """
        Returns:
            n-array: numpy array of all robot-specific proprioceptive observations.
        """
        proprio_dict = self._get_proprioception_dict()
        return np.concatenate([proprio_dict[obs] for obs in self._proprio_obs])

    def _get_proprioception_dict(self):
        """
        Returns:
            dict: keyword-mapped proprioception observations available for this robot.
                Can be extended by subclasses
        """
        joint_positions = self.get_joint_positions(normalized=False)
        joint_velocities = self.get_joint_velocities(normalized=False)
        joint_efforts = self.get_joint_efforts(normalized=False)
        pos, ori = self.get_position(), self.get_rpy()
        return dict(
            joint_qpos=joint_positions,
            joint_qpos_sin=np.sin(joint_positions),
            joint_qpos_cos=np.cos(joint_positions),
            joint_qvel=joint_velocities,
            joint_qeffort=joint_efforts,
            robot_pos=pos,
            robot_ori_cos=np.cos(ori),
            robot_ori_sin=np.sin(ori),
            robot_lin_vel=self.get_linear_velocity(),
            robot_ang_vel=self.get_angular_velocity(),
        )

    def _load_observation_space(self):
        # We compile observation spaces from our sensors
        obs_space = dict()

        for sensor_name, sensor in self._sensors.items():
            # Load the sensor observation space
            sensor_obs_space = sensor.load_observation_space()
            for obs_modality, obs_modality_space in sensor_obs_space.items():
                obs_space[f"{sensor_name}_{obs_modality}"] = obs_modality_space

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_space["proprio"] = self._build_obs_box_space(shape=(self.proprioception_dim,), low=-np.inf, high=np.inf, dtype=np.float64)

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
            obs = sensor.get_obs()
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
        # Remove all sensors
        for sensor in self._sensors.values():
            sensor.remove()

        # Run super
        super().remove()

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
        return len(self.get_proprioception())

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
    @abstractmethod
    def usd_path(self):
        # For all robots, this must be specified a priori, before we actually initialize the USDObject constructor!
        # So we override the parent implementation, and make this an abstract method
        raise NotImplementedError

    @property
    def urdf_path(self):
        """
        Returns:
            str: file path to the robot urdf file.
        """
        raise NotImplementedError

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
