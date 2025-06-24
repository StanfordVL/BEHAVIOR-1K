from abc import ABCMeta

import gymnasium as gym

from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.python_utils import Registerable, assert_valid_key, classproperty

# Registered sensors
REGISTERED_SENSORS = dict()

# All possible modalities across all sensors
ALL_SENSOR_MODALITIES = set()


class BaseSensor(XFormPrim, GymObservable, Registerable, metaclass=ABCMeta):
    """
    Base Sensor class.
    Sensor-specific get_obs method is implemented in subclasses

    Args:
        relative_prim_path (str): Scene-local prim path of the Sensor to encapsulate or create.
        name (str): Name for the sensor. Names need to be unique per scene.
        modalities (str or list of str): Modality(s) supported by this sensor. Default is "all", which corresponds
            to all modalities being used. Otherwise, valid options should be part of cls.all_modalities.
        enabled (bool): Whether this sensor should be enabled by default
        noise (None or BaseSensorNoise): If specified, sensor noise model to apply to this sensor.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this sensor's prim at runtime.
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        modalities="all",
        enabled=True,
        noise=None,
        load_config=None,
    ):
        # Store inputs (and sanity check modalities along the way)
        if modalities == "all":
            modalities = self.all_modalities
        else:
            modalities = [modalities] if isinstance(modalities, str) else modalities
            for modality in modalities:
                assert_valid_key(key=modality, valid_keys=self.all_modalities, name="modality")
        self._modalities = set(modalities)
        self._enabled = enabled
        self._noise = noise

        # Run super method
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _load(self):
        # Sub-sensors must implement this class directly! Cannot use parent XForm class by default
        raise NotImplementedError("Sensor class must implement _load!")

    def _post_load(self):
        # Run super first
        super()._post_load()

        # Set the enabled property based on the internal value
        # This is done so that any subclassed sensors which require simulator specific enabling can handle this now
        self.enabled = self._enabled

    def get_obs(self):
        # Get sensor reading, and optionally corrupt the readings with noise using self.noise if
        # self.noise.enabled is True.
        # Note that the returned dictionary will only be filled in if this sensor is enabled!
        if not self._enabled:
            return dict()

        obs, info = self._get_obs()

        if self._noise is not None:
            for k, v in obs.items():
                if k not in self.no_noise_modalities:
                    obs[k] = self._noise(v)

        return obs, info

    def _get_obs(self):
        """
        Get sensor reading. Should generally be extended by subclass.

        Returns:
            2-tuple:
                dict: Keyword-mapped observations mapping modality names to numpy arrays of arbitrary dimension
                dict: Additional information about the observations.
        """
        # Default is returning an empty dict
        return dict(), dict()

    def _load_observation_space(self):
        # Fill in observation space based on mapping and active modalities
        obs_space = dict()
        for modality, space in self._obs_space_mapping.items():
            if modality in self._modalities:
                if isinstance(space, gym.Space):
                    # Directly add this space
                    obs_space[modality] = space
                else:
                    # Assume we are procedurally generating a box space
                    shape, low, high, dtype = space
                    obs_space[modality] = self._build_obs_box_space(shape=shape, low=low, high=high, dtype=dtype)

        return obs_space

    def add_modality(self, modality):
        """
        Add a modality to this sensor. Must be a valid modality (one of self.all_modalities)

        Args:
            modality (str): Name of the modality to add to this sensor
        """
        assert_valid_key(key=modality, valid_keys=self.all_modalities, name="modality")
        if modality not in self._modalities:
            self._modalities.add(modality)
            # Update observation space
            self.load_observation_space()

    def remove_modality(self, modality):
        """
        Remove a modality from this sensor. Must be a valid modality that is active (one of self.modalities)

        Args:
            modality (str): Name of the modality to remove from this sensor
        """
        assert_valid_key(key=modality, valid_keys=self._modalities, name="modality")
        if modality in self._modalities:
            self._modalities.remove(modality)
            # Update observation space
            self.load_observation_space()

    @property
    def modalities(self):
        """
        Returns:
            set: Name of modalities provided by this sensor. This should correspond to all the keys provided
                in self.get_obs()
        """
        return self._modalities

    @property
    def _obs_space_mapping(self):
        """
        Returns:
            dict: Keyword-mapped observation space settings for each modality. For each modality in
                cls.all_modalities, its name should map directly to the corresponding gym space Space for that modality
                or a 4-tuple entry (shape, low, high, dtype) for procedurally generating the appropriate Box Space
                for that modality
        """
        raise NotImplementedError()

    @classproperty
    def all_modalities(cls):
        """
        Returns:
            set: All possible valid modalities for this sensor. Should be implemented by subclass.
        """
        raise NotImplementedError()

    @property
    def noise(self):
        """
        Returns:
            None or BaseSensorNoise: Noise model to use for this sensor
        """
        return self._noise

    @classproperty
    def no_noise_modalities(cls):
        """
        Returns:
            set: Modalities that should NOT be passed through noise, irregardless of whether noise is enabled or not.
                This is useful for some modalities which are not exclusively numerical arrays.
        """
        raise NotImplementedError()

    @property
    def enabled(self):
        """
        Returns:
            bool: Whether this sensor is enabled or not
        """
        # By default, we simply return the internal value. Subclasses may need to extend this functionality,
        # e.g. by disabling actual sim functionality for better computational efficiency
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        """
        Args:
            enabled (bool): Whether this sensor should be enabled or not
        """
        # By default, we simply store the value internally. Subclasses may need to extend this functionality,
        # e.g. by disabling actual sim functionality for better computational efficiency
        self._enabled = enabled

    @classproperty
    def sensor_type(cls):
        """
        Returns:
            str: Type of this sensor. By default, this is the sensor class name
        """
        return cls.__name__

    @classmethod
    def _register_cls(cls):
        global ALL_SENSOR_MODALITIES

        # Run super first
        super()._register_cls()

        # Also store modalities from this sensor class if we're registering it
        if cls.__name__ not in cls._do_not_register_classes:
            ALL_SENSOR_MODALITIES.union(cls.all_modalities)

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseSensor")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_SENSORS
        return REGISTERED_SENSORS
