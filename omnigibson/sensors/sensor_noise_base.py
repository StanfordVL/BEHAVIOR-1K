from abc import ABCMeta, abstractmethod

from omnigibson.utils.python_utils import Registerable, classproperty

# Registered sensor noises
REGISTERED_SENSOR_NOISES = dict()


class BaseSensorNoise(Registerable, metaclass=ABCMeta):
    """
    Base SensorNoise class.
    Sensor noise-specific add_noise method is implemented in subclasses

    Args:
        enabled (bool): Whether this sensor should be enabled by default
    """

    def __init__(self, enabled=True):
        # Store whether this noise model is enabled or not
        self._enabled = enabled

    def __call__(self, obs):
        """
        If this noise is enabled, corrupts observation @obs by adding sensor noise to sensor reading. This is an
        identical call to self.corrupt(...)

        Args:
            obs (th.tensor): observation numpy array of values of arbitrary dimension normalized to range [0.0, 1.0]

        Returns:
            th.tensor: Corrupted observation numpy array if self.enabled is True, otherwise this is a pass-through
        """
        return self.corrupt(obs=obs)

    def corrupt(self, obs):
        """
        If this noise is enabled, corrupts observation @obs by adding sensor noise to sensor reading.

        Args:
            obs (th.tensor): observation numpy array of values of arbitrary dimension normalized to range [0.0, 1.0]

        Returns:
            th.tensor: Corrupted observation numpy array if self.enabled is True, otherwise this is a pass-through
        """
        # Run sanity check to make sure obs is in acceptable range
        assert len(obs[(obs < 0.0) | (obs > 1.0)]) == 0, "sensor reading has to be between [0.0, 1.0]"

        return self._corrupt(obs=obs) if self._enabled else obs

    @abstractmethod
    def _corrupt(self, obs):
        """
        Corrupts observation @obs by adding sensor noise to sensor reading

        Args:
            obs (th.tensor): observation numpy array of values of arbitrary dimension normalized to range [0.0, 1.0]

        Returns:
            th.tensor: Corrupted observation numpy array
        """
        raise NotImplementedError()

    @property
    def enabled(self):
        """
        Returns:
            bool: Whether this noise model is enabled or not
        """
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        """
        En/disables this noise model

        Args:
            enabled (bool): Whether this noise model should be enabled or not
        """
        self._enabled = enabled

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseSensorNoise")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_SENSOR_NOISES
        return REGISTERED_SENSOR_NOISES
