from abc import ABCMeta

import torch as th

import omnigibson.lazy as lazy


class SettingsBase(metaclass=ABCMeta):
    """
    Base class for all renderer settings classes.

    Settings classes include Common, Real-Time (Ray-Tracing), Path-Tracing and Post Processing.
    """


class SubSettingsBase(metaclass=ABCMeta):
    """
    Base class for all renderer sub-settings classes.
    """

    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

    @property
    def enabled_setting_path(self):
        """
        The path of "enabled" setting for this sub-settings class.

        Subclass with "enabled" mode needs to overwrite this method.

        Returns:
            str or None: The path of "enabled" mode for this sub-setting class.
                Defaults to None, which means this sub-setting group cannot be enabled/disabled.
        """
        return None

    def is_enabled(self):
        """
        Get the enabled status for this sub-setting class.

        Returns:
            bool: Whether this sub-setting group is enabled.
                Returns true if this sub-setting group has no "enabled" mode.
        """
        if not self.enabled_setting_path:
            return True
        return self._carb_settings.get(self.enabled_setting_path)

    def enable(self):
        """
        Enable this sub-setting class.
        """
        if not self.enabled_setting_path:
            print(f"{self.__class__.__name__} has no enabled mode.")
            return
        self._carb_settings.set_bool(self.enabled_setting_path, True)

    def disable(self):
        """
        Disable this sub-setting class.
        """
        if not self.enabled_setting_path:
            print(f"{self.__class__.__name__} has no enabled mode.")
            return
        self._carb_settings.set_bool(self.enabled_setting_path, False)


class SettingItem:
    """
    A wrapper of an individual setting item.

    Args:
        owner (:class:`SubSettingsBase`): The SubSettingsBase object owning this setting.
        setting_type (:class:`SettingType`): Setting type (e.g. float, int).
        name (str): Description of this setting.
        path (str): Path of this setting.
        range_from (float): The lower bound of the values for this setting. Defaults to -inf.
        range_to (float): The upper bound of the values for this settin. Defaults to inf.
        range_list (list): Possible values for this setting. Defaults to None.
        range_dict (dict): Possible values for this setting. Defaults to None.
    """

    def __init__(
        self,
        owner,
        setting_type,
        name,
        path,
        range_from=-float("inf"),
        range_to=float("inf"),
        range_list=None,
        range_dict=None,
    ):
        self._carb_settings = lazy.carb.settings.get_settings()
        self.owner = owner
        self.setting_type = setting_type
        self.name = name
        self.path = path
        self.range_from = range_from
        self.range_to = range_to
        self.range_list = range_list
        self.range_dict = range_dict
        self.initial_value = self.value

    @property
    def value(self):
        """
        Get the current setting value.

        Returns:
            any: The current setting value.
        """
        return self._carb_settings.get(self.path)

    def get(self):
        """
        Get the current setting value.

        Returns:
            any: The current setting value.
        """
        return self.value

    def reset(self):
        """
        Reset the current setting value to default.
        """
        self.set(self.initial_value)

    def set(self, value):
        """
        Set the current setting to @value.

        Args:
            value (any): Value to set for the current setting value.
        """
        print(f"Set setting {self.path} ({self.name}) to {value}.")  # carb.log_info
        if not self.owner.is_enabled():
            print(f"Note: {self.owner.enabled_setting_path} is not enabled.")

        # Validate range list and range dict.
        if self.range_list:
            assert value in self.range_list, f"Setting {self.path} must be chosen from {self.range_list}."
        if self.range_dict:
            assert isinstance(self.range_dict, dict)
            assert (
                value in self.range_dict.values()
            ), f"Setting {self.path} must be chosen from a value (not key) in {self.range_dict}."

        if self.setting_type == lazy.omni.kit.widget.settings.SettingType.FLOAT:
            assert isinstance(value, (int, float)), f"Setting {self.path} must be of type float."
            assert (
                value >= self.range_from and value <= self.range_to
            ), f"Setting {self.path} must be within range ({self.range_from}, {self.range_to})."
            self._carb_settings.set_float(self.path, value)

        elif self.setting_type == lazy.omni.kit.widget.settings.SettingType.INT:
            assert isinstance(value, int), f"Setting {self.path} must be of type int."
            assert (
                value >= self.range_from and value <= self.range_to
            ), f"Setting {self.path} must be within range ({self.range_from}, {self.range_to})."
            self._carb_settings.set_int(self.path, value)

        elif self.setting_type == lazy.omni.kit.widget.settings.SettingType.COLOR3:
            assert (
                isinstance(value, (list, tuple, th.Tensor)) and len(value) == 3
            ), f"Setting {self.path} must be a list of 3 numbers within range [0,1]."
            for v in value:
                assert (
                    isinstance(v, (int, float)) and v >= 0 and v <= 1
                ), f"Setting {self.path} must be a list of 3 numbers within range [0,1]."
            self._carb_settings.set_float_array(self.path, value)

        elif self.setting_type == lazy.omni.kit.widget.settings.SettingType.BOOL:
            assert isinstance(value, bool), f"Setting {self.path} must be of type bool."
            self._carb_settings.set_bool(self.path, value)

        elif self.setting_type == lazy.omni.kit.widget.settings.SettingType.STRING:
            assert isinstance(value, str), f"Setting {self.path} must be of type str."
            self._carb_settings.set_string(self.path, value)

        elif self.setting_type == lazy.omni.kit.widget.settings.SettingType.DOUBLE3:
            assert (
                isinstance(value, (list, tuple, th.Tensor)) and len(value) == 3
            ), f"Setting {self.path} must be a list of 3 floats."
            for v in value:
                assert isinstance(v, (int, float)), f"Setting {self.path} must be a list of 3 floats."
            self._carb_settings.set_float_array(self.path, value)

        elif self.setting_type == lazy.omni.kit.widget.settings.SettingType.INT2:
            assert (
                isinstance(value, (list, tuple, th.Tensor)) and len(value) == 2
            ), f"Setting {self.path} must be a list of 2 ints."
            for v in value:
                assert isinstance(v, int), f"Setting {self.path} must be a list of 2 ints."
            self._carb_settings.set_int_array(self.path, value)

        elif self.setting_type == lazy.omni.kit.widget.settings.SettingType.DOUBLE2:
            assert (
                isinstance(value, (list, tuple, th.Tensor)) and len(value) == 2
            ), f"Setting {self.path} must be a list of 2 floats."
            for v in value:
                assert isinstance(v, (int, float)), f"Setting {self.path} must be a list of 2 floats."
            self._carb_settings.set_float_array(self.path, value)

        else:
            raise TypeError(f"Setting type {self.setting_type} is not supported.")
