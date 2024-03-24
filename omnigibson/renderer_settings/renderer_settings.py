import omnigibson.lazy as lazy
from omnigibson.renderer_settings.common_settings import CommonSettings
from omnigibson.renderer_settings.path_tracing_settings import PathTracingSettings
from omnigibson.renderer_settings.post_processing_settings import PostProcessingSettings
from omnigibson.renderer_settings.real_time_settings import RealTimeSettings


def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


@singleton
class RendererSettings:
    """
    Controller for all renderer settings.
    """

    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()
        self.common_settings = CommonSettings()
        self.path_tracing_settings = PathTracingSettings()
        self.post_processing_settings = PostProcessingSettings()
        self.real_time_settings = RealTimeSettings()

    def set_setting(self, path, value):
        """
        Sets setting @path with value @value.

        Args:
            path (str): Path of the setting to set.
            value (any): Value to set for for setting @path.
        """
        if path not in self.settings:
            raise NotImplementedError(f"Setting {path} is not supported.")
        self.settings[path].set(value)

    def reset_setting(self, path):
        """
        Resets setting @path to default value.

        Args:
            path (str): Path of the setting to reset.
        """
        if path not in self.settings:
            raise NotImplementedError(f"Setting {path} is not supported.")
        self.settings[path].reset()

    def get_setting_from_path(self, path):
        """
        Get the value of setting @path.

        Args:
            path (str): Path of the setting to get.

        Returns:
            any: Value of the requested setting @path.
        """
        return self._carb_settings.get(path)

    def get_current_renderer(self):
        """
        Get the current renderer.

        Args:
            path (str): Path of the setting to get.

        Returns:
            str: the current renderer.
        """
        return lazy.omni.rtx.window.settings.RendererSettingsFactory.get_current_renderer()

    def set_current_renderer(self, renderer):
        """
        Set the current renderer to @renderer.

        Args:
            renderer (str): The renderer to set as current (e.g. Real-Time, Path-Traced).
        """
        assert (
            renderer in lazy.omni.rtx.window.settings.RendererSettingsFactory.get_registered_renderers()
        ), f"renderer must be one of {lazy.omni.rtx.window.settings.RendererSettingsFactory.get_registered_renderers()}"
        print(f"Set current renderer to {renderer}.")
        lazy.omni.rtx.window.settings.RendererSettingsFactory.set_current_renderer(renderer)

    @property
    def settings(self):
        """
        Get all available settings.

        Returns:
            dict: A dictionary of all available settings.
                Keys are setting paths and values are setting item objects.
        """
        settings = {}
        settings.update(self.common_settings.settings)
        settings.update(self.path_tracing_settings.settings)
        settings.update(self.post_processing_settings.settings)
        settings.update(self.real_time_settings.settings)
        return settings
