from omni.rtx.window.settings import RendererSettingsFactory
from igibson.renderer_settings.common_settings import CommonSettings
from igibson.renderer_settings.path_tracing_settings import PathTracedFogSettings
from igibson.renderer_settings.post_processing_settings import PostProcessingSettings
from igibson.renderer_settings.real_time_settings import RealTimeSettings
import carb


class RendererSettings:
    def __init__(self):
        self._carb_settings = carb.settings.get_settings()
        self.common_settings = CommonSettings()
        self.path_tracing_settings = PathTracedFogSettings()
        self.post_processing_settings = PostProcessingSettings()
        self.real_time_settings = RealTimeSettings()

    def set_setting(self, path, value):
        if path not in self.settings:
            raise NotImplementedError(f"Setting {path} is not supported.")
        self.settings[path].set(value)

    def get_setting_from_path(self, path):
        return self._carb_settings.get(path)

    def get_current_renderer(self):
        return RendererSettingsFactory.get_current_renderer()

    def set_current_renderer(self, renderer):
        assert (
            renderer in RendererSettingsFactory.get_registered_renderers()
        ), f"renderer must be one of {RendererSettingsFactory.get_registered_renderers()}"
        print(f"Set current renderer to {renderer}.")
        RendererSettingsFactory.set_current_renderer(renderer)

    @property
    def settings(self):
        settings = {}
        settings.update(self.common_settings.settings)
        settings.update(self.path_tracing_settings.settings)
        settings.update(self.post_processing_settings.settings)
        settings.update(self.real_time_settings.settings)
        return settings
