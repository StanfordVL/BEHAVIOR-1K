from omnigibson.render_backends.base import RenderBackend
from omnigibson.render_backends.usd_stage_services import UsdStageServicesMixin


class NullRenderBackend(UsdStageServicesMixin, RenderBackend):
    def render(self):
        pass

    def create_camera_resource(self, prim_path, resolution, force_new=False):
        raise NotImplementedError("NullRenderBackend does not render; this should be unreachable.")

    def destroy_camera_resource(self, camera_resource):
        raise NotImplementedError("NullRenderBackend does not render; this should be unreachable.")

    def create_annotator(self, raw_modality_name):
        raise NotImplementedError("NullRenderBackend does not render; this should be unreachable.")

    def attach_modality(self, annotator, camera_resource):
        raise NotImplementedError("NullRenderBackend does not render; this should be unreachable.")

    def detach_modality(self, annotator, camera_resource):
        raise NotImplementedError("NullRenderBackend does not render; this should be unreachable.")

    def get_modality_data(self, annotator, device=None):
        raise NotImplementedError("NullRenderBackend does not render; this should be unreachable.")
