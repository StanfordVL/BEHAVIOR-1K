"""
No-op implementation of the RenderBackend interface: renders nothing, needs no Kit application.

This is the default render backend. Its whole purpose is to preserve exactly what "no render backend
at all" used to mean before this abstraction existed -- in particular, a physics backend with
``runs_inside_kit = False`` (e.g. Newton) continues to boot with zero Kit dependency unless the caller
explicitly opts into ``gm.RENDER_BACKEND = "kit"``. Without this class, defaulting straight to
``KitRenderBackend`` would silently force a live Kit application even for callers who never asked for
rendering, since ``Simulator._launch_app()`` launches Kit if *either* backend's ``runs_inside_kit`` is
True.
"""

from omnigibson.render_backends.base import RenderBackend


class NullRenderBackend(RenderBackend):
    """
    Render backend that renders nothing. All capability flags stay at their False defaults.
    """

    runs_inside_kit = False

    def render(self):
        pass

    # Camera capture pipeline: unreachable in practice -- every caller guards on `runs_inside_kit`
    # first (this backend's is always False), so these only exist to satisfy the ABC.

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
