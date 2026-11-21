from omnigibson.render_backends.newton_viewer_backend import NewtonViewerRenderBackend


class GLRenderBackend(NewtonViewerRenderBackend):
    """
    Rasterized rendering through Newton's headless ``ViewerGL``.
    """

    def _create_viewer(self, width, height):
        import newton

        return newton.viewer.ViewerGL(width=width, height=height, headless=True)

    def _rebind_viewer(self, viewer, model):
        # Rebind rather than close and rebuild: once a process's first ViewerGL is closed, every ViewerGL built after
        # it renders all-black frames, with no error.
        viewer.set_model(model)
        return viewer
