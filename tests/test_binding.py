import platform

from omnigibson.render.mesh_renderer.get_available_devices import get_available_devices
from omnigibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings


def test_device():
    assert len(get_available_devices()[0]) > 0


def test_binding():
    if platform.system() == "Darwin":
        from omnigibson.render.mesh_renderer import GLFWRendererContext

        setting = MeshRendererSettings()
        r = GLFWRendererContext.GLFWRendererContext(
            256, 256, setting.glfw_gl_version[0], setting.glfw_gl_version[1], False, False
        )
    else:
        from omnigibson.render.mesh_renderer import EGLRendererContext

        r = EGLRendererContext.EGLRendererContext(256, 256, get_available_devices()[0][0])
    r.init()
    r.release()
