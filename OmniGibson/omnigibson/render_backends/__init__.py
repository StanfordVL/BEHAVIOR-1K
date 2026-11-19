from omnigibson.render_backends.kit_backend import KitRenderBackend
from omnigibson.render_backends.newton_backend import NewtonRenderBackend
from omnigibson.render_backends.null_backend import NullRenderBackend
from omnigibson.render_backends.warp_backend import WarpRenderBackend

RENDER_BACKENDS = {
    "none": NullRenderBackend,
    "kit": KitRenderBackend,
    "newton": NewtonRenderBackend,
    "warp": WarpRenderBackend,
}
