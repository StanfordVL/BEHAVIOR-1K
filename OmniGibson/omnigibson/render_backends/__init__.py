from omnigibson.render_backends.kit_backend import KitRenderBackend
from omnigibson.render_backends.newton_backend import NewtonRenderBackend
from omnigibson.render_backends.null_backend import NullRenderBackend

RENDER_BACKENDS = {
    "none": NullRenderBackend,
    "kit": KitRenderBackend,
    "newton": NewtonRenderBackend,
}
