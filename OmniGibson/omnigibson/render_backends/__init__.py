from omnigibson.render_backends.kit_backend import KitRenderBackend
from omnigibson.render_backends.newton_viewer_backend import GLRenderBackend, OVRTXRenderBackend
from omnigibson.render_backends.null_backend import NullRenderBackend

RENDER_BACKENDS = {
    "none": NullRenderBackend,
    "kit": KitRenderBackend,
    "gl": GLRenderBackend,
    "ovrtx": OVRTXRenderBackend,
}

# (gm.PHYSICS_BACKEND, gm.RENDER_BACKEND) pairs that are supported. PhysX lives inside Kit, so it only renders through
# Kit; Newton runs without Kit, so it renders through Newton's own viewers (gl, ovrtx) or not at all.
SUPPORTED_BACKEND_COMBINATIONS = (
    ("physx", "kit"),
    ("newton", "none"),
    ("newton", "gl"),
    ("newton", "ovrtx"),
)
