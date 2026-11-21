"""
Shared base for render backends built on Newton's standalone viewers, for Newton physics without Kit:

- ``GLRenderBackend`` (``gm.RENDER_BACKEND = "gl"``, gl_backend.py): rasterized, through ``newton.viewer.ViewerGL``
- ``OVRTXRenderBackend`` (``gm.RENDER_BACKEND = "ovrtx"``, ovrtx_backend.py): ray-traced, through
  ``newton.viewer.ViewerRTX`` (ovrtx)

Both give VisionSensor ``rgb`` capture through headless viewers shared between camera resources, synced to a camera
prim's world pose on capture.

Only ``rgb`` is supported: neither viewer has an equivalent to Replicator's
depth/segmentation/bbox/optical-flow/pointcloud annotators, so ``create_annotator()`` raises for any
other raw modality name.
"""

import math

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.render_backends.base import RenderBackend
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import get_world_pose

log = create_module_logger(module_name=__name__)

_SUPPORTED_RAW_MODALITIES = {"rgb"}
# UsdGeom.Camera's universal convention: a camera looks down its own local -Z axis, +Y up.
_LOCAL_FORWARD = th.tensor([0.0, 0.0, -1.0])


class _NewtonCameraResource:
    """One USD camera prim's live world pose, rendered through its backend's headless viewer.

    The viewer is NOT private to this resource: every resource of the same resolution shares a viewer.
    """

    def __init__(self, backend, prim_path, resolution):
        self.backend = backend
        self.sim = backend.sim
        self.prim_path = prim_path
        self.width, self.height = resolution
        self._viewer = None

    def _ensure_viewer(self):
        # Re-fetched every capture since the backend rebinds the shared viewer when the physics model is rebuilt.
        self._viewer = self.backend._get_viewer(self.width, self.height, self._vertical_fov())

    def _vertical_fov(self):
        """
        Returns:
            None or float: vertical FOV in degrees matching the prim's authored intrinsics, or None if unauthored
        """
        # Best-effort FOV match against the tracked prim's camera attributes. This is an
        # approximation since Newton's Camera takes one scalar `fov`, not separate horizontal/vertical/
        # aperture-based intrinsics like a real USD camera.
        usd_camera = lazy.pxr.UsdGeom.Camera(og.sim.stage.GetPrimAtPath(self.prim_path))
        focal_length = usd_camera.GetFocalLengthAttr().Get()
        horizontal_aperture = usd_camera.GetHorizontalApertureAttr().Get()
        if focal_length and horizontal_aperture:
            # Newton's `fov` is the VERTICAL field of view , while the horizontal aperture is the
            # authoritative one on the USD side -- OmniGibson only ever authors horizontalAperture and
            # lets the vertical extent follow the output aspect ratio.
            horizontal_fov = 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
            aspect = self.width / self.height
            return math.degrees(2.0 * math.atan(math.tan(horizontal_fov / 2.0) / aspect))
        return None

    def _sync_camera(self):
        from pyglet.math import Vec3 as PyVec3

        pos, quat = get_world_pose(self.prim_path)
        forward = T.quat_apply(quat, _LOCAL_FORWARD.to(quat.device))
        self._viewer.camera.pos = PyVec3(*pos.tolist())
        self._viewer.camera.look_at((pos + forward).tolist())
        # ViewerRTX only pushes the camera to the renderer when it is flagged dirty
        self._viewer._camera_dirty = True

        fov = self._vertical_fov()
        if not self.backend.fov_fixed_at_build and fov is not None:
            self._viewer.camera.fov = fov

    def capture_rgb(self, device):
        self._ensure_viewer()
        self._sync_camera()

        backend = self.sim.physics_backend
        self._viewer.begin_frame(0.0)
        self._viewer.log_state(backend._state_0)
        self._viewer.end_frame()

        # (H, W, 3) uint8, host copy regardless of the viewer's own rendering device.
        rgb = self._viewer.get_frame().numpy()
        # Match Replicator's "rgb" annotator convention (RGBA, not RGB). The end frame is always opaque in Newton.
        h, w, _ = rgb.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255

        if device is None or device == "cpu":
            return rgba
        return lazy.warp.array(rgba, dtype=lazy.warp.uint8, device=device)

    def destroy(self):
        self._viewer = None


class _NewtonAnnotator:
    """Trivial stand-in for a Replicator annotator: just tracks which modality and which camera
    resource it's attached to."""

    def __init__(self, modality):
        self.modality = modality
        self.camera_resource = None


class NewtonViewerRenderBackend(RenderBackend):
    """
    Shared base for render backends built on a Newton viewer. Only supports ``rgb``, and requires Newton physics.
    """

    supports_camera_capture = True
    # Whether the viewer's FOV is fixed when it is built, so viewers can only be shared between cameras with one FOV
    fov_fixed_at_build = False

    def __init__(self, sim=None):
        super().__init__(sim=sim)
        # Offscreen viewers shared between camera resources, keyed by _get_viewer()'s viewer key.
        self._shared_viewers = {}

    def _get_viewer(self, width, height, fov=None):
        """
        The headless viewer shared by every camera resource of this size (and, under RTX, this FOV), built on
        first use and rebound when the physics model is rebuilt.

        Shared rather than one-per-camera because a ViewerGL owns its own GL context and uploads a
        private copy of every visual mesh in the scene into it: up to ~5 GB on house_double_floor_lower).
        Every caller re-points the camera at its own prim (_sync_camera) immediately before rendering.

        Args:
            width (int): Viewer width in pixels
            height (int): Viewer height in pixels
            fov (None or float): Vertical FOV in degrees. Fixed at build time under RTX, so part of the key there

        Returns:
            ViewerGL or ViewerRTX: Shared viewer for this key, bound to the current Newton model
        """
        key = (width, height, None if fov is None else round(fov, 3)) if self.fov_fixed_at_build else (width, height)
        model = self.sim.physics_backend._model
        viewer = self._shared_viewers.get(key)
        if viewer is not None and viewer.model is not model:
            # Model was rebuilt (dynamic object add/remove): a viewer still bound to the
            # old one renders stale geometry.
            viewer = self._rebind_viewer(viewer, model)
        if viewer is None:
            # _sync_camera() poses the camera from the tracked prim before every capture
            viewer = self._create_viewer(width=width, height=height)
            viewer.set_model(model)
            # Render the model's baked-in visual shapes (physics_backends/newton_visuals.py), not its collision shapes
            viewer.show_collision = False
            viewer.show_visual = True
            if fov is not None:
                # ViewerRTX reads this once, when it initializes on the first end_frame()
                viewer.camera.fov = fov
            self._shared_viewers[key] = viewer
        return viewer

    def _create_viewer(self, width, height):
        """
        Args:
            width (int): Viewer width in pixels
            height (int): Viewer height in pixels

        Returns:
            ViewerGL or ViewerRTX: New headless viewer, not yet bound to a model
        """
        raise NotImplementedError()

    def _rebind_viewer(self, viewer, model):
        """
        Args:
            viewer (ViewerGL or ViewerRTX): Shared viewer still bound to an old Newton model
            model (newton.Model): The rebuilt model

        Returns:
            None or ViewerGL or ViewerRTX: @viewer bound to @model, or None if it was closed and must be rebuilt
        """
        raise NotImplementedError()

    def render(self):
        # Nothing to tick globally: each camera resource renders on-demand in get_modality_data().
        pass

    def create_camera_resource(self, prim_path, resolution, force_new=False):
        return _NewtonCameraResource(self, prim_path, resolution)

    def destroy_camera_resource(self, camera_resource):
        camera_resource.destroy()

    def create_annotator(self, raw_modality_name):
        if raw_modality_name not in _SUPPORTED_RAW_MODALITIES:
            raise NotImplementedError(
                f"{type(self).__name__} only supports {sorted(_SUPPORTED_RAW_MODALITIES)} modalities, got "
                f"{raw_modality_name!r}."
            )
        return _NewtonAnnotator(raw_modality_name)

    def attach_modality(self, annotator, camera_resource):
        annotator.camera_resource = camera_resource

    def detach_modality(self, annotator, camera_resource):
        annotator.camera_resource = None

    def get_modality_data(self, annotator, device=None):
        if annotator.camera_resource is None:
            raise RuntimeError("Annotator is not attached to a camera resource.")
        return annotator.camera_resource.capture_rgb(device)
