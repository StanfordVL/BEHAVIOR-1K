"""
Newton-viewer implementation of the RenderBackend interface: gives VisionSensor ``rgb`` capture through
Newton's own standalone ``ViewerGL`` (no live Kit application at all, no Replicator) -- one headless GL
window per active camera resource, its camera synced to that camera prim's own live world pose/FOV on
every capture.

Only ``rgb`` is supported: ``ViewerGL`` has no equivalent to Replicator's
depth/segmentation/bbox/optical-flow/pointcloud annotators, so ``create_annotator()`` raises for any
other raw modality name -- ``VisionSensor`` never reaches this backend for those anyway once a caller
adds unsupported-modality guards on top of ``supports_camera_capture`` (not yet done as of this writing;
today, requesting e.g. ``depth`` under this backend surfaces this class's own ``NotImplementedError``
instead of a cleaner up-front check).

Uses ``ViewerGL`` specifically, not ``ViewerRTX`` -- confirmed empirically (see
``newton_viewer_recording.py``'s module history) that ``ViewerRTX`` renders this codebase's baked visual
shapes as near-black regardless of lighting/sample-count settings, while ``ViewerGL`` renders them
correctly.
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
    """One USD camera prim's live world pose/FOV, rendered through a headless ``ViewerGL``.

    The viewer is NOT private to this resource: every resource of the same resolution shares one (see
    NewtonRenderBackend._acquire_viewer), because each ``ViewerGL`` owns its own GL context and
    therefore uploads its own private copy of every visual mesh in the scene -- measured at ~5 GB per
    viewer on a full household scene, so one viewer per camera exhausted a 32 GB card at the 4th
    camera. Sharing is safe precisely because _sync_camera() re-points the shared viewer at this
    resource's own prim immediately before every capture, and the frame is read back before control
    returns, so no state carries between two resources' captures.

    Acquired lazily on first capture (not at construction time) since it needs the physics backend's
    finalized Newton model, which may not exist yet when ``create_camera_resource()`` is called (mirrors
    ``newton_viewer_recording.build_viewer()``'s own callers, which all build the model first).
    """

    def __init__(self, backend, prim_path, resolution):
        self.backend = backend
        self.sim = backend.sim
        self.prim_path = prim_path
        self.width, self.height = resolution
        self._viewer = None
        self._released = False
        backend._register_viewer_user(self.width, self.height)

    def _ensure_viewer(self):
        # Re-fetched every capture rather than cached on the resource: the backend rebinds the shared
        # viewer when the physics model is rebuilt, and a cached reference would keep rendering the
        # old model's geometry (or a closed viewer).
        self._viewer = self.backend._get_viewer(self.width, self.height)

    def _sync_camera(self):
        from pyglet.math import Vec3 as PyVec3

        pos, quat = get_world_pose(self.prim_path)
        forward = T.quat_apply(quat, _LOCAL_FORWARD.to(quat.device))
        pos_list = pos.tolist()
        look_at = (pos + forward).tolist()

        self._viewer.camera.pos = PyVec3(*pos_list)
        self._viewer.camera.look_at(look_at)

        # Best-effort FOV match against the tracked prim's own authored camera attributes -- an
        # approximation (Newton's Camera takes one scalar `fov`, not separate horizontal/vertical/
        # aperture-based intrinsics like a real USD camera), not exact intrinsics parity.
        usd_camera = lazy.pxr.UsdGeom.Camera(og.sim.stage.GetPrimAtPath(self.prim_path))
        focal_length = usd_camera.GetFocalLengthAttr().Get()
        horizontal_aperture = usd_camera.GetHorizontalApertureAttr().Get()
        if focal_length and horizontal_aperture:
            # Newton's `fov` is the VERTICAL field of view (it feeds pyglet's perspective_projection,
            # whose fov argument sets the vertical extent), while the horizontal aperture is the
            # authoritative one on the USD side -- OmniGibson only ever authors horizontalAperture and
            # lets the vertical extent follow the output aspect ratio. So convert through this
            # camera's aspect; the two only coincide for a square capture resolution.
            horizontal_fov = 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
            aspect = self.width / self.height
            self._viewer.camera.fov = math.degrees(2.0 * math.atan(math.tan(horizontal_fov / 2.0) / aspect))

    def capture_rgb(self, device):
        self._ensure_viewer()
        self._sync_camera()

        backend = self.sim.physics_backend
        self._viewer.begin_frame(0.0)
        self._viewer.log_state(backend._state_0)
        self._viewer.end_frame()

        # (H, W, 3) uint8, host copy regardless of the GL viewer's own rendering device.
        rgb = self._viewer.get_frame().numpy()
        # Match Replicator's "rgb" annotator convention (RGBA, not RGB) -- VisionSensor's observation
        # space is built assuming a 4th (alpha) channel exists. Newton's GL viewer has no transparency
        # concept for the final composited frame, so this is always fully opaque (255).
        h, w, _ = rgb.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255

        if device is None or device == "cpu":
            return rgba
        return lazy.warp.array(rgba, dtype=lazy.warp.uint8, device=device)

    def destroy(self):
        # Hand the shared viewer back; it only actually closes once its last user is gone.
        if not self._released:
            self._released = True
            self._viewer = None
            self.backend._release_viewer(self.width, self.height)


class _NewtonAnnotator:
    """Trivial stand-in for a Replicator annotator: just tracks which modality and which camera
    resource it's attached to. NewtonRenderBackend has no separate annotator-graph concept of its own --
    ``get_modality_data()`` captures on demand straight from ``camera_resource``."""

    def __init__(self, modality):
        self.modality = modality
        self.camera_resource = None


class NewtonRenderBackend(RenderBackend):
    """
    Renders through Newton's own standalone ``ViewerGL`` -- no Kit application, no Replicator. Only
    supports the ``rgb`` modality (see module docstring).
    """

    supports_camera_capture = True

    def __init__(self, sim=None):
        super().__init__(sim=sim)
        # Offscreen ViewerGLs shared by every camera resource of the same resolution, and how many
        # resources currently want one of that size: (width, height) -> viewer / -> user count. See
        # _get_viewer() for why they are shared rather than one per camera.
        self._shared_viewers = {}
        self._viewer_users = {}

    def _register_viewer_user(self, width, height):
        """Note that one more camera resource wants a shared viewer of this size (see _get_viewer)."""
        self._viewer_users[(width, height)] = self._viewer_users.get((width, height), 0) + 1

    def _get_viewer(self, width, height):
        """
        The headless ViewerGL shared by every camera resource of this size, built on first use and
        rebound when the physics model is rebuilt.

        Shared rather than one-per-camera because a ViewerGL owns its own GL context and so uploads a
        private copy of every visual mesh in the scene into it: measured at ~5 GB apiece on a full
        household scene (house_double_floor_lower), so a main view plus 6 camera panels asked for
        ~35 GB on a 32 GB card and died of GPU OOM partway through building the 4th viewer. Sharing is
        safe because every caller re-points the camera at its own prim (_sync_camera) immediately
        before rendering and reads the frame back before returning, so no state carries across two
        resources' captures.

        Args:
            width (int): Viewer width in pixels
            height (int): Viewer height in pixels

        Returns:
            ViewerGL: Shared viewer for this resolution, bound to the current Newton model
        """
        from omnigibson.utils.newton_viewer_recording import build_viewer

        model = self.sim.physics_backend._model
        viewer = self._shared_viewers.get((width, height))
        if viewer is not None and viewer.model is not model:
            # Model was rebuilt (dynamic object add/remove): a viewer still bound to the old one
            # renders stale geometry. Replace it in place, so the other resources sharing this size
            # pick the new one up on their own next capture without disturbing the user count.
            viewer.close()
            viewer = None
        if viewer is None:
            # cam_pos/look_at are throwaway placeholders -- _sync_camera() overwrites them from the
            # tracked prim's own live pose before every capture.
            viewer = build_viewer(
                "gl",
                model,
                cam_pos=(1.0, 0.0, 0.0),
                look_at=(0.0, 0.0, 0.0),
                width=width,
                height=height,
                headless=True,
            )
            self._shared_viewers[(width, height)] = viewer
        return viewer

    def _release_viewer(self, width, height):
        """Drop one resource's claim on a shared viewer, closing it once nobody is left using it."""
        key = (width, height)
        remaining = self._viewer_users.get(key, 0) - 1
        if remaining > 0:
            self._viewer_users[key] = remaining
            return
        self._viewer_users.pop(key, None)
        viewer = self._shared_viewers.pop(key, None)
        if viewer is not None:
            viewer.close()

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
                f"NewtonRenderBackend only supports {sorted(_SUPPORTED_RAW_MODALITIES)} modalities, got "
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
