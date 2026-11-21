"""CUDA Warp path tracing with OIDN, RGB sensors, and an optional standalone viewer.

Currently supports Newton rigid geometry. Materials/lights are extracted when the model is
rebuilt; dynamic material/light edits and particle surfaces are not yet supported.
"""

import math
import time
from types import SimpleNamespace

import numpy as np
import torch as th

from omnigibson.macros import gm
from omnigibson.render_backends.null_backend import NullRenderBackend


class WarpRenderBackend(NullRenderBackend):
    supports_camera_capture = True
    requires_physics_visual_shapes = False

    def __init__(self, sim=None):
        super().__init__(sim)
        if gm.PHYSICS_BACKEND != "newton":
            raise ValueError("Warp rendering currently requires the Newton physics backend.")
        self._model = self._renderer = self._denoiser = self._viewer = None
        self._last_frame_time = None
        self._closed = False
        self._view_distance = 3.0

    def setup_viewer_camera(self, viewer_width, viewer_height):
        # A real VisionSensor keeps existing examples' viewer-camera pose API working.
        self.sim._set_viewer_camera(viewer_width=viewer_width, viewer_height=viewer_height)
        # Widen only the navigation camera to a 60-degree vertical field of view.
        self.sim.viewer_camera.focal_length = (
            self.sim.viewer_camera.horizontal_aperture * viewer_height / viewer_width / (2 * math.tan(math.radians(30)))
        )

    def focus_viewer_camera(self, target):
        """Aim the navigation camera at a world-space target and use it as the orbit pivot."""
        from omnigibson.render_backends.warp_renderer.renderer import look_at_rotation
        from omnigibson.utils.transform_utils import mat2quat

        sensor = self.sim.viewer_camera
        position, _ = sensor.get_position_orientation()
        target = th.as_tensor(target, dtype=position.dtype, device=position.device)
        self._view_distance = float(th.linalg.vector_norm(position - target))
        rotation = look_at_rotation(position.cpu().numpy(), target.cpu().numpy())
        sensor.set_position_orientation(
            position=position,
            orientation=mat2quat(th.as_tensor(rotation, dtype=position.dtype, device=position.device)),
        )

    def _ensure_renderer(self):
        from omnigibson.render_backends.warp_renderer.extract import extract_scene
        from omnigibson.render_backends.warp_renderer.oidn import Denoiser
        from omnigibson.render_backends.warp_renderer.renderer import WarpRenderer

        physics = self.sim.physics_backend
        model = physics._model
        if model is None or model.body_count == 0:
            return False
        if model is not self._model:
            if self._denoiser is not None:
                self._denoiser.close()
            self._renderer = self._denoiser = None
            labels = [""] * model.body_count
            for path, index in physics._path_body_map.items():
                labels[index] = path
            scene = extract_scene(self.sim.stage, labels)
            # Rendering stays on CUDA even if an example runs its physics on the CPU.
            device = physics._state_0.body_q.device
            device = device if device.is_cuda else "cuda:0"
            self._renderer = WarpRenderer(scene, device=device, ambient=160.0)
            self._denoiser = Denoiser(device=device, quality="high")
            self._model = model
        self._renderer.set_body_q(physics._state_0.body_q.numpy())
        return True

    def _camera_parameters(self, path, width, height):
        from omnigibson.utils.usd_utils import get_world_pose
        from omnigibson.utils.transform_utils import quat2mat

        position, orientation = get_world_pose(path)
        rotation = quat2mat(orientation).cpu().numpy()
        position = position.cpu().numpy()
        prim = self.sim.stage.GetPrimAtPath(path)
        focal = prim.GetAttribute("focalLength").Get()
        aperture = prim.GetAttribute("horizontalAperture").Get()
        return dict(
            pos=position,
            look_at=position - rotation[:, 2],
            up=rotation[:, 1],
            fov=math.degrees(2 * math.atan(aperture / (2 * focal) * height / width)),
        )

    def _capture(self, camera, width, height):
        from omnigibson.render_backends.warp_renderer.oidn import warp_with_oidn

        if not self._ensure_renderer():
            return np.zeros((height, width, 3), dtype=np.uint8)
        return warp_with_oidn(
            self._renderer,
            self._denoiser,
            [camera],
            width,
            height,
            preset="pt4",
            exposure=0.0007717915155850135,
            seed=0,
            spp=1,
        ).numpy()[0]

    def render(self):
        if self._closed or gm.HEADLESS or self.sim.viewer_camera is None:
            return
        # Initialization also ticks rendering before a scene/model is ready.
        if not any(scene.initialized for scene in self.sim.scenes):
            return
        from omnigibson.utils.warp_viewer import _WarpViewer
        from omnigibson.utils.transform_utils import mat2quat
        from omnigibson.render_backends.warp_renderer.renderer import look_at_rotation
        from pyglet.window import key

        sensor = self.sim.viewer_camera
        width, height = sensor.image_width, sensor.image_height
        camera = self._camera_parameters(sensor.prim_path, width, height)
        camera["look_at"] = camera["pos"] + (camera["look_at"] - camera["pos"]) * self._view_distance
        if self._viewer is None:
            self._viewer = _WarpViewer(camera, width, height)

            def on_key_press(symbol, modifiers):
                if symbol == key.ESCAPE:
                    self._viewer.window.has_exit = True

            self._viewer.window.push_handlers(on_key_press=on_key_press)
        else:
            # Respect camera changes made through the public VisionSensor API.
            self._viewer.camera.update(camera)
        window = self._viewer.window
        before_pos = np.array(camera["pos"])
        before_target = np.array(camera["look_at"])
        window.dispatch_events()
        if window.has_exit:
            self.close()
            raise SystemExit(0)
        camera = self._viewer.camera
        self._view_distance = float(np.linalg.norm(np.asarray(camera["pos"]) - camera["look_at"]))
        if not (np.array_equal(before_pos, camera["pos"]) and np.array_equal(before_target, camera["look_at"])):
            rotation = look_at_rotation(camera["pos"], camera["look_at"])
            sensor.set_position_orientation(
                position=th.as_tensor(camera["pos"], dtype=th.float32, device=self.sim.device),
                orientation=mat2quat(th.as_tensor(rotation, dtype=th.float32, device=self.sim.device)),
            )
            camera["up"] = rotation[:, 1]
        frame = self._capture(camera, width, height)
        now = time.monotonic()
        fps = 0 if self._last_frame_time is None else 1 / max(now - self._last_frame_time, 1e-6)
        self._last_frame_time = now
        self._viewer.draw(frame, fps, self.sim.physics_backend._model)

    def create_camera_resource(self, prim_path, resolution, force_new=False):
        return SimpleNamespace(prim_path=prim_path, resolution=resolution)

    def destroy_camera_resource(self, camera_resource):
        pass

    def create_annotator(self, raw_modality_name):
        if raw_modality_name != "rgb":
            raise NotImplementedError(f"Warp currently supports only RGB camera capture, not {raw_modality_name!r}.")
        return SimpleNamespace(camera_resource=None)

    def attach_modality(self, annotator, camera_resource):
        annotator.camera_resource = camera_resource

    def detach_modality(self, annotator, camera_resource):
        annotator.camera_resource = None

    def get_modality_data(self, annotator, device=None):
        resource = annotator.camera_resource
        if resource is None:
            raise RuntimeError("Camera annotator is not attached.")
        width, height = resource.resolution
        rgb = self._capture(self._camera_parameters(resource.prim_path, width, height), width, height)
        rgba = np.full((height, width, 4), 255, dtype=np.uint8)
        rgba[..., :3] = rgb
        if device is None or str(device) == "cpu":
            return rgba
        import warp as wp

        return wp.array(rgba, dtype=wp.uint8, device=device)

    def close(self):
        self._closed = True
        if self._viewer is not None:
            self._viewer.window.close()
        if self._denoiser is not None:
            self._denoiser.close()
        self._viewer = self._denoiser = self._renderer = self._model = None
