"""Third-person video capture driven off the engine's per-tick hook.

``MultiAgentPrimitiveEngine.on_tick`` is called once per ``env.step`` with the
tick index, which is exactly the cadence a recorder wants. The demo scripts
already use that hook for their concurrency monitor, so :func:`chain` composes
several callbacks onto the one slot.

Rendering is off in the normal headless configuration
(``gm.RENDER_VIEWER_CAMERA = False``) because the symbolic runs never need
pixels -- see PORTING_PLAN.md 4.5. :func:`enable_viewer_rendering` flips it
back on, and must be called *before* ``og.Environment`` is constructed, since
that is when the viewer camera is created.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Optional, Sequence

import omnigibson as og
from omnigibson.macros import gm

__all__ = ["ViewerRecorder", "chain", "enable_viewer_rendering"]


def enable_viewer_rendering() -> None:
    """Turn the viewer camera on. Call before building the Environment."""
    gm.RENDER_VIEWER_CAMERA = True


def chain(*callbacks: Optional[Callable[[int], None]]) -> Callable[[int], None]:
    """Compose several ``on_tick`` callbacks into one. ``None`` entries drop out."""
    active = [callback for callback in callbacks if callback is not None]

    def call(env_step: int) -> None:
        for callback in active:
            callback(env_step)

    return call


class ViewerRecorder:
    """Writes a video from the viewer camera as the engine ticks.

    Args:
        path: output file. The extension picks the container (``.mp4`` is the
            safe default; ``.webm`` also works via the bundled ffmpeg).
        every: capture one frame every N ticks. Rendering is the expensive part
            of a symbolic run -- the primitives themselves are nearly free --
            so this is the main speed/smoothness dial. At the default 4, a
            1000-tick episode is 250 frames.
        fps: frames per second written into the file. With ``every=4`` and a 30
            Hz action rate, fps=30 plays back at 4x real time.
        camera: viewer camera to read. Defaults to ``og.sim.viewer_camera``.

    Use as ``engine.on_tick = recorder`` (or via :func:`chain`), then call
    :meth:`close` when the run ends -- the file is only finalised on close.
    """

    def __init__(self, path: str, every: int = 4, fps: int = 30, camera=None):
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.path = path
        self.every = int(every)
        self.fps = int(fps)
        self._camera = camera
        self._writer = None
        self.frames = 0

    @property
    def camera(self):
        if self._camera is None:
            self._camera = og.sim.viewer_camera
        if self._camera is None:
            raise RuntimeError(
                "og.sim.viewer_camera is None. Call enable_viewer_rendering() BEFORE constructing "
                "og.Environment -- gm.RENDER_VIEWER_CAMERA is read when the viewer camera is created."
            )
        return self._camera

    def _writer_handle(self):
        if self._writer is None:
            import imageio  # noqa: PLC0415 - keep the import cost off non-recording runs

            self._writer = imageio.get_writer(self.path, fps=self.fps)
        return self._writer

    def capture(self) -> None:
        """Render one frame and append it. Safe to call directly."""
        # get_obs() reads the sensor's last rendered buffer, so a render has to
        # have happened this tick. env.step does not necessarily render when
        # headless, hence the explicit call.
        og.sim.render()
        frame = self.camera.get_obs()[0]["rgb"][:, :, :3]
        self._writer_handle().append_data(frame.cpu().numpy())
        self.frames += 1

    def __call__(self, env_step: int) -> None:
        """``on_tick`` entry point."""
        if env_step % self.every == 0:
            self.capture()

    def close(self) -> None:
        """Finalise the file. Idempotent."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            print(f"[video] wrote {self.frames} frames to {self.path}")
