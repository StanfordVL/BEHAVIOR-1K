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

__all__ = [
    "OVERVIEW_FOCAL_LENGTH",
    "WIDE_FOCAL_LENGTH",
    "MultiViewRecorder",
    "ViewerRecorder",
    "chain",
    "chase_pose",
    "enable_viewer_rendering",
    "overview_pose",
]

#: Viewer camera intrinsics, read off og.sim.viewer_camera's defaults.
FOCAL_LENGTH = 17.0
HORIZONTAL_APERTURE = 20.995

#: Wide-angle focal length, in mm. Needed for *any* indoor shot here, not
#: just a room-wide one: the ceiling caps the camera at ~2.4 m, an R1 is ~1.5 m
#: tall, so a camera above a robot has under a metre of clearance and a 63 deg
#: lens sees little more than the robot's own head. At 8.5 mm the same height
#: covers 5.7 m of floor.
#:
#: Original note (still true): the room overview at 17 mm was impossible. The default 17 mm is
#: "roughly the human eye" (upstream's words) and gives a 63 deg horizontal
#: FOV: from inside a 4.9 m room with a 2.4 m ceiling that cannot contain the
#: 4.4 m the task spans, from any position. No amount of repositioning fixes
#: an FOV that is too narrow, which is what three rounds of moving the camera
#: were really failing at. At 8.5 mm the FOV is 102 deg and 3.8 m of standoff
#: covers 9.4 m.
WIDE_FOCAL_LENGTH = 8.5
OVERVIEW_FOCAL_LENGTH = WIDE_FOCAL_LENGTH  # kept: overview_pose still references it

#: Hard ceiling on camera height, in metres. NOT derived from the span: doing
#: that put the camera 5.6 m up to fit a 4.4 m living room and filmed the
#: ceiling from inside the roof void, and 45 m up in a 23 m hall to film the
#: outside of the building. Rooms have ceilings; halls have roofs. Fit the span
#: with a horizontal standoff instead, and keep the lens under the ceiling.
MAX_CAMERA_HEIGHT = 2.4


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


def _half_fov_h(focal_length: float = FOCAL_LENGTH) -> float:
    import math

    return math.atan((HORIZONTAL_APERTURE / 2) / focal_length)


def overview_pose(points, room_cells=None, height=None):
    """A pose that sees @points, standing where a robot could stand.

    Args:
        points: world positions the shot has to contain.
        room_cells: ``[(x, y), ...]`` traversable cells of the room. The camera
            is placed on the one FARTHEST from the cluster centre, which
            maximises standoff while guaranteeing the lens is inside the room.
        height: override the derived height. Capped at MAX_CAMERA_HEIGHT.

    Every framing failure before this one came from inferring where the camera
    could stand:

    - deriving height from the span put it 5.6 m up, above the ceiling, filming
      the roof void;
    - backing off by the standoff the FOV wanted put it 6.3 m out of a room
      that is 4.9 m wide, i.e. through the wall;
    - snapping that ideal to "the nearest traversable cell" put it on the far
      side of the west wall, in the next room -- the giveaway was a different
      floor material from the one the robots stand on;
    - and object positions say nothing about where the walls are: the object
      bounding box here is 2.4 x 4.4 m inside a room that is 4.9 x 5.0 m.

    Traversable cells are the one source of truth about where a camera can be,
    so this takes them as an argument rather than re-deriving anything.
    """
    import math

    import torch as th

    from coop2.behavior_env.placement import look_at_quaternion

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    eye_height = min(MAX_CAMERA_HEIGHT, height if height is not None else MAX_CAMERA_HEIGHT)

    if room_cells:
        spread = max(max(xs) - min(xs), max(ys) - min(ys), 2.0)
        want = (spread / 2) / math.tan(_half_fov_h(OVERVIEW_FOCAL_LENGTH)) + spread / 2

        # Distance alone cannot tell whether a wall is in the way, and this
        # room's seg-map label spans an open-plan boundary: the cell at the
        # right distance was on the far side of it, and the frame was that
        # wall. Restrict candidates to the neighbourhood of the objects, which
        # are in the same space as the furniture by construction -- a shot from
        # above a robot's own footprint is the one view that has always worked.
        margin = 0.75
        near_objects = [
            xy for xy in room_cells
            if min(xs) - margin <= xy[0] <= max(xs) + margin
            and min(ys) - margin <= xy[1] <= max(ys) + margin
        ]
        eye_xy = min(
            near_objects or room_cells,
            key=lambda xy: abs(math.hypot(xy[0] - cx, xy[1] - cy) - want),
        )
    else:
        # No cells given: back off along the long axis by half its extent, which
        # at least stays within the cluster's own footprint.
        spread_x, spread_y = max(xs) - min(xs), max(ys) - min(ys)
        eye_xy = (cx, cy - spread_y / 2) if spread_y >= spread_x else (cx - spread_x / 2, cy)

    position = th.tensor([eye_xy[0], eye_xy[1], eye_height], dtype=th.float32)
    # Aim at the cluster centre a little above the floor: what matters sits on
    # furniture, and aiming at the floor tips the furniture out of frame.
    target = th.tensor([cx, cy, 0.6], dtype=th.float32)
    return position, look_at_quaternion(position, target)


def chase_pose(robot, height: float = 2.3, lead: float = 0.0, room_cells=None):
    """Third-person pose for @robot: straight down from above it.

    This is, deliberately and exactly, the one camera geometry that has ever
    produced a usable frame in this scene -- eye directly over the robot at
    ``height``, target the robot itself. Everything else was tried and failed:

    - "behind the robot by a fixed offset" is inside the wall whenever the
      robot stands near one, and the frame is flat grey;
    - a room-wide overview cannot be framed at all (63 deg cannot hold 4.4 m
      of task from inside a 4.9 m room with a 2.4 m ceiling);
    - and aiming even 0.3 m ahead of the robot instead of at it turned the
      identical eye position from "robot centred on the floor" into "a wall",
      which is why @lead now defaults to 0.

    @lead is kept as an argument because a small forward aim is genuinely nicer
    when it works, but it is off by default: this function's job is to return a
    frame with the robot in it, and only straight down does that reliably.

    Args:
        robot: the subject.
        height: metres above the robot's base. Its head is ~1.5 m up and the
            ceiling is ~2.4 m, so there is about a metre of room to work in --
            which is why the recorder pairs this with a wide lens.
        lead: metres ahead of the robot to aim. 0 means straight down.
        room_cells: accepted for signature compatibility; unused, because a
            pose directly above a robot cannot leave the robot's room.
    """
    import math

    import torch as th

    from omnigibson.utils import transform_utils as T

    from coop2.behavior_env.placement import look_at_quaternion

    position, orientation = robot.get_position_orientation()
    x, y, z = (float(v) for v in position)

    # A centimetre off vertical, matching the control shot: exactly vertical is
    # degenerate for look-at (forward parallel to world up) and leaves roll
    # undefined.
    eye = th.tensor([x + 0.01, y, z + height], dtype=th.float32)
    if lead:
        yaw = float(T.quat2euler(orientation)[2])
        target = th.tensor([x + lead * math.cos(yaw), y + lead * math.sin(yaw), z + 0.6],
                           dtype=th.float32)
    else:
        target = th.tensor([x, y, z + 0.6], dtype=th.float32)
    return eye, look_at_quaternion(eye, target)


class ViewerRecorder:
    """Writes a video from the viewer camera as the engine ticks.

    Args:
        path: output file. Written through OmniGibson's own
            ``create_video_writer`` / ``write_video`` (PyAV, libx264, yuv420p),
            so these videos encode identically to the ones its eval pipeline
            produces. ``.mp4``.
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

    def _writer_handle(self, frame):
        """OmniGibson's own PyAV writer, so codec/pix_fmt match its videos."""
        if self._writer is None:
            from omnigibson.eval.utils.obs_utils import create_video_writer  # noqa: PLC0415

            height, width = int(frame.shape[0]), int(frame.shape[1])
            self._writer = create_video_writer(fpath=self.path, resolution=(height, width), rate=self.fps)
        return self._writer

    def capture(self) -> None:
        """Render one frame and append it. Safe to call directly."""
        # get_obs() reads the sensor's last rendered buffer, so a render has to
        # have happened this tick. env.step does not necessarily render when
        # headless, hence the explicit call.
        #
        # Twice, not once. A camera pose set just before this does not reach
        # the render pipeline until a render has been submitted, so the first
        # frame after a move comes back with the *previous* pose. That produced
        # four rounds of identical "the camera is looking at a wall" frames
        # from four completely different computed poses -- the poses were never
        # the problem, they had simply not been applied yet.
        og.sim.render()
        og.sim.render()
        frame = self.camera.get_obs()[0]["rgb"][:, :, :3].cpu().numpy()
        from omnigibson.eval.utils.obs_utils import write_video  # noqa: PLC0415

        write_video(frame[None], self._writer_handle(frame), mode="rgb")
        self.frames += 1

    def __call__(self, env_step: int) -> None:
        """``on_tick`` entry point."""
        if env_step % self.every == 0:
            self.capture()

    def close(self) -> None:
        """Finalise the file. Idempotent."""
        if self._writer is not None:
            container, stream = self._writer
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            self._writer = None
            print(f"[video] wrote {self.frames} frames to {self.path}")


class MultiViewRecorder:
    """One video per named view, all from the single viewer camera.

    OmniGibson gives a scene one viewer camera, so several viewpoints in the
    same episode means moving it, rendering, and moving it back -- which is
    what this does, once per view per captured tick. Rendering is the whole
    cost of a symbolic run (the primitives are nearly free), so N views make
    capture N times more expensive; ``every`` is the dial that pays for it.

    Args:
        views: ``{name: pose_fn}``. Each ``pose_fn()`` returns
            ``(position, orientation)`` and is called at capture time, so a
            view can follow a robot. A static shot just closes over a
            precomputed pose.
        path_for: ``name -> output path``.
        every: capture one frame every N ticks, as in ViewerRecorder.
        fps: frames per second written into each file.
    """

    def __init__(self, views, path_for, every: int = 4, fps: int = 30, focal_lengths=None):
        if not views:
            raise ValueError("MultiViewRecorder needs at least one view")
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.views = dict(views)
        #: ``{view name: focal length in mm}``. A room overview needs a wider
        #: lens than a chase shot, and the viewer camera is shared, so the
        #: focal length is swapped per view along with the pose.
        self.focal_lengths = dict(focal_lengths or {})
        self.every = int(every)
        self._writers = {
            name: ViewerRecorder(path_for(name), every=1, fps=fps) for name in self.views
        }

    @property
    def frames(self) -> int:
        """Frames written per view (they are all captured together)."""
        return next(iter(self._writers.values())).frames

    def capture(self) -> None:
        camera = og.sim.viewer_camera
        restore = camera.get_position_orientation()
        restore_focal = camera.focal_length

        # The FIRST view of every pass comes back wrong -- the scene as it was
        # before the camera moved -- while every later view in the same pass is
        # correct. Established by rendering one identical pose both first and
        # last in a pass: first gave a wall, last gave the robot.
        #
        # Three explanations were tested and none held: rendering twice per
        # view, reading the buffer twice per view, and burning one whole pass
        # at startup. So this does not try to explain it; it spends one
        # throwaway render on the bad slot, every pass, and reads nothing from
        # it. One extra render against N real ones.
        first_pose = next(iter(self.views.values()))()
        if first_pose is not None:
            camera.set_position_orientation(position=first_pose[0], orientation=first_pose[1])
            og.sim.render()
        try:
            for name, pose_fn in self.views.items():
                pose = pose_fn()
                if pose is None:
                    continue
                camera.set_position_orientation(position=pose[0], orientation=pose[1])
                focal = self.focal_lengths.get(name)
                if focal is not None and camera.focal_length != focal:
                    camera.focal_length = focal
                # every=1 on the inner recorders, so this always captures.
                self._writers[name].capture()
        finally:
            # Leave the camera as it was, so anything else reading it (a live
            # viewer, another recorder) is not silently retargeted or rezoomed.
            camera.set_position_orientation(position=restore[0], orientation=restore[1])
            if camera.focal_length != restore_focal:
                camera.focal_length = restore_focal

    def __call__(self, env_step: int) -> None:
        if env_step % self.every == 0:
            self.capture()

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
