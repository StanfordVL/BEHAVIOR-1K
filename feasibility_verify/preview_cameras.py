"""Render one frame per camera view and stop. No episode, no LLM.

Framing cannot be checked by reading pose numbers -- twice now a pose that
looked reasonable turned out to be filming the ceiling from inside the roof
void, and once a wall. The only way to know is to look at a frame, so this
makes looking cheap: one env load instead of a whole run.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/preview_cameras.py \
        --out /tmp/camera_preview
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/camera_preview", help="directory for the PNGs")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene", default="Pomaria_1_int")
    parser.add_argument("--room", default="living_room_0")
    parser.add_argument("--bddl-activity", default="coop_two_apples_pomaria")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import omnigibson as og
    from PIL import Image

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv
    from coop2.behavior_env.recording import WIDE_FOCAL_LENGTH

    os.makedirs(args.out, exist_ok=True)

    env = CooperativeBehaviorEnv(
        n_agents=args.agents,
        seed=args.seed,
        scene_model=args.scene,
        room=args.room,
        bddl_activity=args.bddl_activity or None,
        # A path is what turns the viewer camera on; nothing is written because
        # the recorder is replaced below before any tick happens.
        video_path=os.path.join(args.out, "unused.mp4"),
        length=40000,
    )
    env.reset()

    print("\n--- geometry the framing is derived from ---")
    for robot in env.env.robots:
        position = robot.get_position_orientation()[0]
        print(f"  {robot.name:10s} ({float(position[0]):+.2f}, {float(position[1]):+.2f}, {float(position[2]):+.2f})")
    task = getattr(env.env, "task", None)
    for name, entity in (getattr(task, "object_scope", None) or {}).items():
        if entity is None or hasattr(entity, "controllers"):
            continue
        position = entity.get_position_orientation()[0]
        print(f"  {name:24s} ({float(position[0]):+.2f}, {float(position[1]):+.2f}, {float(position[2]):+.2f})")

    # Where can the camera actually stand? Every framing failure so far came
    # from inferring the room's extent from object positions, which says
    # nothing about walls: poses landed behind the west wall, through the east
    # wall, and above the ceiling.
    from coop2.behavior_env.placement import sample_free_points

    cells = sample_free_points(
        env.env.scene, env.env.robots[0], count=400, room=args.room, max_attempts=4000
    )
    if cells:
        cxs = [c[0] for c in cells]
        cys = [c[1] for c in cells]
        print(f"\n--- {args.room} traversable extent, from {len(cells)} cells ---")
        print(f"  x {min(cxs):+.2f} .. {max(cxs):+.2f}   y {min(cys):+.2f} .. {max(cys):+.2f}")

    recorder = env.recorder
    if recorder is None:
        print("no recorder; nothing to preview")
        return 1

    # Controls. If a robot does not appear in a shot taken from three metres
    # directly at it, the problem is not the framing rule.
    import torch as th

    from coop2.behavior_env.placement import look_at_quaternion

    def straight_at(robot, dx, dy, dz):
        base = robot.get_position_orientation()[0]
        x, y, z = (float(v) for v in base)
        eye = th.tensor([x + dx, y + dy, z + dz], dtype=th.float32)
        target = th.tensor([x, y, z + 0.6], dtype=th.float32)
        return eye, look_at_quaternion(eye, target)

    # Same view, rendered last instead of first. Order is the only variable
    # left: identical poses have produced a wall (first in the loop) and a
    # correct frame (fourth in the loop).
    first_name, first_fn = next(iter(recorder.views.items()))
    recorder.views[f"zz_{first_name}_rendered_last"] = first_fn

    for robot in env.env.robots:
        recorder.views[f"control_{robot.name}_side"] = (
            lambda robot=robot: straight_at(robot, 2.5, 0.0, 1.2)
        )
        recorder.views[f"control_{robot.name}_top"] = (
            lambda robot=robot: straight_at(robot, 0.01, 0.0, 2.3)
        )

    print("\n--- rendering one frame per view ---")
    camera = og.sim.viewer_camera
    for name, pose_fn in recorder.views.items():
        pose = pose_fn()
        camera.set_position_orientation(position=pose[0], orientation=pose[1])
        camera.focal_length = WIDE_FOCAL_LENGTH
        # Experiment: does the buffer lag by one *read* rather than one render?
        # Grab and discard, then grab for real, and save both to compare.
        og.sim.render()
        stale = camera.get_obs()[0]["rgb"][:, :, :3].cpu().numpy()
        og.sim.render()
        frame = camera.get_obs()[0]["rgb"][:, :, :3].cpu().numpy()
        Image.fromarray(stale).save(os.path.join(args.out, f"{name}_first_read.png"))
        import numpy as _np
        print(f"    first-read vs second-read differ: {not _np.array_equal(stale, frame)}")
        path = os.path.join(args.out, f"{name}.png")
        Image.fromarray(frame).save(path)
        print(f"  {name:10s} eye=({float(pose[0][0]):+.2f}, {float(pose[0][1]):+.2f}, "
              f"{float(pose[0][2]):+.2f})  ->  {path}")

    env.recorder = None  # nothing was written; skip the encoder flush
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
