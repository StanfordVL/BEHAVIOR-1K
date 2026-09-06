"""Can we use upstream's collision check as the symbolic sampler's pose filter?

``NavigableSymbolicActionPrimitives`` drops ``_validate_poses`` -- the cuRobo
stage -- and therefore accepts base poses at any distance, including on top of
the target. Restoring the upstream check means building a
``CuRoboMotionGenerator`` per robot purely as a collision oracle. This script
measures whether that is affordable and whether it actually works:

1. Wall-clock and VRAM to build one DEFAULT-embodiment-only generator per robot
   (``check_collisions`` only ever uses ``CuRoboEmbodimentSelection.DEFAULT``).
2. A distance sweep around apple_0: which candidate base poses does
   ``check_collisions`` reject? This gives the *effective* minimum distance the
   oracle enforces, to compare against a hand-picked geometric lower bound.
3. Whether a pose overlapping the other robot is rejected -- the failure mode
   that produced ``agent_0: holding=agent_1`` in the --no-contention control.

Run:
    OMNIGIBSON_HEADLESS=1 python feasibility_verify/measure_curobo_collision_oracle.py
"""

from __future__ import annotations

import math
import os
import sys
import time
import traceback

import torch as th

import omnigibson as og
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
from omnigibson.macros import gm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.behavior_env.env_setup import assert_multi_robot_sanity, build_multi_robot_config, prepare_robots

ROBOT_MODEL = "R1"
SCENE_MODEL = "Rs_int"
LOAD_OBJECT_CATEGORIES = ["floors", "walls", "coffee_table"]
ROBOT_POSES = [
    ([-1.37, 0.536, 0.033], [0.0, 0.0, 0.0, 1.0]),
    ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
]
AGENT_IDS = ["agent_0", "agent_1"]
APPLES = [
    ("apple_0", [-1.694, -3.578, 0.036]),
    ("apple_1", [1.092, 0.925, 0.036]),
]
# cuRobo's batch_size drives the trajopt tensors allocated during mg.warmup().
# batch_size=16 OOMs a 16 GB card (13.25 GiB allocated) before the first
# generator finishes. check_collisions takes an (N, D) q of any N, so the
# trajopt batch size does not bound how many candidate poses we can test.
BATCH = int(os.environ.get('CUROBO_BATCH', 2))


def build_objects():
    return [
        dict(type="DatasetObject", name=name, category="apple", model="agveuv", position=position)
        for name, position in APPLES
    ]


def vram_used_gb():
    free, total = th.cuda.mem_get_info()
    return (total - free) / 1024**3


def candidate_at(target_xy, distance, yaw, robot):
    """One (x, y, yaw) base pose at @distance from @target_xy, facing it.

    Same geometry as the sampler: polar offset, then the upstream facing
    correction ``yaw + pi - mean(arm_workspace_range)``.
    """
    try:
        offset = math.pi - float(th.mean(robot.arm_workspace_range[robot.default_arm]))
    except Exception:  # noqa: BLE001
        offset = math.pi
    return th.tensor(
        [
            float(target_xy[0]) + distance * math.cos(yaw),
            float(target_xy[1]) + distance * math.sin(yaw),
            yaw + offset,
        ],
        dtype=th.float32,
    )


def check(generator, robot, poses, skip_update=False):
    """Run check_collisions on a list of 2d base poses. True = collision-free."""
    current = robot.get_joint_positions()
    batch = []
    for pose in poses:
        q = current.clone()
        q[robot.base_control_idx] = pose
        batch.append(q)
    invalid = generator.check_collisions(
        th.stack(batch), self_collision_check=False, skip_obstacle_update=skip_update
    ).cpu()
    return ~invalid


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = False
    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False

    try:
        config = build_multi_robot_config(
            robot_poses=ROBOT_POSES,
            robot_model=ROBOT_MODEL,
            scene_model=SCENE_MODEL,
            load_object_categories=LOAD_OBJECT_CATEGORIES,
            objects=build_objects(),
            agent_names=AGENT_IDS,
        )
        started = time.time()
        env = og.Environment(configs=config)
        print(f"[timing] environment loaded in {time.time() - started:.1f}s")
        prepare_robots(env)
        assert_multi_robot_sanity(env, expected_robots=len(ROBOT_POSES))

        print(f"\n[vram] before any generator: {vram_used_gb():.2f} GB used (batch_size={BATCH})")

        generators = {}
        for agent_id, robot in zip(AGENT_IDS, env.robots):
            before_mem, before_t = vram_used_gb(), time.time()
            generators[agent_id] = CuRoboMotionGenerator(
                robot, use_default_embodiment_only=True, batch_size=BATCH
            )
            print(
                f"[timing] {agent_id}: DEFAULT-only generator built in {time.time() - before_t:.1f}s, "
                f"VRAM {before_mem:.2f} -> {vram_used_gb():.2f} GB"
            )

        agent_0, agent_1 = AGENT_IDS
        robot_0 = env.robots_by_name[agent_0] if hasattr(env, "robots_by_name") else env.robots[0]
        apple_0 = env.scene.object_registry("name", "apple_0")
        apple_xy = apple_0.get_position_orientation()[0][:2]

        print("\n=== TEST 1: distance sweep around apple_0 (is a near pose rejected?) ===")
        print("Each row: 8 yaws evenly spaced; 'ok' = collision-free per check_collisions.")
        distances = [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5]
        yaws = [i * 2 * math.pi / 8 - math.pi for i in range(8)]
        for distance in distances:
            poses = [candidate_at(apple_xy, distance, yaw, robot_0) for yaw in yaws]
            valid = check(generators[agent_0], robot_0, poses)
            marks = "".join("o" if bool(v) else "." for v in valid)
            print(f"  d={distance:>4.2f} m  [{marks}]  {int(valid.sum())}/8 collision-free")

        print("\n=== TEST 2: is the OTHER ROBOT seen as an obstacle? ===")
        # Park agent_1 right next to apple_0, then ask agent_0's oracle about
        # poses on top of agent_1. If the other robot is in the collision world
        # these must all come back as colliding.
        other = env.robots[1]
        park_xy = (float(apple_xy[0]) + 0.7, float(apple_xy[1]))
        other.set_position_orientation(
            position=th.tensor([park_xy[0], park_xy[1], 0.03]), orientation=th.tensor([0.0, 0.0, 0.0, 1.0])
        )
        for _ in range(30):
            og.sim.step()
        moved = other.get_position_orientation()[0]
        print(f"  parked {agent_1} at {[round(float(v), 3) for v in moved[:2]]}")
        on_top = [candidate_at(moved[:2], 0.0, yaw, robot_0) for yaw in yaws[:4]]
        valid = check(generators[agent_0], robot_0, on_top)
        marks = "".join("o" if bool(v) else "." for v in valid)
        print(f"  poses ON {agent_1}: [{marks}]  {int(valid.sum())}/4 collision-free")
        print(f"  -> other robot {'IS' if int(valid.sum()) == 0 else 'is NOT'} an obstacle in the collision world")

        print("\n=== TEST 3: cost of one oracle query ===")
        poses = [candidate_at(apple_xy, 1.0, yaw, robot_0) for yaw in yaws]
        check(generators[agent_0], robot_0, poses)  # warm
        started = time.time()
        for _ in range(10):
            check(generators[agent_0], robot_0, poses)
        with_update = (time.time() - started) / 10
        started = time.time()
        for _ in range(10):
            check(generators[agent_0], robot_0, poses, skip_update=True)
        without_update = (time.time() - started) / 10
        print(f"  {len(poses)} candidates, with update_obstacles:    {with_update * 1000:.0f} ms")
        print(f"  {len(poses)} candidates, without update_obstacles: {without_update * 1000:.0f} ms")

        print(f"\n[vram] final: {vram_used_gb():.2f} GB used of {th.cuda.mem_get_info()[1] / 1024**3:.1f} GB")
    except BaseException:
        # og.shutdown() below ends in app.close(), which terminates the process
        # while the exception is still propagating -- so the interpreter never
        # gets to print it. Print it here or the run looks like a silent exit 0.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if og.sim is not None:
            og.shutdown()


if __name__ == "__main__":
    main()
