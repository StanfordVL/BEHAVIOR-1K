"""Can trav_map + geometry replace the cuRobo oracle as the pose filter?

With N=9 robots in one scene a per-robot ``CuRoboMotionGenerator`` is out of
reach: measured at ~2.2 GB each, nine of them plus a 1.66 GB env is 21.5 GB on
a 15.4 GB card. This script checks whether the two free filters cover the same
ground, using one cuRobo generator as the reference verdict:

* **trav_map** -- the scene's baked ``floor_trav_0.png``, eroded by the robot's
  own chassis extent. Static: knows walls and the furniture baked into the map,
  knows nothing about objects added through the env config or about robots.
* **geometry** -- planar clearance from the target's AABB, plus a minimum
  separation from every other robot. Live, but blind to static geometry.

The interesting number is the disagreement: poses cuRobo rejects that the free
filters accept (holes we would be introducing) and vice versa.

Run:
    OMNIGIBSON_HEADLESS=1 python feasibility_verify/measure_pose_filters.py
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
APPLES = [("apple_0", [-1.694, -3.578, 0.036]), ("apple_1", [1.092, 0.925, 0.036])]

# Centre-to-centre. The measured R1 base radius is 0.62 m, so anything below
# ~1.24 m still lets two robots overlap -- 0.8 m was too small and let poses
# through that cuRobo rejected.
ROBOT_SEPARATION = float(os.environ.get("ROBOT_SEPARATION", 1.3))


def build_objects():
    return [
        dict(type="DatasetObject", name=name, category="apple", model="agveuv", position=position)
        for name, position in APPLES
    ]


def facing_yaw_offset(robot):
    try:
        return math.pi - float(th.mean(robot.arm_workspace_range[robot.default_arm]))
    except Exception:  # noqa: BLE001
        return math.pi


def curobo_ok(generator, robot, poses, skip_update=False):
    current = robot.get_joint_positions()
    batch = []
    for pose in poses:
        q = current.clone()
        q[robot.base_control_idx] = pose
        batch.append(q)
    invalid = generator.check_collisions(
        th.stack(batch), self_collision_check=False, skip_obstacle_update=skip_update
    ).cpu()
    return [not bool(v) for v in invalid]


class TravMapFilter:
    """Base pose is on traversable floor, eroded by the robot's chassis."""

    def __init__(self, scene, robot):
        self.trav_map = scene.trav_map
        self.eroded = self.trav_map._erode_trav_map(th.clone(self.trav_map.floor_map[0]), robot=robot)
        extent = robot.reset_joint_pos_aabb_extent[:2]
        self.radius = float(th.norm(extent)) / 2.0 + 0.2
        print(f"  [trav_map] erosion radius {self.radius:.2f} m, map {tuple(self.eroded.shape)}")

    def ok(self, xy):
        row, col = self.trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])]))
        row, col = int(row), int(col)
        if not (0 <= row < self.eroded.shape[0] and 0 <= col < self.eroded.shape[1]):
            return False
        return bool(self.eroded[row][col] == 255)


class GeometryFilter:
    """Clearance from the target's AABB plus separation from other robots."""

    def __init__(self, robot, others, margin=0.15):
        extent = robot.reset_joint_pos_aabb_extent[:2]
        self.base_radius = float(th.norm(extent)) / 2.0
        self.margin = margin
        self.others = others

    def clearance_for(self, obj):
        aabb = obj.aabb_extent[:2]
        return float(th.norm(aabb)) / 2.0 + self.base_radius + self.margin

    def ok(self, xy, obj):
        target_xy = obj.get_position_orientation()[0][:2]
        if math.hypot(float(xy[0]) - float(target_xy[0]), float(xy[1]) - float(target_xy[1])) < self.clearance_for(obj):
            return False
        for other in self.others:
            other_xy = other.get_position_orientation()[0][:2]
            if math.hypot(float(xy[0]) - float(other_xy[0]), float(xy[1]) - float(other_xy[1])) < ROBOT_SEPARATION:
                return False
        return True


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

        robot = env.robots[0]
        other = env.robots[1]
        apple_0 = env.scene.object_registry("name", "apple_0")
        apple_xy = apple_0.get_position_orientation()[0][:2]

        print("\n[setup] building filters")
        trav = TravMapFilter(env.scene, robot)
        geom = GeometryFilter(robot, [other])
        print(f"  [geometry] base radius {geom.base_radius:.2f} m, "
              f"apple clearance {geom.clearance_for(apple_0):.2f} m, robot separation {ROBOT_SEPARATION} m")
        started = time.time()
        generator = CuRoboMotionGenerator(robot, use_default_embodiment_only=True, batch_size=2)
        print(f"  [curobo] reference oracle built in {time.time() - started:.1f}s")

        # Park the other robot near apple_0 so the robot-robot case is in play.
        park = th.tensor([float(apple_xy[0]) + 0.9, float(apple_xy[1]), 0.03])
        other.set_position_orientation(position=park, orientation=th.tensor([0.0, 0.0, 0.0, 1.0]))
        for _ in range(30):
            og.sim.step()
        print(f"  parked agent_1 at {[round(float(v), 2) for v in other.get_position_orientation()[0][:2]]}")

        offset = facing_yaw_offset(robot)
        distances = [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5]
        yaws = [i * 2 * math.pi / 12 - math.pi for i in range(12)]

        print("\n=== per-distance agreement (C=cuRobo, T=trav_map, G=geometry) ===")
        print(f"{'dist':>6}  {'C ok':>5} {'T ok':>5} {'G ok':>5}   {'T&!C':>5} {'C&!T':>5} {'G&!C':>5} {'C&!G':>5}")
        totals = dict(t_not_c=0, c_not_t=0, g_not_c=0, c_not_g=0, n=0)
        for distance in distances:
            poses = [
                th.tensor(
                    [
                        float(apple_xy[0]) + distance * math.cos(yaw),
                        float(apple_xy[1]) + distance * math.sin(yaw),
                        yaw + offset,
                    ],
                    dtype=th.float32,
                )
                for yaw in yaws
            ]
            c = curobo_ok(generator, robot, poses)
            t = [trav.ok(p[:2]) for p in poses]
            g = [geom.ok(p[:2], apple_0) for p in poses]
            t_not_c = sum(1 for i in range(len(poses)) if t[i] and not c[i])
            c_not_t = sum(1 for i in range(len(poses)) if c[i] and not t[i])
            g_not_c = sum(1 for i in range(len(poses)) if g[i] and not c[i])
            c_not_g = sum(1 for i in range(len(poses)) if c[i] and not g[i])
            totals["t_not_c"] += t_not_c
            totals["c_not_t"] += c_not_t
            totals["g_not_c"] += g_not_c
            totals["c_not_g"] += c_not_g
            totals["n"] += len(poses)
            print(
                f"{distance:>6.2f}  {sum(c):>5} {sum(t):>5} {sum(g):>5}   "
                f"{t_not_c:>5} {c_not_t:>5} {g_not_c:>5} {c_not_g:>5}"
            )

        print(f"\n  over {totals['n']} candidate poses:")
        print(f"    trav_map accepts but cuRobo rejects : {totals['t_not_c']}  <- holes trav_map alone would leave")
        print(f"    cuRobo accepts but trav_map rejects : {totals['c_not_t']}  <- poses trav_map needlessly drops")
        print(f"    geometry accepts but cuRobo rejects : {totals['g_not_c']}")
        print(f"    cuRobo accepts but geometry rejects : {totals['c_not_g']}")

        print("\n=== union filter (trav_map AND geometry) vs cuRobo ===")
        both_miss = 0
        for distance in distances:
            poses = [
                th.tensor(
                    [
                        float(apple_xy[0]) + distance * math.cos(yaw),
                        float(apple_xy[1]) + distance * math.sin(yaw),
                        yaw + offset,
                    ],
                    dtype=th.float32,
                )
                for yaw in yaws
            ]
            c = curobo_ok(generator, robot, poses)
            for i, pose in enumerate(poses):
                free = trav.ok(pose[:2]) and geom.ok(pose[:2], apple_0)
                if free and not c[i]:
                    both_miss += 1
        print(f"  poses the free filters accept but cuRobo rejects: {both_miss} / {totals['n']}")
        print("  (these are the real regressions from dropping cuRobo)")
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if og.sim is not None:
            og.shutdown()


if __name__ == "__main__":
    main()
