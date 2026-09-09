"""Does teleporting an R1 across the house exceed its virtual base joints?

Reported symptoms: agent_1 vanishes on its first teleport, and agent_0 topples
and convulses partway through an episode. Both are consistent with one cause and
it is worth measuring rather than arguing about.

A holonomic base does not move its root prim. ``Robot.set_position_orientation``
computes the target pose *relative to the root link* and writes it into the six
1-DoF virtual base joints (x, y, z, rx, ry, rz) with ``drive=False``
(``robots/robot.py``). The root prim stays wherever the robot was spawned. So the
robot's world pose is an offset carried entirely by prismatic joints -- and those
joints have finite limits.

coop2 spawns robots at ``[1.5 * i, 0, 0.05]`` (env_setup) and then ``place_robots``
teleports them into the task room, which for Pomaria_1_int/living_room_0 is around
(-10, -2): a ~10 m offset that the base joints must absorb. If a joint's limit is
smaller than that, the joint saturates, the robot does not arrive where it was
sent (looks like vanishing) and the solver fights a joint pinned at its stop
(looks like convulsing).

This prints, per robot, the root anchor, the commanded pose, the pose actually
reached, and every base joint against its own limits -- at build time and after a
long teleport.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/measure_base_joint_limits.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Pomaria_1_int")
    parser.add_argument("--room", default="living_room_0")
    parser.add_argument("--bddl-activity", default="coop_two_apples_pomaria")
    parser.add_argument("--settle", type=int, default=120)
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import torch as th

    import omnigibson as og

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv

    env = CooperativeBehaviorEnv(
        n_agents=2, seed=0, scene_model=args.scene, room=args.room,
        bddl_activity=args.bddl_activity or None, length=40000,
    )
    env.reset()

    def report(label):
        print(f"\n===== {label}")
        for robot in env.env.robots:
            world = robot.get_position_orientation()[0]
            root = robot.root_link.get_position_orientation()[0]
            print(f"  {robot.name}: world=({float(world[0]):+.3f}, {float(world[1]):+.3f}, "
                  f"{float(world[2]):+.3f})  root_anchor=({float(root[0]):+.3f}, "
                  f"{float(root[1]):+.3f}, {float(root[2]):+.3f})")
            print(f"    holonomic_base={robot.is_holonomic_base}  base_idx={list(robot.base_idx)}")
            positions = robot.get_joint_positions()
            for component in ("x", "y", "z", "rx", "ry", "rz"):
                name = f"base_footprint_{component}_joint"
                joint = robot.joints.get(name)
                if joint is None:
                    continue
                index = int(joint.dof_indices[0])
                value = float(positions[index])
                low, high = float(joint.lower_limit), float(joint.upper_limit)
                span = high - low
                # How close is this joint to its stop, as a fraction of its range?
                if span > 0:
                    headroom = min(value - low, high - value)
                    flag = "  <-- AT ITS LIMIT" if headroom <= 1e-3 else (
                        "  <-- near limit" if headroom < 0.05 * span else "")
                else:
                    flag = ""
                print(f"    {name:<28} = {value:+9.3f}   limits [{low:+.3f}, {high:+.3f}]{flag}")

    def placement_sanity(label):
        """Is each robot level, on the floor, in its room, and touching nothing?

        A robot placed intersecting furniture starts tilted and then topples,
        and a toppled robot with a position-mode base controller fights its own
        joint targets -- which is what "convulsing" looks like.
        """
        import math

        from omnigibson.utils.usd_utils import RigidContactAPI

        from coop2.behavior_env.placement import room_of, robot_radius

        print(f"\n***** placement sanity: {label}")
        scene = env.env.scene
        for robot in env.env.robots:
            position, orientation = robot.get_position_orientation()
            xy = (float(position[0]), float(position[1]))
            positions = robot.get_joint_positions()
            tilts = {}
            for component in ("rx", "ry"):
                joint = robot.joints.get(f"base_footprint_{component}_joint")
                if joint is not None:
                    tilts[component] = math.degrees(float(positions[int(joint.dof_indices[0])]))
            tilt = max(abs(v) for v in tilts.values()) if tilts else 0.0
            room = room_of(scene, xy)
            print(f"  {robot.name}: xy=({xy[0]:+.3f}, {xy[1]:+.3f}) z={float(position[2]):+.3f} "
                  f"room={room} tilt={tilt:.1f} deg"
                  + ("   <-- NOT LEVEL" if tilt > 2.0 else ""))

            touching = []
            for obj in scene.objects:
                if obj is robot or getattr(obj, "name", "") in {r.name for r in env.env.robots}:
                    continue
                other = obj.get_position_orientation()[0]
                if float(th.norm(other[:2] - position[:2])) > 2.5:
                    continue
                try:
                    hit = RigidContactAPI.is_in_contact(
                        scene_idx=scene.idx, query_set=[robot], with_set=[obj],
                        ignore_set=None, current_only=False,
                    )
                except Exception:
                    continue
                if hit:
                    touching.append(obj.name)
            floor_only = all(t.startswith("floors") for t in touching)
            print(f"    touching: {touching or 'nothing'}"
                  + ("" if floor_only else "   <-- INTERSECTING FURNITURE"))

        pair = [r.get_position_orientation()[0][:2] for r in env.env.robots]
        if len(pair) == 2:
            gap = float(th.norm(pair[0] - pair[1]))
            need = robot_radius(env.env.robots[0]) * 2
            print(f"  robot separation {gap:.3f} m (needs >= {need:.3f} m)"
                  + ("   <-- TOO CLOSE" if gap < need else ""))

    report("after build + place_robots (this is where agent_1 'vanished')")
    placement_sanity("straight after build")

    # Now the thing a navigate does: teleport to a pose near the task, which is
    # what _navigate_to_pose ends up calling.
    task = getattr(env.env.task, "object_scope", None) or {}
    table = task.get("coffee_table.n.01_1")
    table = getattr(table, "wrapped_obj", table)
    if table is None:
        print("FAIL: no coffee_table.n.01_1 in the object scope")
        return 1
    target_xy = table.get_position_orientation()[0][:2]

    for index, robot in enumerate(env.env.robots):
        # A metre out from the table on opposite sides, i.e. a normal navigate.
        offset = th.tensor([1.2 if index == 0 else -1.2, 0.0])
        goal = th.tensor([float(target_xy[0] + offset[0]), float(target_xy[1] + offset[1]), 0.05])
        print(f"\n[teleport] {robot.name} -> ({float(goal[0]):+.3f}, {float(goal[1]):+.3f})")
        robot.set_position_orientation(position=goal, orientation=th.tensor([0.0, 0.0, 0.0, 1.0]))
        reached = robot.get_position_orientation()[0]
        error = float(th.norm(reached[:2] - goal[:2]))
        print(f"           reached ({float(reached[0]):+.3f}, {float(reached[1]):+.3f}), "
              f"xy error {error:.3f} m" + ("   <-- DID NOT ARRIVE" if error > 0.05 else ""))

    report("immediately after the teleports")

    for _ in range(args.settle):
        og.sim.step()
    report(f"after {args.settle} settle steps (this is where the convulsing shows)")

    print("\nInterpretation: a base joint pinned at its stop means the robot's world "
          "pose is not the one it was sent to, and the solver is fighting a joint it "
          "cannot satisfy. If every joint has headroom, the cause is elsewhere.")
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
