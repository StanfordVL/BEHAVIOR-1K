"""How many robots can stand around one target at once?

`NO_SPACE_AROUND_TARGET` fired 57 times in a row in one episode: both agents
kept failing to reach the second apple. That is either an agent problem or a
geometric one, and the two need different fixes -- a prompt rule cannot create
floor space. This counts, per candidate pose, which filter rejected it, with
and without a teammate parked at the target.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/measure_target_capacity.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Pomaria_1_int")
    parser.add_argument("--room", default="living_room_0")
    parser.add_argument("--bddl-activity", default="coop_two_apples_pomaria")
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import torch as th

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv

    env = CooperativeBehaviorEnv(
        n_agents=2, seed=0, scene_model=args.scene, room=args.room,
        bddl_activity=args.bddl_activity or None, length=40000,
    )
    env.reset()

    scope = getattr(env.env.task, "object_scope", None) or {}
    controller = env.controllers["agent_0"]
    robots = list(env.env.robots)

    def survey(target, label):
        """Reject-reason histogram over @args.samples candidate poses."""
        target_xy = [float(v) for v in target.get_position_orientation()[0][:2]]
        lo, hi = controller.sampling_range_for(target)
        target_rooms = [controller._room_of(target_xy)]
        counts = {"accepted": 0, "room": 0, "trav": 0, "robots": 0}
        for _ in range(args.samples):
            distance = th.rand(1).item() * (hi - lo) + lo
            yaw = th.rand(1).item() * 2.0 * math.pi - math.pi
            xy = th.tensor([target_xy[0] + distance * math.cos(yaw),
                            target_xy[1] + distance * math.sin(yaw)], dtype=th.float32)
            if controller._room_of(xy) not in target_rooms:
                counts["room"] += 1
            elif not controller._is_traversable(xy):
                counts["trav"] += 1
            elif not controller._clear_of_other_robots(xy):
                counts["robots"] += 1
            else:
                counts["accepted"] += 1
        total = float(args.samples)
        print(f"\n{label}")
        print(f"  target at ({target_xy[0]:+.2f}, {target_xy[1]:+.2f})  "
              f"sampling range [{lo:.2f}, {hi:.2f}] m  room={target_rooms[0]}")
        for key in ("accepted", "room", "trav", "robots"):
            print(f"    {key:9s} {counts[key]:5d}  {counts[key] / total * 100:5.1f}%")
        return counts["accepted"]

    print(f"robot radius: {controller.robot_radius if hasattr(controller, 'robot_radius') else '?'}")
    for name in ("apple.n.01_1", "apple.n.01_2", "coffee_table.n.01_1"):
        target = scope.get(name)
        if target is None:
            continue
        survey(target, f"--- {name}, both robots at their start poses ---")

    # Now park agent_1 right next to the coffee table, which is the state the
    # failing episode was in: one robot delivered its apple and stayed there.
    table = scope.get("coffee_table.n.01_1")
    if table is not None and len(robots) > 1:
        table_xy = [float(v) for v in table.get_position_orientation()[0][:2]]
        lo, hi = controller.sampling_range_for(table)
        parked = th.tensor([table_xy[0] + lo, table_xy[1], 0.0], dtype=th.float32)
        robots[1].set_position_orientation(
            position=parked, orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
        )
        for _ in range(30):
            env.env.step(env.engine.idle_action_dict())
        print(f"\nparked {robots[1].name} at ({float(parked[0]):+.2f}, {float(parked[1]):+.2f}), "
              f"{lo:.2f} m from the table")
        for name in ("apple.n.01_2", "coffee_table.n.01_1"):
            target = scope.get(name)
            if target is not None:
                survey(target, f"--- {name}, with a teammate parked at the table ---")

    # The state the failing episode was actually in: the target apple was in a
    # teammate's gripper, so its position coincides with that teammate's. The
    # separation filter then rejects poses near the target *because* the target
    # is where the robot is -- a case no amount of free floor can help.
    apple = scope.get("apple.n.01_2")
    if apple is not None and len(robots) > 1:
        apple_xy = [float(v) for v in apple.get_position_orientation()[0][:2]]
        robots[0].set_position_orientation(
            position=th.tensor([apple_xy[0], apple_xy[1], 0.0], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
        for _ in range(30):
            env.env.step(env.engine.idle_action_dict())
        print(f"\nmoved {robots[0].name} onto apple.n.01_2 at "
              f"({apple_xy[0]:+.2f}, {apple_xy[1]:+.2f}) -- the 'teammate is carrying it' case")
        survey(apple, "--- apple.n.01_2, co-located with a teammate ---")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
