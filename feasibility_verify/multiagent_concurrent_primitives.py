"""Two R1 robots moving at the same time in Rs_int, on real physics primitives.

M1+M2 demo for the COOP2 port. The claim under test is narrow and mechanical:

    two robots' StarterSemanticActionPrimitives generators can be interleaved
    into one env.step({robot_name: action}) per tick, so both robots are
    actually in motion during the same simulator steps.

The script does not just assert that -- it **measures** it. Every tick the
engine records which agents contributed a real (non-idle) action in
``engine.last_advanced``; the monitor below turns that into an overlap ratio
and a per-agent Gantt chart, so "they moved together" is a number and a
picture rather than an impression.

What "simultaneous" does and does not mean here
----------------------------------------------
* **Simulation time: genuinely simultaneous.** Both agents' actions go into
  the same ``env.step``, so physics advances both robots on the same tick.
* **Wall clock: partly serialized, unavoidably.** cuRobo planning runs
  *inside* the generator body, so the ``next()`` that starts a sub-motion
  blocks for seconds while it plans. Two agents therefore plan one after the
  other in wall clock. Threading is not an option (Isaac/cuRobo are not
  thread-safe; COOP2 uses threads only for LLM calls). This costs wall clock,
  not simultaneity.
* **Collision avoidance between the robots: best effort only.** cuRobo
  snapshots the collision world at plan time and ``_execute_motion_plan`` then
  runs for thousands of ticks without re-checking, so each robot is planning
  against where its teammate *was*. ``--obstacle-refresh-every`` refreshes the
  snapshot between sub-motions; ``--mode exclusive`` removes the problem
  entirely and is the control condition.

No LLM: the plans are scripted, so the only thing under test is the execution
layer. ``run_plans`` is a deliberate miniature of COOP2's
``PlanningEnvWrapper.step`` loop minus the ``all(agent.ready)`` barrier (there
is nothing to wait for without an LLM): each agent walks its own primitive
sequence, gets the next primitive as soon as its previous one ends, and a
failed primitive kills the rest of that agent's plan.

Run:
    python feasibility_verify/multiagent_concurrent_primitives.py
    python feasibility_verify/multiagent_concurrent_primitives.py --mode exclusive
    python feasibility_verify/multiagent_concurrent_primitives.py --plan navigate_then_grasp
    python feasibility_verify/multiagent_concurrent_primitives.py --plan asymmetric
    python feasibility_verify/multiagent_concurrent_primitives.py --contend
    python feasibility_verify/multiagent_concurrent_primitives.py --trace-every 200 --keep-viewer

Reading the output:
    * ``overlap ratio`` -- fraction of busy ticks with >= 2 agents moving.
      Concurrent mode on two independent NAVIGATE_TO primitives should be
      high (the primitives are long and start together); exclusive mode must
      be exactly 0.00.
    * the Gantt rows -- each agent's primitives laid out over tick ranges.
      Overlapping bars are the visual proof.
    * ``reason_code`` on failures -- mostly EXECUTION means keep loosening the
      tracking tolerances in ``tune_primitive_macros``; mostly PLANNING /
      SAMPLING means the spawn poses or the room assignment are wrong (check
      the room each robot reports at startup).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import torch as th

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitiveSet as Primitive,
)
from omnigibson.macros import gm

# Allow running this file directly from the repo root without installing coop2.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.behavior_env.env_setup import (  # noqa: E402
    assert_multi_robot_sanity,
    build_multi_robot_config,
    prepare_robots,
    tune_primitive_macros,
)
from coop2.behavior_env.primitive_engine import (  # noqa: E402
    MotionMode,
    MultiAgentPrimitiveEngine,
    PrimitiveOutcome,
)

ROBOT_MODEL = "R1"
SCENE_MODEL = "Rs_int"

# Keep the collision world small so cuRobo planning stays responsive, exactly
# as the single-robot baseline does.
LOAD_OBJECT_CATEGORIES = ["floors", "walls", "coffee_table"]

# Spawn poses, (position_xyz, orientation_xyzw).
# NOTE: the first thing to tune if a robot spawns inside geometry or in the
# wrong room. The sanity check prints each robot's room at startup, and
# ``_sample_pose_near_object`` only accepts base poses in the SAME room as the
# target, so a robot in the wrong room can never reach its target.
ROBOT_POSES = [
    ([-1.37, 0.536, 0.033], [0.0, 0.0, 0.0, 1.0]),
    ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
]
AGENT_IDS = ["agent_0", "agent_1"]

# Apple poses captured from the settled objects in the viewer. They are far
# enough apart that two robots navigating to them do not occupy the same spot.
# apple_0 is the contested target (--contend points both agents at it). Its old
# spot, (-1.694, -3.578), was doubly bad: outside the viewer camera frustum, so
# the contention never showed up on video, and on a patch the segmentation map
# assigns to NO room, which made the sampler's same-room filter degenerate to
# `None in [None]` -- it accepted every candidate unchecked.
#
# (-0.50, -1.25) is in living_room_0, inside the camera frustum, has the highest
# standable-pose fraction of any visible in-room point in this scene (23%), and
# sits 1.99 m from agent_0 against 1.95 m from agent_1 -- near enough to equal
# that the race is real rather than decided by the start positions.
# apple_1 is left alone: it is already in living_room_0 and already on camera.
APPLES = [
    ("apple_0", [-0.5, -1.25, 0.036]),
    ("apple_1", [1.092, 0.925, 0.036]),
]

VIEWER_CAMERA_POSITION = th.tensor([1.8294, -3.2502, 1.6885])
VIEWER_CAMERA_ORIENTATION = th.tensor([0.5770, 0.1719, 0.2280, 0.7652])
VIEWER_CAMERA_FOCAL_LENGTH = 17.0
VIEWER_CAMERA_HORIZONTAL_APERTURE = 20.995


def configure_viewer_camera() -> None:
    """Apply the final viewer pose only after all controller setup has rendered."""
    # Controller construction can trigger deferred Kit / viewport updates.
    # Flush those before fixing the final camera projection and pose.
    for _ in range(2):
        og.sim.render()

    camera = og.sim.viewer_camera
    camera.focal_length = VIEWER_CAMERA_FOCAL_LENGTH
    camera.horizontal_aperture = VIEWER_CAMERA_HORIZONTAL_APERTURE
    camera.set_position_orientation(
        position=VIEWER_CAMERA_POSITION,
        orientation=VIEWER_CAMERA_ORIENTATION,
    )

    # USD camera changes can take two renders to reach the viewport. Complete
    # that synchronization before the robots receive their first action.
    for _ in range(2):
        og.sim.render()
    camera.focal_length = VIEWER_CAMERA_FOCAL_LENGTH
    camera.horizontal_aperture = VIEWER_CAMERA_HORIZONTAL_APERTURE
    camera.set_position_orientation(
        position=VIEWER_CAMERA_POSITION,
        orientation=VIEWER_CAMERA_ORIENTATION,
    )
    og.sim.enable_viewer_camera_teleoperation()


def build_objects():
    return [
        {
            "type": "DatasetObject",
            "name": name,
            "category": "apple",
            "model": "agveuv",
            "position": position,
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
        for name, position in APPLES
    ]


# ---------------------------------------------------------------------------
# Concurrency measurement
# ---------------------------------------------------------------------------


class ConcurrencyMonitor:
    """Counts, per tick, how many agents actually moved.

    Hooked in as ``engine.on_tick``. Reads ``engine.last_advanced``, which the
    engine fills with the agents that contributed a non-idle action to the
    tick that just executed.
    """

    def __init__(self, engine: MultiAgentPrimitiveEngine, trace_every: int = 0):
        self.engine = engine
        self.trace_every = trace_every
        #: number of agents moving -> tick count
        self.histogram: Counter = Counter()
        #: agent id -> ticks in which it moved
        self.per_agent: Counter = Counter()

    def __call__(self, env_step: int) -> None:
        advanced = self.engine.last_advanced
        self.histogram[len(advanced)] += 1
        for agent_id in advanced:
            self.per_agent[agent_id] += 1

        if self.trace_every and env_step % self.trace_every == 0:
            parts = []
            for agent_id in self.engine.agent_ids:
                robot = self.engine.robots_by_id[agent_id]
                position, _ = robot.get_position_orientation()
                moving = "*" if agent_id in advanced else " "
                primitive = self.engine.active_primitive(agent_id) or "-"
                parts.append(
                    f"{moving}{agent_id}[{primitive}] "
                    f"({position[0]:+.2f},{position[1]:+.2f})"
                )
            print(f"[trace {env_step:>6}] " + "  ".join(parts))

    @property
    def busy_ticks(self) -> int:
        """Ticks in which at least one agent moved."""
        return sum(count for n, count in self.histogram.items() if n >= 1)

    @property
    def overlap_ticks(self) -> int:
        """Ticks in which at least two agents moved."""
        return sum(count for n, count in self.histogram.items() if n >= 2)

    @property
    def overlap_ratio(self) -> float:
        return self.overlap_ticks / self.busy_ticks if self.busy_ticks else 0.0


# ---------------------------------------------------------------------------
# Plans and the mini-L3 loop
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=list(MotionMode.ALL),
        default=MotionMode.CONCURRENT,
        help="concurrent: advance every active generator each tick (both robots move together). "
        "exclusive: only the earliest-assigned agent advances, the rest hold. Control condition; "
        "overlap ratio must be 0.00.",
    )
    parser.add_argument(
        "--plan",
        choices=["navigate", "navigate_then_grasp", "grasp", "asymmetric"],
        default="navigate",
        help="navigate (default): the cleanest simultaneity test -- two long, independent base motions. "
        "navigate_then_grasp: the full pick sequence. "
        "asymmetric: agent_0 gets two primitives, agent_1 one, showing agents are not aligned at "
        "primitive boundaries.",
    )
    parser.add_argument(
        "--contend",
        action="store_true",
        help="Point both agents at apple_0. There is no target lock by design -- contention is a "
        "cooperation problem for the topology layer -- so expect the loser to burn a full motion and "
        "fail with POST_CONDITION.",
    )
    parser.add_argument(
        "--attempts", type=int, default=1, help="apply_ref attempts (keep at 1; retries are not idempotent)."
    )
    parser.add_argument("--curobo-batch-size", type=int, default=1)
    parser.add_argument(
        "--obstacle-refresh-every",
        type=int,
        default=0,
        help="Re-snapshot each active cuRobo collision world every N ticks (best effort; it cannot "
        "re-route a trajectory already in flight). 0 disables.",
    )
    parser.add_argument("--max-ticks-per-primitive", type=int, default=20000)
    parser.add_argument("--max-ticks", type=int, default=100000, help="Global tick budget for the whole run.")
    parser.add_argument(
        "--trace-every",
        type=int,
        default=0,
        help="Print both robots' base positions and which of them is moving, every N ticks.",
    )
    parser.add_argument("--keep-viewer", action="store_true", help="Keep the viewer open after the run.")
    parser.add_argument("--headless", action="store_true", help="No viewer at all (faster).")
    return parser.parse_args()


def build_plans(args) -> Dict[str, List[Tuple[Primitive, str]]]:
    """One sequence of high-level primitives per agent."""
    targets = ["apple_0", "apple_0"] if args.contend else [name for name, _ in APPLES]
    per_agent = dict(zip(AGENT_IDS, targets))

    if args.plan == "navigate":
        return {agent: [(Primitive.NAVIGATE_TO, target)] for agent, target in per_agent.items()}
    if args.plan == "grasp":
        return {agent: [(Primitive.GRASP, target)] for agent, target in per_agent.items()}
    if args.plan == "asymmetric":
        first, second = AGENT_IDS
        return {
            first: [(Primitive.NAVIGATE_TO, per_agent[first]), (Primitive.GRASP, per_agent[first])],
            second: [(Primitive.NAVIGATE_TO, per_agent[second])],
        }
    return {
        agent: [(Primitive.NAVIGATE_TO, target), (Primitive.GRASP, target)]
        for agent, target in per_agent.items()
    }


def run_plans(
    engine: MultiAgentPrimitiveEngine,
    plans: Dict[str, List[Tuple[Primitive, str]]],
    max_ticks: int,
) -> List[PrimitiveOutcome]:
    """Miniature of COOP2's PlanningEnvWrapper.step loop (no ready barrier).

    Every agent that is free gets its next primitive **before** the tick, so
    on the first tick both generators are live and both robots start moving on
    the same env.step. After that each agent advances through its own sequence
    independently; the only global thing is the single ``engine.tick()``.
    """
    cursors = {agent_id: 0 for agent_id in plans}
    results: List[PrimitiveOutcome] = []

    while True:
        # Lazy on purpose: the engine never pre-fetches, so the plan cursor
        # stays entirely under the caller's control (which is what lets a
        # message interrupt discard a plan cleanly).
        for agent_id, steps in plans.items():
            if engine.has_active(agent_id) or cursors[agent_id] >= len(steps):
                continue
            primitive, target = steps[cursors[agent_id]]
            cursors[agent_id] += 1
            immediate = engine.assign(agent_id, primitive, target)
            if immediate is not None:
                # Failed before anything moved.
                results.append(immediate)
                cursors[agent_id] = len(steps)

        if not engine.active_agents():
            break
        if engine.env_step >= max_ticks:
            print(f"[demo] global tick budget {max_ticks} exhausted; aborting the rest")
            for agent_id in engine.active_agents():
                aborted = engine.abort(agent_id)
                if aborted is not None:
                    results.append(aborted)
            break

        for agent_id, outcome in engine.tick().items():
            results.append(outcome)
            # A failed primitive kills the remainder of that agent's plan,
            # mirroring plan.complete_failed() in COOP2.
            if not outcome.ok:
                cursors[agent_id] = len(plans[agent_id])

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def gantt(results: Sequence[PrimitiveOutcome], agent_ids: Sequence[str], total_ticks: int, width: int = 68) -> None:
    """ASCII timeline: overlapping bars are the proof of simultaneous motion."""
    if total_ticks <= 0:
        return
    scale = width / total_ticks
    print(f"\ntick timeline (0 .. {total_ticks}, one column ~ {total_ticks / width:.0f} ticks)")
    print("           " + "".join("|" if i % 10 == 0 else "-" for i in range(width)))
    for agent_id in agent_ids:
        row = [" "] * width
        for outcome in results:
            if outcome.agent_id != agent_id:
                continue
            start = min(int(outcome.started_env_step * scale), width - 1)
            end = min(int(outcome.ended_env_step * scale), width - 1)
            glyph = "=" if outcome.ok else "x"
            for column in range(start, max(end, start) + 1):
                row[column] = glyph
            label = outcome.primitive[:1]
            row[start] = label.lower() if not outcome.ok else label
        print(f"{agent_id:<11}" + "".join(row))
    print("           (first char of the primitive at its start; '=' ok, 'x' failed)")


def report(args, plans, results, engine, monitor, total_seconds):
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"mode={args.mode}  plan={args.plan}  attempts={args.attempts}  contend={args.contend}")
    print("\nplans:")
    for agent_id, steps in plans.items():
        pretty = " -> ".join(f"{primitive.name}({target})" for primitive, target in steps)
        print(f"  {agent_id}: {pretty}")

    header = (
        f"{'agent':<10}{'primitive':<14}{'target':<10}{'status':<9}"
        f"{'ticks':>7}{'sec':>8}{'start':>8}{'end':>8}  reason"
    )
    print("\n" + header)
    print("-" * 96)
    for outcome in results:
        reason = "" if outcome.ok else f"{outcome.reason_code}: {outcome.failure_reason[:52]}"
        flag = " [kills plan]" if outcome.terminate_plan else ""
        print(
            f"{outcome.agent_id:<10}{outcome.primitive:<14}{str(outcome.target):<10}{outcome.status:<9}"
            f"{outcome.ticks:>7}{outcome.wall_seconds:>8.1f}"
            f"{outcome.started_env_step:>8}{outcome.ended_env_step:>8}  {reason}{flag}"
        )
    print("-" * 96)

    gantt(results, AGENT_IDS, engine.env_step)

    print("\nCONCURRENCY")
    print(f"  busy ticks (>=1 agent moving): {monitor.busy_ticks}")
    print(f"  overlap ticks (>=2 moving):    {monitor.overlap_ticks}")
    print(f"  overlap ratio:                 {monitor.overlap_ratio:.2f}", end="")
    if args.mode == MotionMode.EXCLUSIVE:
        print("   (exclusive mode: must be 0.00)")
    elif monitor.overlap_ratio > 0:
        print("   <- the two agents were in motion on the same env.step")
    else:
        print("   <- NO overlap; the primitives never ran at the same time")
    for count in sorted(monitor.histogram):
        print(f"    ticks with {count} agent(s) moving: {monitor.histogram[count]}")
    for agent_id in AGENT_IDS:
        print(f"    {agent_id} moved on {monitor.per_agent[agent_id]} ticks")

    print("\nCOST")
    print(f"  env.step ticks:      {engine.env_step}")
    print(f"  primitives issued:   {engine.decision_count}   <- the denominator for COOP2 metrics")
    print(f"  wall clock:          {total_seconds:.1f}s")
    if engine.decision_count:
        print(f"  ticks per primitive: {engine.env_step / engine.decision_count:.0f}")
    if engine.env_step:
        print(f"  wall ms per tick:    {1000 * total_seconds / engine.env_step:.1f}")

    print("\nFINAL STATE")
    for agent_id in AGENT_IDS:
        robot = engine.robots_by_id[agent_id]
        position, _ = robot.get_position_orientation()
        room = engine.env.scene._seg_map.get_room_instance_by_point(position[:2])
        held = engine.held_objects()[agent_id]
        print(f"  {agent_id}: holding={held} pos={[round(v, 3) for v in position.tolist()]} room={room}")
    for name, _ in APPLES:
        obj = engine.env.scene.object_registry("name", name)
        if obj is not None:
            position, _ = obj.get_position_orientation()
            print(f"  {name}: pos={[round(v, 3) for v in position.tolist()]}")

    failures = Counter(o.reason_code for o in results if not o.ok)
    if failures:
        print("\nFAILURE CODES")
        for code, count in failures.most_common():
            print(f"  {code}: {count}")
        print("\nFAILURE DETAILS")
        for outcome in results:
            if not outcome.ok:
                print(f"  {outcome.agent_id} {outcome.primitive}({outcome.target}):")
                for line in outcome.failure_reason.splitlines():
                    print(f"    {line}")
                if outcome.metadata:
                    print(f"    metadata: {outcome.metadata}")
        print("  EXECUTION-heavy -> loosen tracking tolerances in tune_primitive_macros().")
        print("  PLANNING/SAMPLING-heavy -> check the spawn poses and the room each robot is in.")


def keep_viewer_open(env):
    print("\nViewer stays open. Click the viewport, then RMB-drag or W/A/S/D/T/G to move the camera.")
    print("Continuously printing apple positions.")
    print("Ctrl+C to exit.")
    try:
        while True:
            if not og.sim.is_playing():
                og.sim.play()
            og.sim.step()
            positions = []
            for name, _ in APPLES:
                obj = env.scene.object_registry("name", name)
                if obj is not None:
                    position, _ = obj.get_position_orientation()
                    positions.append(f"{name}={[round(v, 3) for v in position.tolist()]}")
            print("[keep-viewer] " + "  ".join(positions))
    except KeyboardInterrupt:
        print("\nexiting")


def main() -> None:
    args = parse_args()

    # og.shutdown() ends in app.close(), which terminates the process without
    # unwinding Python -- so a block-buffered stdout (any run redirected to a
    # file) loses every print, and a traceback from inside the try block is
    # swallowed along with it: the run looks like a silent exit 0. Line
    # buffering here makes redirected runs readable without needing -u.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):  # not a TextIOWrapper
            pass

    # Same performance settings as the single-robot baseline. Transition rules
    # are pure overhead for a pick-and-place smoke test.
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = False
    if args.headless:
        gm.HEADLESS = True
        gm.RENDER_VIEWER_CAMERA = False

    # MUST happen before any controller is constructed: reading a macro locks
    # it against further writes.
    tune_primitive_macros()

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

        started = time.time()
        engine = MultiAgentPrimitiveEngine(
            env,
            agent_ids=AGENT_IDS,
            attempts=args.attempts,
            motion_mode=args.mode,
            obstacle_refresh_every=args.obstacle_refresh_every,
            max_ticks_per_primitive=args.max_ticks_per_primitive,
            enable_head_tracking=False,
            curobo_batch_size=args.curobo_batch_size,
        )
        print(
            f"[timing] {len(engine.robots)} primitive controllers built in {time.time() - started:.1f}s "
            "(this is the cost that decides how many agents are practical)"
        )

        if not args.headless:
            configure_viewer_camera()

        monitor = ConcurrencyMonitor(engine, trace_every=args.trace_every)
        engine.on_tick = monitor

        plans = build_plans(args)
        started = time.time()
        results = run_plans(engine, plans, max_ticks=args.max_ticks)
        report(args, plans, results, engine, monitor, time.time() - started)

        if args.keep_viewer and not args.headless:
            keep_viewer_open(env)
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
