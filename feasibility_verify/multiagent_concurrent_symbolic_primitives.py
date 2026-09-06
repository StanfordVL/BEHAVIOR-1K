"""Two R1 agents executing SymbolicSemanticActionPrimitives concurrently.

It contains its own environment configuration, per-agent plan loop,
concurrency monitor, Gantt chart, and final-state report, and injects one
symbolic controller per robot. No cuRobo controller is constructed and object
state changes are performed by the symbolic primitives directly.

Plan options are named after the primitives they issue, i.e. the lowercased
``SymbolicSemanticActionPrimitiveSet`` members (``navigate_to``, ``grasp``,
``release``).

Run:
    python feasibility_verify/multiagent_concurrent_symbolic_primitives.py
    python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --plan navigate_to
    python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --plan navigate_to_then_grasp
    python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --plan grasp_then_release
    python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --plan asymmetric
    python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --mode exclusive
    python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --contend
    python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --keep-viewer

Note on ``navigate_to`` in the symbolic set: ``_navigate_to_pose`` is a pure
teleport (``robot.set_position_orientation`` followed by ``_settle_robot``), so
it costs only the settling ticks rather than a planned trajectory. It is also
optional before ``grasp`` here -- symbolic ``_grasp`` has its
``_navigate_if_needed`` call commented out and teleports the object to the
end-effector at any distance -- which is exactly why it is worth having as its
own plan: it isolates base motion from manipulation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch as th

import omnigibson as og
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitiveSet as Primitive,
)
from omnigibson.macros import gm

# Allow running this file directly from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.behavior_env.env_setup import assert_multi_robot_sanity, build_multi_robot_config, prepare_robots
from coop2.behavior_env.placement import look_at_quaternion, place_objects, place_robots
from coop2.behavior_env.recording import ViewerRecorder, chain, enable_viewer_rendering
from coop2.behavior_env.primitive_engine import (
    MotionMode,
    MultiAgentPrimitiveEngine,
    PrimitiveOutcome,
)
from coop2.behavior_env.symbolic_contention import ContentiousSymbolicActionPrimitives
from coop2.behavior_env.symbolic_navigation import NavigableSymbolicActionPrimitives
from coop2.behavior_env.symbolic_view import render_symbolic_view, target_hints
from coop2.behavior_env.world_state import BehaviorWorldState

PlanStep = Tuple[Primitive, Optional[str]]

ROBOT_MODEL = "R1"
# Rs_int was measured unusable: 96% of sampled base poses put an R1 where it does
# not fit and 46% of targets have no standable pose at all; its largest connected
# free region after eroding by R1's 0.62 m radius is 0.9 m2, which holds exactly
# one robot. house_single_floor is 29.5% / 1.7% with a 1709 m2 connected region,
# and is a multi-room house -- L1b's room-level world graph needs real rooms, so
# the big undivided halls are not substitutes despite being emptier.
# See feasibility_verify/{measure_teleport_risk,survey_scene_capacity}.py.
SCENE_MODEL = "house_single_floor"
LOAD_OBJECT_CATEGORIES = None  # None = load the whole scene; the trav map assumes it
# Placeholders only. Robots are teleported to scene-derived poses right after
# load (see coop2.behavior_env.placement) -- build_multi_robot_config needs
# poses before the scene, and therefore before its trav map, exists.
ROBOT_POSES = [
    ([0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]),
    ([1.5, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]),
]
AGENT_IDS = ["agent_0", "agent_1"]
APPLES = [
    ("apple_0", [0.0, 0.0, 0.05]),
    ("apple_1", [1.0, 0.0, 0.05]),
]

VIEWER_CAMERA_POSITION = th.tensor([1.8294, -3.2502, 1.6885])
VIEWER_CAMERA_ORIENTATION = th.tensor([0.5770, 0.1719, 0.2280, 0.7652])
VIEWER_CAMERA_FOCAL_LENGTH = 17.0
VIEWER_CAMERA_HORIZONTAL_APERTURE = 20.995


def configure_viewer_camera(env=None) -> None:
    """Frame the camera on the robots and apples.

    A fixed pose cannot survive a scene change, and placement is randomised per
    seed, so derive the framing: sit back from the centroid along the shorter
    horizontal axis and look at it.
    """
    if env is None:
        og.sim.viewer_camera.set_position_orientation(
            position=VIEWER_CAMERA_POSITION, orientation=VIEWER_CAMERA_ORIENTATION
        )
    else:
        points = [robot.get_position_orientation()[0] for robot in env.robots]
        for name, _ in APPLES:
            obj = env.scene.object_registry("name", name)
            if obj is not None:
                points.append(obj.get_position_orientation()[0])
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        spread = max(max(xs) - min(xs), max(ys) - min(ys), 2.0)
        # Back off along -y and lift, so the whole spread fits the 63 deg FOV.
        distance = spread * 1.3 + 2.0
        eye = th.tensor([cx, cy - distance, 0.6 * distance], dtype=th.float32)
        target = th.tensor([cx, cy, 0.4], dtype=th.float32)
        og.sim.viewer_camera.set_position_orientation(
            position=eye, orientation=look_at_quaternion(eye, target)
        )
    og.sim.viewer_camera.focal_length = VIEWER_CAMERA_FOCAL_LENGTH
    og.sim.viewer_camera.horizontal_aperture = VIEWER_CAMERA_HORIZONTAL_APERTURE


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


class ConcurrencyMonitor:
    def __init__(self, engine: MultiAgentPrimitiveEngine, trace_every: int = 0):
        self.engine = engine
        self.trace_every = trace_every
        self.histogram: Counter = Counter()
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
                parts.append(f"{moving}{agent_id}[{primitive}] ({position[0]:+.2f},{position[1]:+.2f})")
            print(f"[trace {env_step:>6}] " + "  ".join(parts))

    @property
    def busy_ticks(self) -> int:
        return sum(count for n_agents, count in self.histogram.items() if n_agents >= 1)

    @property
    def overlap_ticks(self) -> int:
        return sum(count for n_agents, count in self.histogram.items() if n_agents >= 2)

    @property
    def overlap_ratio(self) -> float:
        return self.overlap_ticks / self.busy_ticks if self.busy_ticks else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=list(MotionMode.ALL),
        default=MotionMode.CONCURRENT,
        help="concurrent: advance all symbolic generators on each tick; exclusive: advance one agent at a time.",
    )
    parser.add_argument(
        "--plan",
        choices=["navigate_to", "grasp", "navigate_to_then_grasp", "grasp_then_release", "asymmetric"],
        default="grasp",
        help="Named after the primitives issued. "
        "navigate_to: each agent teleports to a base pose near its apple (base motion only). "
        "grasp (default): each agent grasps its apple. "
        "navigate_to_then_grasp: approach first, then grasp. "
        "grasp_then_release: both grasp then release. "
        "asymmetric: agent_0 grasps then releases while agent_1 only grasps.",
    )
    parser.add_argument(
        "--contend",
        action="store_true",
        help="Point both agents at apple_0 to exercise symbolic target contention.",
    )
    parser.add_argument("--attempts", type=int, default=1, help="Number of symbolic apply_ref attempts.")
    parser.add_argument("--max-ticks-per-primitive", type=int, default=2000)
    parser.add_argument("--max-ticks", type=int, default=10000, help="Global tick budget for the whole run.")
    parser.add_argument(
        "--trace-every",
        type=int,
        default=0,
        help="Print both robots' base positions and active symbolic primitives every N ticks.",
    )
    parser.add_argument(
        "--no-contention",
        action="store_true",
        help=(
            "Use the bare NavigableSymbolicActionPrimitives: no interaction radius, no travel cost, "
            "no cross-robot claims. This is the upstream behaviour and it lets two robots 'hold' the "
            "same object at once -- useful only as a control."
        ),
    )
    parser.add_argument(
        "--interaction-radius",
        type=float,
        default=None,
        help="Metres within which an object can be grasped/placed/opened. Default: nav upper bound + 0.35.",
    )
    parser.add_argument(
        "--travel-ticks-per-meter",
        type=float,
        default=60.0,
        help="Hold-position ticks charged per metre of base travel before the teleport. 0 disables.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed torch's global RNG, which _sample_pose_near_object draws from. The sampler picks the base "
            "pose uniformly in distance_range, so travel cost carries +/- (range/2 * ticks_per_meter) ticks "
            "of noise -- larger than the head start --contend gives the closer agent. Seed it to make a race "
            "reproducible."
        ),
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Record a third-person video of the run to this path (.mp4 or .webm). Forces the viewer "
            "camera on even with --headless, which is otherwise disabled for speed."
        ),
    )
    parser.add_argument(
        "--video-every",
        type=int,
        default=1,
        help=(
            "Capture one frame every N ticks. 1 (default) plays back at real time. Raise it to "
            "compress a long run -- but a short one then vanishes: --plan grasp is ~100 ticks, "
            "which at 4 was a 0.8 s file."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Frames per second in the file.")
    parser.add_argument(
        "--show-observation",
        action="store_true",
        help=(
            "Build the L1a world model and print each agent's L1b text observation before and "
            "after the run. This is what the LLM will be shown, so it is the thing to eyeball."
        ),
    )
    parser.add_argument(
        "--n-robots",
        type=int,
        default=2,
        help=(
            "How many robots to spawn. Start poses are derived from the scene's trav map, so this "
            "just scales -- but the chosen room has to hold them: house_single_floor's living_room "
            "is 16.7 m2, enough for nine R1s at 1.24 m separation, while Rs_int's best room is 0.9 m2."
        ),
    )
    parser.add_argument(
        "--room",
        type=str,
        default=None,
        help=(
            "Room instance to place robots and apples in. Default picks the largest indoor room -- "
            "the largest connected free region is often the garden, which is not where a household "
            "cooperation benchmark belongs."
        ),
    )
    parser.add_argument("--keep-viewer", action="store_true", help="Keep the viewer open after the run.")
    parser.add_argument("--headless", action="store_true", help="Run without the viewer.")
    return parser.parse_args()


def build_symbolic_plans(args) -> Dict[str, List[PlanStep]]:
    # zip() truncates to the shorter sequence, so with N agents and 2 apples only
    # the first 2 agents used to get a plan at all. Cycle instead.
    names = [name for name, _ in APPLES]
    if args.contend:
        targets = [names[0]] * len(AGENT_IDS)
    else:
        targets = [names[i % len(names)] for i in range(len(AGENT_IDS))]
    per_agent = dict(zip(AGENT_IDS, targets))

    if args.plan == "navigate_to":
        return {agent_id: [(Primitive.NAVIGATE_TO, target)] for agent_id, target in per_agent.items()}
    if args.plan == "grasp":
        return {agent_id: [(Primitive.GRASP, target)] for agent_id, target in per_agent.items()}
    if args.plan == "navigate_to_then_grasp":
        return {
            agent_id: [(Primitive.NAVIGATE_TO, target), (Primitive.GRASP, target)]
            for agent_id, target in per_agent.items()
        }
    if args.plan == "grasp_then_release":
        return {
            agent_id: [(Primitive.GRASP, target), (Primitive.RELEASE, None)]
            for agent_id, target in per_agent.items()
        }

    first, second = AGENT_IDS[0], AGENT_IDS[1 % len(AGENT_IDS)]
    return {
        first: [(Primitive.GRASP, per_agent[first]), (Primitive.RELEASE, None)],
        second: [(Primitive.GRASP, per_agent[second])],
    }


def run_plans(
    engine: MultiAgentPrimitiveEngine,
    plans: Dict[str, List[PlanStep]],
    max_ticks: int,
) -> List[PrimitiveOutcome]:
    cursors = {agent_id: 0 for agent_id in plans}
    results: List[PrimitiveOutcome] = []

    while True:
        for agent_id, steps in plans.items():
            if engine.has_active(agent_id) or cursors[agent_id] >= len(steps):
                continue
            primitive, target = steps[cursors[agent_id]]
            cursors[agent_id] += 1
            immediate = engine.assign(agent_id, primitive, target)
            if immediate is not None:
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
            if not outcome.ok:
                cursors[agent_id] = len(plans[agent_id])

    return results


def print_gantt(
    results: Sequence[PrimitiveOutcome],
    agent_ids: Sequence[str],
    total_ticks: int,
    width: int = 68,
) -> None:
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
            row[start] = outcome.primitive[:1] if outcome.ok else outcome.primitive[:1].lower()
        print(f"{agent_id:<11}" + "".join(row))


def report(args, plans, results, engine, monitor, total_seconds) -> None:
    print("\n" + "=" * 96)
    print("SYMBOLIC ACTION SUMMARY")
    print("=" * 96)
    print(f"mode={args.mode}  plan={args.plan}  attempts={args.attempts}  contend={args.contend}")
    if args.no_contention:
        print("contention: OFF (bare NavigableSymbolicActionPrimitives -- upstream, distance/holder blind)")
    else:
        controller = next(iter(engine.controllers.values()))
        print(
            f"contention: travel={controller.travel_ticks_per_meter:g} ticks/m  "
            f"claims={controller.enforce_claims}  seed={args.seed}"
        )
        # The interaction radius is per object now -- a fixed one cannot serve
        # both an apple and a table -- so report it per target instead.
        print(
            f"sampler:    robot_radius={controller.robot_radius:.2f}m  "
            f"separation={controller.robot_separation:.2f}m  "
            f"traversable_filter={controller._nav_require_traversable}"
        )
        for name, _ in APPLES:
            obj = engine.env.scene.object_registry("name", name)
            if obj is None:
                continue
            lo, hi = controller.sampling_range_for(obj)
            print(
                f"  {name}: sample [{lo:.2f}, {hi:.2f}] m  "
                f"interaction radius {controller.interaction_radius_for(obj):.2f} m"
            )
    print("\nplans:")
    for agent_id, steps in plans.items():
        pretty = " -> ".join(f"{primitive.name}({target})" for primitive, target in steps)
        print(f"  {agent_id}: {pretty}")

    print(f"\n{'agent':<10}{'primitive':<14}{'target':<10}{'status':<9}{'ticks':>7}{'sec':>8}  reason")
    print("-" * 96)
    for outcome in results:
        reason = "" if outcome.ok else f"{outcome.reason_code}: {outcome.failure_reason}"
        print(
            f"{outcome.agent_id:<10}{outcome.primitive:<14}{str(outcome.target):<10}{outcome.status:<9}"
            f"{outcome.ticks:>7}{outcome.wall_seconds:>8.1f}  {reason}"
        )
    print("-" * 96)

    print_gantt(results, AGENT_IDS, engine.env_step)
    print("\nCONCURRENCY")
    print(f"  busy ticks:    {monitor.busy_ticks}")
    print(f"  overlap ticks: {monitor.overlap_ticks}")
    print(f"  overlap ratio: {monitor.overlap_ratio:.2f}")
    for agent_id in AGENT_IDS:
        print(f"  {agent_id} advanced on {monitor.per_agent[agent_id]} ticks")

    print("\nCOST")
    print(f"  env.step ticks:    {engine.env_step}")
    print(f"  primitives issued: {engine.decision_count}")
    print(f"  wall clock:        {total_seconds:.1f}s")

    print("\nFINAL STATE")
    held_objects = engine.held_objects()
    for agent_id in AGENT_IDS:
        robot = engine.robots_by_id[agent_id]
        position, _ = robot.get_position_orientation()
        room = engine.env.scene.seg_map.get_room_instance_by_point(position[:2])
        print(f"  {agent_id}: holding={held_objects[agent_id]} pos={position.tolist()} room={room}")
    for name, _ in APPLES:
        obj = engine.env.scene.object_registry("name", name)
        position, _ = obj.get_position_orientation()
        print(f"  {name}: pos={position.tolist()}")


def keep_viewer_open(env) -> None:
    print("\nViewer stays open; continuously printing apple positions. Ctrl+C to exit.")
    try:
        while True:
            if not og.sim.is_playing():
                og.sim.play()
            og.sim.step()
            positions = []
            for name, _ in APPLES:
                obj = env.scene.object_registry("name", name)
                position, _ = obj.get_position_orientation()
                positions.append(f"{name}={[round(value, 3) for value in position.tolist()]}")
            # print("[keep-viewer] " + "  ".join(positions))
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

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = False
    if args.headless:
        gm.HEADLESS = True
        gm.RENDER_VIEWER_CAMERA = False
    if args.video:
        # Must precede og.Environment: the viewer camera is created during
        # scene load and gm.RENDER_VIEWER_CAMERA is read at that point.
        enable_viewer_rendering()

    try:
        robot_poses = [([1.5 * i, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]) for i in range(args.n_robots)]
        agent_ids = [f"agent_{i}" for i in range(args.n_robots)]
        globals()["AGENT_IDS"] = agent_ids
        config = build_multi_robot_config(
            robot_poses=robot_poses,
            robot_model=ROBOT_MODEL,
            scene_model=SCENE_MODEL,
            load_object_categories=LOAD_OBJECT_CATEGORIES,
            objects=build_objects(),
            agent_names=agent_ids,
        )

        if args.seed is not None:
            th.manual_seed(args.seed)
            print(f"[setup] torch seed = {args.seed}")

        started = time.time()
        env = og.Environment(configs=config)
        print(f"[timing] environment loaded in {time.time() - started:.1f}s")

        # Placement BEFORE prepare_robots: the config poses are placeholders,
        # and nothing has stepped yet, so moving here costs nothing.
        starts, placement_room = place_robots(env, seed=args.seed, room=args.room)
        print(f"[placement] robots -> {[(round(x, 2), round(y, 2)) for x, y in starts]} in {placement_room!r}")
        prepare_robots(env)
        assert_multi_robot_sanity(env, expected_robots=args.n_robots)

        started = time.time()
        # NavigableSymbolicActionPrimitives instead of the stock symbolic set:
        # NAVIGATE_TO is in the symbolic enum but unusable as shipped (its
        # inherited sampler dereferences the cuRobo motion generator that the
        # symbolic constructor never builds, and then passes a keyword the
        # symbolic _navigate_to_pose does not accept). See
        # coop2/behavior_env/symbolic_navigation.py.
        # ContentiousSymbolicActionPrimitives adds the interaction radius,
        # distance-proportional travel cost, and cross-robot claims on top --
        # without them the symbolic set is distance- and holder-blind, so
        # there is no resource competition to cooperate about (and a second
        # grasp of a held object silently double-joints it).
        if args.no_contention:
            controllers = {
                agent_id: NavigableSymbolicActionPrimitives(env, robot)
                for agent_id, robot in zip(AGENT_IDS, env.robots)
            }
        else:
            controllers = {
                agent_id: ContentiousSymbolicActionPrimitives(
                    env,
                    robot,
                    interaction_radius=args.interaction_radius,
                    travel_ticks_per_meter=args.travel_ticks_per_meter,
                )
                for agent_id, robot in zip(AGENT_IDS, env.robots)
            }
        engine = MultiAgentPrimitiveEngine(
            env,
            agent_ids=agent_ids,
            attempts=args.attempts,
            motion_mode=args.mode,
            obstacle_refresh_every=0,
            max_ticks_per_primitive=args.max_ticks_per_primitive,
            enable_head_tracking=False,
            controllers=controllers,
        )
        print(
            f"[timing] {len(engine.robots)} symbolic primitive controllers built in "
            f"{time.time() - started:.1f}s (cuRobo skipped)"
        )

        # The camera pose is the third-person framing, needed whenever we
        # render -- for the live viewer or for the recording.
        probe = next(iter(controllers.values()))
        apple = env.scene.object_registry("name", APPLES[0][0])
        place_objects(
            env,
            [name for name, _ in APPLES],
            annulus=probe.sampling_range_for(apple),
            seed=args.seed,
            room=placement_room,
        )

        if not args.headless or args.video:
            configure_viewer_camera(env)

        monitor = ConcurrencyMonitor(engine, trace_every=args.trace_every)
        recorder = (
            ViewerRecorder(args.video, every=args.video_every, fps=args.video_fps) if args.video else None
        )
        engine.on_tick = chain(monitor, recorder)

        world = None
        if args.show_observation:
            world = BehaviorWorldState(env)
            world.start()
            world.step()
            print("\n" + "=" * 96)
            print("L1b TEXT OBSERVATION (what the LLM sees)")
            print("=" * 96)
            for agent_id in agent_ids:
                radius = engine.controllers[agent_id].interaction_radius_for(apple)
                print(f"\n--- {agent_id} ---")
                print(render_symbolic_view(world.observation_for(agent_id), interaction_radius=radius))

        plans = build_symbolic_plans(args)
        started = time.time()
        try:
            results = run_plans(engine, plans, max_ticks=args.max_ticks)
        finally:
            # Close before reporting so the file is finalised even if the run
            # raised -- a partial video of a crash is worth having.
            if recorder is not None:
                recorder.close()
        report(args, plans, results, engine, monitor, time.time() - started)

        if world is not None:
            world.step()
            print("\n" + "=" * 96)
            print("L1b TEXT OBSERVATION AFTER THE RUN")
            print("=" * 96)
            for agent_id in agent_ids:
                radius = engine.controllers[agent_id].interaction_radius_for(apple)
                observation = world.observation_for(agent_id)
                print(f"\n--- {agent_id} ---")
                print(render_symbolic_view(observation, interaction_radius=radius))
                legal = target_hints(observation, interaction_radius=radius)
                print(f"  ({len(legal)} legal (primitive, target) pairs)")

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
