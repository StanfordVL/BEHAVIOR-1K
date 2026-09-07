"""Where to put robots and objects, built on OmniGibson's own samplers.

Hard-coded poses do not survive a scene change, and they did not really work in
the scene they were written for either: measured on Rs_int, 96% of sampled base
poses put an R1 somewhere it does not fit (see
``feasibility_verify/measure_teleport_risk.py``). So placement is derived from
the scene -- but by **calling** OmniGibson rather than re-deriving it.

Two upstream APIs do the heavy lifting:

* ``scene.get_random_point(floor, reference_point, robot)`` erodes the
  traversability map by the robot's own footprint, restricts to the connected
  component containing ``reference_point``, and samples uniformly. An earlier
  version of this module reimplemented all three by hand; it was slower (a
  whole-map room scan that was fine on a 35 m2 corridor took minutes on a
  4560 m2 hall) and needed a correctness assertion to guard coordinate
  transforms that calling the API avoids entirely.
* ``obj.states[OnTop].set_value(surface, True)`` samples a physically valid
  pose through ``sample_kinematics``. Scattering objects on bare floor cells,
  as this module used to, is neither realistic for a household benchmark nor
  reliable -- it is what forced the ``standable_fraction`` heuristics.

What genuinely has no upstream equivalent, and is therefore still here:
mutual separation between N robots (upstream placement is single-robot),
destination reservation under concurrency (see
:class:`~coop2.behavior_env.symbolic_navigation.DestinationRegistry`), and
room-scoped placement, which L1b's room-level observation depends on.

Placement runs **after** ``og.Environment`` is constructed even though
``build_multi_robot_config`` wants poses up front: the scene's maps only exist
once it is loaded. Build the config with placeholders, load, then teleport --
nothing has stepped yet, so the placeholders never matter.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

__all__ = [
    "OUTDOOR_ROOM_TAGS",
    "look_at_quaternion",
    "pick_room",
    "place_objects",
    "place_robots",
    "sample_free_points",
]

#: Rooms a household cooperation benchmark should not run in. The largest
#: connected free region of house_single_floor is ``garden_0``; placing the
#: agents there defeats the point of choosing a multi-room house.
OUTDOOR_ROOM_TAGS = ("garden", "lawn", "yard", "porch", "patio", "driveway", "deck", "balcony")


def _seg_map(scene):
    return getattr(scene, "seg_map", None) or getattr(scene, "_seg_map", None)


def room_of(scene, xy) -> Optional[str]:
    """Room instance at a world xy, or None off-map / on a boundary."""
    seg_map = _seg_map(scene)
    if seg_map is None:
        return None
    try:
        return seg_map.get_room_instance_by_point(th.as_tensor([float(xy[0]), float(xy[1])]))
    except Exception:  # noqa: BLE001 - off-map points
        return None


def sample_free_points(
    scene,
    robot,
    count: int,
    reference_point=None,
    room: Optional[str] = None,
    max_attempts: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """@count standable world points, optionally restricted to @room.

    Thin wrapper over ``scene.get_random_point``: erosion by the robot's
    footprint and connected-component restriction come from there. The room
    filter is applied to the **sampled candidates**, not to the whole map --
    which is the difference between a handful of lookups and one per free cell.
    """
    attempts = max_attempts if max_attempts is not None else max(200, count * 200)
    points: List[Tuple[float, float]] = []
    for _ in range(attempts):
        if len(points) >= count:
            break
        try:
            _, point = scene.get_random_point(floor=0, reference_point=reference_point, robot=robot)
        except Exception:  # noqa: BLE001 - scenes without a trav map
            return points
        if point is None:
            continue
        xy = (float(point[0]), float(point[1]))
        if room is not None and room_of(scene, xy) != room:
            continue
        points.append(xy)
    return points


def pick_room(scene, robot, prefer_indoor: bool = True, samples: int = 250) -> Optional[str]:
    """The room with the most standable floor, estimated by sampling.

    Sampling rather than measuring: ``get_random_point`` is uniform over
    standable floor, so the room a sample lands in is proportional to that
    room's standable area. 250 lookups replaces a scan of every free cell,
    which on a 4560 m2 hall is ~450k of them.
    """
    counts: Dict[str, int] = {}
    for _ in range(samples):
        try:
            _, point = scene.get_random_point(floor=0, robot=robot)
        except Exception:  # noqa: BLE001
            return None
        if point is None:
            continue
        room = room_of(scene, (float(point[0]), float(point[1])))
        if room is None:
            continue
        if prefer_indoor and any(tag in room.lower() for tag in OUTDOOR_ROOM_TAGS):
            continue
        counts[room] = counts.get(room, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def look_at_quaternion(eye, target) -> th.Tensor:
    """xyzw quaternion for a camera at @eye looking at @target (USD: -Z forward)."""
    eye = [float(v) for v in eye]
    target = [float(v) for v in target]
    forward = [target[i] - eye[i] for i in range(3)]
    norm = math.sqrt(sum(c * c for c in forward)) or 1.0
    forward = [c / norm for c in forward]
    z = [-c for c in forward]  # USD cameras look down -Z with +Y up
    world_up = [0.0, 0.0, 1.0]
    if abs(sum(z[i] * world_up[i] for i in range(3))) > 0.999:
        world_up = [0.0, 1.0, 0.0]
    x = [
        world_up[1] * z[2] - world_up[2] * z[1],
        world_up[2] * z[0] - world_up[0] * z[2],
        world_up[0] * z[1] - world_up[1] * z[0],
    ]
    norm = math.sqrt(sum(c * c for c in x)) or 1.0
    x = [c / norm for c in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]

    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        qw, qx, qy, qz = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        qw, qx, qy, qz = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        qw, qx, qy, qz = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    return th.tensor([qx, qy, qz, qw], dtype=th.float32)


def robot_radius(robot) -> float:
    """Circumscribed radius at the reset pose, arms included.

    Half the planar diagonal of the whole-robot AABB, so it bounds the robot at
    any yaw. Conservative for a rectangular chassis; an oriented-box test could
    pack tighter at the cost of real code.
    """
    try:
        return float(th.norm(robot.reset_joint_pos_aabb_extent[:2])) / 2.0
    except Exception:  # noqa: BLE001
        return 0.6


def place_robots(
    env,
    separation: Optional[float] = None,
    seed: Optional[int] = None,
    z: float = 0.05,
    room: Optional[str] = None,
    prefer_indoor: bool = True,
    cluster_radius: Optional[float] = 6.0,
) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    """Teleport every robot into one room, mutually separated and connected.

    Call right after ``og.Environment(...)`` and before ``prepare_robots``.

    Everything lands in a **single room** on purpose. Sampling the whole
    traversable region instead put two agents 47 m apart in house_single_floor
    (both outdoors, in ``garden_0``), which cost one a 2000-tick NAVIGATE_TO
    timeout before it ever reached the shared target. Cooperation cannot be
    measured across a distance neither agent can cross inside the budget.

    Args:
        separation: centre-to-centre minimum. ``None`` uses twice the robot's
            circumscribed radius -- just-touching, which is all that is needed:
            navigation teleports, so there is no path to keep clear.
        room: room instance to use. ``None`` picks the roomiest indoor one.
        cluster_radius: keep every robot within this distance of the first one.
            A room can be long and thin -- the roomiest indoor space in
            house_single_floor is a 20 m corridor -- and agents spread along it
            never meet.
    """
    robots = list(env.robots)
    if not robots:
        raise ValueError("No robots in the environment.")
    if seed is not None:
        th.manual_seed(int(seed))
    if separation is None:
        separation = 2.0 * robot_radius(robots[0])

    scene = env.scene
    chosen_room = room if room is not None else pick_room(scene, robots[0], prefer_indoor=prefer_indoor)

    anchor = None
    chosen: List[Tuple[float, float]] = []
    for robot in robots:
        placed = None
        # Sample in batches so a crowded room fails fast rather than spinning.
        for candidate in sample_free_points(
            scene, robot, count=400, reference_point=anchor, room=chosen_room, max_attempts=4000
        ):
            if any(math.hypot(candidate[0] - x, candidate[1] - y) < separation for x, y in chosen):
                continue
            if cluster_radius is not None and chosen:
                if math.hypot(candidate[0] - chosen[0][0], candidate[1] - chosen[0][1]) > cluster_radius:
                    continue
            placed = candidate
            break
        if placed is None:
            raise RuntimeError(
                f"Could not place {robot.name} in {chosen_room!r} at {separation:.2f} m separation "
                f"within {cluster_radius} m of the group. Use a bigger room or fewer robots "
                "(feasibility_verify/survey_scene_capacity.py --by-room)."
            )
        chosen.append(placed)
        if anchor is None:
            anchor = th.tensor([placed[0], placed[1], z], dtype=th.float32)
        robot.set_position_orientation(
            position=th.tensor([placed[0], placed[1], z], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
    print(f"[placement] room {chosen_room!r}: {[(round(x, 2), round(y, 2)) for x, y in chosen]}")
    return chosen, chosen_room


#: Resolved object layouts, keyed by (scene, room, seed, objects). Checked in
#: so a run reproduces exactly and starts without re-sampling; delete an entry
#: (or pass use_cache=False) to force a fresh layout.
_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement_cache.json")


def _cache_key(scene_model, room, seed, names) -> str:
    return "|".join([
        str(scene_model or "?"),
        str(room or "?"),
        str(seed if seed is not None else "?"),
        ",".join(sorted(names)),
    ])


def _load_cache(path: str) -> Dict[str, Any]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _store_cache(path: str, key: str, scene, names) -> None:
    """Record where the objects actually ended up, after settling."""
    poses = {}
    for name in names:
        obj = scene.object_registry("name", name)
        if obj is None:
            return
        position, orientation = obj.get_position_orientation()
        poses[name] = {
            "position": [float(v) for v in position],
            "orientation": [float(v) for v in orientation],
        }
    data = _load_cache(path)
    data[key] = poses
    try:
        with open(path, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        print(f"[placement] cached layout under {key!r}")
    except OSError:
        pass


def _apply_cached(scene, names, cached) -> List[Tuple[float, float]]:
    """Replay a cached layout instead of sampling one."""
    placed: List[Tuple[float, float]] = []
    for name in names:
        obj = scene.object_registry("name", name)
        entry = cached.get(name)
        if obj is None or entry is None:
            raise KeyError(f"Cached layout does not cover {name!r}.")
        obj.set_position_orientation(
            position=th.tensor(entry["position"], dtype=th.float32),
            orientation=th.tensor(entry["orientation"], dtype=th.float32),
        )
        placed.append((float(entry["position"][0]), float(entry["position"][1])))
        print(f"[placement] {name} -> ({placed[-1][0]:.2f}, {placed[-1][1]:.2f}) from cache")
    return placed


class _SurfaceCandidate:
    """The three fields L1b's predicates read, taken off a scene object.

    Placement and the agent must agree on what counts as a surface: if they
    disagree, either the agent is offered a place_on_top it cannot perform, or
    an object is placed somewhere the agent is never told about.
    """

    __slots__ = ("entity_id", "abilities", "is_fixed")

    def __init__(self, obj, fixed_names):
        synset = _synset_for(obj)
        self.entity_id = f"{synset}_1" if synset else (getattr(obj, "name", "") or "")
        self.abilities = sorted(getattr(obj, "abilities", None) or [])
        self.is_fixed = getattr(obj, "name", None) in fixed_names


def _synset_for(obj):
    """WordNet synset for @obj's category, or None outside the taxonomy."""
    category = getattr(obj, "category", None)
    if not category:
        return None
    try:
        from bddl.object_taxonomy import ObjectTaxonomy  # noqa: PLC0415

        global _TAXONOMY
        if _TAXONOMY is None:
            _TAXONOMY = ObjectTaxonomy()
        return _TAXONOMY.get_synset_from_category(category)
    except Exception:  # noqa: BLE001 - category outside the taxonomy
        return None


_TAXONOMY = None


def _is_support_surface(obj, scene) -> bool:
    """Would the agent be allowed to place something onto @obj?

    Reuses L1b's own support test -- the same one that decides whether the
    agent is offered place_on_top -- rather than "has an OnTop state", which
    nearly every kinematic object has. That version treated light switches,
    sliding doors and downlights as surfaces, and a single refused
    ``OnTop.set_value`` on one of them cost 113 seconds.
    """
    from coop2.behavior_env.symbolic_view import is_support_surface  # noqa: PLC0415

    fixed = getattr(scene, "fixed_objects", None) or {}
    fixed_names = set(fixed.keys()) if isinstance(fixed, dict) else {
        getattr(o, "name", o) for o in fixed
    }
    candidate = _SurfaceCandidate(obj, fixed_names)
    try:
        return is_support_surface(candidate)
    except Exception:  # noqa: BLE001 - taxonomy unavailable
        return False


def place_objects(
    env,
    names: Sequence[str],
    room: Optional[str] = None,
    seed: Optional[int] = None,
    min_separation: float = 1.0,
    near_robots: Optional[float] = 8.0,
    surfaces: Optional[Sequence[Any]] = None,
    z: float = 0.05,
    max_surface_tries: int = 3,
    scene_model: Optional[str] = None,
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> List[Tuple[float, float]]:
    """Put each named object somewhere reachable, preferring a real surface.

    Tries ``OnTop.set_value(surface, True)`` first -- that runs OmniGibson's
    kinematic sampler, so the object ends up physically resting on a table or
    counter the way a household object actually would. Only if no surface in the
    room accepts it does this fall back to a standable floor point.

    ``set_value`` is a *stochastic physical* sampler: it raycasts and steps
    physics internally, costing anywhere from milliseconds to several seconds
    per call, and it can simply refuse. Trying every surface in the room was
    therefore unbounded in practice -- corridor_0 holds a dozen shelves, and
    one unlucky reset spent over fifteen minutes in this loop without placing
    a single apple, while a lucky one finished in five seconds. So: try the
    nearest few surfaces only (@max_surface_tries), then take the floor. The
    floor path is a cheap trav-map sample and always terminates.
    """
    robots = list(env.robots)
    scene = env.scene
    if seed is not None:
        th.manual_seed(int(seed) + 9973)  # decorrelate from robot placement

    # A resolved layout is worth keeping. Sampling is stochastic and a single
    # refused OnTop.set_value costs OmniGibson ~112 s, so the same (scene,
    # room, seed, objects) can take anywhere from 0 to several minutes to place
    # identically. Replaying the cached poses makes startup constant-time and
    # makes two runs of the same seed literally identical, which is what the
    # experiment wants. Where objects *should* go is a task-design question;
    # this only remembers an answer once one has been found.
    cache_path = cache_path or _DEFAULT_CACHE
    key = _cache_key(scene_model, room, seed, names)
    if use_cache:
        cached = _load_cache(cache_path).get(key)
        if cached and len(cached) == len(names):
            return _apply_cached(scene, names, cached)

    from omnigibson import object_states  # noqa: PLC0415

    if surfaces is None:
        candidates = scene.object_registry("in_rooms", room, default_val=[]) if room else []
        surfaces = [obj for obj in candidates if _is_support_surface(obj, scene)]

    centre = None
    if near_robots is not None and robots:
        xs = [float(r.get_position_orientation()[0][0]) for r in robots]
        ys = [float(r.get_position_orientation()[0][1]) for r in robots]
        centre = (sum(xs) / len(xs), sum(ys) / len(ys))

    placed: List[Tuple[float, float]] = []
    for name in names:
        obj = scene.object_registry("name", name)
        if obj is None:
            raise KeyError(f"No object named {name!r} in the scene.")

        # Nearest first: the closest surface is both the likeliest to accept a
        # sample and the most useful place for the object to be.
        in_range = []
        for surface in surfaces:
            sx, sy = surface.get_position_orientation()[0][:2]
            if centre is None:
                in_range.append((0.0, surface))
                continue
            distance = math.hypot(float(sx) - centre[0], float(sy) - centre[1])
            if distance <= near_robots:
                in_range.append((distance, surface))
        in_range.sort(key=lambda pair: pair[0])

        on_surface = False
        for _, surface in in_range[:max_surface_tries]:
            started = time.time()
            try:
                if obj.states[object_states.OnTop].set_value(surface, True):
                    on_surface = True
                    break
            except Exception:  # noqa: BLE001 - surface refuses the sample
                pass
            elapsed = time.time() - started
            if elapsed > 5.0:
                print(f"[placement] {surface.name} refused {name} after {elapsed:.0f}s")

        if not on_surface:
            spot = None
            for candidate in sample_free_points(
                scene, robots[0], count=200, room=room, max_attempts=2000
            ):
                if centre is not None and math.hypot(candidate[0] - centre[0], candidate[1] - centre[1]) > near_robots:
                    continue
                if any(math.hypot(candidate[0] - x, candidate[1] - y) < min_separation for x, y in placed):
                    continue
                taken = [
                    (float(r.get_position_orientation()[0][0]), float(r.get_position_orientation()[0][1]))
                    for r in robots
                ]
                if any(math.hypot(candidate[0] - x, candidate[1] - y) < min_separation for x, y in taken):
                    continue
                spot = candidate
                break
            if spot is None:
                raise RuntimeError(f"Could not place {name} in {room!r}.")
            obj.set_position_orientation(position=th.tensor([spot[0], spot[1], z], dtype=th.float32))

        position = obj.get_position_orientation()[0]
        placed.append((float(position[0]), float(position[1])))
        where = "on a surface" if on_surface else "on the floor"
        print(f"[placement] {name} -> ({placed[-1][0]:.2f}, {placed[-1][1]:.2f}) {where}")

    if use_cache:
        _store_cache(cache_path, key, scene, names)
    return placed
