"""Where to put robots and objects in a scene, from the scene's own trav map.

Hard-coded poses do not survive a scene change, and they did not really work in
the scene they were written for either: measured on Rs_int, 96% of sampled base
poses put an R1 somewhere it does not fit (see
``feasibility_verify/measure_teleport_risk.py``). This module derives placement
from the scene instead.

Placement runs **after** ``og.Environment`` is constructed, not before, even
though ``build_multi_robot_config`` wants poses up front. The scene's trav map
only exists once the scene is loaded, and replicating OmniGibson's
world<->map transform offline (it flips xy into row/col, and resizes the baked
PNG by ``map_default_resolution / trav_map_resolution``) is a good way to be
subtly wrong. So: build the config with placeholders, load, then teleport.
Nothing has stepped yet at that point, so the placeholders never matter.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch as th

__all__ = ["TraversabilityIndex", "look_at_quaternion", "place_objects", "place_robots"]


class TraversabilityIndex:
    """The scene's trav map, eroded by the robot, as a point query and sampler.

    Erosion uses the map's own ``_erode_trav_map`` so this agrees exactly with
    the filter in :mod:`coop2.behavior_env.symbolic_navigation` -- a placement
    that disagreed with the navigation filter would drop robots on spots the
    sampler then refuses to navigate back to.
    """

    def __init__(self, scene, robot):
        self.scene = scene
        self.robot = robot
        self.trav_map = getattr(scene, "trav_map", None)
        self._eroded = None
        self._component = None
        self._rooms = None

    @property
    def available(self) -> bool:
        return self.trav_map is not None and bool(getattr(self.trav_map, "floor_map", None))

    @property
    def eroded(self):
        if self._eroded is None and self.available:
            self._eroded = self.trav_map._erode_trav_map(th.clone(self.trav_map.floor_map[0]), robot=self.robot)
        return self._eroded

    def is_free(self, xy) -> bool:
        """Does the robot fit at world @xy? True when the scene has no map."""
        eroded = self.eroded
        if eroded is None:
            return True
        row, col = self.trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])]))
        row, col = int(row), int(col)
        if not (0 <= row < eroded.shape[0] and 0 <= col < eroded.shape[1]):
            return False
        return bool(eroded[row][col] == 255)

    def largest_component(self):
        """World-frame ``(N, 2)`` points of the biggest connected free region.

        Connectivity matters: robots scattered across disconnected pockets can
        never reach each other or a shared target, which would silently make
        every cooperation metric meaningless.
        """
        if self._component is not None:
            return self._component
        eroded = self.eroded
        if eroded is None:
            return None
        import cv2  # noqa: PLC0415 - only needed here, and OmniGibson already depends on it

        mask = (eroded.cpu().numpy() == 255).astype("uint8")
        count, labels = cv2.connectedComponents(mask, connectivity=4)
        best_label, best_size = 0, 0
        for label in range(1, count):
            size = int((labels == label).sum())
            if size > best_size:
                best_label, best_size = label, size
        if best_size == 0:
            self._component = th.zeros((0, 2))
            return self._component
        rows, cols = (labels == best_label).nonzero()
        points = [self.trav_map.map_to_world(th.tensor([int(r), int(c)])) for r, c in zip(rows, cols)]
        self._component = th.stack(points) if points else th.zeros((0, 2))
        return self._component

    def _free_cells_map_space(self):
        """``(rows, cols)`` of every free cell, in map indices."""
        eroded = self.eroded
        if eroded is None:
            return None
        import numpy as np  # noqa: PLC0415

        return np.nonzero((eroded.cpu().numpy() == 255))

    def cells_by_room(self, seg_map=None):
        """``{room_instance: [(x, y), ...]}`` over **all** free cells.

        Deliberately not restricted to the map's largest connected component.
        Doing that bucketed only the cells connected to the biggest region --
        the garden, in house_single_floor -- so the real living room (16.7 m2,
        enough for nine R1s) never appeared as a candidate at all and placement
        fell back to a 2.75 m2 corridor that merely touched the garden.
        """
        if self._rooms is not None:
            return self._rooms
        free = self._free_cells_map_space()
        seg_map = seg_map if seg_map is not None else getattr(self.scene, "_seg_map", None)
        if free is None or seg_map is None:
            self._rooms = {}
            return self._rooms
        rooms = {}
        for row, col in zip(*free):
            xy = self.trav_map.map_to_world(th.tensor([int(row), int(col)]))
            room = seg_map.get_room_instance_by_point(xy)
            if room is None:
                continue
            rooms.setdefault(room, []).append((float(xy[0]), float(xy[1])))
        self._rooms = rooms
        return rooms

    def room_component(self, room: str) -> List[Tuple[float, float]]:
        """The largest *connected* run of free cells inside @room.

        Connectivity is checked within the room, not globally: robots dropped
        into two disconnected pockets of the same room could never reach each
        other, and every cooperation metric would quietly read zero.
        """
        cells = self.cells_by_room().get(room)
        if not cells:
            return []
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        indices = [self.trav_map.world_to_map(th.tensor([x, y])) for x, y in cells]
        rows = np.array([int(i[0]) for i in indices])
        cols = np.array([int(i[1]) for i in indices])
        mask = np.zeros(tuple(self.eroded.shape), dtype="uint8")
        mask[rows, cols] = 1
        count, labels = cv2.connectedComponents(mask, connectivity=4)
        best_label, best_size = 0, 0
        for label in range(1, count):
            size = int((labels == label).sum())
            if size > best_size:
                best_label, best_size = label, size
        if best_size == 0:
            return []
        keep = labels == best_label
        return [(x, y) for (x, y), r, c in zip(cells, rows, cols) if keep[r][c]]

    def pick_room(self, prefer_indoor: bool = True, seg_map=None) -> Optional[str]:
        """The room with the most connected free space, preferring indoor ones."""
        rooms = self.cells_by_room(seg_map=seg_map)
        if not rooms:
            return None
        outdoor = ("garden", "lawn", "yard", "porch", "patio", "driveway", "deck", "balcony")
        indoor = {k: v for k, v in rooms.items() if not any(tag in k.lower() for tag in outdoor)}
        pool = indoor if (prefer_indoor and indoor) else rooms
        # Rank on the connected run, not the raw cell count: a room chopped in
        # half by furniture is not twice as useful as one half of it.
        return max(pool, key=lambda k: len(self.room_component(k)))

    def sample_separated(
        self,
        count: int,
        separation: float,
        seed: Optional[int] = None,
        attempts_per_point: int = 4000,
        pool: Optional[Sequence[Tuple[float, float]]] = None,
        exclude: Optional[Sequence[Tuple[float, float]]] = None,
        cluster_radius: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """@count points from @pool, >= @separation apart and clear of @exclude.

        ``exclude`` are positions already taken (other robots, objects); they
        constrain the result but are not returned.
        """
        if pool is None:
            points = self.largest_component()
            pool = [] if points is None else [(float(p[0]), float(p[1])) for p in points]
        if not len(pool):
            raise RuntimeError(
                "No traversable region in this scene once eroded by the robot radius. "
                "The scene cannot host even one robot -- pick another (see "
                "feasibility_verify/survey_scene_capacity.py)."
            )
        points = pool
        generator = th.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
        if cluster_radius is not None:
            # Picking a room is not enough: house_single_floor's roomiest space
            # is a 20 m corridor, which spread nine robots from y=-1.9 to
            # y=18.6. Agents that cannot reach each other inside the episode
            # budget make every cooperation metric read zero, so anchor on one
            # cell and keep the team within reach of it.
            anchor_index = int(th.randint(0, len(points), (1,), generator=generator).item())
            ax, ay = points[anchor_index]
            near = [(x, y) for x, y in points if math.hypot(x - ax, y - ay) <= cluster_radius]
            if len(near) >= count:
                points = near
        chosen: List[Tuple[float, float]] = list(exclude or [])
        keep_from = len(chosen)
        for _ in range(count):
            for _ in range(attempts_per_point):
                index = int(th.randint(0, len(points), (1,), generator=generator).item())
                x, y = float(points[index][0]), float(points[index][1])
                if all(math.hypot(x - px, y - py) >= separation for px, py in chosen):
                    chosen.append((x, y))
                    break
            else:
                raise RuntimeError(
                    f"Could only place {len(chosen) - keep_from} of {count} points at "
                    f"{separation:.2f} m separation in a region of {len(points)} cells. "
                    "Use a bigger room, a bigger scene, or fewer robots."
                )
        return chosen[keep_from:]

    def standable_fraction(self, xy, lo: float, hi: float, samples: int = 60, seed: Optional[int] = None) -> float:
        """Fraction of poses in the ``[lo, hi]`` annulus around @xy that fit.

        This is what decides whether an object is actually usable: a target with
        zero standable poses around it makes NAVIGATE_TO fail outright, and no
        filter can rescue it.
        """
        generator = th.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
        ok = 0
        for _ in range(samples):
            distance = th.rand(1, generator=generator).item() * (hi - lo) + lo
            yaw = th.rand(1, generator=generator).item() * 2 * math.pi - math.pi
            if self.is_free((float(xy[0]) + distance * math.cos(yaw), float(xy[1]) + distance * math.sin(yaw))):
                ok += 1
        return ok / samples


def look_at_quaternion(eye, target) -> th.Tensor:
    """xyzw quaternion for a camera at @eye looking at @target (USD: -Z forward)."""
    eye = [float(v) for v in eye]
    target = [float(v) for v in target]
    forward = [target[i] - eye[i] for i in range(3)]
    norm = math.sqrt(sum(c * c for c in forward)) or 1.0
    forward = [c / norm for c in forward]
    # USD cameras look down -Z with +Y up.
    z = [-c for c in forward]
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


def place_robots(
    env,
    separation: Optional[float] = None,
    seed: Optional[int] = None,
    z: float = 0.05,
    room: Optional[str] = None,
    prefer_indoor: bool = True,
    cluster_radius: Optional[float] = 6.0,
) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    """Teleport every robot into one room, mutually separated and traversable.

    Call right after ``og.Environment(...)`` and before ``prepare_robots``.

    Everything lands in a **single room** on purpose. Sampling the whole
    traversable region instead put two agents 47 m apart in house_single_floor
    (both outdoors, in ``garden_0``), which cost one of them a 2000-tick
    NAVIGATE_TO timeout before it ever reached the shared target. Cooperation
    cannot be measured across a distance neither agent can cross inside the
    episode budget.

    Args:
        separation: centre-to-centre minimum. ``None`` uses twice the robot's
            circumscribed radius -- just-touching, which is all that is needed:
            navigation teleports, so there is no path to keep clear.
        room: room instance to use. ``None`` picks the largest, preferring
            indoor ones (the biggest connected region is often the garden).
        cluster_radius: keep every robot within this distance of a random
            anchor cell in the room. A room can be long and thin -- the roomiest
            indoor space in house_single_floor is a 20 m corridor -- and agents
            spread along it never meet.

    Returns:
        ``(positions, room)``.
    """
    robots = list(env.robots)
    if not robots:
        raise ValueError("No robots in the environment.")
    index = TraversabilityIndex(env.scene, robots[0])
    if separation is None:
        extent = robots[0].reset_joint_pos_aabb_extent[:2]
        separation = float(th.norm(extent))

    chosen_room = room if room is not None else index.pick_room(prefer_indoor=prefer_indoor)
    pool = None
    if chosen_room is not None:
        pool = index.cells_by_room().get(chosen_room)
        if pool:
            print(f"[placement] room {chosen_room!r}: {len(pool)} free cells")
    positions = index.sample_separated(
        len(robots), separation, seed=seed, pool=pool, cluster_radius=cluster_radius
    )
    for robot, (x, y) in zip(robots, positions):
        robot.set_position_orientation(
            position=th.tensor([x, y, z], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
    return positions, chosen_room


def place_objects(
    env,
    names: Sequence[str],
    annulus: Tuple[float, float],
    seed: Optional[int] = None,
    min_standable: float = 0.15,
    min_separation: float = 1.0,
    z: float = 0.05,
    candidates: int = 400,
    room: Optional[str] = None,
    near_robots: Optional[float] = 8.0,
) -> List[Tuple[float, float]]:
    """Move each named object somewhere reachable, in @room, clear of the robots.

    Rejects positions whose surrounding annulus has fewer than @min_standable
    standable poses -- those are the "dead targets" that make NAVIGATE_TO fail
    outright no matter what the sampler does.

    Robot positions are excluded: sampling from the same free cells as
    :func:`place_robots` once dropped an apple exactly on top of a robot, which
    made its NAVIGATE_TO trivially free and the race meaningless.
    """
    robots = list(env.robots)
    index = TraversabilityIndex(env.scene, robots[0])
    pool = None
    if room is not None:
        pool = index.room_component(room)
    if not pool:
        points = index.largest_component()
        pool = [] if points is None else [(float(p[0]), float(p[1])) for p in points]
    if not pool:
        raise RuntimeError("No free cells to place objects in.")

    generator = th.Generator()
    if seed is not None:
        generator.manual_seed(int(seed) + 9973)  # decorrelate from robot placement

    if near_robots is not None and robots:
        # place_robots clusters the team, but objects were still sampled from
        # the whole room. In a 20 m corridor that put the contested apple 11 m
        # from the nearest agent: a 1728-tick NAVIGATE_TO, most of an episode
        # budget spent walking before any cooperation could happen.
        centre_x = sum(float(r.get_position_orientation()[0][0]) for r in robots) / len(robots)
        centre_y = sum(float(r.get_position_orientation()[0][1]) for r in robots) / len(robots)
        near = [(x, y) for x, y in pool if math.hypot(x - centre_x, y - centre_y) <= near_robots]
        if len(near) > len(names):
            pool = near

    taken = [
        (float(r.get_position_orientation()[0][0]), float(r.get_position_orientation()[0][1])) for r in robots
    ]
    chosen: List[Tuple[float, float]] = []
    for name in names:
        obj = env.scene.object_registry("name", name)
        if obj is None:
            raise KeyError(f"No object named {name!r} in the scene.")
        best = None
        for _ in range(candidates):
            i = int(th.randint(0, len(pool), (1,), generator=generator).item())
            x, y = pool[i]
            if any(math.hypot(x - px, y - py) < min_separation for px, py in chosen + taken):
                continue
            fraction = index.standable_fraction((x, y), annulus[0], annulus[1])
            if fraction >= min_standable:
                best = (x, y, fraction)
                break
            if best is None or fraction > best[2]:
                best = (x, y, fraction)
        if best is None:
            raise RuntimeError(f"Could not place {name} anywhere clear of the robots in this region.")
        x, y, fraction = best
        obj.set_position_orientation(position=th.tensor([x, y, z], dtype=th.float32))
        chosen.append((x, y))
        print(f"[placement] {name} -> ({x:.2f}, {y:.2f})  standable poses around it: {fraction:.0%}")
    return chosen
