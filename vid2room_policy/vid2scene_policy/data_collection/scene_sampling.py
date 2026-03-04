import math
import logging
import random
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch as th

import omnigibson as og

logger = logging.getLogger(__name__)

MIN_SUPPORT_HEIGHT = 0.2
MAX_GRIPPER_OPENING = 0.12
MIN_OBJECT_HEIGHT = 0.03
SUPPORT_XY_MARGIN_M = 0.05
SUPPORT_Z_LOWER_MARGIN_M = 0.05
SUPPORT_Z_UPPER_MARGIN_M = 0.5


def get_support_position_2d(support) -> th.Tensor:
    """Get the XY world position of a support object center."""
    pos = support.get_position_orientation()[0]
    return th.tensor([pos[0].item(), pos[1].item()])


def _distance_point_to_aabb_2d(
    x: float,
    y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> float:
    """Compute Euclidean distance from a 2D point to an axis-aligned rectangle."""
    dx = max(min_x - x, 0.0, x - max_x)
    dy = max(min_y - y, 0.0, y - max_y)
    return math.sqrt(dx * dx + dy * dy)


def sample_robot_start_near_support(
    scene,
    robot,
    support,
    floor_z: float,
    max_dist_m: float,
    search_radius_m: float = 2.5,
    erosion_extra_margin_m: float = 0.15,
    ignored_obstacle_categories: tuple[str, ...] | None = None,
) -> tuple | None:
    """Find a deterministic base start pose near a support.

    This searches the eroded traversability map locally around the support and picks
    the traversable pose that minimizes distance to the support AABB in XY.
    """
    trav_map = scene._trav_map
    floor_np = trav_map.floor_map[0].cpu().numpy().copy()
    floor_np = _clear_ignored_obstacles_from_floor_map(
        scene=scene,
        trav_map=trav_map,
        floor_np=floor_np,
        ignored_obstacle_categories=ignored_obstacle_categories,
    )
    eroded = _erode_for_support_sampling(
        floor_np=floor_np,
        trav_map=trav_map,
        robot=robot,
        erosion_extra_margin_m=erosion_extra_margin_m,
    )

    support_pos_2d = get_support_position_2d(support)
    center_map = trav_map.world_to_map(support_pos_2d)
    center_r, center_c = int(center_map[0].item()), int(center_map[1].item())

    search_radius_px = int(math.ceil(search_radius_m / trav_map.map_resolution))
    h, w = eroded.shape
    r0 = max(0, center_r - search_radius_px)
    r1 = min(h - 1, center_r + search_radius_px)
    c0 = max(0, center_c - search_radius_px)
    c1 = min(w - 1, center_c + search_radius_px)

    support_aabb = support.aabb
    min_x = float(support_aabb[0][0].item())
    max_x = float(support_aabb[1][0].item())
    min_y = float(support_aabb[0][1].item())
    max_y = float(support_aabb[1][1].item())
    center_x = float(support_pos_2d[0].item())
    center_y = float(support_pos_2d[1].item())

    best_xy = None
    best_edge_dist = float("inf")
    best_center_dist = float("inf")

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if eroded[r, c] != 255:
                continue

            world_pos = trav_map.map_to_world(th.tensor([r, c]))
            x = float(world_pos[0].item())
            y = float(world_pos[1].item())

            edge_dist = _distance_point_to_aabb_2d(x, y, min_x, max_x, min_y, max_y)
            if edge_dist > max_dist_m:
                continue

            dx = center_x - x
            dy = center_y - y
            center_dist = math.sqrt(dx * dx + dy * dy)

            if (edge_dist < best_edge_dist) or (
                math.isclose(edge_dist, best_edge_dist) and center_dist < best_center_dist
            ):
                best_edge_dist = edge_dist
                best_center_dist = center_dist
                best_xy = (x, y)

    if best_xy is None:
        return None

    facing_yaw = math.atan2(center_y - best_xy[1], center_x - best_xy[0])
    return (best_xy[0], best_xy[1], floor_z, facing_yaw)


def _erode_for_support_sampling(
    floor_np: np.ndarray,
    trav_map,
    robot,
    erosion_extra_margin_m: float,
) -> np.ndarray:
    """Erode map for support sampling with configurable safety margin."""
    robot_chassis_extent = robot.reset_joint_pos_aabb_extent[:2]
    base_radius_m = float((th.norm(robot_chassis_extent) / 2.0).item())
    erosion_radius_m = max(0.0, base_radius_m + float(erosion_extra_margin_m))
    erosion_radius_px = int(math.ceil(erosion_radius_m / trav_map.map_resolution))

    if erosion_radius_px <= 0:
        return floor_np

    kernel = np.ones((erosion_radius_px, erosion_radius_px), dtype=np.uint8)
    return cv2.erode(floor_np, kernel)


def _clear_ignored_obstacles_from_floor_map(
    scene,
    trav_map,
    floor_np: np.ndarray,
    ignored_obstacle_categories: tuple[str, ...] | None,
) -> np.ndarray:
    """Mark ignored-category object footprints as traversable before erosion."""
    if not ignored_obstacle_categories:
        return floor_np

    ignored = set(ignored_obstacle_categories)
    h, w = floor_np.shape
    cleared = 0

    for obj in scene.objects:
        category = getattr(obj, "category", None)
        if category not in ignored:
            continue

        try:
            aabb = obj.aabb
            min_xy = th.tensor([aabb[0][0].item(), aabb[0][1].item()])
            max_xy = th.tensor([aabb[1][0].item(), aabb[1][1].item()])
            min_rc = trav_map.world_to_map(min_xy)
            max_rc = trav_map.world_to_map(max_xy)

            r0 = max(0, min(int(min_rc[0].item()), int(max_rc[0].item())))
            r1 = min(h - 1, max(int(min_rc[0].item()), int(max_rc[0].item())))
            c0 = max(0, min(int(min_rc[1].item()), int(max_rc[1].item())))
            c1 = min(w - 1, max(int(min_rc[1].item()), int(max_rc[1].item())))

            if r0 <= r1 and c0 <= c1:
                floor_np[r0 : r1 + 1, c0 : c1 + 1] = 255
                cleared += 1
        except Exception:
            logger.debug(
                "Failed clearing ignored obstacle footprint for object %s",
                getattr(obj, "name", "<unknown>"),
                exc_info=True,
            )
            continue

    if cleared > 0:
        print(
            f"[Episode] Cleared {cleared} ignored obstacles from trav map: {sorted(ignored)}",
            flush=True,
        )
    return floor_np


def map_supports_to_rooms(scene, supports: list) -> dict[int, list]:
    """Map supports to rooms using scene._seg_map (BEHAVIOR-1K built-in).

    Uses get_room_instance_by_point() which handles coordinate conversion internally.

    Returns:
        room_supports dict keyed by room instance ID.
    """
    seg_map = scene._seg_map

    room_ins_np = seg_map.room_ins_map.cpu().numpy()
    unique_rooms = np.unique(room_ins_np)
    print(f"[Debug] seg_map.room_ins_map: shape={room_ins_np.shape}, rooms={list(unique_rooms)}", flush=True)

    room_supports: dict[int, list] = {}
    for sup in supports:
        pos, _ = sup.get_position_orientation()
        x, y = pos[0].item(), pos[1].item()

        room_name = seg_map.get_room_instance_by_point(th.tensor([x, y]))

        if room_name is not None:
            room_ins_id = seg_map.room_ins_name_to_ins_id.get(room_name, 0)
            if room_ins_id > 0:
                print(
                    f"[Debug] Support {sup.name} at ({x:.2f}, {y:.2f}) -> {room_name} (id={room_ins_id})",
                    flush=True,
                )
                if room_ins_id not in room_supports:
                    room_supports[room_ins_id] = []
                room_supports[room_ins_id].append(sup)
            else:
                print(
                    f"[Debug] Support {sup.name} at ({x:.2f}, {y:.2f}) -> {room_name} (invalid id)",
                    flush=True,
                )
        else:
            print(f"[Debug] Support {sup.name} at ({x:.2f}, {y:.2f}) -> outside rooms", flush=True)

    return room_supports


def get_valid_supports(scene, is_support_fn: Callable[[str], bool], max_support_height: float) -> list:
    """Return supports that are within the allowed height range and pass the support filter."""
    valid = []
    print(
        f"[Debug] Checking supports. Height range: {MIN_SUPPORT_HEIGHT} - {max_support_height:.2f}",
        flush=True,
    )
    for obj in scene.objects:
        category = getattr(obj, "category", None)
        if category is None:
            continue
        is_support = is_support_fn(category)
        if is_support:
            try:
                aabb = obj.aabb
                surface_height = aabb[1][2].item()
                height_ok = MIN_SUPPORT_HEIGHT < surface_height < max_support_height
                print(
                    f"[Debug] Support candidate: {obj.name} (cat={category}) height={surface_height:.2f} -> "
                    f"{'OK' if height_ok else 'REJECTED'}",
                    flush=True,
                )
                if height_ok:
                    valid.append(obj)
            except Exception as exc:
                print(
                    f"[Debug] Support candidate: {obj.name} (cat={category}) -> EXCEPTION: {exc}",
                    flush=True,
                )
    return valid


def object_fits_gripper(obj) -> bool:
    """Check if an object fits inside the gripper aperture and is tall enough."""
    try:
        aabb = obj.aabb
        size_x = aabb[1][0].item() - aabb[0][0].item()
        size_y = aabb[1][1].item() - aabb[0][1].item()
        size_z = aabb[1][2].item() - aabb[0][2].item()
        min_xy = min(size_x, size_y)
        return min_xy <= MAX_GRIPPER_OPENING and size_z >= MIN_OBJECT_HEIGHT
    except Exception:
        logger.debug(
            "Failed to evaluate gripper fit for object %s",
            getattr(obj, "name", "<unknown>"),
            exc_info=True,
        )
        return False


def find_graspable_objects_on_support(scene, support, is_graspable_fn: Callable[[str], bool]) -> list:
    """Return graspable objects that lie on top of the given support."""
    support_aabb = support.aabb
    support_min_x = support_aabb[0][0].item()
    support_max_x = support_aabb[1][0].item()
    support_min_y = support_aabb[0][1].item()
    support_max_y = support_aabb[1][1].item()
    support_surface_z = support_aabb[1][2].item()

    objects_on_support = []
    for obj in scene.objects:
        category = getattr(obj, "category", None)
        if category is None:
            continue
        if not is_graspable_fn(category):
            continue
        if not object_fits_gripper(obj):
            continue

        try:
            obj_pos, _ = obj.get_position_orientation()
            obj_x = obj_pos[0].item()
            obj_y = obj_pos[1].item()
            obj_z = obj_pos[2].item()

            x_on = support_min_x - SUPPORT_XY_MARGIN_M <= obj_x <= support_max_x + SUPPORT_XY_MARGIN_M
            y_on = support_min_y - SUPPORT_XY_MARGIN_M <= obj_y <= support_max_y + SUPPORT_XY_MARGIN_M
            z_on = (
                support_surface_z - SUPPORT_Z_LOWER_MARGIN_M
                <= obj_z
                <= support_surface_z + SUPPORT_Z_UPPER_MARGIN_M
            )

            if x_on and y_on and z_on:
                objects_on_support.append(obj)
        except Exception:
            logger.debug(
                "Failed support-membership check for object %s on support %s",
                getattr(obj, "name", "<unknown>"),
                getattr(support, "name", "<unknown>"),
                exc_info=True,
            )
            continue

    return objects_on_support


def _compute_object_approachability(obj, scene, robot) -> float:
    trav_map = scene._trav_map
    eroded_map = trav_map._erode_trav_map(trav_map.floor_map[0].clone(), robot=robot)

    obj_pos, _ = obj.get_position_orientation()
    obj_x = obj_pos[0].item()
    obj_y = obj_pos[1].item()

    score = 0.0
    good_directions = 0

    for angle_deg in range(0, 360, 30):
        angle = math.radians(angle_deg)
        for dist in [0.4, 0.5, 0.6]:
            check_x = obj_x + dist * math.cos(angle)
            check_y = obj_y + dist * math.sin(angle)

            check_map = trav_map.world_to_map(th.tensor([check_x, check_y]))
            cr, cc = int(check_map[0].item()), int(check_map[1].item())

            if 0 <= cr < eroded_map.shape[0] and 0 <= cc < eroded_map.shape[1]:
                val = eroded_map[cr, cc]
                if hasattr(val, "item"):
                    val = val.item()
                if val == 255:
                    clearance = 0
                    for clear_angle in range(0, 360, 45):
                        clear_rad = math.radians(clear_angle)
                        for clear_dist in [0.3, 0.5]:
                            cx = check_x + clear_dist * math.cos(clear_rad)
                            cy = check_y + clear_dist * math.sin(clear_rad)
                            cm = trav_map.world_to_map(th.tensor([cx, cy]))
                            cmr, cmc = int(cm[0].item()), int(cm[1].item())
                            if 0 <= cmr < eroded_map.shape[0] and 0 <= cmc < eroded_map.shape[1]:
                                cv = eroded_map[cmr, cmc]
                                if hasattr(cv, "item"):
                                    cv = cv.item()
                                if cv == 255:
                                    clearance += 1

                    if clearance >= 8:
                        good_directions += 1
                        score += clearance

    return score if good_directions >= 2 else 0


def select_best_graspable_object(objects: list, scene, robot):
    """Choose the most approachable graspable object from the candidate list."""
    if not objects:
        return None
    if len(objects) == 1:
        return objects[0]

    scored = []
    for obj in objects:
        score = _compute_object_approachability(obj, scene, robot)
        scored.append((obj, score))

    scored.sort(key=lambda x: -x[1])

    good_options = [s for s in scored if s[1] > 0]
    if good_options:
        top_options = good_options[: min(3, len(good_options))]
        chosen = random.choice(top_options)
        print(f"[Episode] Object scores: {[(s[0].name, s[1]) for s in scored[:5]]}", flush=True)
        return chosen[0]
    else:
        return random.choice(objects)

