import logging
import math
import random
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from .config import DataCollectionConfig, get_object_filters
from .data_collector import DataCollector
from .scene_management import spawn_and_place_object, safe_remove_object
from .omnigibson_lerobot_wrapper import OmniGibsonLeRobotWrapper, OmniGibsonLeRobotConfig

logger = logging.getLogger(__name__)

DEBUG_EPISODE = False
DEBUG_OUTPUT_DIR = "/home/yalcintr/workspace/vid2scene_policy/vid2scene_policy"

MIN_SUPPORT_HEIGHT = 0.3
MAX_GRIPPER_OPENING = 0.12
MIN_OBJECT_HEIGHT = 0.03


def _find_nearest_traversable(trav_map, pos_2d: th.Tensor, robot, max_dist: float = 2.0) -> th.Tensor | None:
    """Find nearest traversable point on the eroded map."""
    floor_map = trav_map.floor_map[0].clone()
    # Erode like get_shortest_path does
    robot_chassis_extent = robot.reset_joint_pos_aabb_extent[:2]
    radius = th.norm(robot_chassis_extent) / 2.0 + 0.5
    radius_pixel = int(math.ceil(radius.item() / trav_map.map_resolution))
    eroded = th.tensor(cv2.erode(floor_map.cpu().numpy(), th.ones((radius_pixel, radius_pixel)).cpu().numpy()))

    # Check if already traversable
    map_pos = trav_map.world_to_map(pos_2d)
    r, c = int(map_pos[0].item()), int(map_pos[1].item())
    if 0 <= r < eroded.shape[0] and 0 <= c < eroded.shape[1] and eroded[r, c] == 255:
        return pos_2d

    # Search in expanding circles
    for dist in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        if dist > max_dist:
            break
        for angle in range(0, 360, 15):
            rad = math.radians(angle)
            test_x = pos_2d[0].item() + dist * math.cos(rad)
            test_y = pos_2d[1].item() + dist * math.sin(rad)
            test_map = trav_map.world_to_map(th.tensor([test_x, test_y]))
            tr, tc = int(test_map[0].item()), int(test_map[1].item())
            if 0 <= tr < eroded.shape[0] and 0 <= tc < eroded.shape[1] and eroded[tr, tc] == 255:
                return th.tensor([test_x, test_y])
    return None


def _check_path_exists(scene, robot, source_pos_2d: th.Tensor, target_pos_2d: th.Tensor) -> bool:
    """Check if a path exists between two points using trav_map.get_shortest_path.

    Finds nearest traversable points first (supports are on furniture = non-traversable).
    """
    trav_map = scene._trav_map

    # Find nearest traversable points (supports are on furniture)
    src_trav = _find_nearest_traversable(trav_map, source_pos_2d, robot)
    tgt_trav = _find_nearest_traversable(trav_map, target_pos_2d, robot)

    if src_trav is None or tgt_trav is None:
        return False

    path, dist = trav_map.get_shortest_path(
        floor=0,
        source_world=src_trav,
        target_world=tgt_trav,
        entire_path=False,
        robot=robot
    )
    return path is not None


def _get_support_position_2d(support) -> th.Tensor:
    """Get 2D position of a support surface."""
    pos = support.get_position_orientation()[0]
    return th.tensor([pos[0].item(), pos[1].item()])


def _compute_room_connected_components(scene, room_ins_id: int) -> tuple[np.ndarray, int]:
    """Compute connected components for traversable areas within a room.

    Uses scene._seg_map.room_ins_map (already at trav_map resolution).
    Uses same erosion (0.3m) and room mask as navigation.

    Returns: (labels array at trav_map resolution, num_components)
    """
    trav_map = scene._trav_map
    seg_map = scene._seg_map

    floor_np = trav_map.floor_map[0].cpu().numpy().copy()
    room_ins_np = seg_map.room_ins_map.cpu().numpy()

    # Create room mask and AND with trav_map (already same resolution)
    room_mask = ((room_ins_np == room_ins_id) * 255).astype(np.uint8)
    room_trav = np.minimum(floor_np, room_mask)

    # Apply 0.3m erosion (same as navigation)
    erosion_radius_m = 0.35
    erosion_radius_px = int(math.ceil(erosion_radius_m / trav_map.map_resolution))
    if erosion_radius_px > 0:
        kernel = np.ones((erosion_radius_px, erosion_radius_px), dtype=np.uint8)
        room_trav = cv2.erode(room_trav, kernel)

    # Compute connected components (binary: 0 or 1)
    binary = (room_trav == 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)

    return labels, num_labels


def _get_support_component(support, labels: np.ndarray, trav_map, max_search_dist: float = 2.0) -> int:
    """Find which connected component a support belongs to.

    Searches for nearest traversable pixel since supports are on furniture (non-traversable).

    Returns: component label (0 = no component found)
    """
    pos, _ = support.get_position_orientation()
    x, y = pos[0].item(), pos[1].item()

    # Convert to map coordinates
    map_pos = trav_map.world_to_map(th.tensor([x, y]))
    center_r, center_c = int(map_pos[0].item()), int(map_pos[1].item())

    h, w = labels.shape

    # Check center first
    if 0 <= center_r < h and 0 <= center_c < w:
        label = labels[center_r, center_c]
        if label > 0:
            return int(label)

    # Search in expanding circles
    max_dist_px = int(max_search_dist / trav_map.map_resolution)
    for dist in range(1, max_dist_px + 1):
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                if abs(dr) != dist and abs(dc) != dist:
                    continue  # Only check perimeter
                r, c = center_r + dr, center_c + dc
                if 0 <= r < h and 0 <= c < w:
                    label = labels[r, c]
                    if label > 0:
                        return int(label)

    return 0  # No component found


def _find_robot_start_in_component(scene, robot, labels: np.ndarray, component_id: int,
                                    floor_z: float, max_attempts: int = 100) -> tuple | None:
    """Find a random robot start position within a specific connected component.

    This guarantees the robot can reach any point in the same component.
    """
    trav_map = scene._trav_map

    # Get all pixels in this component
    component_pixels = np.argwhere(labels == component_id)
    if len(component_pixels) == 0:
        return None

    for _ in range(max_attempts):
        # Sample random position in component
        idx = random.randint(0, len(component_pixels) - 1)
        r, c = component_pixels[idx]

        # Convert to world coordinates
        world_pos = trav_map.map_to_world(th.tensor([r, c]))
        x, y = world_pos[0].item(), world_pos[1].item()

        yaw = random.uniform(-math.pi, math.pi)
        return (x, y, floor_z, yaw)

    return None


def _save_trav_map_raw(scene, room_supports: dict):
    """Save raw traversability map with all supports marked (no path)."""
    
    try:
        # Load both maps directly from files at original resolution (1692x1692)
        layout_dir = Path(scene.scene_dir) / "layout"
        trav_path = layout_dir / "floor_trav_0.png"
        insseg_path = layout_dir / "floor_insseg_0.png"

        floor_map = cv2.imread(str(trav_path), cv2.IMREAD_GRAYSCALE)
        room_ins_map = cv2.imread(str(insseg_path), cv2.IMREAD_GRAYSCALE)
        print(f"[Debug] Loaded floor_trav_0.png: shape={floor_map.shape}", flush=True)
        print(f"[Debug] Loaded floor_insseg_0.png: shape={room_ins_map.shape}", flush=True)

        # Create color image
        img = np.zeros((floor_map.shape[0], floor_map.shape[1], 3), dtype=np.uint8)
        img[floor_map == 0] = [0, 0, 100]  # Dark red for walls
        img[floor_map == 255] = [200, 200, 200]  # Gray for traversable

        # World to map conversion at original resolution (0.01m/pixel)
        def world_to_map_cv(x, y):
            resolution = 0.01
            col = int(x / resolution + floor_map.shape[1] / 2)
            row = int(y / resolution + floor_map.shape[0] / 2)
            return col, row

        # Mark all supports
        for room_name, supports in room_supports.items():
            for sup in supports:
                pos = sup.get_position_orientation()[0]
                col, row = world_to_map_cv(pos[0].item(), pos[1].item())
                cv2.circle(img, (col, row), 8, (0, 0, 255), -1)
                cv2.putText(img, f"{sup.name[:15]}({room_name[:10]})", (col + 10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/trav_map_debug.png", img)
        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/trav_map_raw.png", floor_map)

        room_vis = (room_ins_map * 80).astype(np.uint8)
        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/room_segmap.png", room_vis)

        # Logical AND of trav_map and room segmap
        room_mask = (room_ins_map > 0).astype(np.uint8) * 255
        trav_and_room = cv2.bitwise_and(floor_map, room_mask)
        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/trav_and_room.png", trav_and_room)

        # Combined visualization - each room gets a different color
        combined = np.zeros((floor_map.shape[0], floor_map.shape[1], 3), dtype=np.uint8)
        combined[floor_map == 0] = [50, 50, 50]  # Dark gray for walls
        # Define distinct colors for rooms (BGR format)
        room_colors = [
            [0, 255, 0],    # Green
            [255, 0, 0],    # Blue
            [0, 255, 255],  # Yellow
            [255, 0, 255],  # Magenta
            [255, 255, 0],  # Cyan
            [0, 165, 255],  # Orange
            [147, 20, 255], # Pink
            [0, 128, 0],    # Dark green
        ]
        unique_rooms = np.unique(room_ins_map)
        for i, room_id in enumerate(unique_rooms):
            if room_id == 0:
                continue  # Skip background
            color = room_colors[i % len(room_colors)]
            # Only color traversable areas within this room
            room_trav = (room_ins_map == room_id) & (floor_map == 255)
            combined[room_trav] = color
        # Mark non-traversable room areas darker
        for i, room_id in enumerate(unique_rooms):
            if room_id == 0:
                continue
            color = room_colors[i % len(room_colors)]
            dark_color = [c // 3 for c in color]  # Darker version
            room_non_trav = (room_ins_map == room_id) & (floor_map == 0)
            combined[room_non_trav] = dark_color
        # Mark supports
        for room_name, supports in room_supports.items():
            for sup in supports:
                pos = sup.get_position_orientation()[0]
                col, row = world_to_map_cv(pos[0].item(), pos[1].item())
                cv2.circle(combined, (col, row), 4, (0, 0, 255), -1)
                cv2.circle(combined, (col, row), 4, (255, 255, 255), 1)
                cv2.putText(combined, f"{sup.name[:12]}({room_name[:8]})", (col + 6, row), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/trav_combined.png", combined)

        print(f"[TravMap] Saved to {DEBUG_OUTPUT_DIR}/trav_map_debug.png, trav_map_raw.png, room_segmap.png, trav_and_room.png, trav_combined.png", flush=True)
    except Exception as e:
        import traceback
        print(f"[TravMap] Failed to save: {e}", flush=True)
        traceback.print_exc()


def _save_trav_map_visualization(scene, robot, room_supports: dict, source_support, target_support):
    """Save traversability map with supports marked and path between selected pair."""
    
    try:
        # Load both maps directly from files at original resolution (1692x1692)
        layout_dir = Path(scene.scene_dir) / "layout"
        trav_path = layout_dir / "floor_trav_0.png"
        insseg_path = layout_dir / "floor_insseg_0.png"

        floor_map = cv2.imread(str(trav_path), cv2.IMREAD_GRAYSCALE)
        room_ins_map = cv2.imread(str(insseg_path), cv2.IMREAD_GRAYSCALE)
        print(f"[Debug] Loaded floor_trav_0.png: shape={floor_map.shape}", flush=True)
        print(f"[Debug] Loaded floor_insseg_0.png: shape={room_ins_map.shape}", flush=True)

        # Create color image - walls/obstacles in dark red, traversable in gray
        img = np.zeros((floor_map.shape[0], floor_map.shape[1], 3), dtype=np.uint8)
        img[floor_map == 0] = [0, 0, 100]  # Dark red (BGR) for walls/obstacles
        img[floor_map == 255] = [200, 200, 200]  # Gray for traversable

        # World to map conversion at original resolution (0.01m/pixel)
        def world_to_map_cv(x, y):
            resolution = 0.01
            col = int(x / resolution + floor_map.shape[1] / 2)
            row = int(y / resolution + floor_map.shape[0] / 2)
            return col, row

        # Mark all supports
        print(f"[TravMap] Supports by room:", flush=True)
        for room_name, supports in room_supports.items():
            for sup in supports:
                pos = sup.get_position_orientation()[0]
                x, y = pos[0].item(), pos[1].item()
                col, row = world_to_map_cv(x, y)
                cv2.circle(img, (col, row), 8, (0, 0, 255), -1)  # Red for supports
                cv2.putText(img, sup.name[:15], (col + 10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        print(f"[TravMap] Selected pair: {source_support.name} -> {target_support.name}", flush=True)

        # Save trav map
        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/trav_map_debug.png", img)
        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/trav_map_raw.png", floor_map)

        room_vis = (room_ins_map * 80).astype(np.uint8)
        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/room_segmap.png", room_vis)

        # Logical AND of trav_map and room segmap
        room_mask = (room_ins_map > 0).astype(np.uint8) * 255
        trav_and_room = cv2.bitwise_and(floor_map, room_mask)
        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/trav_and_room.png", trav_and_room)

        # Combined visualization - each room gets a different color
        combined = np.zeros((floor_map.shape[0], floor_map.shape[1], 3), dtype=np.uint8)
        combined[floor_map == 0] = [50, 50, 50]  # Dark gray for walls
        room_colors = [
            [0, 255, 0],    # Green
            [255, 0, 0],    # Blue
            [0, 255, 255],  # Yellow
            [255, 0, 255],  # Magenta
            [255, 255, 0],  # Cyan
            [0, 165, 255],  # Orange
            [147, 20, 255], # Pink
            [0, 128, 0],    # Dark green
        ]
        unique_rooms = np.unique(room_ins_map)
        for i, room_id in enumerate(unique_rooms):
            if room_id == 0:
                continue
            color = room_colors[i % len(room_colors)]
            room_trav = (room_ins_map == room_id) & (floor_map == 255)
            combined[room_trav] = color
            dark_color = [c // 3 for c in color]
            room_non_trav = (room_ins_map == room_id) & (floor_map == 0)
            combined[room_non_trav] = dark_color
        # Mark supports
        for room_name, supports in room_supports.items():
            for sup in supports:
                pos = sup.get_position_orientation()[0]
                col, row = world_to_map_cv(pos[0].item(), pos[1].item())
                cv2.circle(combined, (col, row), 4, (0, 0, 255), -1)
                cv2.circle(combined, (col, row), 4, (255, 255, 255), 1)
                cv2.putText(combined, f"{sup.name[:12]}({room_name[:8]})", (col + 6, row), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/trav_combined.png", combined)

        print(f"[TravMap] Saved to {DEBUG_OUTPUT_DIR}/trav_map_debug.png, trav_map_raw.png, room_segmap.png, trav_and_room.png, trav_combined.png", flush=True)

    except Exception as e:
        import traceback
        print(f"[TravMap] Failed to save visualization: {e}", flush=True)
        traceback.print_exc()


def _map_supports_to_rooms(scene, supports: list) -> dict[int, list]:
    """Map supports to rooms using scene._seg_map (BEHAVIOR-1K built-in).

    Uses get_room_instance_by_point() which handles coordinate conversion internally.

    Returns: room_supports dict keyed by room instance ID
    """
    seg_map = scene._seg_map
    trav_map = scene._trav_map

    # Get room_ins_map for ID lookup
    room_ins_np = seg_map.room_ins_map.cpu().numpy()
    unique_rooms = np.unique(room_ins_np)
    print(f"[Debug] seg_map.room_ins_map: shape={room_ins_np.shape}, rooms={list(unique_rooms)}", flush=True)

    room_supports = {}
    for sup in supports:
        pos, _ = sup.get_position_orientation()
        x, y = pos[0].item(), pos[1].item()

        # Use BEHAVIOR-1K's built-in function
        room_name = seg_map.get_room_instance_by_point(th.tensor([x, y]))

        if room_name is not None:
            # Get room instance ID from name
            room_ins_id = seg_map.room_ins_name_to_ins_id.get(room_name, 0)
            if room_ins_id > 0:
                print(f"[Debug] Support {sup.name} at ({x:.2f}, {y:.2f}) -> {room_name} (id={room_ins_id})", flush=True)
                if room_ins_id not in room_supports:
                    room_supports[room_ins_id] = []
                room_supports[room_ins_id].append(sup)
            else:
                print(f"[Debug] Support {sup.name} at ({x:.2f}, {y:.2f}) -> {room_name} (invalid id)", flush=True)
        else:
            print(f"[Debug] Support {sup.name} at ({x:.2f}, {y:.2f}) -> outside rooms", flush=True)

    return room_supports


def _get_valid_supports(scene, is_support_fn: Callable[[str], bool], max_support_height: float) -> list:
    valid = []
    print(f"[Debug] Checking supports. Height range: {MIN_SUPPORT_HEIGHT} - {max_support_height:.2f}", flush=True)
    for obj in scene.objects:
        category = getattr(obj, 'category', None)
        if category is None:
            continue
        is_support = is_support_fn(category)
        if is_support:
            try:
                aabb = obj.aabb
                surface_height = aabb[1][2].item()
                height_ok = MIN_SUPPORT_HEIGHT < surface_height < max_support_height
                print(f"[Debug] Support candidate: {obj.name} (cat={category}) height={surface_height:.2f} -> {'OK' if height_ok else 'REJECTED'}", flush=True)
                if height_ok:
                    valid.append(obj)
            except Exception as e:
                print(f"[Debug] Support candidate: {obj.name} (cat={category}) -> EXCEPTION: {e}", flush=True)
    return valid


def _object_fits_gripper(obj) -> bool:
    try:
        aabb = obj.aabb
        size_x = aabb[1][0].item() - aabb[0][0].item()
        size_y = aabb[1][1].item() - aabb[0][1].item()
        size_z = aabb[1][2].item() - aabb[0][2].item()
        min_xy = min(size_x, size_y)
        return min_xy <= MAX_GRIPPER_OPENING and size_z >= MIN_OBJECT_HEIGHT
    except Exception:
        return False


def _find_graspable_objects_on_support(scene, support, is_graspable_fn: Callable[[str], bool]) -> list:
    support_aabb = support.aabb
    support_min_x = support_aabb[0][0].item()
    support_max_x = support_aabb[1][0].item()
    support_min_y = support_aabb[0][1].item()
    support_max_y = support_aabb[1][1].item()
    support_surface_z = support_aabb[1][2].item()

    objects_on_support = []
    for obj in scene.objects:
        category = getattr(obj, 'category', None)
        if category is None:
            continue
        if not is_graspable_fn(category):
            continue
        if not _object_fits_gripper(obj):
            continue

        try:
            obj_pos, _ = obj.get_position_orientation()
            obj_x = obj_pos[0].item()
            obj_y = obj_pos[1].item()
            obj_z = obj_pos[2].item()

            MARGIN = 0.05
            x_on = support_min_x - MARGIN <= obj_x <= support_max_x + MARGIN
            y_on = support_min_y - MARGIN <= obj_y <= support_max_y + MARGIN
            z_on = support_surface_z - 0.05 <= obj_z <= support_surface_z + 0.5

            if x_on and y_on and z_on:
                objects_on_support.append(obj)
        except Exception:
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
                if hasattr(val, 'item'):
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
                                if hasattr(cv, 'item'):
                                    cv = cv.item()
                                if cv == 255:
                                    clearance += 1

                    if clearance >= 8:
                        good_directions += 1
                        score += clearance

    return score if good_directions >= 2 else 0


def _select_best_graspable_object(objects: list, scene, robot) -> object:
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
        top_options = good_options[:min(3, len(good_options))]
        chosen = random.choice(top_options)
        print(f"[Episode] Object scores: {[(s[0].name, s[1]) for s in scored[:5]]}", flush=True)
        return chosen[0]
    else:
        return random.choice(objects)


def collect_episode(
    env, scene, robot, collector: DataCollector,
    is_graspable_fn: Callable[[str], bool],
    is_support_fn: Callable[[str], bool],
    wrapper: OmniGibsonLeRobotWrapper = None,
    failed_objects: dict[str, int] = None,
    max_object_failures: int = 3,
    dataset_name: str = "behavior-1k-assets",
    cached_pairs: list = None,
    max_support_height: float = 1.0,
) -> tuple[bool, list, list, str | None, list]:
    """Returns (success, observations, actions, failed_obj_name, cached_pairs)"""
    if failed_objects is None:
        failed_objects = {}

    robot.set_position_orientation(position=[0, 0, -10], orientation=[0, 0, 0, 1])
    robot.keep_still()
    for _ in range(5):
        og.sim.step()

    # Use cached pairs if available, otherwise compute them
    if cached_pairs is not None:
        all_pairs = cached_pairs
    else:
        valid_supports = _get_valid_supports(scene, is_support_fn, max_support_height)
        if len(valid_supports) < 2:
            print(f"[Episode] Not enough valid supports: {len(valid_supports)}", flush=True)
            return False, [], [], None, None

        # Map supports to rooms using scene._seg_map (BEHAVIOR-1K built-in)
        room_supports = _map_supports_to_rooms(scene, valid_supports)

        num_rooms = len(room_supports)
        print(f"[Episode] {num_rooms} rooms, supports by room: {[(r, len(s)) for r, s in room_supports.items()]}", flush=True)

        if num_rooms == 0:
            print(f"[Episode] No rooms found with supports - scene unsuitable", flush=True)
            return False, [], [], "__SCENE_UNSUITABLE__", None

        # Pair supports in same room AND same connected component
        # This ensures there's actually a traversable path between them
        all_pairs = []
        trav_map = scene._trav_map
        print(f"[Episode] Pairing supports by connectivity...", flush=True)

        for room_ins_id, supports in room_supports.items():
            # Compute connected components for this room (uses 0.3m erosion + room mask)
            labels, num_components = _compute_room_connected_components(scene, room_ins_id)
            print(f"[Episode] Room {room_ins_id}: {num_components - 1} connected components", flush=True)

            # Map each support to its component
            support_components = {}
            for sup in supports:
                comp = _get_support_component(sup, labels, trav_map)
                support_components[sup.name] = comp
                print(f"[Episode]   {sup.name} -> component {comp}", flush=True)

            # Group supports by component
            component_supports = {}
            for sup in supports:
                comp = support_components[sup.name]
                if comp > 0:  # Only include supports with valid components
                    if comp not in component_supports:
                        component_supports[comp] = []
                    component_supports[comp].append(sup)

            # Pair supports within same component - include labels for robot placement
            for comp, comp_supports in component_supports.items():
                if len(comp_supports) >= 2:
                    for i, s1 in enumerate(comp_supports):
                        for j, s2 in enumerate(comp_supports):
                            if i < j:
                                # Store (room_ins_id, comp_id, labels, s1, s2)
                                all_pairs.append((room_ins_id, comp, labels, s1, s2))
                                all_pairs.append((room_ins_id, comp, labels, s2, s1))
                                print(f"[Episode] Connected pair (room={room_ins_id}, comp={comp}): {s1.name} <-> {s2.name}", flush=True)

        if not all_pairs:
            print(f"[Episode] No valid pairs - scene unsuitable", flush=True)
            if DEBUG_EPISODE:
                # Convert room_supports to room_name keyed dict for visualization
                room_supports_viz = {f"room_{k}": v for k, v in room_supports.items()}
                _save_trav_map_raw(scene, room_supports_viz)
            return False, [], [], "__SCENE_UNSUITABLE__", None

        print(f"[Episode] Found {len(all_pairs)//2} valid support pairs (path existence guaranteed)", flush=True)

    chosen_room_id, chosen_comp_id, chosen_labels, source_support, target_support = random.choice(all_pairs)
    print(f"[Episode] Selected pair:", flush=True)
    print(f"[Episode]   Room: {chosen_room_id}, Component: {chosen_comp_id}", flush=True)
    print(f"[Episode]   Source: {source_support.name} -> Target: {target_support.name}", flush=True)
    print(f"[Episode]   Path guaranteed: supports + robot will be in same component", flush=True)

    if DEBUG_EPISODE and cached_pairs is None:
        # Only visualize when pairs were just computed (room_supports is defined)
        room_supports_viz = {f"room_{k}": v for k, v in room_supports.items()}
        _save_trav_map_visualization(scene, robot, room_supports_viz, source_support, target_support)

    floors = list(scene.object_registry("category", "floors"))
    if floors:
        floor_top_z = max([obj.aabb[1][2] for obj in floors])
        floor_z = floor_top_z  # Robot at floor surface level
        print(f"[Episode] Floor top Z: {floor_top_z:.3f}, robot Z: {floor_z:.3f}", flush=True)
    else:
        floor_z = 0.0
        print(f"[Episode] No floors found, using floor_z={floor_z:.3f}", flush=True)

    # Set room filter for navigation (uses room instance ID from seg_map)
    collector.nav.set_room_filter(chosen_room_id)

    robot_start = _find_robot_start_in_component(
        scene, robot, chosen_labels, chosen_comp_id, floor_z
    )
    if robot_start is None:
        print(f"[Episode] No valid robot position in component {chosen_comp_id}", flush=True)
        return False, [], [], None, all_pairs

    start_x, start_y, start_z, facing_yaw = robot_start
    print(f"[Episode] Robot placed in component {chosen_comp_id}: ({start_x:.2f}, {start_y:.2f}, z={start_z:.2f})", flush=True)

    start_quat = [0, 0, math.sin(facing_yaw / 2), math.cos(facing_yaw / 2)]
    robot.set_position_orientation(position=[start_x, start_y, start_z], orientation=start_quat)
    robot.keep_still()
    robot.reset()

    for _ in range(10):
        og.sim.step()

    spawned_obj = False

    existing_objects = _find_graspable_objects_on_support(scene, source_support, is_graspable_fn)

    valid_objects = [
        obj for obj in existing_objects
        if failed_objects.get(obj.name, 0) < max_object_failures
    ]

    skipped = len(existing_objects) - len(valid_objects)
    if skipped > 0:
        print(f"[Episode] Skipped {skipped} failed objects", flush=True)

    if valid_objects:
        target_obj = _select_best_graspable_object(valid_objects, scene, robot)
        print(f"[Episode] Using {target_obj.name} on {source_support.name}", flush=True)
    else:
        robot_pos, _ = robot.get_position_orientation()
        target_obj = None
        for _ in range(20):
            spawned = spawn_and_place_object(scene, source_support, robot_pos=robot_pos)
            if spawned is not None:
                target_obj = spawned
                break

        if target_obj is None:
            print(f"[Episode] Failed to spawn object on {source_support.name}", flush=True)
            return False, [], [], None, all_pairs
        spawned_obj = True
        print(f"[Episode] Spawned {target_obj.name} on {source_support.name}", flush=True)

    # Remove all loose objects from the scene
    for j, obj in enumerate(scene.objects):
        if not obj.fixed_base and obj not in (target_obj, source_support, target_support):
            obj.set_position_orientation(position=th.as_tensor([100 + j, 0, 10.]))

    if wrapper is not None:
        wrapper.set_target_objects(target_obj, source_support)

    all_observations = []
    all_actions = []

    success, obs, acts = collector.pick_object(target_obj, source_support=source_support)
    print(f"[Episode] Pick: success={success}, steps={len(acts)}", flush=True)
    all_observations.extend(obs)
    all_actions.extend(acts)

    if not success:
        failed_obj_name = target_obj.name
        if spawned_obj:
            safe_remove_object(scene, target_obj, robot)
        return False, all_observations, all_actions, failed_obj_name, all_pairs

    if wrapper is not None:
        wrapper.set_target_objects(target_obj, target_support)

    success, obs, acts = collector.place_object(target_support)
    all_observations.extend(obs)
    all_actions.extend(acts)

    if spawned_obj:
        safe_remove_object(scene, target_obj, robot)

    print(f"[Episode] Complete: {len(all_actions)} steps, success={success}", flush=True)
    return success, all_observations, all_actions, None if success else target_obj.name, all_pairs


def run_data_collection(config: DataCollectionConfig):
    is_support_fn, is_graspable_fn = get_object_filters(config)
    logger.info("Using object filter method: %s", config.object_filter_method)

    og_config_path = Path(og.example_config_path) / "stretch_vid2scene.yaml"
    og_config = yaml.safe_load(open(og_config_path))
    og_config["scene"]["scene_model"] = config.scene_model
    og_config["scene"]["dataset_name"] = config.dataset_name
    og_config["scene"]["not_load_object_categories"] = ["ceilings", "armchair", "ottoman"]

    if config.dataset_name != "behavior-1k-assets":
        og_config["scene"]["scene_instance"] = f"{config.scene_model}_best"

    # For SPOC scenes, add floor plane
    if config.dataset_name == "spoc":
        og_config["scene"]["use_floor_plane"] = True
        og_config["scene"]["floor_plane_visible"] = True

    env = og.Environment(configs=og_config)
    scene = env.scene
    robot = env.robots[0]
    logger.info("Environment created: scene=%s", config.scene_model)

    # Load segmentation map (not auto-loaded by OmniGibson)
    scene._seg_map.load_map()
    logger.info("Segmentation map loaded: %d rooms", len(scene._seg_map.room_ins_name_to_ins_id))

    for obj in scene.objects:
        if getattr(obj, 'category', '') == 'floors':
            for link in obj.links.values():
                for mesh in link.collision_meshes.values():
                    mat_name = f"{obj.name}_floor_physics_mat"
                    physics_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
                        prim_path=f"{obj.prim_path}/Looks/{mat_name}",
                        name=mat_name,
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    )
                    mesh.apply_physics_material(physics_mat)
    logger.info("Floor friction applied")

    with og.sim.stopped():
        original_mass = robot.base_footprint_link.mass
        robot.base_footprint_link.mass = original_mass * 2.0
        logger.info("Robot base mass: %.1f -> %.1f kg", original_mass, robot.base_footprint_link.mass)

    wrapper_config = OmniGibsonLeRobotConfig(
        repo_id=config.repo_id,
        root=config.output_dir,
        task_description=f"pick_and_place_{config.scene_model}",
        fps=config.fps,
        num_episodes=config.num_episodes,
        max_steps=config.max_steps_per_episode,
        use_videos=True,
        include_depth=True,
        include_segmentation=True,
    )
    wrapper = OmniGibsonLeRobotWrapper(env, wrapper_config)

    # Run simulation steps to ensure robot articulation is initialized
    for _ in range(30):
        og.sim.step()

    collector = DataCollector(env, robot, config)

    for _ in range(10):
        og.sim.step()

    env.scene.update_initial_file()
    env.scene.reset()


    for _ in range(30):
        og.sim.step()

    # Get gripper's initial Z at reset pose (for support height filtering)
    robot.reset()
    for _ in range(10):
        og.sim.step()
    # eef_pos, _ = robot.get_eef_pose(robot.arm_names[0])
    # initial_gripper_z = eef_pos[2].item()
    max_support_height = 1.0  # Support must be at least 10cm below gripper
    logger.info("max support height: %.3f", max_support_height)

    sample_obs, _ = wrapper.reset_env()
    wrapper.start_recording()

    

    successful_episodes = 0
    attempt = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 20
    failed_objects = {}
    MAX_OBJECT_FAILURES = 1
    cached_pairs = None
    position_failures = 0  # Track failures at current robot position
    MAX_POSITION_FAILURES = 3  # After 3 failures, try new robot position

    try:
        while successful_episodes < config.num_episodes:
            attempt += 1
            print(f"[Episode] Attempt {attempt}, success: {successful_episodes}/{config.num_episodes}", flush=True)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[Episode] {MAX_CONSECUTIVE_FAILURES} consecutive failures", flush=True)
                break

            if config.dataset_name == "spoc":
                floors = list(env.scene.object_registry("category", "floors"))
                for floor in floors:
                    top_surface = floor.aabb[1][2].item()

                    logger.info("SPOC: Floor %s top surface at Z=%.3f", floor.name, top_surface)
                    if np.isclose(top_surface, 0, atol=0.1):
                        continue
                    floor_pos, floor_ori = floor.get_position_orientation()
                    offset = -top_surface
                    floor_pos[2] += offset
                    logger.info("SPOC: Moving scene down by %.3f to align floor top at Z=0", -offset)
                    floor.set_position_orientation(floor_pos, floor_ori)   

            success, observations, actions, failed_obj_name, cached_pairs = collect_episode(
                env, scene, robot, collector,
                is_graspable_fn, is_support_fn,
                wrapper=wrapper,
                failed_objects=failed_objects,
                max_object_failures=MAX_OBJECT_FAILURES,
                dataset_name=config.dataset_name,
                cached_pairs=cached_pairs,
                max_support_height=max_support_height,
            )

            if not success and failed_obj_name:
                if failed_obj_name == "__SCENE_UNSUITABLE__":
                    print(f"[Episode] Scene is unsuitable for pick-and-place, exiting", flush=True)
                    break
                failed_objects[failed_obj_name] = failed_objects.get(failed_obj_name, 0) + 1
                position_failures += 1
                print(f"[Episode] {failed_obj_name} failed {failed_objects[failed_obj_name]}/{MAX_OBJECT_FAILURES}, position failures: {position_failures}/{MAX_POSITION_FAILURES}", flush=True)

                if position_failures >= MAX_POSITION_FAILURES:
                    print(f"[Episode] {MAX_POSITION_FAILURES} failures at current position, trying new robot position", flush=True)
                    failed_objects.clear()
                    position_failures = 0


            if success and observations and actions:
                for i, (obs, action) in enumerate(zip(observations, actions)):
                    lerobot_obs = wrapper._convert_observation(obs)
                    is_last = (i == len(observations) - 1)
                    wrapper.record_frame(lerobot_obs, np.array(action, dtype=np.float32), 1.0 if is_last else 0.0, is_last)
                wrapper.save_episode()
                successful_episodes += 1
                consecutive_failures = 0
                position_failures = 0  # Reset on success
                print(f"[Episode] Saved episode {successful_episodes} with {len(actions)} frames", flush=True)
            else:
                consecutive_failures += 1

            for arm in robot.arm_names:
                if robot._ag_obj_in_hand.get(arm) is not None:
                    try:
                        robot.release_grasp_immediately(arm=arm)
                    except Exception:
                        pass

            robot.reset()
            env.scene.reset()

            for _ in range(50):
                og.sim.step()

    finally:
        wrapper.stop_recording()
        print(f"[Episode] Done: {successful_episodes}/{config.num_episodes} episodes in {attempt} attempts", flush=True)

    og.shutdown()
