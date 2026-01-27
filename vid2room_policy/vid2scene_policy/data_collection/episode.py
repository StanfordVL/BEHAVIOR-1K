import logging
import math
import random
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import torch as th
import yaml
from PIL import Image
from scipy.ndimage import label as scipy_label

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.asset_utils import get_all_object_categories, get_all_object_category_models

from .config import DataCollectionConfig, get_object_filters
from .data_collector import DataCollector
from .scene_management import get_scene_objects_by_category, spawn_and_place_object, safe_remove_object
from .omnigibson_lerobot_wrapper import OmniGibsonLeRobotWrapper, OmniGibsonLeRobotConfig

logger = logging.getLogger(__name__)

MIN_SUPPORT_HEIGHT = 0.3
MAX_SUPPORT_HEIGHT = 1.2
ROBOT_RADIUS_PIXELS = 40  # Robot footprint radius in map pixels (at 0.01m resolution)

# Gripper size constraints
MAX_GRIPPER_OPENING = 0.12  # 12cm - Stretch gripper max opening
MIN_OBJECT_HEIGHT = 0.03    # 3cm minimum object height


def _find_rooms_from_no_obj_map(scene) -> tuple[np.ndarray, int, dict]:
    """Find rooms from the no-object traversability map.

    Returns:
        labeled_map: Array where each pixel has room ID (0=non-traversable, 1..N=room IDs)
        num_rooms: Number of rooms found
        map_info: Dict with 'resolution', 'offset' for world<->map coordinate conversion
    """
    trav_map = scene._trav_map

    # Get map directory from scene
    map_dir = Path(scene.scene_dir) / "layout"
    no_obj_path = map_dir / "floor_trav_no_obj_0.png"

    if not no_obj_path.exists():
        # Fallback to regular map if no-obj doesn't exist
        no_obj_path = map_dir / "floor_trav_0.png"

    no_obj_map = np.array(Image.open(no_obj_path))
    traversable = (no_obj_map == 255).astype(np.uint8)

    # Find connected components (rooms)
    labeled, num_rooms = scipy_label(traversable)

    # Get map info for coordinate conversion
    # The map is at default resolution (0.01m per pixel), centered at world origin
    map_info = {
        'resolution': trav_map.map_default_resolution,  # 0.01m per pixel
        'shape': no_obj_map.shape,
    }

    logger.info("Found %d rooms in no-object map (%dx%d)", num_rooms, *no_obj_map.shape)
    return labeled, num_rooms, map_info


def _world_to_no_obj_map(x, y, map_info) -> tuple[int, int]:
    """Convert world coordinates to no-object map pixel coordinates.

    The map is centered at world origin (0,0). Coordinate axes are flipped
    between world and map frames (OmniGibson convention).
    """
    resolution = map_info['resolution']
    shape = map_info['shape']

    # World to map: flip axes and center
    # world (x, y) -> map (col, row) with flip
    col = int(x / resolution + shape[1] / 2)
    row = int(y / resolution + shape[0] / 2)
    return row, col


def _no_obj_map_to_world(row, col, map_info) -> tuple[float, float]:
    """Convert no-object map pixel coordinates to world coordinates."""
    resolution = map_info['resolution']
    shape = map_info['shape']

    # Map to world: flip axes and uncenter
    x = (col - shape[1] / 2) * resolution
    y = (row - shape[0] / 2) * resolution
    return x, y


def _get_room_for_position(x, y, labeled_map, map_info) -> int:
    """Get room ID for a world position. Returns 0 if not in any room."""
    row, col = _world_to_no_obj_map(x, y, map_info)
    h, w = labeled_map.shape
    if 0 <= row < h and 0 <= col < w:
        return labeled_map[row, col]
    return 0


def _map_supports_to_rooms(supports: list, labeled_map: np.ndarray, map_info: dict) -> dict[int, list]:
    """Map support objects to their rooms.

    Returns:
        Dict mapping room_id -> list of supports in that room
    """
    room_supports = {}
    for sup in supports:
        pos, _ = sup.get_position_orientation()
        x, y = pos[0].item(), pos[1].item()
        room_id = _get_room_for_position(x, y, labeled_map, map_info)
        if room_id > 0:  # Valid room
            if room_id not in room_supports:
                room_supports[room_id] = []
            room_supports[room_id].append(sup)
    return room_supports


def _find_robot_square_with_bfs(labeled_map: np.ndarray, room_id: int,
                                  start_row: int, start_col: int,
                                  robot_radius: int = ROBOT_RADIUS_PIXELS) -> tuple[int, int] | None:
    """Find a position where robot-sized square fits using BFS from start position.

    Args:
        labeled_map: Room-labeled map
        room_id: Which room to search in
        start_row, start_col: Starting pixel position
        robot_radius: Half-size of robot square in pixels

    Returns:
        (row, col) of valid position center, or None if not found
    """
    h, w = labeled_map.shape

    def square_fits(r, c):
        """Check if robot square centered at (r,c) is entirely within the room."""
        for dr in range(-robot_radius, robot_radius + 1):
            for dc in range(-robot_radius, robot_radius + 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    return False
                if labeled_map[nr, nc] != room_id:
                    return False
        return True

    # BFS to find nearest valid position
    visited = set()
    queue = deque([(start_row, start_col, 0)])  # (row, col, distance)
    visited.add((start_row, start_col))

    while queue:
        r, c, dist = queue.popleft()

        # Check if this position works
        if square_fits(r, c):
            return (r, c)

        # Limit search radius
        if dist > 200:  # ~2m at 0.01m resolution
            continue

        # Add neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and 0 <= nr < h and 0 <= nc < w:
                if labeled_map[nr, nc] == room_id:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))

    return None


def _find_random_robot_start_in_room(labeled_map: np.ndarray, room_id: int,
                                      map_info: dict, floor_z: float,
                                      scene=None, robot=None,
                                      max_attempts: int = 50) -> tuple | None:
    """Find a random valid robot start position in a room.

    Returns:
        (x, y, z, yaw) or None if no valid position found
    """
    h, w = labeled_map.shape

    # Get all pixels in this room
    room_pixels = np.argwhere(labeled_map == room_id)
    if len(room_pixels) == 0:
        return None

    # Get runtime eroded map for validation
    eroded_map = None
    trav_map = None
    if scene is not None:
        trav_map = scene._trav_map
        eroded_map = trav_map._erode_trav_map(trav_map.floor_map[0].clone(), robot=robot)

    for _ in range(max_attempts):
        # Pick random pixel in room
        idx = random.randint(0, len(room_pixels) - 1)
        start_row, start_col = room_pixels[idx]

        # Use BFS to find valid square position on no-obj map
        valid_pos = _find_robot_square_with_bfs(labeled_map, room_id, start_row, start_col)
        if valid_pos is None:
            continue

        # Convert to world coordinates
        x, y = _no_obj_map_to_world(valid_pos[0], valid_pos[1], map_info)

        # Validate against runtime eroded map (with objects)
        if eroded_map is not None and trav_map is not None:
            runtime_map_pos = trav_map.world_to_map(th.tensor([x, y]))
            rr, rc = int(runtime_map_pos[0].item()), int(runtime_map_pos[1].item())
            if not (0 <= rr < eroded_map.shape[0] and 0 <= rc < eroded_map.shape[1]):
                continue
            val = eroded_map[rr, rc]
            if hasattr(val, 'item'):
                val = val.item()
            if val != 255:
                continue  # Position blocked by object in actual scene

        yaw = random.uniform(-math.pi, math.pi)
        return (x, y, floor_z, yaw)

    return None


def _get_valid_supports(scene, is_support_fn: Callable[[str], bool]) -> list:
    """Get all support objects at valid heights using filter function."""
    valid = []

    for obj in scene.objects:
        category = getattr(obj, 'category', None)
        if category is None:
            continue

        if not is_support_fn(category):
            continue

        try:
            aabb = obj.aabb
            surface_height = aabb[1][2].item()
            if MIN_SUPPORT_HEIGHT < surface_height < MAX_SUPPORT_HEIGHT:
                valid.append(obj)
        except Exception:
            continue

    return valid


def _object_fits_gripper(obj) -> bool:
    """Check if object dimensions fit gripper constraints.

    Returns True if:
    - min(X, Y) bbox dimension is <= MAX_GRIPPER_OPENING (gripper can grasp along longer axis)
    - Z bbox dimension is >= MIN_OBJECT_HEIGHT (tall enough to grasp)
    """
    try:
        aabb = obj.aabb
        size_x = aabb[1][0].item() - aabb[0][0].item()
        size_y = aabb[1][1].item() - aabb[0][1].item()
        size_z = aabb[1][2].item() - aabb[0][2].item()

        # Smaller of X,Y must fit in gripper (grasp along longer axis), Z must be tall enough
        min_xy = min(size_x, size_y)
        fits_gripper = min_xy <= MAX_GRIPPER_OPENING
        tall_enough = size_z >= MIN_OBJECT_HEIGHT

        return fits_gripper and tall_enough
    except Exception:
        return False


def _find_graspable_objects_on_support(scene, support, is_graspable_fn: Callable[[str], bool]) -> list:
    """Find graspable objects that are on or near a support surface using filter function."""
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

        # Check if object fits gripper size constraints
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
    """Compute how easy an object is to approach based on surrounding clearance."""
    trav_map = scene._trav_map
    eroded_map = trav_map._erode_trav_map(trav_map.floor_map[0].clone(), robot=robot)

    obj_pos, _ = obj.get_position_orientation()
    obj_x = obj_pos[0].item()
    obj_y = obj_pos[1].item()

    # Check clearance around the object at approach distance
    score = 0.0
    good_directions = 0

    for angle_deg in range(0, 360, 30):
        angle = math.radians(angle_deg)
        # Check at typical approach distance
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
                    # Check further clearance at this position
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

                    if clearance >= 8:  # At least half the checks clear
                        good_directions += 1
                        score += clearance

    return score if good_directions >= 2 else 0  # Need at least 2 good approach directions


def _select_best_graspable_object(objects: list, scene, robot) -> object:
    """Select the object with the best approachability score."""
    if not objects:
        return None

    if len(objects) == 1:
        return objects[0]

    # Score each object
    scored = []
    for obj in objects:
        score = _compute_object_approachability(obj, scene, robot)
        scored.append((obj, score))

    # Sort by score descending
    scored.sort(key=lambda x: -x[1])

    # Return best, or random from top 3 if multiple good options
    good_options = [s for s in scored if s[1] > 0]
    if good_options:
        top_options = good_options[:min(3, len(good_options))]
        chosen = random.choice(top_options)
        print(f"[Episode] Object scores: {[(s[0].name, s[1]) for s in scored[:5]]}", flush=True)
        return chosen[0]
    else:
        # Fallback to random if no good scores
        return random.choice(objects)


def collect_episode(
    env, scene, robot, collector: DataCollector,
    is_graspable_fn: Callable[[str], bool],
    is_support_fn: Callable[[str], bool],
    wrapper: OmniGibsonLeRobotWrapper = None,
    failed_objects: dict[str, int] = None,
    max_object_failures: int = 3,
) -> tuple[bool, list, list, str | None]:
    """Collect one episode of pick and place using room-based logic.

    Returns:
        (success, observations, actions, failed_obj_name)
        failed_obj_name is set if grasp failed, None otherwise
    """
    if failed_objects is None:
        failed_objects = {}
    # Move robot underground while we compute placement
    robot.set_position_orientation(position=[0, 0, -10], orientation=[0, 0, 0, 1])
    robot.keep_still()
    og.sim.step()

    # Find rooms from structural map
    labeled_map, num_rooms, map_info = _find_rooms_from_no_obj_map(scene)
    if num_rooms == 0:
        print(f"[Episode] No rooms found in scene", flush=True)
        return False, [], [], None

    # Find valid supports
    valid_supports = _get_valid_supports(scene, is_support_fn)
    if len(valid_supports) < 2:
        print(f"[Episode] Not enough valid supports: {len(valid_supports)}", flush=True)
        return False, [], [], None

    # Map supports to rooms
    room_supports = _map_supports_to_rooms(valid_supports, labeled_map, map_info)
    print(f"[Episode] {num_rooms} rooms, supports by room: {[(r, len(s)) for r, s in room_supports.items()]}", flush=True)

    # Find rooms with at least 2 supports (can form pairs)
    valid_rooms = [r for r, supports in room_supports.items() if len(supports) >= 2]
    if not valid_rooms:
        print(f"[Episode] No room has 2+ supports for pairing", flush=True)
        return False, [], [], None

    # Pick a random room with supports and create pairs within it
    chosen_room = random.choice(valid_rooms)
    room_support_list = room_supports[chosen_room]

    # Create all pairs within this room
    pairs = []
    for i, s1 in enumerate(room_support_list):
        for j, s2 in enumerate(room_support_list):
            if i != j:
                pairs.append((s1, s2))

    source_support, target_support = random.choice(pairs)
    print(f"[Episode] Room {chosen_room}: {source_support.name} -> {target_support.name} "
          f"({len(pairs)} pairs, {len(room_support_list)} supports)", flush=True)

    # Find robot start position - get floor z from floor objects
    # Group floors within 1.5m and choose the highest in the group
    floor_candidates = []
    for obj in scene.objects:
        cat = getattr(obj, 'category', '') or ''
        name = getattr(obj, 'name', '') or ''
        if ('floor' in cat.lower() or 'floor' in name.lower()) and hasattr(obj, 'aabb'):
            try:
                floor_top_z = obj.aabb[1][2].item()  # Top of floor bbox
                floor_candidates.append(floor_top_z)
            except Exception:
                pass

    floor_z = 0.0
    if floor_candidates:
        floor_candidates.sort()
        # Group floors within 1.5m - find the highest in the lowest group
        groups = []
        current_group = [floor_candidates[0]]
        for z in floor_candidates[1:]:
            if z - current_group[0] < 1.5:
                current_group.append(z)
            else:
                groups.append(current_group)
                current_group = [z]
        groups.append(current_group)
        # Use the highest floor in the first (lowest) group
        floor_z = max(groups[0])

    # Add offset to ensure robot doesn't clip into floor
    ROBOT_BASE_OFFSET = 0.02
    floor_z = floor_z + ROBOT_BASE_OFFSET
    print(f"[Episode] Floor Z: {floor_z:.3f}m (from {len(floor_candidates)} floor objects)", flush=True)

    robot_start = _find_random_robot_start_in_room(
        labeled_map, chosen_room, map_info, floor_z,
        scene=scene, robot=robot
    )
    if robot_start is None:
        print(f"[Episode] No valid robot position in room {chosen_room}", flush=True)
        return False, [], [], None

    start_x, start_y, start_z, facing_yaw = robot_start
    print(f"[Episode] Robot start: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f}) facing={math.degrees(facing_yaw):.0f}°", flush=True)

    # Place robot
    start_quat = [0, 0, math.sin(facing_yaw / 2), math.cos(facing_yaw / 2)]
    robot.set_position_orientation(position=[start_x, start_y, start_z], orientation=start_quat)
    robot.keep_still()
    robot.reset()

    for _ in range(10):
        og.sim.step()

    # Check for existing graspable objects on source support
    spawned_obj = False
    existing_objects = _find_graspable_objects_on_support(scene, source_support, is_graspable_fn)

    # Filter out objects that have failed too many times
    valid_objects = [
        obj for obj in existing_objects
        if failed_objects.get(obj.name, 0) < max_object_failures
    ]
    skipped = len(existing_objects) - len(valid_objects)
    if skipped > 0:
        print(f"[Episode] Skipped {skipped} objects with {max_object_failures}+ failures", flush=True)

    if valid_objects:
        target_obj = _select_best_graspable_object(valid_objects, scene, robot)
        print(f"[Episode] Using {target_obj.name} on {source_support.name} ({len(valid_objects)} available)", flush=True)
    else:
        # Spawn a new graspable object - get categories that pass the filter and have models
        available_categories = []
        for category in get_all_object_categories():
            if is_graspable_fn(category):
                models = get_all_object_category_models(category)
                if models:
                    available_categories.append(category)

        if not available_categories:
            print(f"[Episode] No graspable categories available", flush=True)
            return False, [], [], None

        # Try spawning objects until we find one that fits gripper constraints
        random.shuffle(available_categories)
        target_obj = None
        robot_pos, _ = robot.get_position_orientation()
        for target_category in available_categories[:20]:  # Try up to 20 categories
            print(f"[Episode] Spawning {target_category} on {source_support.name}...", flush=True)
            spawned = spawn_and_place_object(scene, target_category, source_support, robot_pos=robot_pos)
            if spawned is None:
                continue

            # Check if spawned object fits gripper size constraints
            if _object_fits_gripper(spawned):
                target_obj = spawned
                break
            else:
                # Object too big or too small, remove and try another
                aabb = spawned.aabb
                size_x = aabb[1][0].item() - aabb[0][0].item()
                size_y = aabb[1][1].item() - aabb[0][1].item()
                size_z = aabb[1][2].item() - aabb[0][2].item()
                print(f"[Episode] {spawned.name} size ({size_x:.2f}x{size_y:.2f}x{size_z:.2f}) doesn't fit gripper, trying another", flush=True)
                safe_remove_object(scene, spawned, robot)

        if target_obj is None:
            print(f"[Episode] Failed to spawn object that fits gripper on {source_support.name}", flush=True)
            return False, [], [], None
        spawned_obj = True
        print(f"[Episode] Spawned {target_obj.name}, starting pick...", flush=True)

    if wrapper is not None:
        wrapper.set_target_objects(target_obj, source_support)

    all_observations = []
    all_actions = []

    # Pick
    success, obs, acts = collector.pick_object(target_obj, source_support=source_support)
    print(f"[Episode] Pick result: success={success}, steps={len(acts)}", flush=True)
    all_observations.extend(obs)
    all_actions.extend(acts)

    if not success:
        failed_obj_name = target_obj.name
        if spawned_obj:
            safe_remove_object(scene, target_obj, robot)
        return False, all_observations, all_actions, failed_obj_name

    if wrapper is not None:
        wrapper.set_target_objects(target_obj, target_support)

    # Place
    success, obs, acts = collector.place_object(target_support)
    all_observations.extend(obs)
    all_actions.extend(acts)

    if spawned_obj:
        safe_remove_object(scene, target_obj, robot)

    print(f"[Episode] Complete: {len(all_actions)} steps, success={success}", flush=True)
    return success, all_observations, all_actions, None if success else target_obj.name


def run_data_collection(config: DataCollectionConfig):
    """Main data collection loop."""
    is_support_fn, is_graspable_fn = get_object_filters(config)
    logger.info("Using object filter method: %s", config.object_filter_method)

    og_config_path = Path(og.example_config_path) / "stretch_vid2scene.yaml"
    og_config = yaml.safe_load(open(og_config_path))
    og_config["scene"]["scene_model"] = config.scene_model
    og_config["scene"]["dataset_name"] = config.dataset_name
    og_config["scene"]["not_load_object_categories"] = ["ceilings", "armchair", "ottoman"]

    # Non-BEHAVIOR-1K scenes use {scene_model}_best instead of {scene_model}_with_clutter
    if config.dataset_name != "behavior-1k-assets":
        og_config["scene"]["scene_instance"] = f"{config.scene_model}_best"

    env = og.Environment(configs=og_config)
    scene = env.scene
    robot = env.robots[0]
    logger.info("Environment created: scene=%s", config.scene_model)

    # Apply friction to floor collision meshes
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

    # Increase robot base mass slightly for stability
    with og.sim.stopped():
        original_mass = robot.base_footprint_link.mass
        robot.base_footprint_link.mass = original_mass * 2.0
        logger.info("Robot base mass: %.1f -> %.1f kg", original_mass, robot.base_footprint_link.mass)

    # Create LeRobot wrapper for recording
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

    collector = DataCollector(env, robot, config)

    for _ in range(5):
        og.sim.step()

    env.scene.update_initial_file()
    env.scene.reset()

    for _ in range(30):
        og.sim.step()

    # Initialize wrapper observation structure
    sample_obs, _ = wrapper.reset_env()
    wrapper.start_recording()

    successful_episodes = 0
    attempt = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 20
    failed_objects = {}  # Track failed grasp attempts per object name
    MAX_OBJECT_FAILURES = 3

    try:
        while successful_episodes < config.num_episodes:
            attempt += 1
            print(f"[Episode] Attempt {attempt}, success: {successful_episodes}/{config.num_episodes}", flush=True)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[Episode] {MAX_CONSECUTIVE_FAILURES} consecutive failures - scene may be incompatible", flush=True)
                break

            success, observations, actions, failed_obj_name = collect_episode(
                env, scene, robot, collector,
                is_graspable_fn, is_support_fn,
                wrapper=wrapper,
                failed_objects=failed_objects,
                max_object_failures=MAX_OBJECT_FAILURES,
            )

            # Track failed object
            if not success and failed_obj_name:
                failed_objects[failed_obj_name] = failed_objects.get(failed_obj_name, 0) + 1
                print(f"[Episode] Object {failed_obj_name} failed {failed_objects[failed_obj_name]}/{MAX_OBJECT_FAILURES} times", flush=True)

            if success and observations and actions:
                # Record episode to LeRobot dataset
                for i, (obs, action) in enumerate(zip(observations, actions)):
                    lerobot_obs = wrapper._convert_observation(obs)
                    is_last = (i == len(observations) - 1)
                    wrapper.record_frame(lerobot_obs, np.array(action, dtype=np.float32), 1.0 if is_last else 0.0, is_last)
                wrapper.save_episode()
                successful_episodes += 1
                consecutive_failures = 0
                print(f"[Episode] Saved episode {successful_episodes} with {len(actions)} frames", flush=True)
            else:
                consecutive_failures += 1

            # Reset between episodes
            for arm in robot.arm_names:
                if robot._ag_obj_in_hand.get(arm) is not None:
                    try:
                        robot.release_grasp_immediately(arm=arm)
                    except Exception:
                        pass

            robot.reset()
            env.scene.reset()
            collector.nav.invalidate_map_cache()

            for _ in range(30):
                og.sim.step()

    finally:
        wrapper.stop_recording()
        print(f"[Episode] Done: {successful_episodes}/{config.num_episodes} episodes in {attempt} attempts", flush=True)

    og.shutdown()
