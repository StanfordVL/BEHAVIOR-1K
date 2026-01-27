import logging

import numpy as np
import torch as th
from scipy.ndimage import label

logger = logging.getLogger(__name__)


def find_nearest_traversable_on_map(floor_map, map_pos, search_radius=50):
    h, w = floor_map.shape
    row, col = int(map_pos[0]), int(map_pos[1])

    if 0 <= row < h and 0 <= col < w and floor_map[row, col] == 255:
        return (row, col)

    for r in range(1, search_radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr) != r and abs(dc) != r:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if floor_map[nr, nc] == 255:
                        return (nr, nc)
    return None


def _find_on_eroded(trav_map, eroded_map, world_pos, max_radius=50):
    map_pos = trav_map.world_to_map(th.tensor([world_pos[0], world_pos[1]]))
    r, c = int(map_pos[0].item()), int(map_pos[1].item())
    h, w = eroded_map.shape

    if 0 <= r < h and 0 <= c < w:
        val = eroded_map[r, c]
        if hasattr(val, 'item'):
            val = val.item()
        if val == 255:
            return world_pos

    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    val = eroded_map[nr, nc]
                    if hasattr(val, 'item'):
                        val = val.item()
                    if val == 255:
                        world_pt = trav_map.map_to_world(th.tensor([nr, nc]))
                        return (world_pt[0].item(), world_pt[1].item())
    return None


def check_path_exists(scene, pos1: tuple, pos2: tuple, robot=None, eroded_map=None) -> bool:
    trav_map = scene._trav_map
    if eroded_map is None:
        eroded_map = trav_map._erode_trav_map(trav_map.floor_map[0].clone(), robot=robot)

    trav1 = _find_on_eroded(trav_map, eroded_map, pos1)
    trav2 = _find_on_eroded(trav_map, eroded_map, pos2)

    if trav1 is None or trav2 is None:
        return False

    try:
        source = th.tensor([trav1[0], trav1[1]]) if isinstance(trav1, tuple) else th.tensor(trav1)
        target = th.tensor([trav2[0], trav2[1]]) if isinstance(trav2, tuple) else th.tensor(trav2)
        result = scene.get_shortest_path(
            floor=0, source_world=source, target_world=target, entire_path=False, robot=robot,
        )
        return result is not None and result[0] is not None
    except Exception:
        return False


def find_connected_support_pairs(scene, supports: list, robot, eroded_map=None) -> list:
    """Find support pairs with verified navigable paths between them.

    Uses actual path planning to verify connectivity - only returns pairs
    where a valid path exists. Tries with robot erosion first, then falls back
    to default erosion if no pairs found.
    """
    trav_map = scene._trav_map

    # Try with robot erosion first
    if eroded_map is None:
        eroded_map = trav_map._erode_trav_map(trav_map.floor_map[0].clone(), robot=robot)

    pairs = _find_pairs_with_eroded_map(scene, supports, robot, eroded_map, trav_map)

    if not pairs:
        # Fallback: try with default erosion (no robot size consideration)
        logger.info("No pairs with robot erosion, trying default erosion")
        default_eroded = trav_map._erode_trav_map(trav_map.floor_map[0].clone(), robot=None)
        pairs = _find_pairs_with_eroded_map(scene, supports, None, default_eroded, trav_map)

    return pairs


def _find_pairs_with_eroded_map(scene, supports, robot, eroded_map, trav_map) -> list:
    """Helper to find connected pairs using a specific eroded map."""
    # First, find supports that have a nearby traversable point
    valid_supports = []
    for sup in supports:
        sup_pos, _ = sup.get_position_orientation()
        sup_xy = (sup_pos[0].item(), sup_pos[1].item())
        approach = _find_on_eroded(trav_map, eroded_map, sup_xy, max_radius=40)
        if approach is not None:
            valid_supports.append((sup, approach))

    if len(valid_supports) < 2:
        logger.info("Only %d supports have traversable approach points", len(valid_supports))
        return []

    # Now verify actual path connectivity between each pair
    valid_pairs = []
    checked = 0
    for i, (source_sup, source_approach) in enumerate(valid_supports):
        for j, (target_sup, target_approach) in enumerate(valid_supports):
            if i >= j:  # Skip self and duplicates (we'll add both directions if valid)
                continue

            checked += 1
            # Check if path exists using actual path planning
            if check_path_exists(scene, source_approach, target_approach, robot=robot, eroded_map=eroded_map):
                valid_pairs.append((source_sup, target_sup))
                valid_pairs.append((target_sup, source_sup))  # Both directions

    logger.info("Checked %d pairs, found %d with valid paths (%d supports)",
                checked, len(valid_pairs) // 2, len(valid_supports))
    return valid_pairs
