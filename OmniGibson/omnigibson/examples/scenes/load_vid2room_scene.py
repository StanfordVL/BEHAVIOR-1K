import pathlib
import json
import traceback
from collections import defaultdict

from omnigibson.scenes.scene_base import Scene
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
import torch as th
from omnigibson.objects.dataset_object import DatasetObject
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
import trimesh
import trimesh.proximity
import numpy as np
from tqdm.auto import tqdm
from shapely.geometry import Polygon, Point

import omnigibson as og
from omnigibson.macros import gm


PYTORCH_TO_OPENCV = R.from_euler("z", [180], degrees=True)
PYTORCH_TO_OPENCV_4 = np.eye(4)
PYTORCH_TO_OPENCV_4[:3, :3] = PYTORCH_TO_OPENCV.as_matrix()
Z_UP_TO_Y_UP = R.from_matrix(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
Z_UP_TO_Y_UP_4 = np.eye(4)
Z_UP_TO_Y_UP_4[:3, :3] = Z_UP_TO_Y_UP.as_matrix()
COLLISION_THRESHOLD = 0.30

# Objects whose AABB bottom is more than this many meters below the floor (z=0)
# are considered broken and removed. Everything else is pushed up.
MAX_FLOOR_PENETRATION_DEPTH = 1.0

# Positional corrections smaller than this (in meters) are ignored.
STABILIZATION_TOLERANCE = 0.005

# Maximum vertical gap (in meters) between an object's bottom and a potential
# support's top for the support relationship to be considered in the support graph.
SUPPORT_SEARCH_DISTANCE = 0.5

# Minimum lateral tolerance (in meters) when checking if an object is above a support.
# The actual tolerance used is max(LATERAL_TOLERANCE_FACTOR * obj_xy_extent, this value).
LATERAL_TOLERANCE_MIN = 0.2

# Fraction of the object's horizontal extent used as lateral tolerance in the
# support graph. Combined with LATERAL_TOLERANCE_MIN via max().
LATERAL_TOLERANCE_FACTOR = 0.5

# Maximum distance (in meters) from an object to the nearest wall mesh for
# the object to be classified as wall-mounted.
WALL_PROXIMITY_THRESHOLD = 0.5

# Minimum AABB bottom z (in meters) for an object to be classified as wall-mounted.
# Objects resting on the floor (bottom near z=0) are never wall-mounted.
WALL_MOUNT_MIN_HEIGHT = 1.2

# An object whose AABB top exceeds this fraction of the ceiling height, and has
# no convincing support below, is classified as ceiling-mounted.
CEILING_PROXIMITY_FRACTION = 0.95

# When two objects' collision meshes overlap and the AABB intersection volume
# exceeds this fraction of the smaller object's AABB volume, the smaller object
# is removed as a duplicate.
DUPLICATE_OVERLAP_FRACTION = 0.3

# During the final depenetration pass, if the AABB intersection volume between
# two colliding objects exceeds this fraction of the smaller object's AABB volume,
# the smaller object is removed. Otherwise the objects are pushed apart.
DEPENETRATION_REMOVAL_FRACTION = 0.2


def snap_rotation(rot, threshold_degrees=15):
    # 1. Create rotation object
    matrix = rot.as_matrix()  # 3x3 matrix: columns are Local X, Y, Z

    # Extract basis vectors (columns)
    # X=0, Y=1, Z=2
    axes = [matrix[:, 0], matrix[:, 1], matrix[:, 2]]

    best_axis_idx = -1
    closest_dot = 0
    sign = 1

    # 2. Find which local axis is closest to World Up (0,0,1)
    # We check dot product with (0,0,1), which is just the z-component of the vector
    for i, axis in enumerate(axes):
        z_component = axis[2]
        if abs(z_component) > abs(closest_dot):
            closest_dot = z_component
            best_axis_idx = i
            # Is it pointing Up (+1) or Down (-1)?
            sign = 1 if z_component > 0 else -1

    # 3. Check Threshold
    # Dot product of 1.0 = 0 degrees.
    # We need to convert degrees to dot product threshold.
    # cos(10 degrees) ~= 0.9848
    threshold_dot = np.cos(np.deg2rad(threshold_degrees))

    if abs(closest_dot) < threshold_dot:
        return rot  # Not close enough to snap

    # 4. Construct Snapped Basis
    # The 'vertical' axis is forced to be exactly World Z
    new_vertical = np.array([0.0, 0.0, float(sign)])

    # We need a 'horizontal' axis to preserve the Yaw.
    # We pick a different axis (e.g., if Z is vertical, pick X)
    # If X (idx 0) is vertical, pick Y (idx 1).
    horizontal_idx = (best_axis_idx + 1) % 3
    raw_horizontal = axes[horizontal_idx].copy()

    # Flatten horizontal axis to XY plane and normalize
    raw_horizontal[2] = 0
    new_horizontal = raw_horizontal / np.linalg.norm(raw_horizontal)

    # Compute the third axis using Cross Product
    # Order depends on which slot we are filling to maintain Right-Hand Rule
    # We have two known vectors, we need to arrange them into a matrix

    new_matrix = np.zeros((3, 3))

    # Place the vertical axis
    new_matrix[:, best_axis_idx] = new_vertical

    # Place the horizontal axis
    new_matrix[:, horizontal_idx] = new_horizontal

    # Calculate the remaining axis via cross product
    # To determine cross order (A x B vs B x A), recall: X x Y = Z.
    # It is safer to re-cross depending on indices,
    # but a simple trick is to fill the matrix and use SVD or QR to orthonormalize,
    # OR just cross manually:

    # Simplified Cross Logic:
    # If we snapped Local Z (2) and used Local X (0): Local Y (1) = Z cross X
    if best_axis_idx == 2:  # Z is vertical
        # Y = Z cross X
        new_matrix[:, 1] = np.cross(new_vertical, new_horizontal)
    elif best_axis_idx == 0:  # X is vertical
        # Z = X cross Y (Horizontal was Y)
        new_matrix[:, 2] = np.cross(new_vertical, new_horizontal)
    elif best_axis_idx == 1:  # Y is vertical
        # X = Y cross Z (Horizontal was Z)
        new_matrix[:, 0] = np.cross(new_vertical, new_horizontal)

    # Convert back to rotation representation
    return R.from_matrix(new_matrix)


def load_collision_meshes_from_npz(npz_path: pathlib.Path) -> list:
    """
    Load pre-computed collision meshes from an NPZ file.

    Args:
        npz_path: Path to the NPZ file containing collision mesh data

    Returns:
        List of trimesh.Trimesh collision meshes
    """
    data = np.load(npz_path)
    collision_meshes = []
    i = 0
    while f"vertices_{i}" in data:
        vertices = data[f"vertices_{i}"]
        faces = data[f"faces_{i}"]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        collision_meshes.append(mesh)
        i += 1
    return collision_meshes


def build_wall_meshes(vertices, ceiling_height, wall_thickness, additional_buffer=0.1):
    """Build trimesh wall segments from room boundary vertices."""
    inner_polygon = Polygon(vertices).buffer(additional_buffer, join_style="mitre")
    outer_polygon = inner_polygon.buffer(wall_thickness, join_style="mitre")

    inner_coords = np.array(inner_polygon.exterior.coords)
    outer_coords = np.array(outer_polygon.exterior.coords)

    wall_meshes = []
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        inner_i = inner_coords[i]
        inner_j = inner_coords[j]
        outer_i = outer_coords[i]
        outer_j = outer_coords[j]

        wall_polygon = Polygon([inner_i, inner_j, outer_j, outer_i])
        if wall_polygon.is_valid and wall_polygon.area > 0:
            try:
                wall_mesh = trimesh.creation.extrude_polygon(wall_polygon, height=ceiling_height)
                wall_meshes.append(wall_mesh)
            except Exception as e:
                print(f"Failed to create wall mesh {i}: {e}")
    return wall_meshes


def build_floor_mesh(vertices, wall_thickness, additional_buffer=0.1, floor_thickness=0.05):
    """Build a trimesh floor with top surface at z=0."""
    inner_polygon = Polygon(vertices).buffer(additional_buffer, join_style="mitre")
    outer_polygon = inner_polygon.buffer(wall_thickness, join_style="mitre")
    floor_mesh = trimesh.creation.extrude_polygon(outer_polygon, height=floor_thickness)
    floor_mesh.vertices -= np.array([0, 0, floor_thickness])
    return floor_mesh


def build_wall_collision_manager(wall_meshes):
    """Build a CollisionManager containing only wall meshes."""
    manager = trimesh.collision.CollisionManager()
    for i, wm in enumerate(wall_meshes):
        manager.add_object(f"wall_{i}", wm)
    return manager


def _compute_object_transform(output_entry):
    """Recompute a 4x4 transform from output_data position/rotation/scale."""
    tf = np.eye(4)
    rot = R.from_quat(output_entry["rotation"])
    tf[:3, :3] = rot.as_matrix() @ np.diag(output_entry["scale"])
    tf[:3, 3] = output_entry["position"]
    return tf


def _recompute_transformed_mesh(name, output_data, original_meshes, transformed_meshes):
    """Recompute a transformed mesh after position has been updated."""
    tf = _compute_object_transform(output_data[name])
    mesh = original_meshes[name].copy()
    mesh.apply_transform(tf)
    transformed_meshes[name] = mesh


def remove_duplicate_objects(output_data, original_meshes, transformed_meshes, collision_manager):
    """
    Remove duplicate objects that have significant collision mesh overlap.

    Uses the trimesh CollisionManager to find pairs in collision, then computes
    what fraction of each object's volume is overlapping. When the overlap fraction
    exceeds DUPLICATE_OVERLAP_FRACTION for the smaller object, the smaller one
    is removed.
    """
    in_collision, name_pairs, contacts = collision_manager.in_collision_internal(
        return_names=True, return_data=True)
    if not in_collision:
        return

    # Aggregate max penetration depth per object-pair (collision mesh names are "{obj}-{i}")
    pair_depths = {}
    for contact in contacts:
        names_list = list(contact.names)
        obj_a = names_list[0].rsplit("-", 1)[0]
        obj_b = names_list[1].rsplit("-", 1)[0]
        if obj_a == obj_b:
            continue
        pair = tuple(sorted([obj_a, obj_b]))
        pair_depths[pair] = max(pair_depths.get(pair, 0), contact.depth)

    # For each pair with significant penetration, check volume overlap fraction
    to_remove = {}
    for (name_a, name_b), depth in pair_depths.items():
        if name_a not in output_data or name_b not in output_data:
            continue
        if name_a in to_remove or name_b in to_remove:
            continue

        mesh_a = transformed_meshes[name_a]
        mesh_b = transformed_meshes[name_b]
        aabb_a = mesh_a.bounds
        aabb_b = mesh_b.bounds
        extent_a = aabb_a[1] - aabb_a[0]
        extent_b = aabb_b[1] - aabb_b[0]
        vol_a = np.prod(extent_a)
        vol_b = np.prod(extent_b)

        # Compute AABB intersection volume
        overlap_min = np.maximum(aabb_a[0], aabb_b[0])
        overlap_max = np.minimum(aabb_a[1], aabb_b[1])
        overlap_extent = np.maximum(overlap_max - overlap_min, 0)
        overlap_vol = np.prod(overlap_extent)

        if overlap_vol == 0:
            continue

        smaller_vol = min(vol_a, vol_b)
        overlap_frac = overlap_vol / smaller_vol if smaller_vol > 0 else 0

        if overlap_frac > DUPLICATE_OVERLAP_FRACTION:
            victim = name_a if vol_a <= vol_b else name_b
            survivor = name_b if victim == name_a else name_a
            to_remove[victim] = (survivor, overlap_frac)

    for name, (survivor, frac) in to_remove.items():
        print(f"  Removing duplicate: {name} ({frac*100:.0f}% overlap with {survivor})")
        del output_data[name]
        del transformed_meshes[name]
        if name in original_meshes:
            del original_meshes[name]
        i = 0
        while True:
            try:
                collision_manager.remove_object(f"{name}-{i}")
                i += 1
            except ValueError:
                break


def build_support_graph(transformed_meshes, output_data):
    """
    Build a support graph that is tolerant of lateral drift, clipping, and hovering.

    Returns dict mapping object name -> (support_name_or_"floor"_or_None, score).
    """
    support_map = {}

    for name_a, mesh_a in transformed_meshes.items():
        aabb_a = mesh_a.bounds
        center_a_xy = (aabb_a[0][:2] + aabb_a[1][:2]) / 2
        bottom_a_z = aabb_a[0][2]
        extent_a = aabb_a[1] - aabb_a[0]
        extent_a_z = extent_a[2]
        extent_a_xy = extent_a[:2]

        best_support = None
        best_score = float("inf")

        for name_b, mesh_b in transformed_meshes.items():
            if name_b == name_a:
                continue
            aabb_b = mesh_b.bounds
            top_b_z = aabb_b[1][2]

            vertical_gap = bottom_a_z - top_b_z
            if vertical_gap > SUPPORT_SEARCH_DISTANCE:
                continue
            if vertical_gap < -extent_a_z * 0.5:
                continue

            clamped = np.clip(center_a_xy, aabb_b[0][:2], aabb_b[1][:2])
            horiz_dist = np.linalg.norm(center_a_xy - clamped)

            lateral_tolerance = max(np.max(extent_a_xy) * LATERAL_TOLERANCE_FACTOR, LATERAL_TOLERANCE_MIN)
            if horiz_dist > lateral_tolerance:
                continue

            score = abs(vertical_gap) + horiz_dist * 0.5
            if score < best_score:
                best_score = score
                best_support = name_b

        floor_gap = bottom_a_z
        floor_score = abs(floor_gap)
        if (floor_gap > -MAX_FLOOR_PENETRATION_DEPTH
                and floor_gap < SUPPORT_SEARCH_DISTANCE
                and floor_score < best_score):
            best_support = "floor"
            best_score = floor_score

        support_map[name_a] = (best_support, best_score)
    return support_map


def classify_objects(support_map, transformed_meshes, wall_manager, ceiling_height):
    """
    Classify each object geometrically as one of:
    'ceiling-mounted', 'wall-mounted', 'floor-standing', 'surface-resting', 'floating'.
    """
    classifications = {}

    for name, (support, support_score) in support_map.items():
        mesh = transformed_meshes[name]
        aabb = mesh.bounds
        bottom_z = aabb[0][2]
        top_z = aabb[1][2]
        extent_z = top_z - bottom_z

        no_convincing_support = (support is None) or (support_score > SUPPORT_SEARCH_DISTANCE)

        if top_z > ceiling_height * CEILING_PROXIMITY_FRACTION and no_convincing_support:
            classifications[name] = "ceiling-mounted"
            continue

        try:
            wall_dist, _, _ = wall_manager.min_distance_single(mesh)
        except Exception:
            wall_dist = float("inf")

        if (bottom_z > WALL_MOUNT_MIN_HEIGHT
                and wall_dist < WALL_PROXIMITY_THRESHOLD
                and no_convincing_support):
            classifications[name] = "wall-mounted"
        elif support == "floor":
            classifications[name] = "floor-standing"
        elif support is not None and support != "floor":
            classifications[name] = "surface-resting"
        else:
            classifications[name] = "floating"

    return classifications


def remove_invalid_objects(output_data, original_meshes, transformed_meshes,
                           collision_manager, wall_manager, ceiling_height):
    """
    Iteratively remove invalid objects until the scene is stable:
    - Objects significantly below floor
    - Floating objects with no support
    - Wall-classified objects not actually near a wall
    Chain-removes objects that become unsupported after removals.
    """
    changed = True
    while changed:
        changed = False
        support_map = build_support_graph(transformed_meshes, output_data)
        classifications = classify_objects(support_map, transformed_meshes, wall_manager, ceiling_height)

        to_remove = {}
        for name in list(output_data.keys()):
            aabb = transformed_meshes[name].bounds
            extent_z = aabb[1][2] - aabb[0][2]
            cls = classifications[name]

            if aabb[0][2] < -MAX_FLOOR_PENETRATION_DEPTH:
                to_remove[name] = (
                    f"bottom z={aabb[0][2]:.3f} is {abs(aabb[0][2]):.2f}m below floor "
                    f"(max allowed {MAX_FLOOR_PENETRATION_DEPTH:.1f}m)"
                )
            elif cls == "floating":
                support, score = support_map[name]
                floor_dist = aabb[0][2]
                to_remove[name] = (
                    f"classified as floating (no convincing support; "
                    f"best candidate={support}, score={score:.3f}, "
                    f"bottom z={floor_dist:.3f}, floor distance={abs(floor_dist):.3f}m)"
                )

        if to_remove:
            changed = True
            for name, reason in to_remove.items():
                print(f"  Removing {name}: {reason}")
                del output_data[name]
                del transformed_meshes[name]
                if name in original_meshes:
                    del original_meshes[name]
                # Remove from collision manager
                i = 0
                while True:
                    try:
                        collision_manager.remove_object(f"{name}-{i}")
                        i += 1
                    except ValueError:
                        break

    final_support_map = build_support_graph(transformed_meshes, output_data)
    final_classifications = classify_objects(
        final_support_map, transformed_meshes, wall_manager, ceiling_height)
    return final_support_map, final_classifications


def stabilize_bottom_up(output_data, original_meshes, transformed_meshes,
                        support_map, classifications, collision_manager):
    """
    Unified bottom-up stabilization pass. For each object in topological order:
    1. Fix lateral drift (clamp XY onto support footprint)
    2. Fix vertical position (place AABB bottom on support AABB top)
    3. Recompute mesh so dependents see corrected geometry
    """
    dependents = defaultdict(list)
    for name, (support, _) in support_map.items():
        if support is not None:
            dependents[support].append(name)

    topo_order = []
    visited = set()
    queue = [n for n, (s, _) in support_map.items() if s == "floor"]
    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        topo_order.append(name)
        for dep in dependents.get(name, []):
            queue.append(dep)

    for name in output_data:
        if name not in visited:
            topo_order.append(name)

    for name in topo_order:
        if name not in output_data:
            continue
        cls = classifications.get(name, "floating")
        if cls in ("wall-mounted", "ceiling-mounted"):
            continue

        support, _ = support_map[name]
        aabb = transformed_meshes[name].bounds
        obj_center = (aabb[0] + aabb[1]) / 2.0
        delta = np.array([0.0, 0.0, 0.0])

        if support not in (None, "floor") and support in transformed_meshes:
            sup_aabb = transformed_meshes[support].bounds
            center_xy = obj_center[:2]
            clamped_xy = np.clip(center_xy, sup_aabb[0][:2], sup_aabb[1][:2])
            delta[:2] = clamped_xy - center_xy

        if support == "floor":
            desired_bottom_z = 0.0
        elif support is not None and support in transformed_meshes:
            sup_aabb = transformed_meshes[support].bounds
            desired_bottom_z = sup_aabb[1][2]
        else:
            desired_bottom_z = None

        if desired_bottom_z is not None:
            current_bottom_z = aabb[0][2]
            delta[2] = desired_bottom_z - current_bottom_z

        if np.linalg.norm(delta) > STABILIZATION_TOLERANCE:
            output_data[name]["position"][0] += delta[0]
            output_data[name]["position"][1] += delta[1]
            output_data[name]["position"][2] += delta[2]

            _recompute_transformed_mesh(name, output_data, original_meshes, transformed_meshes)


def fix_wall_mounted(obj_name, obj_mesh, wall_meshes, output_data, original_meshes, transformed_meshes):
    """
    Snap a wall-mounted object flush against the nearest wall.
    Uses trimesh.proximity.closest_point for position and face_normals for rotation.
    """
    best_wall_dist = float("inf")
    best_wall_point = None
    best_wall_normal = None

    obj_center = obj_mesh.centroid
    for wall_mesh in wall_meshes:
        try:
            closest_pts, distances, triangle_ids = trimesh.proximity.closest_point(
                wall_mesh, [obj_center])
        except Exception:
            continue
        if distances[0] < best_wall_dist:
            best_wall_dist = distances[0]
            best_wall_point = closest_pts[0]
            best_wall_normal = wall_mesh.face_normals[triangle_ids[0]]

    if best_wall_point is None:
        return

    wall_normal_xy = best_wall_normal[:2].copy()
    norm = np.linalg.norm(wall_normal_xy)
    if norm < 1e-6:
        return
    wall_normal_xy /= norm
    wall_normal_3d = np.array([wall_normal_xy[0], wall_normal_xy[1], 0.0])

    obj_aabb = obj_mesh.bounds
    obj_extent = obj_aabb[1] - obj_aabb[0]
    half_depth = abs(np.dot(obj_extent, wall_normal_3d)) / 2.0

    desired_xy = best_wall_point[:2] + wall_normal_xy * half_depth
    output_data[obj_name]["position"][0] = float(desired_xy[0])
    output_data[obj_name]["position"][1] = float(desired_xy[1])

    rot = R.from_quat(output_data[obj_name]["rotation"])
    matrix = rot.as_matrix()

    best_axis = -1
    best_dot = 0
    for i in range(3):
        local_axis_xy = matrix[:2, i]
        dot = abs(np.dot(local_axis_xy / (np.linalg.norm(local_axis_xy) + 1e-9), wall_normal_xy))
        if dot > best_dot:
            best_dot = dot
            best_axis = i

    if best_axis >= 0 and best_dot > 0.5:
        current_axis = matrix[:, best_axis].copy()
        current_axis_xy = current_axis[:2]
        sign = 1.0 if np.dot(current_axis_xy, wall_normal_xy) < 0 else -1.0
        target_axis = sign * wall_normal_3d
        target_axis[2] = current_axis[2]
        target_norm = np.linalg.norm(target_axis)
        if target_norm > 1e-6:
            target_axis /= target_norm
            new_matrix = matrix.copy()
            new_matrix[:, best_axis] = target_axis

            other_horiz = (best_axis + 1) % 3
            if other_horiz == np.argmax(np.abs(matrix[2, :])):
                other_horiz = (best_axis + 2) % 3
            vertical_idx = 3 - best_axis - other_horiz

            new_matrix[:, other_horiz] = np.cross(new_matrix[:, vertical_idx], new_matrix[:, best_axis])
            nh_norm = np.linalg.norm(new_matrix[:, other_horiz])
            if nh_norm > 1e-6:
                new_matrix[:, other_horiz] /= nh_norm
                new_matrix[:, vertical_idx] = np.cross(new_matrix[:, best_axis], new_matrix[:, other_horiz])
                nv_norm = np.linalg.norm(new_matrix[:, vertical_idx])
                if nv_norm > 1e-6:
                    new_matrix[:, vertical_idx] /= nv_norm
                    try:
                        new_rot = R.from_matrix(new_matrix)
                        output_data[obj_name]["rotation"] = new_rot.as_quat().tolist()
                    except Exception:
                        pass

    _recompute_transformed_mesh(obj_name, output_data, original_meshes, transformed_meshes)


def depenetrate_objects(output_data, original_meshes, transformed_meshes):
    """
    Final depenetration pass using current transformed meshes.
    For each pair of colliding objects:
    - If overlap volume > DEPENETRATION_REMOVAL_FRACTION of the smaller object, remove it.
    - Otherwise, push the two objects apart along the axis of least AABB overlap.
    """
    cm = trimesh.collision.CollisionManager()
    for name, mesh in transformed_meshes.items():
        if name not in output_data:
            continue
        cm.add_object(name, mesh)

    in_collision, _, contacts = cm.in_collision_internal(
        return_names=True, return_data=True)
    if not in_collision:
        return

    pair_depths = {}
    for contact in contacts:
        names_list = list(contact.names)
        obj_a = names_list[0]
        obj_b = names_list[1]
        if obj_a == obj_b:
            continue
        pair = tuple(sorted([obj_a, obj_b]))
        pair_depths[pair] = max(pair_depths.get(pair, 0), contact.depth)

    to_remove = {}
    pushed = set()

    for (name_a, name_b), depth in sorted(pair_depths.items(), key=lambda x: -x[1]):
        if name_a not in output_data or name_b not in output_data:
            continue
        if name_a in to_remove or name_b in to_remove:
            continue

        mesh_a = transformed_meshes[name_a]
        mesh_b = transformed_meshes[name_b]
        aabb_a = mesh_a.bounds
        aabb_b = mesh_b.bounds
        vol_a = np.prod(aabb_a[1] - aabb_a[0])
        vol_b = np.prod(aabb_b[1] - aabb_b[0])

        overlap_min = np.maximum(aabb_a[0], aabb_b[0])
        overlap_max = np.minimum(aabb_a[1], aabb_b[1])
        overlap_extent = np.maximum(overlap_max - overlap_min, 0)
        overlap_vol = np.prod(overlap_extent)
        if overlap_vol == 0:
            continue

        smaller_vol = min(vol_a, vol_b)
        overlap_frac = overlap_vol / smaller_vol if smaller_vol > 0 else 0

        if overlap_frac > DEPENETRATION_REMOVAL_FRACTION:
            victim = name_a if vol_a <= vol_b else name_b
            survivor = name_b if victim == name_a else name_a
            to_remove[victim] = (
                f"penetrating {survivor} by {overlap_frac*100:.0f}% "
                f"(threshold {DEPENETRATION_REMOVAL_FRACTION*100:.0f}%)"
            )
        else:
            if name_a in pushed or name_b in pushed:
                continue
            sep_axis = _find_separation_axis(aabb_a, aabb_b)
            center_a = (aabb_a[0] + aabb_a[1]) / 2.0
            center_b = (aabb_b[0] + aabb_b[1]) / 2.0
            half_push = overlap_extent[sep_axis] / 2.0 + STABILIZATION_TOLERANCE
            direction = 1.0 if center_a[sep_axis] < center_b[sep_axis] else -1.0

            output_data[name_a]["position"][sep_axis] -= float(direction * half_push)
            output_data[name_b]["position"][sep_axis] += float(direction * half_push)
            _recompute_transformed_mesh(name_a, output_data, original_meshes, transformed_meshes)
            _recompute_transformed_mesh(name_b, output_data, original_meshes, transformed_meshes)
            pushed.update([name_a, name_b])
            print(f"  Depenetrating {name_a} and {name_b}: pushed apart {2*half_push:.3f}m along axis {sep_axis}")

    for name, reason in to_remove.items():
        print(f"  Removing {name}: {reason}")
        del output_data[name]
        del transformed_meshes[name]
        if name in original_meshes:
            del original_meshes[name]


def _find_separation_axis(aabb_a, aabb_b):
    """Find the axis with the smallest AABB overlap (cheapest to separate along)."""
    overlap_min = np.maximum(aabb_a[0], aabb_b[0])
    overlap_max = np.minimum(aabb_a[1], aabb_b[1])
    overlap_extent = np.maximum(overlap_max - overlap_min, 0)
    return int(np.argmin(overlap_extent))


def parse_object_meshes(scene_dir: pathlib.Path) -> dict:
    meshes_root = scene_dir / "obj_meshes_2"
    collision_root = scene_dir / "obj_meshes_2_collision"
    poses_root = scene_dir / "obj_meshes_2"

    cnp = np.load(scene_dir / "sparse_pi3x/0/cameras_and_points.npz")
    filenames_in_cnp = list(cnp["filenames"])
    pointmaps = cnp["local_points"]
    camera_poses = cnp["camera_poses"]
    meshes = {}
    pose_files = list(poses_root.glob("*.json"))
    for pose_json in tqdm(pose_files):
        # Load the visual mesh
        mesh_path = meshes_root / (pose_json.stem + ".glb")
        if not mesh_path.exists():
            continue
        mesh = trimesh.load(mesh_path, force="mesh")

        # Load the collision meshes
        collision_path = collision_root / (pose_json.stem + ".npz")
        if not collision_path.exists():
            continue
        collision_meshes = load_collision_meshes_from_npz(collision_path)

        # Load pose information
        with open(pose_json, "r") as f:
            frame_data = json.load(f)

        # Convert to the correct frames
        for frame_name, frame_info in frame_data.items():
            index = filenames_in_cnp.index(frame_name)
            frame_info["pointmap"] = pointmaps[index].reshape(-1, 3)
            frame_info["camera_pose"] = camera_poses[index]

            frame_info["scale"] = np.array(frame_info["post_postprocess"]["scale"]).reshape(-1)
            frame_info["rotation"] = frame_info["post_postprocess"]["rotation"]
            frame_info["translation"] = frame_info["post_postprocess"]["translation"]

            scale = np.array(frame_info["scale"]).reshape(-1).tolist()
            scale = np.array(scale + [1])
            scale_transform = np.diag(scale)
            rotation_transform = np.eye(4)
            rotation_transform[:3, :3] = (
                R.from_quat(np.array(frame_info["rotation"]).reshape(-1), scalar_first=True).as_matrix().T
            )
            translation_transform = np.eye(4)
            translation_transform[:3, 3] = np.array(frame_info["translation"])
            transform = (
                PYTORCH_TO_OPENCV_4 @ translation_transform @ rotation_transform @ scale_transform @ Z_UP_TO_Y_UP_4
            )

            frame_info["obj_in_cam"] = transform
            frame_info["obj_in_world"] = frame_info["camera_pose"] @ transform

            frame_info["world_position"] = frame_info["obj_in_world"][:3, 3]
            frame_info["world_rotation"] = frame_info["obj_in_world"][:3, :3] / frame_info["scale"]

        # Package the data
        data = {
            "frames": frame_data,
            "mesh": mesh,
            "mesh_path": mesh_path,
            "collision_meshes": collision_meshes,
        }
        meshes[mesh_path.stem] = data

    # Load room geometry
    floorplan_path = scene_dir / "floorplan" / "room_parameters.json"
    scene_data = json.loads(floorplan_path.read_text())

    vlm_analysis_data = json.loads((scene_dir / "vlm_analysis.json").read_text())
    ceiling_height = vlm_analysis_data["ceiling_height"]

    vertices = [x[:2] for x in scene_data["boundary3d"]]
    vertices = vertices[: len(vertices) // 2]
    vertices = np.array(vertices)

    wall_thickness = 0.1
    additional_buffer = 0.1

    # Step 1: Build structure geometry
    print("Building structure geometry...")
    wall_meshes = build_wall_meshes(vertices, ceiling_height, wall_thickness, additional_buffer)
    wall_manager = build_wall_collision_manager(wall_meshes)
    floor_mesh = build_floor_mesh(vertices, wall_thickness, additional_buffer)

    collision_manager = trimesh.collision.CollisionManager()
    output_data = {}
    original_meshes = {}
    transformed_meshes = {}

    for mesh_name, data in meshes.items():
        if mesh_name.rsplit("-", 2)[0] in ("curtain", "pillow"):
            continue

        avg_position = np.median([frame["world_position"] for frame in data["frames"].values()], axis=0)

        scales = [np.array(frame["scale"]).reshape(-1) for frame in data["frames"].values()]
        frames_list = list(data["frames"].values())
        scale_norms = [np.linalg.norm(s) for s in scales]
        median_scale_idx = np.argsort(scale_norms)[len(scale_norms) // 2]
        avg_scale = scales[median_scale_idx]

        avg_rotation = R.from_matrix(frames_list[median_scale_idx]["world_rotation"])
        snapped_rotation = snap_rotation(avg_rotation)

        avg_tf = np.eye(4)
        avg_tf[:3, :3] = snapped_rotation.as_matrix() @ np.diag(avg_scale)
        avg_tf[:3, 3] = avg_position

        mesh = data["mesh"].copy()
        mesh.apply_transform(avg_tf)

        original_meshes[mesh_name] = data["mesh"]
        transformed_meshes[mesh_name] = mesh

        for i, cmesh in enumerate(data["collision_meshes"]):
            cmesh_copy = cmesh.copy()
            cmesh_copy.apply_transform(avg_tf)
            collision_manager.add_object(f"{mesh_name}-{i}", cmesh_copy)

        output_data[mesh_name] = {
            "scale": avg_scale.tolist(),
            "rotation": snapped_rotation.as_quat().tolist(),
            "position": avg_position.tolist(),
        }

    print(f"Loaded {len(output_data)} objects. Running fixup pipeline...")

    # Step 0: Remove duplicate objects (significant collision mesh overlap)
    print("Removing duplicate objects...")
    remove_duplicate_objects(output_data, original_meshes, transformed_meshes, collision_manager)
    print(f"  {len(output_data)} objects remain after deduplication.")

    # Step 2 + 3 + 4: Build support graph, classify, remove invalid objects
    print("Removing invalid objects...")
    support_map, classifications = remove_invalid_objects(
        output_data, original_meshes, transformed_meshes,
        collision_manager, wall_manager, ceiling_height)
    print(f"  {len(output_data)} objects remain after removal.")

    # Step 5: Unified bottom-up stabilization
    print("Stabilizing object placements (bottom-up)...")
    stabilize_bottom_up(output_data, original_meshes, transformed_meshes,
                        support_map, classifications, collision_manager)

    # Step 6: Fix wall-mounted objects
    print("Fixing wall-mounted objects...")
    for name, cls in classifications.items():
        if cls == "wall-mounted" and name in output_data:
            fix_wall_mounted(name, transformed_meshes[name], wall_meshes,
                             output_data, original_meshes, transformed_meshes)

    # Step 7: Depenetration -- remove or push apart remaining colliding objects
    print("Depenetrating objects...")
    depenetrate_objects(output_data, original_meshes, transformed_meshes)
    print(f"  {len(output_data)} objects remain after depenetration.")

    print("Fixup pipeline complete.")
    return output_data


def load_object(room_dir, mesh_name, scale):
    """Load a regular object (furniture, etc.) from the dataset."""
    scene_id = get_scene_id(room_dir)
    in_rooms = [room_dir.name]
    # Match naming from import_vid2room_objects.py
    base_category = mesh_name.rsplit("-", 2)[0]
    category = "".join(c if c.isalnum() or c == "_" else "_" for c in base_category.lower())
    model = f"{scene_id}_{mesh_name}".replace("-", "_")
    model = "".join(c if c.isalnum() or c == "_" else "" for c in model.lower())
    fixed_base = True

    i = len(og.sim.scenes[0].objects)
    obj = DatasetObject(
        name=f"{category}_{i}",
        category=category,
        model=model,
        fixed_base=fixed_base,
        dataset_name="vid2room",
        scale=scale,
        in_rooms=in_rooms,
    )

    og.sim.scenes[0].add_object(obj)

    return obj


def load_structure_objects(room_dir):
    """
    Load all pre-generated structure objects (floor, wall, ceiling) as a DatasetObject.

    The material is already baked into the USD during the import process via
    import_vid2room_scene_structures.py.

    Args:
        scene_id: The Vid2Room scene identifier (e.g., "train_0")

    Returns:
        List of DatasetObject: The loaded structure objects
    """
    # Map structure type to category
    scene_id = get_scene_id(room_dir)
    in_rooms = [room_dir.name]
    dataset_root = pathlib.Path(gm.DATA_PATH) / "vid2room"
    objects = []
    for category in ("floors", "walls"):
        category_root = dataset_root / "objects" / category
        model_prefix = f"vid2room_{scene_id}_".replace("-", "_")
        models = []
        if category_root.exists():
            for model_dir in category_root.iterdir():
                if not model_dir.is_dir():
                    continue
                if not model_dir.name.startswith(model_prefix):
                    continue
                if not (model_dir / "import.success").exists():
                    continue
                models.append(model_dir.name)
        for model in models:
            i = len(og.sim.scenes[0].objects)
            obj = DatasetObject(
                name=f"{category}_{i}",
                category=category,
                model=model,
                fixed_base=True,
                dataset_name="vid2room",
                in_rooms=in_rooms,
            )

            og.sim.scenes[0].add_object(obj)
            objects.append(obj)

    return objects


def get_scene_id(room_dir):
    """
    Generate a unique scene ID from scene path.

    For vid2room scenes, the path structure is typically:
    .../vid_XXXXX/rooms/room_type_N

    We want to create a unique ID like: vid_XXXXX_room_type_N
    """
    room_name = room_dir.name  # e.g., "living_room_0"
    # Go up to find the video ID (parent of "rooms" directory)
    video_id = room_dir.parent.parent.name  # e.g., "vid_1vdXN7X4Af4"
    assert video_id.startswith("vid_"), f"Video ID {video_id} does not start with 'vid_'"
    return f"{video_id}_{room_name}"


def load_vid2room_scene(room_dir):
    """
    Process and load a Vid2Room scene.

    Args:
        room_dir: Path to the room root directory
    """
    object_data = parse_object_meshes(room_dir)

    ogscene = Scene(use_floor_plane=True, floor_plane_visible=False)
    og.sim.import_scene(ogscene)

    load_structure_objects(room_dir)

    for mesh_name, data in object_data.items():
        obj = load_object(room_dir, mesh_name, data["scale"])
        obj.set_position_orientation(
            position=th.as_tensor(data["position"]), orientation=th.as_tensor(data["rotation"])
        )

    og.sim.play()

    for _ in range(10):
        og.sim.step()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_vid2room_scene.py <room_root>")
        sys.exit(1)

    room_root = pathlib.Path(sys.argv[1])

    if og.sim:
        og.clear()
    else:
        og.launch()

    load_vid2room_scene(room_root)

    while True:
        og.sim.render()
