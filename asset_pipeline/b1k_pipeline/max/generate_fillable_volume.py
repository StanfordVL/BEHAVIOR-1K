import pymxs

rt = pymxs.runtime

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from collections import defaultdict
from scipy.spatial import ConvexHull
import networkx as nx

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name
from b1k_pipeline.max.collision_vertex_reduction import reduce_mesh

REQUIRE_ADDITIONAL_BASE_POINTS = False


def anorm(x, axis=None, keepdims=False):
    """Compute L2 norms alogn specified axes."""
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    norm = anorm(v, axis=axis, keepdims=True)
    return v / np.where(norm < eps, eps, norm)


def convert_to_trimesh(obj):
    # Get vertices and faces into numpy arrays for conversion
    verts = np.array(
        [rt.polyop.getVert(obj, i + 1) for i in range(rt.polyop.GetNumVerts(obj))]
    )
    faces = (
        np.array(
            rt.polyop.getFacesVerts(
                obj, rt.execute("#{1..%d}" % rt.polyop.GetNumFaces(obj))
            )
        )
        - 1
    )
    assert faces.shape[1] == 3, f"{obj.name} has non-triangular faces"

    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    assert m.is_volume, f"{obj.name} element is not a volume"

    return m


def vecs2quat(vec0, vec1, normalized=False):
    """
    Converts the angle from unnormalized 3D vectors @vec0 to @vec1 into a quaternion representation of the angle

    Args:
        vec0 (np.array): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        vec1 (np.array): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        normalized (bool): If True, @vec0 and @vec1 are assumed to already be normalized and we will skip the
            normalization step (more efficient)
    """
    # Normalize vectors if requested
    if not normalized:
        vec0 = normalize(vec0, axis=-1)
        vec1 = normalize(vec1, axis=-1)

    # Half-way Quaternion Solution -- see https://stackoverflow.com/a/11741520
    cos_theta = np.sum(vec0 * vec1, axis=-1, keepdims=True)
    quat_unnormalized = np.where(
        cos_theta == -1,
        np.array([1.0, 0, 0, 0]),
        np.concatenate([np.cross(vec0, vec1), 1 + cos_theta], axis=-1),
    )
    return quat_unnormalized / np.linalg.norm(quat_unnormalized, axis=-1, keepdims=True)


def sample_radial_rays(tm, point, normal, n=40, dist=4.0):
    """
    Shoots @n rays radially from @point in directions orthogonal to @normal, returning any hits with @tm within @dist

    Args:
        tm (Trimesh): mesh used to sample rays
        point (3-array): (x,y,z) origin point of rays
        normal (3-array): normal direction of the plane for sampling radial rays
        n (int): number of rays to shoot
        dist (float): max distance of rays

    Returns:
        2-tuple:
            - float: Proportion of rays cast that returned a valid hit
            - dict: Index-mapped hit (x,y,z) locations
    """
    angles = np.arange(n) * 2 * np.pi / n
    x = np.cos(angles)
    y = np.sin(angles)
    start_points = np.ones((n, 3)) * point.reshape(1, 3)
    directions = np.array([x, y, np.zeros(n)]).T

    # Rotate points appropriately
    rot = R.from_quat(vecs2quat(np.array([0, 0, 1.0]), np.array(normal))).as_matrix()
    directions = directions @ rot.T

    # Run raytest
    locations, index_ray, index_tri = tm.ray.intersects_location(
        ray_origins=start_points,
        ray_directions=directions,
    )

    results = defaultdict(list)

    # Loop through all hits, and add to results
    for location, idx in zip(locations, index_ray):
        results[idx].append(location)

    # Filter out for closest
    pruned_results = dict()
    for i in range(n):
        if i in results:
            result = results[i]
            dist = np.linalg.norm(np.array(result) - point.reshape(1, 3), axis=-1)
            min_idx = np.argmin(dist)
            pruned_results[i] = result[min_idx]

    # Return pruned results
    return len(pruned_results) / n, pruned_results


def shoot_ray(tm, point, direction):
    """
    Shoots a single ray from @start_point in direction @direction

    Args:
        tm (Trimesh): mesh used to shoot ray
        point (3-array): (x,y,z) origin point of ray
        direction (3-array): direction to shoot ray

    Returns:
        None or np.ndarray: None if no hit, else (x,y,z) location of nearest hit
    """
    # Run raytest
    locations, index_ray, index_tri = tm.ray.intersects_location(
        ray_origins=point.reshape(1, 3),
        ray_directions=direction.reshape(1, 3),
    )

    if len(locations) == 0:
        hit = None
    else:
        # Only keep closest
        dists = np.linalg.norm(np.array(locations) - point.reshape(1, 3), axis=-1)
        min_idx = np.argmin(dists)
        hit = locations[min_idx]

    return hit


def sample_fillable_volume(
    tm,
    start_point,
    direction=(0, 0, 1.0),
    hit_threshold=0.75,
    n_rays=100,
):
    """
    Samples fillable volume within a given mesh @tm's cavity using raycasting

    Args:
        tm (Trimesh): Trimesh mesh to within which to sample fillable
        start_point (3-array): (x,y,z) value used as seed starting point within the mesh
        direction (3-array): (x,y,z) direction vector determining which direction raycasting should occur
        hit_threshold (float): Proportion of rays that should hit each iteration of the fillable volume generation
            for generation to continue
        n_rays (int): Number of radial rays to shoot during each iteration. Must be divisible by 4!

    Returns:
        Trimesh: The sampled trimesh fillable volume
    """
    n_rays_half = int(n_rays / 2)
    n_rays_quarter = int(n_rays / 4)

    # Make sure n_rays is divisible by 4
    assert n_rays_half == n_rays / 2.0
    assert n_rays_quarter == n_rays / 4.0

    # Shoot rays at the initial start point
    direction = np.array(direction)
    prop_hit, hits = sample_radial_rays(
        tm=tm, point=start_point, normal=direction, n=n_rays
    )
    all_points = np.array([hit for hit in hits.values()])
    if not prop_hit >= hit_threshold:
        print(
            f"Expected at least {hit_threshold} raycast hits, got {prop_hit} instead."
        )
        # Plot the pruned results
        sources = np.stack([start_point] * len(hits), axis=0)
        rays = np.stack([sources, all_points], axis=1)
        trimesh_path = trimesh.load_path(rays)
        trimesh.Scene([tm, trimesh_path]).show()

    # Move point to centroid of all points of circle
    center = all_points.mean(axis=0)

    # Shoot ray in downwards direction, make sure we hit some surface
    bottom_hit = shoot_ray(tm=tm, point=center, direction=-direction)
    assert (
        bottom_hit is not None
    ), "Got no valid hit when trying to shoot ray towards opposite direction!"

    # Find top hit by shooting ray in positive @direction
    top_hit = shoot_ray(tm=tm, point=center, direction=direction)
    if top_hit is None:
        # Only when shooting up, we are OK hitting the convex hull. try that
        top_hit = shoot_ray(tm=tm.convex_hull, point=center, direction=direction)
    assert (
        top_hit is not None
    ), "Got no valid hit within convex hull when trying to shoot ray towards positive direction!"

    # Transform mesh to such that z points in @direction, and compute average diameter in this direction
    tf = np.eye(4)
    rot = R.from_quat(vecs2quat(np.array([0, 0, 1.0]), np.array(direction))).as_matrix()
    tf[:3, :3] = rot
    tm.apply_transform(tf)
    bbox_min_rot, bbox_max_rot = tm.bounding_box.bounds
    bbox_extent_rot = bbox_max_rot - bbox_min_rot
    avg_diameter = np.mean(bbox_extent_rot[:2])
    min_diameter = avg_diameter / 5.0
    offset = min(
        0.005, avg_diameter / 20.0
    )  # Offset to prevent direct collisions. Defaults to 5mm

    # Make sure to rotate trimesh mesh back
    tf_inv = np.eye(4)
    tf_inv[:3, :3] = rot.T
    tm.apply_transform(tf_inv)

    # Starting at the bottom, sample points radially and iteratively move in @direction
    mesh_points = np.array([]).reshape(0, 3)

    total_distance = np.linalg.norm(top_hit - bottom_hit) - (offset / 2.0)
    delta = 0.005
    delta = np.clip(
        delta, total_distance / 20, total_distance / 3
    )  # between 3 and 20 entries
    i = 0

    cur_distance = offset / 2.0

    while cur_distance <= total_distance:
        point = bottom_hit + cur_distance * direction
        prop_hit, hits = sample_radial_rays(
            tm=tm, point=point, normal=direction, n=n_rays
        )
        if prop_hit < hit_threshold:
            if i == 0:
                # Failed to sample at all, probably a degenerate mesh, so raise an error
                raise ValueError("Failed to sample any valid points!")
            # Terminate, this is assumed to be the end of a valid sampleable volume
            break

        # Get all hit positions transformed into rotated direction
        positions = np.array([hit for hit in hits.values()])
        positions_rot = positions @ rot.T
        # If the distance between points is less than a threshold, also break
        if len(mesh_points) > 0 and np.any(
            (positions_rot.max(axis=0)[:2] - positions_rot.min(axis=0)[:2])
            < min_diameter
        ):
            break

        # Compute points -- slightly offset them by @offset so they don't directly collide with edge
        layer_rays = positions - point.reshape(1, 3)
        layer_points = positions - offset * layer_rays / np.linalg.norm(layer_rays)
        mesh_points = np.concatenate([mesh_points, layer_points], axis=0)
        cur_distance += delta
        i += 1

    # Prune previous layer
    mesh_points = mesh_points[: -len(layer_points)]

    # Make sure we have nonzero set of points
    assert (
        len(mesh_points) > 0
    ), "Got no valid mesh points for generating fillable mesh!"

    if REQUIRE_ADDITIONAL_BASE_POINTS:
        # 5. Create fillable trimesh
        tm_fillable_tmp = trimesh.Trimesh(vertices=np.array(mesh_points))

        # Rotate it so that the z-axis points in @direction
        tm_fillable_tmp.apply_transform(tf)

        # 6. Create convex hull
        ctm_rot = tm_fillable_tmp.convex_hull
        ctm_rot_unmerged = ctm_rot.copy()
        ctm_rot_unmerged.unmerge_vertices()

        # 7. Take the projection of the convex hull with normal @direction (which, since already rotated, is simply
        # [0, 0, 1]), and then sample rays shooting against that normal to compensate for the original offset
        # We know the bounding box is [1, 1, 1], so sample points uniformly in XY plane
        proj = ConvexHull(ctm_rot_unmerged.vertices[:, :2].copy())
        equations = np.array(proj.equations, dtype=np.float32)

        n_dim_samples = 10
        ctm_rot_min, ctm_rot_max = ctm_rot_unmerged.bounding_box.bounds
        x_range = np.linspace(ctm_rot_min[0], ctm_rot_max[0], n_dim_samples)
        y_range = np.linspace(ctm_rot_min[1], ctm_rot_max[1], n_dim_samples)
        ray_grid = np.stack(np.meshgrid(x_range, y_range, indexing="ij"), axis=-1)
        ray_grid_flattened = ray_grid.reshape(-1, 2)

        # Check which rays are within the polygon
        # Each inequality is of the form Ax + By + C <= 0
        # We need to check if the point satisfies all inequalities
        is_within = np.all(
            (ray_grid_flattened @ equations[:, :-1].T) + equations[:, -1] <= 0, axis=1
        )
        xy_ray_positions = ray_grid_flattened[is_within]

        # Shoot these rays downwards and record their poses -- add them to the point set
        z_range = np.linspace(bbox_min_rot[2], bbox_max_rot[2], n_dim_samples)
        additional_points_rot = []
        for xy_ray_pos in xy_ray_positions:
            # Find a corresponding z value within the convex hull to use as the start raycasting point
            start_samples = np.zeros((n_dim_samples, 3))
            start_samples[:, :2] = xy_ray_pos
            start_samples[:, 2] = z_range
            is_contained = ctm_rot.contains(start_samples)
            if not np.any(is_contained):
                # Just skip this sample
                continue
            # Use the lowest point (i.e.: the first idx that is True) as the raycasting start point
            z = z_range[np.where(is_contained)[0][0]]

            # Raycast downwards and record the hit point
            start = np.array([*xy_ray_pos, z])
            hit = shoot_ray(
                tm=ctm_rot_unmerged, point=start, direction=np.array([0, 0, -1.0])
            )

            # If we have a valid hit, record this point
            if hit is not None:
                additional_points_rot.append(hit)

        assert (
            len(additional_points_rot) > 0
        ), "Failed to generate any additional points for the fillable mesh!"

        # Rotate the points back into the original frame
        # Note: forward transform is points @ rot.T so inverse is points @ rot
        additional_points = np.array(additional_points_rot) @ rot

        # Append all additional points to our existing set of points
        mesh_points = np.concatenate([mesh_points, additional_points], axis=0)

    # Re-write to trimesh and take the finalized convex hull
    tm_fillable = trimesh.Trimesh(vertices=np.array(mesh_points))

    # Return the convex hull of this mesh
    return reduce_mesh(tm_fillable.convex_hull)


def generate_fillable_mesh_from_seed(is_open):
    # Assert that the currently selected object is a seed point
    assert (
        len(rt.selection) == 1
    ), "Expected exactly one object (seed point) to be selected"
    (seed,) = rt.selection
    assert (
        seed.name == "fillable_seed"
    ), "Expected the selected object to be the seed point"
    assert (
        rt.classOf(seed) == rt.Point
    ), "Expected the selected object to be the seed point"

    # Some checks about the target object
    target_obj = seed.parent
    assert target_obj is not None, "Expected the seed point to have a parent object"
    parsed_name = parse_name(target_obj.name)
    assert parsed_name, "Failed to parse the name of the target object"
    assert not parsed_name.group("bad"), "Expected the target object to not be bad"
    assert (
        parsed_name.group("instance_id") == "0"
    ), "Expected the target object to be the 0 instance"
    assert not parsed_name.group(
        "meta_type"
    ), "Expected the target object to not have a meta type"
    assert (
        parsed_name.group("joint_side") != "upper"
    ), "Expected the target object to not be an upper joint"
    target_link_name = parsed_name.group("link_name") or "base_link"

    # Get the position and orientation info
    start_point = np.array(seed.position)
    up_dir = np.array(rt.Point3(0, 0, 1.0) * seed.rotation)

    # Browse the entire scene to find the articulation tree of the target object
    articulation_tree = nx.DiGraph()
    link_objs = {}
    for obj in rt.objects:
        candidate_parsed_name = parse_name(obj.name)
        if not candidate_parsed_name:
            continue
        if candidate_parsed_name.group("model_id") != parsed_name.group("model_id"):
            continue
        if candidate_parsed_name.group("instance_id") != parsed_name.group(
            "instance_id"
        ):
            continue
        if candidate_parsed_name.group("bad"):
            continue
        if candidate_parsed_name.group("meta_type"):
            continue
        if candidate_parsed_name.group("joint_side") == "upper":
            continue
        link_name = candidate_parsed_name.group("link_name") or "base_link"

        # Assert that it has a collision mesh
        collision_mesh_objects = []
        for child in obj.children:
            if rt.classOf(child) == rt.Editable_Poly and "Mcollision" in child.name:
                collision_mesh_objects.append(child)
        assert (
            len(collision_mesh_objects) == 1
        ), f"Expected {obj.name} to have exactly one collision mesh"
        collision_mesh_object = collision_mesh_objects[0]

        # Add it into the graph
        articulation_tree.add_node(link_name)
        link_objs[link_name] = collision_mesh_object
        if link_name != "base_link":
            parent_link_name = candidate_parsed_name.group("parent_link_name")
            articulation_tree.add_edge(parent_link_name, link_name)

    # Get all the descendants of the target link. These will be used when computing the fillable volume
    links_to_include = set(nx.descendants(articulation_tree, target_link_name)) | {
        target_link_name
    }
    collision_mesh_objects_to_include = [
        link_objs[link_name] for link_name in links_to_include
    ]

    # Convert to trimesh
    tm = trimesh.util.concatenate(
        [convert_to_trimesh(col_obj) for col_obj in collision_mesh_objects_to_include]
    )

    # Scale by 1/1000 to match canonical scale requirements
    scale_mat = np.eye(4)
    scale_mat[:3, :3] *= 1e-3
    tm.apply_transform(scale_mat)

    if is_open:
        tm = trimesh.util.concatenate([tm, tm.convex_hull])

    tm.export(r"D:\tmp\fillable.obj")

    fillable_hull = sample_fillable_volume(
        tm=tm,
        start_point=start_point / 1000.0,  # Scale to meters
        direction=up_dir,
        hit_threshold=0.75,
        n_rays=100,
    )

    # Scale back to original scale
    scale_mat_inv = np.eye(4)
    scale_mat_inv[:3, :3] *= 1e3
    fillable_hull.apply_transform(scale_mat_inv)

    # Check if the appropriate kind of fillable obj exists already, if not, create it
    fillable_kind = "Mopenfillable" if is_open else "Mfillable"
    fillable_obj = None
    for candidate in target_obj.children:
        if (
            rt.classOf(candidate) == rt.Editable_Poly
            and fillable_kind in candidate.name
        ):
            fillable_obj = candidate
            break

    if not fillable_obj:
        fillable_obj = rt.Editable_Mesh()
        rt.ConvertToPoly(fillable_obj)
        fillable_obj.name = f"{parsed_name.group('mesh_basename')}-{fillable_kind}"
        fillable_obj.rotation = target_obj.rotation
        fillable_obj.position = target_obj.position
        fillable_obj.wirecolor = rt.yellow
        fillable_obj.parent = target_obj

    # Add the vertices
    vertex_idxes = []  # map from trimesh vertex idxes to max vertex idxes
    for v in fillable_hull.vertices:
        vertex_idxes.append(rt.polyop.createVert(fillable_obj, rt.Point3(*v.tolist())))

    # Add the faces
    for f in fillable_hull.faces:
        face = [vertex_idxes[v] for v in f]
        rt.polyop.createPolygon(fillable_obj, face)

    # Update the mesh to reflect changes
    rt.update(fillable_obj)

    return fillable_hull


if __name__ == "__main__":
    generate_fillable_mesh_from_seed(is_open=False)
