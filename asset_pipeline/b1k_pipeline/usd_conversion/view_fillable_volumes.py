import hashlib
import os
import random
import sys
import glob
import pathlib
import numpy as np
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.macros import gm
from omnigibson.prims import XFormPrim
from omnigibson.utils.ui_utils import KeyboardEventHandler
from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
import trimesh
import json
import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import torch as th
import networkx as nx
from collections import defaultdict
from fs.zipfs import ZipFS
import fs.path

gm.HEADLESS = False
gm.USE_ENCRYPTED_ASSETS = True
gm.ENABLE_FLATCACHE = False
gm.DATASET_PATH = r"D:\fillable-10-21"

ASSIGNMENT_FILE = os.path.join(gm.DATASET_PATH, "fillable_assignments_2.json")

MAX_BBOX = 0.3

DRAWING_MESHES = []


def get_assignments():
    if not os.path.exists(ASSIGNMENT_FILE):
        return {}

    with open(ASSIGNMENT_FILE, "r") as f:
        return json.load(f)


def add_assignment(mdl, assignment):
    assignments = get_assignments()
    assignments[mdl] = assignment
    with open(ASSIGNMENT_FILE, "w") as f:
        json.dump(assignments, f)


def _draw_meshes():
    draw = lazy.omni.isaac.debug_draw._debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()
    for item in DRAWING_MESHES:
        if item is None:
            continue
        mesh, parent_pos, color, size, visible = item
        if not visible:
            continue
        edge_vert_idxes = mesh.edges_unique
        N = len(edge_vert_idxes)
        colors = [color for _ in range(N)]
        sizes = [1.0 for _ in range(N)]
        points1 = [
            tuple(x)
            for x in (
                mesh.vertices[edge_vert_idxes[:, 0]] + parent_pos.numpy().copy()
            ).tolist()
        ]
        points2 = [
            tuple(x)
            for x in (
                mesh.vertices[edge_vert_idxes[:, 1]] + parent_pos.numpy().copy()
            ).tolist()
        ]
        draw.draw_lines(points1, points2, colors, sizes)


def draw_mesh(mesh, parent_pos, color=(1.0, 0.0, 0.0, 1.0), size=1.0):
    DRAWING_MESHES.append((mesh, parent_pos, color, size, True))
    _draw_meshes()
    return len(DRAWING_MESHES) - 1


def toggle_draw_visibility(idx):
    mesh, parent_pos, color, size, visible = DRAWING_MESHES[idx]
    DRAWING_MESHES[idx] = (mesh, parent_pos, color, size, not visible)
    _draw_meshes()


def erase_mesh(idx):
    DRAWING_MESHES[idx] = None
    _draw_meshes()


def clear_meshes():
    DRAWING_MESHES.clear()
    _draw_meshes()


def anorm(x, axis=None, keepdims=False):
    """Compute L2 norms alogn specified axes."""
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    norm = anorm(v, axis=axis, keepdims=True)
    return v / np.where(norm < eps, eps, norm)


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
    scale=(1.0, 1.0, 1.0),
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
        scale (3-array): (x,y,z) scale of the object. Default is (1, 1, 1)

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

    # 5. Create fillable trimesh
    tm_fillable_tmp = trimesh.Trimesh(vertices=np.array(mesh_points))

    # Rotate it so that the z-axis points in @direction
    tm_fillable_tmp.apply_transform(tf)

    # 6. Create convex hull
    ctm_rot = tm_fillable_tmp.convex_hull
    ctm_rot.unmerge_vertices()

    # 7. Take the projection of the convex hull with normal @direction (which, since already rotated, is simply
    # [0, 0, 1]), and then sample rays shooting against that normal to compensate for the original offset
    # We know the bounding box is [1, 1, 1], so sample points uniformly in XY plane
    proj = ConvexHull(ctm_rot.vertices[:, :2].copy())
    equations = np.array(proj.equations, dtype=np.float32)

    n_dim_samples = 10
    ctm_rot_min, ctm_rot_max = ctm_rot.bounding_box.bounds
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
        hit = shoot_ray(tm=ctm_rot, point=start, direction=np.array([0, 0, -1.0]))

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

    # Denormalize the mesh points based on the objects' scale
    scale = 1.0 / np.array(scale)
    mesh_points = mesh_points * scale.reshape(1, 3)

    # Re-write to trimesh and take the finalized convex hull
    tm_fillable = trimesh.Trimesh(vertices=np.array(mesh_points))

    # Return the convex hull of this mesh
    return tm_fillable.convex_hull


def generate_fillable_mesh_for_object(
    obj, selected_link_name, start_point, up_dir, allow_convex_hull_hit
):
    # Walk through the link hierarchy to select the stuff below this one
    links_to_include = set(
        nx.descendants(obj.articulation_tree, selected_link_name)
    ) | {selected_link_name}

    tm = trimesh.util.concatenate(
        [
            mesh_prim_to_trimesh_mesh(cm.prim, world_frame=True)
            for link_name in links_to_include
            for cm in obj.links[link_name].collision_meshes.values()
        ]
    )
    if allow_convex_hull_hit:
        tm = trimesh.util.concatenate([tm, tm.convex_hull])

    try:
        fillable_hull = sample_fillable_volume(
            tm=tm,
            start_point=start_point,
            direction=up_dir,
            hit_threshold=0.75,
            n_rays=100,
            scale=obj.scale.cpu().numpy(),
        )
    except Exception as e:
        raise
        # print(f"Failed to generate fillable volume for {obj.category}/{obj.model} at {start_point}: {e}")
        # return None

    # Move the fillable hull to the object's position
    # note that we don't include the orientation here because it is assumed to be identity (e.g. even
    # if the object has an orientation, that's just a correction that should be ignored)
    obj_pos = obj.get_position_orientation()[0].numpy()
    obj_transform = np.eye(4)
    obj_transform[:3, 3] = obj_pos
    fillable_hull.apply_transform(np.linalg.inv(obj_transform))

    return fillable_hull


def view_object(cat, mdl):
    if og.sim:
        og.clear()
    else:
        og.launch()

    clear_meshes()

    if og.sim.is_playing():
        og.sim.stop()

    orn = [0, 0, 0, 1]

    cfg = {
        "scene": {
            "type": "Scene",
            "use_floor_plane": False,
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "fillable",
                "category": cat,
                "model": mdl,
                "orientation": orn,
                "kinematic_only": False,
                "fixed_base": True,
            },
        ],
    }

    env = og.Environment(configs=cfg)
    og.sim.step()

    fillable = env.scene.object_registry("name", "fillable")
    fillable.set_position([0, 0, fillable.aabb_extent[2]])
    og.sim.step()

    # Reset keyboard bindings
    KeyboardEventHandler.KEYBOARD_CALLBACKS = {}

    print("\n\nNow processing:", cat, mdl)

    # Create the stopper function
    keep_rendering = True

    def stop_rendering():
        nonlocal keep_rendering
        keep_rendering = False

    def save_assignment_and_stop(assignment, meshes=None):
        # meshes should look like {link_name: [(mesh, enclosed)]}
        print(f"Chose option {assignment} for {cat}/{mdl}")
        add_assignment(mdl, assignment)
        if meshes:
            for link_name, meshlist in meshes.items():
                for i, (mesh, enclosed) in enumerate(meshlist):
                    enclosed = "enclosed" if enclosed else "open"
                    mesh.export(
                        f"{gm.DATASET_PATH}/objects/{cat}/{mdl}/fillable---{link_name}---{i}---{enclosed}.obj"
                    )
        stop_rendering()

    # Skip without any assignment
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.J,
        callback_fn=stop_rendering,
    )
    print("Press J to skip")

    # Skip with assignment that says we should remove the fillable annotation from the object
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.U,
        callback_fn=lambda: save_assignment_and_stop("notfillable"),
    )
    print(
        "Press U to indicate we should remove the fillable annotation from the object."
    )

    dip_path = (
        pathlib.Path(gm.DATASET_PATH) / "objects" / cat / mdl / "fillable_dip.obj"
    )
    if dip_path.exists():
        # Find the scale the mesh was generated at
        scale = np.minimum(1, MAX_BBOX / np.max(np.asarray(fillable.native_bbox)))

        dip_mesh = trimesh.load(dip_path, force="mesh")
        inv_scale = 1 / scale
        transform = np.diag([inv_scale, inv_scale, inv_scale, 1])
        dip_mesh.apply_transform(transform)

        # Draw the mesh
        dip_viz_idx = draw_mesh(
            dip_mesh, fillable.get_position_orientation()[0], color=(1.0, 0.0, 0.0, 1.0)
        )

        # Add the dip option chooser
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.X,
            callback_fn=lambda: save_assignment_and_stop(
                "dip", meshes={fillable.root_link_name: [(dip_mesh, True)]}
            ),
        )
        print("Press X to choose the dip (red) option.")

        # Add the dip visibility toggler
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.B,
            callback_fn=lambda: toggle_draw_visibility(dip_viz_idx),
        )
        print("Press B to toggle visibility of the dip (red) mesh.")

    ray_path = (
        pathlib.Path(gm.DATASET_PATH) / "objects" / cat / mdl / "fillable_ray.obj"
    )
    if ray_path.exists():
        ray_mesh = trimesh.load(ray_path, force="mesh")

        # Draw the mesh
        ray_viz_idx = draw_mesh(
            ray_mesh, fillable.get_position_orientation()[0], color=(0.0, 0.0, 1.0, 1.0)
        )

        # Add the ray option chooser
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.S,
            callback_fn=lambda: save_assignment_and_stop(
                "ray", meshes={fillable.root_link_name: [(ray_mesh, True)]}
            ),
        )
        print("Press S to choose the ray (blue) option.")

        # Add the ray visibility toggler
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.N,
            callback_fn=lambda: toggle_draw_visibility(ray_viz_idx),
        )
        print("Press N to toggle visibility of the ray (blue) mesh.")

    # Now the combined version
    if dip_path.exists() and ray_path.exists():
        # Check if either mesh contains the entire other mesh
        combined_mesh = trimesh.convex.convex_hull(
            np.concatenate([dip_mesh.vertices, ray_mesh.vertices], axis=0)
        )
        if not np.allclose(
            combined_mesh.volume, dip_mesh.volume, rtol=1e-3
        ) and not np.allclose(combined_mesh.volume, ray_mesh.volume, rtol=1e-3):
            # Draw the mesh
            combined_viz_idx = draw_mesh(
                combined_mesh,
                fillable.get_position_orientation()[0],
                color=(1.0, 0.0, 1.0, 1.0),
                size=0.5,
            )

            # Add the combined option chooser
            KeyboardEventHandler.add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.W,
                callback_fn=lambda: save_assignment_and_stop(
                    "combined",
                    meshes={fillable.root_link_name: [(combined_mesh, True)]},
                ),
            )
            print("Press W to choose the combined (purple) option.")

            # Add the combined visibility toggler
            KeyboardEventHandler.add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.M,
                callback_fn=lambda: toggle_draw_visibility(combined_viz_idx),
            )
            print("Press M to toggle visibility of the combined (purple) mesh.")

    # Add the features for the GENERATE NOW option next
    selected_link = fillable.root_link_name

    # Add a seed prim
    seed_prim = XFormPrim(name="seed", relative_prim_path="/seed")
    seed_prim.load(env.scene)

    def _move_seed():
        descendants = nx.descendants(fillable.articulation_tree, selected_link) | {
            selected_link
        }
        aabbs = th.cat(
            [th.stack(list(fillable.links[link].aabb), dim=0) for link in descendants],
            dim=0,
        )
        aabb_low = th.min(aabbs, dim=0).values
        aabb_high = th.max(aabbs, dim=0).values
        aabb_extent = aabb_high - aabb_low
        aabb_center = (aabb_high + aabb_low) / 2
        start_point = aabb_center + th.as_tensor([0, 0, aabb_extent[2] * 0.125])
        seed_prim.set_position_orientation(start_point, th.as_tensor([0, 0, 0, 1]))

    _move_seed()

    # Store the original hiddenness of prims
    hide_visuals = True
    hide_collision = False
    original_hiddenness = {
        geom: "default" if geom.visible else "guide"
        for link in fillable.links.values()
        for geom in link.visual_meshes.values()
    }

    def _update_visibility():
        # Hide links that are not descendants of the selected link
        descendants = nx.descendants(fillable.articulation_tree, selected_link) | {
            selected_link
        }
        for link_name, link in fillable.links.items():
            for geom in link.visual_meshes.values():
                geom.purpose = (
                    "guide"
                    if hide_visuals or link_name not in descendants
                    else original_hiddenness[geom]
                )
            for geom in link.collision_meshes.values():
                geom.purpose = (
                    "guide"
                    if hide_collision or link_name not in descendants
                    else "default"
                )

    _update_visibility()

    def _toggle_visuals_visibility():
        nonlocal hide_visuals
        hide_visuals = not hide_visuals
        _update_visibility()

    # Add the ability to hide or show visuals
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.V,
        callback_fn=_toggle_visuals_visibility,
    )
    print("Press V to toggle visibility of the object.")

    def _toggle_collision_visibility():
        nonlocal hide_collision
        hide_collision = not hide_collision
        _update_visibility()

    # Add the ability to hide or show collision meshes
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.C,
        callback_fn=_toggle_collision_visibility,
    )
    print("Press C to toggle visibility of the collision meshes.")

    def _increment_selection(inc):
        nonlocal selected_link
        available_links = [
            k for k, v in fillable.links.items() if v.collision_meshes
        ]  # only pick links with meshes
        selected_idx = available_links.index(selected_link)
        new_selected_idx = (selected_idx + inc) % len(available_links)
        selected_link = available_links[new_selected_idx]

        _update_visibility()
        _move_seed()
        print(f"Selected link: {selected_link}")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.A,
        callback_fn=lambda: _increment_selection(1),
    )
    print("Press A to select the next link.")
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Q,
        callback_fn=lambda: _increment_selection(-1),
    )
    print("Press Q to select the previous link.")

    generated_meshes = defaultdict(list)

    def _generate_mesh(allow_convex_hull_hit):
        # First check that no parent or child already has a generated mesh
        ancestors = nx.ancestors(fillable.articulation_tree, selected_link)
        descendants = nx.descendants(fillable.articulation_tree, selected_link)
        if any(ancestors & set(generated_meshes)) or any(
            descendants & set(generated_meshes)
        ):
            print(
                f"Cannot generate mesh for {selected_link} because a parent or child already has a generated mesh."
            )
            return

        seed_pos, seed_orn = seed_prim.get_position_orientation()
        seed_mat = T.quat2mat(seed_orn).numpy()
        up_dir = seed_mat @ np.array([0, 0, 1.0])
        generated_mesh = generate_fillable_mesh_for_object(
            fillable,
            selected_link,
            seed_pos.numpy().copy(),
            up_dir,
            allow_convex_hull_hit,
        )
        if generated_mesh is None:
            return
        draw_idx = draw_mesh(
            generated_mesh,
            fillable.get_position_orientation()[0],
            color=(0.0, 1.0, 0.0, 1.0),
        )
        generated_meshes[selected_link].append(
            (generated_mesh, draw_idx, allow_convex_hull_hit)
        )
        print(
            f"Generated mesh {len(generated_meshes[selected_link])} for selected link {selected_link}"
        )

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.K,
        callback_fn=lambda: _generate_mesh(allow_convex_hull_hit=False),
    )
    print("Press K to generate an ENCLOSED fillable mesh from the current seed point.")
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.L,
        callback_fn=lambda: _generate_mesh(allow_convex_hull_hit=True),
    )
    print(
        "Press L to generate an OPEN fillable mesh from the current seed point, allowing convex hull hits."
    )
    print(
        "    Note that this marks the fillable volume as unenclosed, meaning it provides fewer guarantees with sampling."
    )
    print("    Use this option if the volume is not fully enclosed, e.g. a shelf.")

    def _remove_last_generated():
        if len(generated_meshes[selected_link]) == 0:
            return
        generated_mesh, draw_idx, allow_convex_hull_hit = generated_meshes[
            selected_link
        ].pop()
        erase_mesh(draw_idx)
        print(f"Removed last generated mesh")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.O,
        callback_fn=_remove_last_generated,
    )
    print("Press O to remove the last generated mesh.")

    def _clear_generated_meshes():
        for link, meshes in generated_meshes.items():
            while meshes:
                mesh, idx, allow_convex_hull_hit = meshes.pop()
                erase_mesh(idx)
        print(f"Cleared all generated meshes.")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.P,
        callback_fn=_clear_generated_meshes,
    )
    print("Press P to clear all generated meshes.")

    # Add the generated option chooser
    def _pick_generated():
        if sum(len(meshes) for meshes in generated_meshes.values()) == 0:
            print("You currently do NOT have any generated meshes.")
            return
        generated_concat = {
            # enclosed = not allow_convex_hull_hit
            link: [
                (mesh, not allow_convex_hull_hit)
                for mesh, idx, allow_convex_hull_hit in meshes
            ]
            for link, meshes in generated_meshes.items()
            if len(meshes) > 0
        }
        save_assignment_and_stop("generated", meshes=generated_concat)

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Z,
        callback_fn=_pick_generated,
    )
    print("Press Z to pick the generated option.")
    print()

    while keep_rendering:
        og.sim.step()

    clear_meshes()


def main():
    idx = int(sys.argv[1])
    idxes = int(sys.argv[2])
    salt = sys.argv[3]

    print("Fillable annotator version 11.6.0")

    # Get all the models that are fillable-annotated
    from bddl.knowledge_base import Object

    fillables = sorted(
        o.name.split("-")
        for o in Object.all_objects()
        if any(p.name == "fillable" for p in o.category.synset.properties)
    )
    assert (
        len(fillables) == 1031
    ), f"Expected 1031 fillable objects, got {len(fillables)}. Uninstall BDDL and reinstall the develop branch."

    # Get the ones that don't have a fillable assignment
    assignments = get_assignments()
    fillables = [(cat, mdl) for cat, mdl in fillables if mdl not in assignments]

    # Get the ones whose model hash match our ID
    fillables = [
        (cat, mdl)
        for cat, mdl in fillables
        if int(hashlib.md5((mdl + salt).encode()).hexdigest(), 16) % idxes == idx
    ]

    for cat, mdl in tqdm.tqdm(fillables):
        if not os.path.exists(
            DatasetObject.get_usd_path(cat, mdl).replace(".usd", ".encrypted.usd")
        ):
            print(f"Skipping {cat}/{mdl} because it does not exist")
            continue
        view_object(cat, mdl)


if __name__ == "__main__":
    main()
