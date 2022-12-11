import itertools
from collections import Counter, defaultdict

import numpy as np

import time
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.stats import truncnorm

from omni.physx import get_physx_scene_query_interface

import omnigibson as og
from omnigibson.macros import create_module_macros
import omnigibson.utils.transform_utils as T


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_AABB_OFFSET = 0.01
m.DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE = 1.0  # Around 60 degrees
m.DEFAULT_HIT_TO_PLANE_THRESHOLD = 0.05
m.DEFAULT_MAX_ANGLE_WITH_Z_AXIS = 3 * np.pi / 4
m.DEFAULT_MAX_SAMPLING_ATTEMPTS = 10
m.DEFAULT_CUBOID_BOTTOM_PADDING = 0.005
# We will cast an additional parallel ray for each additional this much distance.
m.DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE = 0.1
m.DEFAULT_HIT_PROPORTION = 0.6


def fit_plane(points):
    """
    Fits a plane to the given 3D points.
    Copied from https://stackoverflow.com/a/18968498

    :param points: np.array of shape (k, 3)
    :return Tuple[np.array, np.array] where first element is the points' centroid and the second is the normal of the fitted plane
    """
    assert points.shape[1] <= points.shape[0], "Cannot fit plane with only {} points in {} dimensions.".format(
        points.shape[0], points.shape[1]
    )
    ctr = points.mean(axis=0)
    x = points - ctr
    normal = np.linalg.svd(np.dot(x.T, x))[0][:, -1]
    normal /= np.linalg.norm(normal)
    return ctr, normal


def check_distance_to_plane(points, plane_centroid, plane_normal, hit_to_plane_threshold, refusal_log):
    distances = get_distance_to_plane(points, plane_centroid, plane_normal)
    if np.any(distances > hit_to_plane_threshold):
        if og.debug_sampling:
            refusal_log.append("distances to plane: %r" % distances)
        return False
    return True


def get_distance_to_plane(points, plane_centroid, plane_normal):
    return np.abs(np.dot(points - plane_centroid, plane_normal))


def get_projection_onto_plane(points, plane_centroid, plane_normal):
    distances_to_plane = get_distance_to_plane(points, plane_centroid, plane_normal)
    return points - np.outer(distances_to_plane, plane_normal)


def draw_debug_markers(hit_positions, radius=0.01):
    # Import here to avoid circular imports
    from omnigibson.objects.primitive_object import PrimitiveObject

    color = np.concatenate([np.random.rand(3), [1]])
    for vec in hit_positions:
        time_str = str(time.time())
        cur_time = time_str[(time_str.index(".") + 1):]
        obj = PrimitiveObject(
            prim_path=f"/World/debug_marker_{cur_time}",
            name=f"debug_marker_{cur_time}",
            primitive_type="Sphere",
            visual_only=True,
            rgba=color,
            radius=radius,
        )
        og.sim.import_object(obj)
        obj.set_position(vec)


def get_parallel_rays(
    source, destination, offset, new_ray_per_horizontal_distance
):
    """Given an input ray described by a source and a destination, sample parallel rays around it as the center.

    The parallel rays start at the corners of a square of edge length `offset` centered on `source`, with the square
    orthogonal to the ray direction. That is, the cast rays are the height edges of a square-base cuboid with bases
    centered on `source` and `destination`.

    :param source: Source of the ray to sample parallel rays of.
    :param destination: Source of the ray to sample parallel rays of.
    :param offset: Orthogonal distance of parallel rays from input ray.
    :param new_ray_per_horizontal_distance: Step in offset beyond which an additional split will be applied in the
        parallel ray grid (which at minimum is 3x3 at the AABB corners & center).
    :return Tuple[List, List, Array[W, H, 3]] containing sources and destinations of original ray and the unflattened,
        untransformed grid in object coordinates.
    """
    ray_direction = destination - source

    # Get an orthogonal vector using a random vector.
    random_vector = np.random.rand(3)
    orthogonal_vector_1 = np.cross(ray_direction, random_vector)
    orthogonal_vector_1 /= np.linalg.norm(orthogonal_vector_1)

    # Get a second vector orthogonal to both the ray and the first vector.
    orthogonal_vector_2 = -np.cross(ray_direction, orthogonal_vector_1)
    orthogonal_vector_2 /= np.linalg.norm(orthogonal_vector_2)

    orthogonal_vectors = np.array([orthogonal_vector_1, orthogonal_vector_2])
    assert np.all(np.isfinite(orthogonal_vectors))

    # Convert the offset into a 2-vector if it already isn't one.
    offset = np.array([1, 1]) * offset

    # Compute the grid of rays
    steps = (offset / new_ray_per_horizontal_distance).astype(int) * 2 + 1
    steps = np.maximum(steps, 3)
    x_range = np.linspace(-offset[0], offset[0], steps[0])
    y_range = np.linspace(-offset[1], offset[1], steps[1])
    ray_grid = np.dstack(np.meshgrid(x_range, y_range, indexing="ij"))
    ray_grid_flattened = ray_grid.reshape(-1, 2)

    # Apply the grid onto the orthogonal vectors to obtain the rays in the world frame.
    sources = [source + np.dot(offsets, orthogonal_vectors) for offsets in ray_grid_flattened]
    destinations = [destination + np.dot(offsets, orthogonal_vectors) for offsets in ray_grid_flattened]

    return sources, destinations, ray_grid


def sample_origin_positions(mins, maxes, count, bimodal_mean_fraction, bimodal_stdev_fraction, axis_probabilities):
    """
    Sample ray casting origin positions with a given distribution.

    The way the sampling works is that for each particle, it will sample two coordinates uniformly and one
    using a symmetric, bimodal truncated normal distribution. This way, the particles will mostly be close to the faces
    of the AABB (given a correctly parameterized bimodal truncated normal) and will be spread across each face,
    but there will still be a small number of particles spawned inside the object if it has an interior.

    :param mins: Array of shape (3, ), the minimum coordinate along each axis.
    :param maxes: Array of shape (3, ), the maximum coordinate along each axis.
    :param count: int, Number of origins to sample.
    :param bimodal_mean_fraction: float, the mean of one side of the symmetric bimodal distribution as a fraction of the
        min-max range.
    :param bimodal_stdev_fraction: float, the standard deviation of one side of the symmetric bimodal distribution as a
        fraction of the min-max range.
    :param axis_probabilities: Array of shape (3, ), the probability of ray casting along each axis.
    :return: List of (ray cast axis index, bool whether the axis was sampled from the top side, [x, y, z]) tuples.
    """
    assert len(mins.shape) == 1
    assert mins.shape == maxes.shape

    results = []
    for i in range(count):
        # Get the uniform sample first.
        position = np.random.rand(3)

        # Sample the bimodal normal.
        bottom = (0 - bimodal_mean_fraction) / bimodal_stdev_fraction
        top = (1 - bimodal_mean_fraction) / bimodal_stdev_fraction
        bimodal_sample = truncnorm.rvs(bottom, top, loc=bimodal_mean_fraction, scale=bimodal_stdev_fraction)

        # Pick which axis the bimodal normal sample should go to.
        bimodal_axis = np.random.choice([0, 1, 2], p=axis_probabilities)

        # Choose which side of the axis to sample from. We only sample from the top for the Z axis.
        if bimodal_axis == 2:
            bimodal_axis_top_side = True
        else:
            bimodal_axis_top_side = np.random.choice([True, False])

        # Move sample based on chosen side.
        position[bimodal_axis] = bimodal_sample if bimodal_axis_top_side else 1 - bimodal_sample

        # Scale the position from the standard normal range to the min-max range.
        scaled_position = mins + (maxes - mins) * position

        # Save the result.
        results.append((bimodal_axis, bimodal_axis_top_side, scaled_position))

    return results


def raytest_batch(start_points, end_points, closest=True, ignore_bodies=None, ignore_collisions=None):
    """
    Computes raytest collisions for a set of rays cast from @start_points to @end_points.

    Args:
        start_points (list of 3-array): Array of start locations to cast rays, where each is (x,y,z) global
            start location of the ray
        end_points (list of 3-array): Array of end locations to cast rays, where each is (x,y,z) global
            end location of the ray
        closest (bool): Whether we report the first (closest) hit from the ray or grab all hits
        ignore_bodies (None or list of str): If specified, specifies absolute USD paths to rigid bodies
            whose collisions should be ignored
        ignore_collisions (None or list of str): If specified, specifies absolute USD paths to collision geoms
            whose collisions should be ignored

    Returns:
        list of dict or list of list of dict: Results for all rays, where each entry corresponds to the result for the
            ith ray cast. If @closest=True, each entry in the list is the closest hit. Otherwise, each entry is its own
            (unordered) list of hits for that ray. Each dict is composed of:

            "hit" (bool): Whether an object was hit or not
            "position" (3-array): Location of the hit position
            "normal" (3-array): normal vector of the face hit
            "distance" (float): distance from @start_point the hit occurred
            "collision" (str): absolute USD path to the collision body hit
            "rigidBody" (str): absolute USD path to the associated rigid body hit

            Note that only "hit" = False exists in the dict if no hit was found
    """
    # For now, we do a naive for loop over individual raytests until a better API comes out
    results = []
    for start_point, end_point in zip(start_points, end_points):
        results.append(raytest(
            start_point=start_point,
            end_point=end_point,
            closest=closest,
            ignore_bodies=ignore_bodies,
            ignore_collisions=ignore_collisions,
        ))

    return results


def raytest(
    start_point,
    end_point,
    closest=True,
    ignore_bodies=None,
    ignore_collisions=None,
):
    """
    Computes raytest collision for ray cast from @start_point to @end_point

    Args:
        start_point (3-array): (x,y,z) global start location of the ray
        end_point (3-array): (x,y,z) global end location of the ray
        closest (bool): Whether we report the first (closest) hit from the ray or grab all hits
        ignore_bodies (None or list of str): If specified, specifies absolute USD paths to rigid bodies
            whose collisions should be ignored
        ignore_collisions (None or list of str): If specified, specifies absolute USD paths to collision geoms
            whose collisions should be ignored

    Returns:
        dict or list of dict: Results for this raytest. If @closest=True, then we only return the information from the
            closest hit. Otherwise, we return an (unordered) list of information for all hits encountered.
            Each dict is composed of:

            "hit" (bool): Whether an object was hit or not
            "position" (3-array): Location of the hit position
            "normal" (3-array): normal vector of the face hit
            "distance" (float): distance from @start_point the hit occurred
            "collision" (str): absolute USD path to the collision body hit
            "rigidBody" (str): absolute USD path to the associated rigid body hit

            Note that only "hit" = False exists in the dict if no hit was found
    """
    # Make sure start point, end point are numpy arrays
    start_point, end_point = np.array(start_point), np.array(end_point)
    point_diff = end_point - start_point
    distance = np.linalg.norm(point_diff)
    direction = point_diff / distance

    # For efficiency's sake, we handle special case of no ignore_bodies, ignore_collisions, and closest_hit
    if closest and ignore_bodies is None and ignore_collisions is None:
        return get_physx_scene_query_interface().raycast_closest(
            origin=start_point,
            dir=direction,
            distance=distance,
        )
    else:
        # Compose callback function for finding raycasts
        hits = []
        ignore_bodies = set() if ignore_bodies is None else set(ignore_bodies)
        ignore_collisions = set() if ignore_collisions is None else set(ignore_collisions)

        def callback(hit):
            # Only add to hits if we're not ignoring this body or collision
            if hit.rigid_body not in ignore_bodies and hit.collision not in ignore_collisions:
                hits.append({
                    "hit": True,
                    "position": np.array(hit.position),
                    "normal": np.array(hit.normal),
                    "distance": hit.distance,
                    "collision": hit.collision,
                    "rigidBody": hit.rigid_body,
                })
            # We always want to continue traversing to collect all hits
            return True

        # Grab all collisions
        get_physx_scene_query_interface().raycast_all(
            origin=start_point,
            dir=direction,
            distance=distance,
            reportFn=callback,
        )

        # If we only want the closest, we need to sort these hits, otherwise we return them all
        if closest:
            # Return the empty hit dictionary if our ray did not hit anything, otherwise we return the closest
            return {"hit": False} if len(hits) == 0 else sorted(hits, key=lambda hit: hit["distance"])[0]
        else:
            # Return all hits (list)
            return hits


def sample_raytest_start_end_symmetric_bimodal_distribution(
    obj,
    num_samples,
    bimodal_mean_fraction,
    bimodal_stdev_fraction,
    axis_probabilities,
    aabb_offset=m.DEFAULT_AABB_OFFSET,
    max_sampling_attempts=m.DEFAULT_MAX_SAMPLING_ATTEMPTS,
):
    """
    Sample the start points and end points around a given object by a symmetric bimodal distribution

    :param obj: The object to sample points on.
    :param num_samples: int, the number of points to try to sample.
    :param bimodal_mean_fraction: float, the mean of one side of the symmetric bimodal distribution as a fraction of the
        min-max range.
    :param bimodal_stdev_fraction: float, the standard deviation of one side of the symmetric bimodal distribution as a
        fraction of the min-max range.
    :param axis_probabilities: Array of shape (3, ), the probability of ray casting along each axis.
    :param aabb_offset: float or numpy array, padding for AABB to initiate ray-testing.
    :param max_sampling_attempts: int, how many times sampling will be attempted for each requested point.

    Returns:
        start_points: the start points for raycasting, defined in the world frame,
            numpy array of shape [num_samples, max_sampling_attempts, 3]
        end_points: the end points for raycasting, defined in the world frame,
            numpy array of shape [num_samples, max_sampling_attempts, 3]
    """
    bbox_center, bbox_orn, bbox_bf_extent, _ = obj.get_base_aligned_bbox(xy_aligned=True, fallback_to_aabb=True)
    half_extent_with_offset = (bbox_bf_extent / 2) + aabb_offset

    start_points = np.zeros((num_samples, max_sampling_attempts, 3))
    end_points = np.zeros((num_samples, max_sampling_attempts, 3))
    for i in range(num_samples):
        # Sample the starting positions in advance.
        # TODO: Narrow down the sampling domain so that we don't sample scenarios where the center is in-domain but the
        # full extent isn't. Currently a lot of samples are being wasted because of this.
        samples = sample_origin_positions(
            -half_extent_with_offset,
            half_extent_with_offset,
            max_sampling_attempts,
            bimodal_mean_fraction,
            bimodal_stdev_fraction,
            axis_probabilities,
        )

        # Try each sampled position in the AABB.
        for j, (axis, is_top, start_point) in enumerate(samples):
            # Compute the ray's destination using the sampling & AABB information.
            end_point = compute_ray_destination(
                axis, is_top, start_point, -half_extent_with_offset, half_extent_with_offset
            )
            start_points[i][j] = start_point
            end_points[i][j] = end_point

    # Convert the points into the world frame
    orig_shape = start_points.shape
    to_wf_transform = T.pose2mat((bbox_center, bbox_orn))
    start_points = trimesh.transformations.transform_points(start_points.reshape(-1, 3), to_wf_transform).reshape(orig_shape)
    end_points = trimesh.transformations.transform_points(end_points.reshape(-1, 3), to_wf_transform).reshape(orig_shape)

    return start_points, end_points


def sample_raytest_start_end_full_grid_topdown(
    obj,
    ray_spacing,
    aabb_offset=m.DEFAULT_AABB_OFFSET,
):
    """
    Sample the start points and end points around a given object by a dense grid from top down.

    :param obj: The object to sample points on.
    :param ray_spacing: float, spacing between the rays, or equivalently, size of the grid cell
    :param aabb_offset: float or numpy array, padding for AABB to initiate ray-testing.
    Returns:
        start_points: the start points for raycasting, defined in the world frame,
            numpy array of shape [num_samples, max_sampling_attempts, 3]
        end_points: the end points for raycasting, defined in the world frame,
            numpy array of shape [num_samples, max_sampling_attempts, 3]
    """
    bbox_center, bbox_orn, bbox_bf_extent, _ = obj.get_base_aligned_bbox(xy_aligned=True, fallback_to_aabb=True)

    half_extent_with_offset = (bbox_bf_extent / 2) + aabb_offset
    x = np.linspace(-half_extent_with_offset[0], half_extent_with_offset[0], int(half_extent_with_offset[0] * 2 / ray_spacing))
    y = np.linspace(-half_extent_with_offset[1], half_extent_with_offset[1], int(half_extent_with_offset[1] * 2 / ray_spacing))
    n_rays = len(x) * len(y)

    start_points = np.stack([
        np.tile(x, len(y)),
        np.repeat(y, len(x)),
        np.ones(n_rays) * half_extent_with_offset[2],
    ]).T

    end_points = np.copy(start_points)
    end_points[:, 2] = -half_extent_with_offset[2]

    # Convert the points into the world frame
    to_wf_transform = T.pose2mat((bbox_center, bbox_orn))
    start_points = trimesh.transformations.transform_points(start_points, to_wf_transform)
    end_points = trimesh.transformations.transform_points(end_points, to_wf_transform)

    start_points = np.expand_dims(start_points, axis=1)
    end_points = np.expand_dims(end_points, axis=1)

    return start_points, end_points


def sample_cuboid_on_object_symmetric_bimodal_distribution(
    obj,
    num_samples,
    cuboid_dimensions,
    bimodal_mean_fraction,
    bimodal_stdev_fraction,
    axis_probabilities,
    new_ray_per_horizontal_distance=m.DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE,
    hit_proportion=m.DEFAULT_HIT_PROPORTION,
    aabb_offset=m.DEFAULT_AABB_OFFSET,
    max_sampling_attempts=m.DEFAULT_MAX_SAMPLING_ATTEMPTS,
    max_angle_with_z_axis=m.DEFAULT_MAX_ANGLE_WITH_Z_AXIS,
    parallel_ray_normal_angle_tolerance=m.DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE,
    hit_to_plane_threshold=m.DEFAULT_HIT_TO_PLANE_THRESHOLD,
    cuboid_bottom_padding=m.DEFAULT_CUBOID_BOTTOM_PADDING,
    undo_cuboid_bottom_padding=True,
    refuse_downwards=False,
):
    """
    Samples points on an object's surface using ray casting.
    Rays are sampled with a symmetric bimodal distribution.

    :param obj: The object to sample points on.
    :param num_samples: int, the number of points to try to sample.
    :param cuboid_dimensions: Float sequence of len 3, the size of the empty cuboid we are trying to sample. Can also
        provide list of cuboid dimension triplets in which case each i'th sample will be sampled using the i'th triplet.
        Alternatively, cuboid_dimensions can be set to be all zeros if the user just want to sample points (instead of
        cuboids) for significantly better performance. This applies when the user wants to sample very small particles.
    :param bimodal_mean_fraction: float, the mean of one side of the symmetric bimodal distribution as a fraction of the
        min-max range.
    :param bimodal_stdev_fraction: float, the standard deviation of one side of the symmetric bimodal distribution as a
        fraction of the min-max range.
    :param axis_probabilities: Array of shape (3, ), the probability of ray casting along each axis.
    :param new_ray_per_horizontal_distance: float, per this distance of the cuboid dimension, increase the grid size of
        the parallel ray-testing by 1. This controls how fine-grained the grid ray-casting should be with respect to
        the size of the sampled cuboid.
    :param hit_proportion: float, the minimum percentage of the hits required across the grid.
    :param aabb_offset: float or numpy array, padding for AABB to initiate ray-testing.
    :param max_sampling_attempts: int, how many times sampling will be attempted for each requested point.
    :param max_angle_with_z_axis: float, maximum angle between hit normal and positive Z axis allowed. Can be used to
        disallow downward-facing hits when refuse_downwards=True.
    :param parallel_ray_normal_angle_tolerance: float, maximum angle difference between the normal of the center hit
        and the normal of other hits allowed.
    :param hit_to_plane_threshold: float, how far any given hit position can be from the least-squares fit plane to
        all of the hit positions before the sample is rejected.
    :param cuboid_bottom_padding: float, additional padding applied to the bottom of the cuboid. This is needed for the
        emptiness check (@check_cuboid_empty) within the cuboid. un_padding=True can be set if the user wants to remove
        the padding after the emptiness check.
    :param refuse_downwards: bool, whether downward-facing hits (as defined by max_angle_with_z_axis) are allowed.
    :param undo_cuboid_bottom_padding: bool. Whether the bottom padding that's applied to the cuboid should be removed before return.
        Useful when the cuboid needs to be flush with the surface for whatever reason. Note that the padding will still
        be applied initially (since it's not possible to do the cuboid emptiness check without doing this - otherwise
        the rays will hit the sampled-on object), so the emptiness check still checks a padded cuboid. This flag will
        simply make the sampler undo the padding prior to returning.
    :return: List of num_samples elements where each element is a tuple in the form of
        (cuboid_centroid, cuboid_up_vector, cuboid_rotation, {refusal_reason: [refusal_details...]}). Cuboid positions
        are set to None when no successful sampling happens within the max number of attempts. Refusal details are only
        filled if the debug_sampling flag is globally set to True.
    """
    start_points, end_points = sample_raytest_start_end_symmetric_bimodal_distribution(
        obj,
        num_samples,
        bimodal_mean_fraction,
        bimodal_stdev_fraction,
        axis_probabilities,
        aabb_offset=aabb_offset,
        max_sampling_attempts=max_sampling_attempts,
    )
    return sample_cuboid_on_object(
        obj,
        start_points,
        end_points,
        cuboid_dimensions,
        undo_cuboid_bottom_padding=undo_cuboid_bottom_padding,
        new_ray_per_horizontal_distance=new_ray_per_horizontal_distance,
        hit_proportion=hit_proportion,
        max_angle_with_z_axis=max_angle_with_z_axis,
        parallel_ray_normal_angle_tolerance=parallel_ray_normal_angle_tolerance,
        hit_to_plane_threshold=hit_to_plane_threshold,
        cuboid_bottom_padding=cuboid_bottom_padding,
        refuse_downwards=refuse_downwards,
    )


def sample_cuboid_on_object_full_grid_topdown(
    obj,
    ray_spacing,
    cuboid_dimensions,
    new_ray_per_horizontal_distance=m.DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE,
    hit_proportion=m.DEFAULT_HIT_PROPORTION,
    aabb_offset=m.DEFAULT_AABB_OFFSET,
    max_angle_with_z_axis=m.DEFAULT_MAX_ANGLE_WITH_Z_AXIS,
    parallel_ray_normal_angle_tolerance=m.DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE,
    hit_to_plane_threshold=m.DEFAULT_HIT_TO_PLANE_THRESHOLD,
    cuboid_bottom_padding=m.DEFAULT_CUBOID_BOTTOM_PADDING,
    undo_cuboid_bottom_padding=True,
    refuse_downwards=False,
):
    """
    Samples points on an object's surface using ray casting.
    Rays are sampled with a dense grid from top down.

    :param obj: The object to sample points on.
    :param ray_spacing: float, spacing between the rays, or equivalently, size of the grid cell, when sampling the
        start and end points. This implicitly determines the number of cuboids that will be sampled.
    :param cuboid_dimensions: Float sequence of len 3, the size of the empty cuboid we are trying to sample. Can also
        provide list of cuboid dimension triplets in which case each i'th sample will be sampled using the i'th triplet.
        Alternatively, cuboid_dimensions can be set to be all zeros if the user just want to sample points (instead of
        cuboids) for significantly better performance. This applies when the user wants to sample very small particles.
    :param new_ray_per_horizontal_distance: float, per this distance of the cuboid dimension, increase the grid size of
        the parallel ray-testing by 1. This controls how fine-grained the grid ray-casting should be with respect to
        the size of the sampled cuboid.
    :param hit_proportion: float, the minimum percentage of the hits required across the grid.
    :param aabb_offset: float or numpy array, padding for AABB to initiate ray-testing.
    :param max_angle_with_z_axis: float, maximum angle between hit normal and positive Z axis allowed. Can be used to
        disallow downward-facing hits when refuse_downwards=True.
    :param parallel_ray_normal_angle_tolerance: float, maximum angle difference between the normal of the center hit
        and the normal of other hits allowed.
    :param hit_to_plane_threshold: float, how far any given hit position can be from the least-squares fit plane to
        all of the hit positions before the sample is rejected.
    :param cuboid_bottom_padding: float, additional padding applied to the bottom of the cuboid. This is needed for the
        emptiness check (@check_cuboid_empty) within the cuboid. un_padding=True can be set if the user wants to remove
        the padding after the emptiness check.
    :param refuse_downwards: bool, whether downward-facing hits (as defined by max_angle_with_z_axis) are allowed.
    :param undo_cuboid_bottom_padding: bool. Whether the bottom padding that's applied to the cuboid should be removed before return.
        Useful when the cuboid needs to be flush with the surface for whatever reason. Note that the padding will still
        be applied initially (since it's not possible to do the cuboid emptiness check without doing this - otherwise
        the rays will hit the sampled-on object), so the emptiness check still checks a padded cuboid. This flag will
        simply make the sampler undo the padding prior to returning.
    :return: List of num_samples elements where each element is a tuple in the form of
        (cuboid_centroid, cuboid_up_vector, cuboid_rotation, {refusal_reason: [refusal_details...]}). Cuboid positions
        are set to None when no successful sampling happens within the max number of attempts. Refusal details are only
        filled if the debug_sampling flag is globally set to True.
    """
    start_points, end_points = sample_raytest_start_end_full_grid_topdown(
        obj,
        ray_spacing,
        aabb_offset=aabb_offset,
    )
    return sample_cuboid_on_object(
        obj,
        start_points,
        end_points,
        cuboid_dimensions,
        undo_cuboid_bottom_padding=undo_cuboid_bottom_padding,
        new_ray_per_horizontal_distance=new_ray_per_horizontal_distance,
        hit_proportion=hit_proportion,
        max_angle_with_z_axis=max_angle_with_z_axis,
        parallel_ray_normal_angle_tolerance=parallel_ray_normal_angle_tolerance,
        hit_to_plane_threshold=hit_to_plane_threshold,
        cuboid_bottom_padding=cuboid_bottom_padding,
        refuse_downwards=refuse_downwards,
    )


def sample_cuboid_on_object(
    obj,
    start_points,
    end_points,
    cuboid_dimensions,
    ignore_objs=None,
    new_ray_per_horizontal_distance=m.DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE,
    hit_proportion=m.DEFAULT_HIT_PROPORTION,
    max_angle_with_z_axis=m.DEFAULT_MAX_ANGLE_WITH_Z_AXIS,
    parallel_ray_normal_angle_tolerance=m.DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE,
    hit_to_plane_threshold=m.DEFAULT_HIT_TO_PLANE_THRESHOLD,
    cuboid_bottom_padding=m.DEFAULT_CUBOID_BOTTOM_PADDING,
    undo_cuboid_bottom_padding=True,
    refuse_downwards=False,
):
    """
    Samples points on an object's surface using ray casting.

    :param obj: (None or EntityPrim) The object to sample points on. If None, will sample points on arbitrary
        objects hit
    :param start_points: the start points for raycasting, defined in the world frame,
        numpy array of shape [num_samples, max_sampling_attempts, 3]
    :param end_points: the end points for raycasting, defined in the world frame,
        numpy array of shape [num_samples, max_sampling_attempts, 3]
    :param cuboid_dimensions: Float sequence of len 3, the size of the empty cuboid we are trying to sample. Can also
        provide list of cuboid dimension triplets in which case each i'th sample will be sampled using the i'th triplet.
        Alternatively, cuboid_dimensions can be set to be all zeros if the user just wants to sample points (instead of
        cuboids) for significantly better performance. This applies when the user wants to sample very small particles.
    :param ignore_objs: (None or list of EntityPrim) If @obj is None, this can be used to filter objects when checking
        for valid cuboid locations. Any sampled rays that hit an object in @ignore_objs will be ignored. If None,
        no filtering will be used
    :param new_ray_per_horizontal_distance: float, per this distance of the cuboid dimension, increase the grid size of
        the parallel ray-testing by 1. This controls how fine-grained the grid ray-casting should be with respect to
        the size of the sampled cuboid.
    :param hit_proportion: float, the minimum percentage of the hits required across the grid.
    :param max_angle_with_z_axis: float, maximum angle between hit normal and positive Z axis allowed. Can be used to
        disallow downward-facing hits when refuse_downwards=True.
    :param parallel_ray_normal_angle_tolerance: float, maximum angle difference between the normal of the center hit
        and the normal of other hits allowed.
    :param hit_to_plane_threshold: float, how far any given hit position can be from the least-squares fit plane to
        all of the hit positions before the sample is rejected.
    :param cuboid_bottom_padding: float, additional padding applied to the bottom of the cuboid. This is needed for the
        emptiness check (@check_cuboid_empty) within the cuboid. un_padding=True can be set if the user wants to remove
        the padding after the emptiness check.
    :param undo_cuboid_bottom_padding: bool. Whether the bottom padding that's applied to the cuboid should be removed before return.
        Useful when the cuboid needs to be flush with the surface for whatever reason. Note that the padding will still
        be applied initially (since it's not possible to do the cuboid emptiness check without doing this - otherwise
        the rays will hit the sampled-on object), so the emptiness check still checks a padded cuboid. This flag will
        simply make the sampler undo the padding prior to returning.
    :param refuse_downwards: bool, whether downward-facing hits (as defined by max_angle_with_z_axis) are allowed.
    :return: List of num_samples elements where each element is a tuple in the form of
        (cuboid_centroid, cuboid_up_vector, cuboid_rotation, {refusal_reason: [refusal_details...]}). Cuboid positions
        are set to None when no successful sampling happens within the max number of attempts. Refusal details are only
        filled if the debug_sampling flag is globally set to True.
    """

    assert start_points.shape == end_points.shape, \
        "the start and end points of raycasting are expected to have the same shape."
    num_samples = start_points.shape[0]

    cuboid_dimensions = np.array(cuboid_dimensions)
    if np.any(cuboid_dimensions > 50.0):
        print("WARNING: Trying to sample for a very large cuboid (at least one dimensions > 50)."
              "This will take a prohibitively large amount of time!")
    assert cuboid_dimensions.ndim <= 2
    assert cuboid_dimensions.shape[-1] == 3, "Cuboid dimensions need to contain all three dimensions."
    if cuboid_dimensions.ndim == 2:
        assert cuboid_dimensions.shape[0] == num_samples, "Need as many offsets as samples requested."

    results = [(None, None, None, None, defaultdict(list)) for _ in range(num_samples)]
    rigid_bodies = None if obj is None else [link.prim_path for link in obj.links.values()]
    ignore_rigid_bodies = None if ignore_objs is None else \
        [link.prim_path for ignore_obj in ignore_objs for link in ignore_obj.links.values()]

    for i in range(num_samples):
        refusal_reasons = results[i][4]
        # Try each sampled position in the AABB.
        for start_pos, end_pos in zip(start_points[i], end_points[i]):
            # If we have a list of cuboid dimensions, pick the one that corresponds to this particular sample.
            this_cuboid_dimensions = cuboid_dimensions if cuboid_dimensions.ndim == 1 else cuboid_dimensions[i]

            zero_cuboid_dimension = (this_cuboid_dimensions == 0.0).all()

            if not zero_cuboid_dimension:
                # Make sure we have valid (nonzero) x and y values
                assert (this_cuboid_dimensions[:-1] > 0).all(), \
                    f"Cuboid x and y dimensions must not be zero if z dimension is nonzero! Got: {this_cuboid_dimensions}"
                # Obtain the parallel rays using the direction sampling method.
                sources, destinations, grid = np.array(get_parallel_rays(
                    start_pos, end_pos, this_cuboid_dimensions[:2] / 2.0, new_ray_per_horizontal_distance,
                ))
            else:
                sources = np.array([start_pos])
                destinations = np.array([end_pos])

            # Time to cast the rays.
            cast_results = raytest_batch(start_points=sources, end_points=destinations)

            # Check whether sufficient number of rays hit the object
            hits = check_rays_hit_object(
                cast_results, hit_proportion, refusal_reasons["missed_object"], rigid_bodies, ignore_rigid_bodies)
            if hits is None:
                continue

            center_idx = int(len(hits) / 2)
            # Only consider objects whose center idx has a ray hit
            if not hits[center_idx]:
                continue

            filtered_cast_results = []
            filtered_center_idx = None
            for idx, hit in enumerate(hits):
                if hit:
                    filtered_cast_results.append(cast_results[idx])
                    if idx == center_idx:
                        filtered_center_idx = len(filtered_cast_results) - 1

            # Process the hit positions and normals.
            hit_positions = np.array([ray_res["position"] for ray_res in filtered_cast_results])
            hit_normals = np.array([ray_res["normal"] for ray_res in filtered_cast_results])
            hit_normals /= np.linalg.norm(hit_normals, axis=1, keepdims=True)

            assert filtered_center_idx is not None
            hit_link = filtered_cast_results[filtered_center_idx]["rigidBody"]
            center_hit_pos = hit_positions[filtered_center_idx]
            center_hit_normal = hit_normals[filtered_center_idx]

            # Reject anything facing more than 45deg downwards if requested.
            if refuse_downwards:
                if not check_hit_max_angle_from_z_axis(
                    center_hit_normal, max_angle_with_z_axis, refusal_reasons["downward_normal"]
                ):
                    continue

            # Check that none of the parallel rays' hit normal differs from center ray by more than threshold.
            if not zero_cuboid_dimension:
                if not check_normal_similarity(center_hit_normal, hit_normals, parallel_ray_normal_angle_tolerance, refusal_reasons["hit_normal_similarity"]):
                    continue

                # Fit a plane to the points.
                plane_centroid, plane_normal = fit_plane(hit_positions)

                # The fit_plane normal can be facing either direction on the normal axis, but we want it to face away from
                # the object for purposes of normal checking and padding. To do this:
                # We get a vector from the centroid towards the center ray source, and flip the plane normal to match it.
                # The cosine has positive sign if the two vectors are similar and a negative one if not.
                plane_to_source = sources[center_idx] - plane_centroid
                plane_normal *= np.sign(np.dot(plane_to_source, plane_normal))

                # Check that the plane normal is similar to the hit normal
                if not check_normal_similarity(
                    center_hit_normal, plane_normal[None, :], parallel_ray_normal_angle_tolerance, refusal_reasons["plane_normal_similarity"]
                ):
                    continue

                # Check that the points are all within some acceptable distance of the plane.
                if not check_distance_to_plane(
                    hit_positions, plane_centroid, plane_normal, hit_to_plane_threshold, refusal_reasons["dist_to_plane"]
                ):
                    continue

                # Get projection of the base onto the plane, fit a rotation, and compute the new center hit / corners.
                hit_positions = np.array([ray_res.get("position", np.zeros(3)) for ray_res in cast_results])
                projected_hits = get_projection_onto_plane(hit_positions, plane_centroid, plane_normal)
                padding = cuboid_bottom_padding * plane_normal
                projected_hits += padding
                center_projected_hit = projected_hits[center_idx]
                cuboid_centroid = center_projected_hit + plane_normal * this_cuboid_dimensions[2] / 2.0

                rotation = compute_rotation_from_grid_sample(
                    grid, projected_hits, cuboid_centroid, this_cuboid_dimensions,
                    hits, refusal_reasons["rotation_not_computable"])

                # Make sure there are enough hit points that can be used for alignment to find the rotation
                if rotation is None:
                    continue

                corner_positions = cuboid_centroid[None, :] + (
                    rotation.apply(
                        0.5
                        * this_cuboid_dimensions
                        * np.array(
                            [
                                [1, 1, -1],
                                [-1, 1, -1],
                                [-1, -1, -1],
                                [1, -1, -1],
                            ]
                        )
                    )
                )

                # Now we use the cuboid's diagonals to check that the cuboid is actually empty
                if not check_cuboid_empty(
                        plane_normal,
                        corner_positions,
                        this_cuboid_dimensions,
                        refusal_reasons["cuboid_not_empty"],
                        ignore_body_names=ignore_rigid_bodies,
                ):
                    continue

                if undo_cuboid_bottom_padding:
                    cuboid_centroid -= padding

            else:
                cuboid_centroid = center_hit_pos
                if not undo_cuboid_bottom_padding:
                    padding = cuboid_bottom_padding * center_hit_normal
                    cuboid_centroid += padding
                plane_normal = np.zeros(3)
                rotation = R.from_quat([0, 0, 0, 1])

            # We've found a nice attachment point. Continue onto next point to sample.
            results[i] = (cuboid_centroid, plane_normal, rotation.as_quat(), hit_link, refusal_reasons)
            break

    if og.debug_sampling:
        print("Sampling rejection reasons:")
        counter = Counter()

        for instance in results:
            for reason, refusals in instance[-1].items():
                counter[reason] += len(refusals)

        print("\n".join("%s: %d" % pair for pair in counter.items()))

    return results


def compute_rotation_from_grid_sample(two_d_grid, projected_hits, cuboid_centroid, this_cuboid_dimensions, hits, refusal_log):
    if np.sum(hits) < 3:
        if og.debug_sampling:
            refusal_log.append(f"insufficient hits to compute the rotation of the grid: needs 3, has {np.sum(hits)}")
        return None

    grid_in_planar_coordinates = two_d_grid.reshape(-1, 2)
    grid_in_planar_coordinates = grid_in_planar_coordinates[hits]
    grid_in_object_coordinates = np.zeros((len(grid_in_planar_coordinates), 3))
    grid_in_object_coordinates[:, :2] = grid_in_planar_coordinates
    grid_in_object_coordinates[:, 2] = -this_cuboid_dimensions[2] / 2.0

    projected_hits = projected_hits[hits]
    sampled_grid_relative_vectors = projected_hits - cuboid_centroid

    rotation, _ = R.align_vectors(sampled_grid_relative_vectors, grid_in_object_coordinates)

    return rotation


def check_normal_similarity(center_hit_normal, hit_normals, tolerance, refusal_log):
    parallel_hit_main_hit_dot_products = np.clip(
        np.dot(hit_normals, center_hit_normal)
        / (np.linalg.norm(hit_normals, axis=1) * np.linalg.norm(center_hit_normal)),
        -1.0,
        1.0,
    )
    parallel_hit_normal_angles_to_hit_normal = np.arccos(parallel_hit_main_hit_dot_products)
    all_rays_hit_with_similar_normal = np.all(
        parallel_hit_normal_angles_to_hit_normal < tolerance
    )
    if not all_rays_hit_with_similar_normal:
        if og.debug_sampling:
            refusal_log.append("angles %r" % (np.rad2deg(parallel_hit_normal_angles_to_hit_normal),))

        return False

    return True


def check_rays_hit_object(cast_results, threshold, refusal_log, body_names=None, ignore_body_names=None):
    """
    Checks whether rays hit a specific object, as specified by a list of @body_names

    Args:
        cast_results (list of dict): Output from raycast_batch.
        threshold (float): Relative ratio in [0, 1] specifying proportion of rays from @cast_results are
            required to hit @body_names to count as the object being hit
        refusal_log (list of str): Logging array for adding debug logs
        body_names (None or list or set of str): absolute USD paths to rigid bodies to check for hit. If not
            specified, then any valid hit will be accepted
        ignore_body_names (None or list or set of str): absolute USD paths to rigid bodies to ignore for hit. If not
            specified, then any valid hit will be accepted

    Returns:
        tuple:
            - list of bool or None: Individual T/F for each ray -- whether it hit the object or not
    """
    body_names = None if body_names is None else set(body_names)
    ray_hits = [
        ray_res["hit"] and
        (body_names is None or ray_res["rigidBody"] in body_names) and
        (ignore_body_names is None or ray_res["rigidBody"] not in ignore_body_names)
        for ray_res in cast_results
    ]
    if sum(ray_hits) / len(cast_results) < threshold:
        if og.debug_sampling:
            refusal_log.append(f"{sum(ray_hits)} / {len(cast_results)} < {threshold} hits: {[ray_res['rigidBody'] for ray_res in cast_results if ray_res['hit']]}")

        return None

    return ray_hits


def check_hit_max_angle_from_z_axis(hit_normal, max_angle_with_z_axis, refusal_log):
    hit_angle_with_z = np.arccos(np.clip(np.dot(hit_normal, np.array([0, 0, 1])), -1.0, 1.0))
    if hit_angle_with_z > max_angle_with_z_axis:
        if og.debug_sampling:
            refusal_log.append("normal %r" % hit_normal)

        return False

    return True


def compute_ray_destination(axis, is_top, start_pos, aabb_min, aabb_max):
    # Get the ray casting direction - we want to do it parallel to the sample axis.
    ray_direction = np.array([0, 0, 0])
    ray_direction[axis] = 1
    ray_direction *= -1 if is_top else 1

    # We want to extend our ray until it intersects one of the AABB's faces.
    # Start by getting the distances towards the min and max boundaries of the AABB on each axis.
    point_to_min = aabb_min - start_pos
    point_to_max = aabb_max - start_pos

    # Then choose the distance to the point in the correct direction on each axis.
    closer_point_on_each_axis = np.where(ray_direction < 0, point_to_min, point_to_max)

    # For each axis, find how many times the ray direction should be multiplied to reach the AABB's boundary.
    multiple_to_face_on_each_axis = closer_point_on_each_axis / ray_direction

    # Choose the minimum of these multiples, e.g. how many times the ray direction should be multiplied
    # to reach the nearest boundary.
    multiple_to_face = np.min(multiple_to_face_on_each_axis[np.isfinite(multiple_to_face_on_each_axis)])

    # Finally, use the multiple we found to calculate the point on the AABB boundary that we want to cast our
    # ray until.
    point_on_face = start_pos + ray_direction * multiple_to_face

    # Make sure that we did not end up with all NaNs or infinities due to division issues.
    assert not np.any(np.isnan(point_on_face)) and not np.any(np.isinf(point_on_face))

    return point_on_face


def check_cuboid_empty(hit_normal, bottom_corner_positions, this_cuboid_dimensions, refusal_log, ignore_body_names=None):
    if og.debug_sampling:
        draw_debug_markers(bottom_corner_positions)

    # Compute top corners.
    top_corner_positions = bottom_corner_positions + hit_normal * this_cuboid_dimensions[2]

    # We only generate valid rays that have nonzero distances. If the inputted cuboid is flat (i.e.: one dimension
    # is zero, i.e.: it is in fact a rectangle), some of our generated rays will have zero distance

    # Get all the top-to-bottom corner pairs. When we cast these rays, we check for two things: that the cuboid
    # height is actually available, and the faces & volume of the cuboid are unoccupied.
    top_to_bottom_pairs = [] if this_cuboid_dimensions[2] == 0 else \
        list(itertools.product(top_corner_positions, bottom_corner_positions))

    # Get all the same-height pairs. These also check that the surfaces areas are empty.
    # Note: These are redundant if our cuboid has zero height!
    bottom_pairs = list(itertools.combinations(bottom_corner_positions, 2))
    top_pairs = [] if this_cuboid_dimensions[2] == 0 else list(itertools.combinations(top_corner_positions, 2))

    # Combine all these pairs, cast the rays, and make sure the rays don't hit anything.
    all_pairs = np.array(top_to_bottom_pairs + bottom_pairs + top_pairs)
    check_cast_results = raytest_batch(start_points=all_pairs[:, 0, :], end_points=all_pairs[:, 1, :], ignore_bodies=ignore_body_names)
    if any(ray["hit"] for ray in check_cast_results):
        if og.debug_sampling:
            refusal_log.append("check ray info: %r" % (check_cast_results))

        return False

    return True
