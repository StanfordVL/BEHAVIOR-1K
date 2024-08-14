import itertools
import math
import random
from collections import Counter, defaultdict

import torch as th
from scipy.stats import truncnorm

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.utils.ui_utils import create_module_logger, draw_line

# Create module logger
log = create_module_logger(module_name=__name__)


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEBUG_SAMPLING = False
m.DEFAULT_AABB_OFFSET_FRACTION = 0.02
m.DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE = 1.0  # Around 60 degrees
m.DEFAULT_HIT_TO_PLANE_THRESHOLD = 0.05
m.DEFAULT_MAX_ANGLE_WITH_Z_AXIS = 3 * math.pi / 4
m.DEFAULT_MAX_SAMPLING_ATTEMPTS = 10
m.DEFAULT_CUBOID_BOTTOM_PADDING = 0.005
# We will cast an additional parallel ray for each additional this much distance.
m.DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE = 0.1
m.DEFAULT_HIT_PROPORTION = 0.8


def fit_plane(points, refusal_log):
    """
    Fits a plane to the given 3D points.
    Copied from https://stackoverflow.com/a/18968498

    Args:
        points ((k, 3)-array): th.tensor of shape (k, 3)
        refusal_log (dict): Debugging dictionary to add error messages to

    Returns:
        2-tuple:
            - 3-array: (x,y,z) points' centroid
            - 3-array: (x,y,z) normal of the fitted plane
    """
    if points.shape[0] < points.shape[1]:
        if m.DEBUG_SAMPLING:
            refusal_log.append(f"insufficient points to fit a 3D plane: needs 3, has {points.shape[0]}.")
        return None, None

    ctr = points.mean(dim=0)
    x = points - ctr
    normal = th.linalg.svd(x.T @ x).U[:, -1]
    normal /= th.norm(normal)
    return ctr, normal


def check_distance_to_plane(points, plane_centroid, plane_normal, hit_to_plane_threshold, refusal_log):
    """
    Calculates whether points are within @hit_to_plane_threshold distance to plane defined by @plane_centroid
    and @plane_normal

    Args:
        points ((k, 3)-array): th.tensor of shape (k, 3)
        plane_centroid (3-array): (x,y,z) points' centroid
        plane_normal (3-array): (x,y,z) normal of the fitted plane
        hit_to_plane_threshold (float): Threshold distance to check between @points and plane
        refusal_log (dict): Debugging dictionary to add error messages to

    Returns:
        bool: True if all points are within @hit_to_plane_threshold distance to plane, otherwise False
    """
    distances = get_distance_to_plane(points, plane_centroid, plane_normal)
    if th.any(distances > hit_to_plane_threshold):
        if m.DEBUG_SAMPLING:
            refusal_log.append("distances to plane: %r" % distances)
        return False
    return True


def get_distance_to_plane(points, plane_centroid, plane_normal):
    """
    Computes distance from @points to plane defined by @plane_centroid and @plane_normal

    Args:
        points ((k, 3)-array): th.tensor of shape (k, 3)
        plane_centroid (3-array): (x,y,z) points' centroid
        plane_normal (3-array): (x,y,z) normal of the fitted plane

    Returns:
        k-array: Absolute distances from each point to the plane
    """
    return th.abs((points - plane_centroid) @ plane_normal)


def get_projection_onto_plane(points, plane_centroid, plane_normal):
    """
    Computes @points' projection onto the plane defined by @plane_centroid and @plane_normal

    Args:
        points ((k, 3)-array): th.tensor of shape (k, 3)
        plane_centroid (3-array): (x,y,z) points' centroid
        plane_normal (3-array): (x,y,z) normal of the fitted plane

    Returns:
        (k,3)-array: Points' positions projected onto the plane
    """
    distances_to_plane = get_distance_to_plane(points, plane_centroid, plane_normal)
    return points - th.outer(distances_to_plane, plane_normal)


def draw_debug_markers(hit_positions, radius=0.01):
    """
    Helper method to generate and place debug markers at @hit_positions

    Args:
        hit_positions ((n, 3)-array): Desired positions to place markers at
        radius (float): Radius of the generated virtual marker
    """
    color = th.cat([th.rand(3), [1]])
    for vec in hit_positions:
        for dim in range(3):
            start_point = vec + th.eye(3)[dim] * radius
            end_point = vec - th.eye(3)[dim] * radius
            draw_line(start_point, end_point, color)


def get_parallel_rays(source, destination, offset, new_ray_per_horizontal_distance):
    """
    Given an input ray described by a source and a destination, sample parallel rays around it as the center.

    The parallel rays start at the corners of a square of edge length `offset` centered on `source`, with the square
    orthogonal to the ray direction. That is, the cast rays are the height edges of a square-base cuboid with bases
    centered on `source` and `destination`.

    Args:
        source (3-array): (x,y,z) source of the ray to sample parallel rays of.
        destination (3-array): Source of the ray to sample parallel rays of.
        offset (float): Orthogonal distance of parallel rays from input ray.
        new_ray_per_horizontal_distance (float): Step in offset beyond which an additional split will be applied in the
            parallel ray grid (which at minimum is 3x3 at the AABB corners & center).

    Returns:
        3-tuple:
            - list: generated sources from the original ray
            - list: generated destinations from the original ray
            - (W, H, 3)-array: unflattened, untransformed grid of parallel rays in object coordinates
    """
    ray_direction = destination - source

    # Get an orthogonal vector using a random vector.
    random_vector = th.rand(3)
    orthogonal_vector_1 = th.linalg.cross(ray_direction, random_vector)
    orthogonal_vector_1 /= th.norm(orthogonal_vector_1)

    # Get a second vector orthogonal to both the ray and the first vector.
    orthogonal_vector_2 = -th.linalg.cross(ray_direction, orthogonal_vector_1)
    orthogonal_vector_2 /= th.norm(orthogonal_vector_2)

    orthogonal_vectors = th.stack([orthogonal_vector_1, orthogonal_vector_2])
    assert th.all(th.isfinite(orthogonal_vectors))

    # Convert the offset into a 2-vector if it already isn't one.
    offset = th.tensor([1, 1]) * offset

    # Compute the grid of rays
    steps = (offset / new_ray_per_horizontal_distance).int() * 2 + 1
    steps = th.maximum(steps, th.tensor(3))
    x_range = th.linspace(-offset[0], offset[0], steps[0])
    y_range = th.linspace(-offset[1], offset[1], steps[1])
    ray_grid = th.stack(th.meshgrid(x_range, y_range, indexing="ij"), dim=-1)
    ray_grid_flattened = ray_grid.reshape(-1, 2)

    # Apply the grid onto the orthogonal vectors to obtain the rays in the world frame.
    sources = [source + offsets @ orthogonal_vectors for offsets in ray_grid_flattened]
    destinations = [destination + offsets @ orthogonal_vectors for offsets in ray_grid_flattened]

    return sources, destinations, ray_grid


def sample_origin_positions(mins, maxes, count, bimodal_mean_fraction, bimodal_stdev_fraction, axis_probabilities):
    """
    Sample ray casting origin positions with a given distribution.

    The way the sampling works is that for each particle, it will sample two coordinates uniformly and one
    using a symmetric, bimodal truncated normal distribution. This way, the particles will mostly be close to the faces
    of the AABB (given a correctly parameterized bimodal truncated normal) and will be spread across each face,
    but there will still be a small number of particles spawned inside the object if it has an interior.

    Args:
        mins (3-array): the minimum coordinate along each axis.
        maxes (3-array): the maximum coordinate along each axis.
        count (int): Number of origins to sample.
        bimodal_mean_fraction (float): the mean of one side of the symmetric bimodal distribution as a fraction of the
            min-max range.
        bimodal_stdev_fraction (float): the standard deviation of one side of the symmetric bimodal distribution as a
            fraction of the min-max range.
        axis_probabilities (3-array): the probability of ray casting along each axis.

    Returns:
        list: List where each element is (ray cast axis index, bool whether the axis was sampled from the top side,
            [x, y, z]) tuples.
    """
    assert len(mins.shape) == 1
    assert mins.shape == maxes.shape

    results = []
    for i in range(count):
        # Get the uniform sample first.
        position = th.rand(3)

        # Sample the bimodal normal.
        bottom = (0 - bimodal_mean_fraction) / bimodal_stdev_fraction
        top = (1 - bimodal_mean_fraction) / bimodal_stdev_fraction
        bimodal_sample = truncnorm.rvs(bottom, top, loc=bimodal_mean_fraction, scale=bimodal_stdev_fraction)

        # Pick which axis the bimodal normal sample should go to.
        bimodal_axis = th.multinomial(th.tensor(axis_probabilities, dtype=th.float32), 1).item()

        # Choose which side of the axis to sample from. We only sample from the top for the Z axis.
        if bimodal_axis == 2:
            bimodal_axis_top_side = True
        else:
            bimodal_axis_top_side = random.choice([True, False])

        # Move sample based on chosen side.
        position[bimodal_axis] = bimodal_sample if bimodal_axis_top_side else 1 - bimodal_sample

        # Scale the position from the standard normal range to the min-max range.
        scaled_position = mins + (maxes - mins) * position

        # Save the result.
        results.append((bimodal_axis, bimodal_axis_top_side, scaled_position))

    return results


def raytest_batch(
    start_points, end_points, only_closest=True, ignore_bodies=None, ignore_collisions=None, callback=None
):
    """
    Computes raytest collisions for a set of rays cast from @start_points to @end_points.

    Args:
        start_points (list of 3-array): Array of start locations to cast rays, where each is (x,y,z) global
            start location of the ray
        end_points (list of 3-array): Array of end locations to cast rays, where each is (x,y,z) global
            end location of the ray
        only_closest (bool): Whether we report the first (closest) hit from the ray or grab all hits
        ignore_bodies (None or list of str): If specified, specifies absolute USD paths to rigid bodies
            whose collisions should be ignored
        ignore_collisions (None or list of str): If specified, specifies absolute USD paths to collision geoms
            whose collisions should be ignored
        callback (None or function): If specified and @only_closest is False, the custom callback to use per-hit.
            This can be efficient if raytests are meant to terminate early. If None, no custom callback will be used.
            Expected signature is callback(hit) -> bool, which returns True if the raycast should continue or not

    Returns:
        list of dict or list of list of dict: Results for all rays, where each entry corresponds to the result for the
            ith ray cast. If @only_closest=True, each entry in the list is the closest hit. Otherwise, each entry is
            its own (unordered) list of hits for that ray. Each dict is composed of:

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
        results.append(
            raytest(
                start_point=start_point,
                end_point=end_point,
                only_closest=only_closest,
                ignore_bodies=ignore_bodies,
                ignore_collisions=ignore_collisions,
                callback=callback,
            )
        )

    return results


def raytest(
    start_point,
    end_point,
    only_closest=True,
    ignore_bodies=None,
    ignore_collisions=None,
    callback=None,
):
    """
    Computes raytest collision for ray cast from @start_point to @end_point

    Args:
        start_point (3-array): (x,y,z) global start location of the ray
        end_point (3-array): (x,y,z) global end location of the ray
        only_closest (bool): Whether we report the first (closest) hit from the ray or grab all hits
        ignore_bodies (None or list of str): If specified, specifies absolute USD paths to rigid bodies
            whose collisions should be ignored
        ignore_collisions (None or list of str): If specified, specifies absolute USD paths to collision geoms
            whose collisions should be ignored
        callback (None or function): If specified and @only_closest is False, the custom callback to use per-hit.
            This can be efficient if raytests are meant to terminate early. If None, no custom callback will be used.
            Expected signature is callback(hit) -> bool, which returns True if the raycast should continue or not

    Returns:
        dict or list of dict: Results for this raytest. If @only_closest=True, then we only return the information from
            the closest hit. Otherwise, we return an (unordered) list of information for all hits encountered.
            Each dict is composed of:

            "hit" (bool): Whether an object was hit or not
            "position" (3-array): Location of the hit position
            "normal" (3-array): normal vector of the face hit
            "distance" (float): distance from @start_point the hit occurred
            "collision" (str): absolute USD path to the collision body hit
            "rigidBody" (str): absolute USD path to the associated rigid body hit

            Note that only "hit" = False exists in the dict if no hit was found
    """
    # Make sure start point, end point are torch tensors
    start_point = th.tensor(start_point) if not isinstance(start_point, th.Tensor) else start_point
    end_point = th.tensor(end_point) if not isinstance(end_point, th.Tensor) else end_point
    point_diff = end_point - start_point
    distance = th.norm(point_diff)
    direction = point_diff / distance

    # For efficiency's sake, we handle special case of no ignore_bodies, ignore_collisions, and closest_hit
    if only_closest and ignore_bodies is None and ignore_collisions is None:
        result = og.sim.psqi.raycast_closest(
            origin=start_point.tolist(),
            dir=direction.tolist(),
            distance=distance.tolist(),
        )
        if result["hit"]:
            result["position"] = th.tensor(result["position"])
            result["normal"] = th.tensor(result["normal"])
        return result
    else:
        # Compose callback function for finding raycasts
        hits = []
        ignore_bodies = set() if ignore_bodies is None else set(ignore_bodies)
        ignore_collisions = set() if ignore_collisions is None else set(ignore_collisions)

        def hit_callback(hit):
            # Only add to hits if we're not ignoring this body or collision
            if hit.rigid_body not in ignore_bodies and hit.collision not in ignore_collisions:
                hits.append(
                    {
                        "hit": True,
                        "position": th.tensor(hit.position),
                        "normal": th.tensor(hit.normal),
                        "distance": hit.distance,
                        "collision": hit.collision,
                        "rigidBody": hit.rigid_body,
                    }
                )
            # We always want to continue traversing to collect all hits
            return True if callback is None else callback(hit)

        # Grab all collisions
        og.sim.psqi.raycast_all(
            origin=start_point.tolist(),
            dir=direction.tolist(),
            distance=distance.tolist(),
            reportFn=hit_callback,
        )

        # If we only want the closest, we need to sort these hits, otherwise we return them all
        if only_closest:
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
    aabb_offset=None,
    aabb_offset_fraction=None,
    max_sampling_attempts=None,
):
    """
    Sample the start points and end points around a given object by a symmetric bimodal distribution

    obj (DatasetObject): The object to sample points on.
    num_samples (int): the number of points to try to sample.
    bimodal_mean_fraction (float): the mean of one side of the symmetric bimodal distribution as a fraction of the
        min-max range.
    bimodal_stdev_fraction (float): the standard deviation of one side of the symmetric bimodal distribution as a
        fraction of the min-max range.
    axis_probabilities (3-array): probability of ray casting along each axis.
    aabb_offset (None or float or 3-array): padding for AABB to initiate ray-testing, in absolute units. If specified,
        will override @aabb_offset_fraction
    aabb_offset_fraction (float or 3-array): padding for AABB to initiate ray-testing, as a fraction of overall AABB.
    max_sampling_attempts (int): how many times sampling will be attempted for each requested point.

    Returns:
        2-tuple:
            - (n, s, 3)-array: (num_samples, max_sampling_attempts, 3) shaped array representing the start points for
                raycasting defined in the world frame
            - (n, s, 3)-array: (num_samples, max_sampling_attempts, 3) shaped array representing the end points for
                raycasting defined in the world frame
    """
    aabb_offset_fraction = aabb_offset_fraction if aabb_offset_fraction is not None else m.DEFAULT_AABB_OFFSET_FRACTION
    max_sampling_attempts = (
        max_sampling_attempts if max_sampling_attempts is not None else m.DEFAULT_MAX_SAMPLING_ATTEMPTS
    )
    bbox_center, bbox_orn, bbox_bf_extent, _ = obj.get_base_aligned_bbox(xy_aligned=True)
    aabb_offset = aabb_offset_fraction * bbox_bf_extent if aabb_offset is None else aabb_offset
    half_extent_with_offset = (bbox_bf_extent / 2) + aabb_offset

    start_points = th.zeros((num_samples, max_sampling_attempts, 3))
    end_points = th.zeros((num_samples, max_sampling_attempts, 3))
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
    start_points = T.transform_points(start_points.reshape(-1, 3), to_wf_transform).reshape(orig_shape)
    end_points = T.transform_points(end_points.reshape(-1, 3), to_wf_transform).reshape(orig_shape)

    return start_points, end_points


def sample_raytest_start_end_full_grid_topdown(
    obj,
    ray_spacing,
    aabb_offset=None,
    aabb_offset_fraction=None,
):
    """
    Sample the start points and end points around a given object by a dense grid from top down.

    Args:
        obj (DatasetObject): The object to sample points on.
        ray_spacing (float): spacing between the rays, or equivalently, size of the grid cell
        aabb_offset (None or float or 3-array): padding for AABB to initiate ray-testing, in absolute units. If specified,
            will override @aabb_offset_fraction
        aabb_offset_fraction (float or 3-array): padding for AABB to initiate ray-testing, as a fraction of overall AABB.

    Returns:
        2-tuple:
            - (n, s, 3)-array: (num_samples, max_sampling_attempts, 3) shaped array representing the start points for
                raycasting defined in the world frame
            - (n, s, 3)-array: (num_samples, max_sampling_attempts, 3) shaped array representing the end points for
                raycasting defined in the world frame
    """
    aabb_offset_fraction = aabb_offset_fraction if aabb_offset_fraction is not None else m.DEFAULT_AABB_OFFSET_FRACTION
    bbox_center, bbox_orn, bbox_bf_extent, _ = obj.get_base_aligned_bbox(xy_aligned=True)
    aabb_offset = aabb_offset_fraction * bbox_bf_extent if aabb_offset is None else aabb_offset

    half_extent_with_offset = (bbox_bf_extent / 2) + aabb_offset
    x = th.linspace(
        -half_extent_with_offset[0], half_extent_with_offset[0], int(half_extent_with_offset[0] * 2 / ray_spacing) + 1
    )
    y = th.linspace(
        -half_extent_with_offset[1], half_extent_with_offset[1], int(half_extent_with_offset[1] * 2 / ray_spacing) + 1
    )
    n_rays = len(x) * len(y)

    start_points = th.stack(
        [
            th.tile(x, (len(y),)),
            th.repeat_interleave(y, len(x)),
            th.ones(n_rays) * half_extent_with_offset[2],
        ]
    ).T

    end_points = th.clone(start_points)
    end_points[:, 2] = -half_extent_with_offset[2]

    # Convert the points into the world frame
    to_wf_transform = T.pose2mat((bbox_center, bbox_orn))
    start_points = T.transform_points(start_points, to_wf_transform)
    end_points = T.transform_points(end_points, to_wf_transform)

    start_points = th.unsqueeze(start_points, dim=1)
    end_points = th.unsqueeze(end_points, dim=1)

    return start_points, end_points


def sample_cuboid_on_object_symmetric_bimodal_distribution(
    obj,
    num_samples,
    cuboid_dimensions,
    bimodal_mean_fraction,
    bimodal_stdev_fraction,
    axis_probabilities,
    new_ray_per_horizontal_distance=None,
    hit_proportion=None,
    aabb_offset=None,
    aabb_offset_fraction=None,
    max_sampling_attempts=None,
    max_angle_with_z_axis=None,
    parallel_ray_normal_angle_tolerance=None,
    hit_to_plane_threshold=None,
    cuboid_bottom_padding=None,
    undo_cuboid_bottom_padding=True,
    verify_cuboid_empty=True,
    refuse_downwards=False,
):
    """
    Samples points on an object's surface using ray casting.
    Rays are sampled with a symmetric bimodal distribution.

    Args:
        obj (DatasetObject): The object to sample points on.
        num_samples (int): the number of points to try to sample.
        cuboid_dimensions ((n, 3)-array): Float sequence of len 3, the size of the empty cuboid we are trying to sample.
            Can also provide list of cuboid dimension triplets in which case each i'th sample will be sampled using
            the i'th triplet. Alternatively, cuboid_dimensions can be set to be all zeros if the user just want to
            sample points (instead of cuboids) for significantly better performance. This applies when the user wants
            to sample very small particles.
        bimodal_mean_fraction (float): the mean of one side of the symmetric bimodal distribution as a fraction of the
            min-max range.
        bimodal_stdev_fraction (float): the standard deviation of one side of the symmetric bimodal distribution as a
            fraction of the min-max range.
        axis_probabilities (3-array): the probability of ray casting along each axis.
        new_ray_per_horizontal_distance (float): per this distance of the cuboid dimension, increase the grid size of
            the parallel ray-testing by 1. This controls how fine-grained the grid ray-casting should be with respect to
            the size of the sampled cuboid.
        hit_proportion (float): the minimum percentage of the hits required across the grid.
        aabb_offset (None or float or 3-array): padding for AABB to initiate ray-testing, in absolute units. If specified,
            will override @aabb_offset_fraction
        aabb_offset_fraction (float or 3-array): padding for AABB to initiate ray-testing, as a fraction of overall AABB.
        max_sampling_attempts (int): how many times sampling will be attempted for each requested point.
        max_angle_with_z_axis (float): maximum angle between hit normal and positive Z axis allowed. Can be used to
            disallow downward-facing hits when refuse_downwards=True.
        parallel_ray_normal_angle_tolerance (float): maximum angle difference between the normal of the center hit
            and the normal of other hits allowed.
        hit_to_plane_threshold (float): how far any given hit position can be from the least-squares fit plane to
            all of the hit positions before the sample is rejected.
        cuboid_bottom_padding (float): additional padding applied to the bottom of the cuboid. This is needed for the
            emptiness check (@check_cuboid_empty) within the cuboid. un_padding=True can be set if the user wants to remove
            the padding after the emptiness check.
        undo_cuboid_bottom_padding (bool): Whether the bottom padding that's applied to the cuboid should be removed before return.
            Useful when the cuboid needs to be flush with the surface for whatever reason. Note that the padding will still
            be applied initially (since it's not possible to do the cuboid emptiness check without doing this - otherwise
            the rays will hit the sampled-on object), so the emptiness check still checks a padded cuboid. This flag will
            simply make the sampler undo the padding prior to returning.
        verify_cuboid_empty (bool): Whether to filter out sampled cuboid locations that are not collision-free. Note
            that this check will only potentially occur if nonzero cuboid dimensions are specified.
        refuse_downwards (bool): whether downward-facing hits (as defined by max_angle_with_z_axis) are allowed.

    Returns:
        list of tuple: list of length num_samples elements where each element is a tuple in the form of
            (cuboid_centroid, cuboid_up_vector, cuboid_rotation, {refusal_reason: [refusal_details...]}). Cuboid positions
            are set to None when no successful sampling happens within the max number of attempts. Refusal details are only
            filled if the m.DEBUG_SAMPLING flag is globally set to True.
    """
    new_ray_per_horizontal_distance = (
        new_ray_per_horizontal_distance
        if new_ray_per_horizontal_distance is not None
        else m.DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE
    )
    hit_proportion = hit_proportion if hit_proportion is not None else m.DEFAULT_HIT_PROPORTION
    aabb_offset_fraction = aabb_offset_fraction if aabb_offset_fraction is not None else m.DEFAULT_AABB_OFFSET_FRACTION
    max_sampling_attempts = (
        max_sampling_attempts if max_sampling_attempts is not None else m.DEFAULT_MAX_SAMPLING_ATTEMPTS
    )
    max_angle_with_z_axis = (
        max_angle_with_z_axis if max_angle_with_z_axis is not None else m.DEFAULT_MAX_ANGLE_WITH_Z_AXIS
    )
    parallel_ray_normal_angle_tolerance = (
        parallel_ray_normal_angle_tolerance
        if parallel_ray_normal_angle_tolerance is not None
        else m.DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE
    )
    hit_to_plane_threshold = (
        hit_to_plane_threshold if hit_to_plane_threshold is not None else m.DEFAULT_HIT_TO_PLANE_THRESHOLD
    )
    cuboid_bottom_padding = (
        cuboid_bottom_padding if cuboid_bottom_padding is not None else m.DEFAULT_CUBOID_BOTTOM_PADDING
    )
    start_points, end_points = sample_raytest_start_end_symmetric_bimodal_distribution(
        obj,
        num_samples,
        bimodal_mean_fraction,
        bimodal_stdev_fraction,
        axis_probabilities,
        aabb_offset=aabb_offset,
        aabb_offset_fraction=aabb_offset_fraction,
        max_sampling_attempts=max_sampling_attempts,
    )

    return sample_cuboid_on_object(
        obj,
        start_points,
        end_points,
        cuboid_dimensions,
        new_ray_per_horizontal_distance=new_ray_per_horizontal_distance,
        hit_proportion=hit_proportion,
        max_angle_with_z_axis=max_angle_with_z_axis,
        parallel_ray_normal_angle_tolerance=parallel_ray_normal_angle_tolerance,
        hit_to_plane_threshold=hit_to_plane_threshold,
        cuboid_bottom_padding=cuboid_bottom_padding,
        undo_cuboid_bottom_padding=undo_cuboid_bottom_padding,
        verify_cuboid_empty=verify_cuboid_empty,
        refuse_downwards=refuse_downwards,
    )


def sample_cuboid_on_object_full_grid_topdown(
    obj,
    ray_spacing,
    cuboid_dimensions,
    new_ray_per_horizontal_distance=None,
    hit_proportion=None,
    aabb_offset=None,
    aabb_offset_fraction=None,
    max_angle_with_z_axis=None,
    parallel_ray_normal_angle_tolerance=None,
    hit_to_plane_threshold=None,
    cuboid_bottom_padding=None,
    undo_cuboid_bottom_padding=True,
    verify_cuboid_empty=True,
    refuse_downwards=False,
):
    """
    Samples points on an object's surface using ray casting.
    Rays are sampled with a dense grid from top down.

    Args:
        obj (DatasetObject): The object to sample points on.
        ray_spacing (float): spacing between the rays, or equivalently, size of the grid cell, when sampling the
            start and end points. This implicitly determines the number of cuboids that will be sampled.
        cuboid_dimensions ((n, 3)-array): Float sequence of len 3, the size of the empty cuboid we are trying to sample.
            Can also provide list of cuboid dimension triplets in which case each i'th sample will be sampled using
            the i'th triplet. Alternatively, cuboid_dimensions can be set to be all zeros if the user just want to
            sample points (instead of cuboids) for significantly better performance. This applies when the user wants
            to sample very small particles.
        new_ray_per_horizontal_distance (float): per this distance of the cuboid dimension, increase the grid size of
            the parallel ray-testing by 1. This controls how fine-grained the grid ray-casting should be with respect to
            the size of the sampled cuboid.
        hit_proportion (float): the minimum percentage of the hits required across the grid.
        aabb_offset (None or float or 3-array): padding for AABB to initiate ray-testing, in absolute units. If specified,
            will override @aabb_offset_fraction
        aabb_offset_fraction (float or 3-array): padding for AABB to initiate ray-testing, as a fraction of overall AABB.
        max_angle_with_z_axis (float): maximum angle between hit normal and positive Z axis allowed. Can be used to
            disallow downward-facing hits when refuse_downwards=True.
        parallel_ray_normal_angle_tolerance (float): maximum angle difference between the normal of the center hit
            and the normal of other hits allowed.
        hit_to_plane_threshold (float): how far any given hit position can be from the least-squares fit plane to
            all of the hit positions before the sample is rejected.
        cuboid_bottom_padding (float): additional padding applied to the bottom of the cuboid. This is needed for the
            emptiness check (@check_cuboid_empty) within the cuboid. un_padding=True can be set if the user wants to remove
            the padding after the emptiness check.
        undo_cuboid_bottom_padding (bool): Whether the bottom padding that's applied to the cuboid should be removed before return.
            Useful when the cuboid needs to be flush with the surface for whatever reason. Note that the padding will still
            be applied initially (since it's not possible to do the cuboid emptiness check without doing this - otherwise
            the rays will hit the sampled-on object), so the emptiness check still checks a padded cuboid. This flag will
            simply make the sampler undo the padding prior to returning.
        verify_cuboid_empty (bool): Whether to filter out sampled cuboid locations that are not collision-free. Note
            that this check will only potentially occur if nonzero cuboid dimensions are specified.
        refuse_downwards (bool): whether downward-facing hits (as defined by max_angle_with_z_axis) are allowed.

    Returns:
        list of tuple: list of length num_samples elements where each element is a tuple in the form of
            (cuboid_centroid, cuboid_up_vector, cuboid_rotation, {refusal_reason: [refusal_details...]}). Cuboid positions
            are set to None when no successful sampling happens within the max number of attempts. Refusal details are only
            filled if the m.DEBUG_SAMPLING flag is globally set to True.
    """
    new_ray_per_horizontal_distance = (
        new_ray_per_horizontal_distance
        if new_ray_per_horizontal_distance is not None
        else m.DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE
    )
    hit_proportion = hit_proportion if hit_proportion is not None else m.DEFAULT_HIT_PROPORTION
    aabb_offset_fraction = aabb_offset_fraction if aabb_offset_fraction is not None else m.DEFAULT_AABB_OFFSET_FRACTION
    max_angle_with_z_axis = (
        max_angle_with_z_axis if max_angle_with_z_axis is not None else m.DEFAULT_MAX_ANGLE_WITH_Z_AXIS
    )
    parallel_ray_normal_angle_tolerance = (
        parallel_ray_normal_angle_tolerance
        if parallel_ray_normal_angle_tolerance is not None
        else m.DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE
    )
    hit_to_plane_threshold = (
        hit_to_plane_threshold if hit_to_plane_threshold is not None else m.DEFAULT_HIT_TO_PLANE_THRESHOLD
    )
    cuboid_bottom_padding = (
        cuboid_bottom_padding if cuboid_bottom_padding is not None else m.DEFAULT_CUBOID_BOTTOM_PADDING
    )
    start_points, end_points = sample_raytest_start_end_full_grid_topdown(
        obj,
        ray_spacing,
        aabb_offset=aabb_offset,
        aabb_offset_fraction=aabb_offset_fraction,
    )
    return sample_cuboid_on_object(
        obj,
        start_points,
        end_points,
        cuboid_dimensions,
        new_ray_per_horizontal_distance=new_ray_per_horizontal_distance,
        hit_proportion=hit_proportion,
        max_angle_with_z_axis=max_angle_with_z_axis,
        parallel_ray_normal_angle_tolerance=parallel_ray_normal_angle_tolerance,
        hit_to_plane_threshold=hit_to_plane_threshold,
        cuboid_bottom_padding=cuboid_bottom_padding,
        undo_cuboid_bottom_padding=undo_cuboid_bottom_padding,
        verify_cuboid_empty=verify_cuboid_empty,
        refuse_downwards=refuse_downwards,
    )


def sample_cuboid_on_object(
    obj,
    start_points,
    end_points,
    cuboid_dimensions,
    ignore_objs=None,
    new_ray_per_horizontal_distance=None,
    hit_proportion=None,
    max_angle_with_z_axis=None,
    parallel_ray_normal_angle_tolerance=None,
    hit_to_plane_threshold=None,
    cuboid_bottom_padding=None,
    undo_cuboid_bottom_padding=True,
    verify_cuboid_empty=True,
    refuse_downwards=False,
):
    """
    Samples points on an object's surface using ray casting.

    Args:
        obj (DatasetObject): The object to sample points on.
        start_points ((n, s, 3)-array): (num_samples, max_sampling_attempts, 3) shaped array representing the start points for
            raycasting defined in the world frame
        end_points ((n, s, 3)-array): (num_samples, max_sampling_attempts, 3) shaped array representing the end points for
            raycasting defined in the world frame
        cuboid_dimensions ((n, 3)-array): Float sequence of len 3, the size of the empty cuboid we are trying to sample.
            Can also provide list of cuboid dimension triplets in which case each i'th sample will be sampled using
            the i'th triplet. Alternatively, cuboid_dimensions can be set to be all zeros if the user just want to
            sample points (instead of cuboids) for significantly better performance. This applies when the user wants
            to sample very small particles.
        ignore_objs (None or list of EntityPrim): If @obj is None, this can be used to filter objects when checking
            for valid cuboid locations. Any sampled rays that hit an object in @ignore_objs will be ignored. If None,
            no filtering will be used
        new_ray_per_horizontal_distance (float): per this distance of the cuboid dimension, increase the grid size of
            the parallel ray-testing by 1. This controls how fine-grained the grid ray-casting should be with respect to
            the size of the sampled cuboid.
        hit_proportion (float): the minimum percentage of the hits required across the grid.
        max_angle_with_z_axis (float): maximum angle between hit normal and positive Z axis allowed. Can be used to
            disallow downward-facing hits when refuse_downwards=True.
        parallel_ray_normal_angle_tolerance (float): maximum angle difference between the normal of the center hit
            and the normal of other hits allowed.
        hit_to_plane_threshold (float): how far any given hit position can be from the least-squares fit plane to
            all of the hit positions before the sample is rejected.
        cuboid_bottom_padding (float): additional padding applied to the bottom of the cuboid. This is needed for the
            emptiness check (@check_cuboid_empty) within the cuboid. un_padding=True can be set if the user wants to remove
            the padding after the emptiness check.
        undo_cuboid_bottom_padding (bool): Whether the bottom padding that's applied to the cuboid should be removed before return.
            Useful when the cuboid needs to be flush with the surface for whatever reason. Note that the padding will still
            be applied initially (since it's not possible to do the cuboid emptiness check without doing this - otherwise
            the rays will hit the sampled-on object), so the emptiness check still checks a padded cuboid. This flag will
            simply make the sampler undo the padding prior to returning.
        verify_cuboid_empty (bool): Whether to filter out sampled cuboid locations that are not collision-free. Note
            that this check will only potentially occur if nonzero cuboid dimensions are specified.
        refuse_downwards (bool): whether downward-facing hits (as defined by max_angle_with_z_axis) are allowed.

    Returns:
        list of tuple: list of length num_samples elements where each element is a tuple in the form of
            (cuboid_centroid, cuboid_up_vector, cuboid_rotation, {refusal_reason: [refusal_details...]}). Cuboid positions
            are set to None when no successful sampling happens within the max number of attempts. Refusal details are only
            filled if the m.DEBUG_SAMPLING flag is globally set to True.
    """
    new_ray_per_horizontal_distance = (
        new_ray_per_horizontal_distance
        if new_ray_per_horizontal_distance is not None
        else m.DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE
    )
    hit_proportion = hit_proportion if hit_proportion is not None else m.DEFAULT_HIT_PROPORTION
    max_angle_with_z_axis = (
        max_angle_with_z_axis if max_angle_with_z_axis is not None else m.DEFAULT_MAX_ANGLE_WITH_Z_AXIS
    )
    parallel_ray_normal_angle_tolerance = (
        parallel_ray_normal_angle_tolerance
        if parallel_ray_normal_angle_tolerance is not None
        else m.DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE
    )
    hit_to_plane_threshold = (
        hit_to_plane_threshold if hit_to_plane_threshold is not None else m.DEFAULT_HIT_TO_PLANE_THRESHOLD
    )
    cuboid_bottom_padding = (
        cuboid_bottom_padding if cuboid_bottom_padding is not None else m.DEFAULT_CUBOID_BOTTOM_PADDING
    )

    assert (
        start_points.shape == end_points.shape
    ), "the start and end points of raycasting are expected to have the same shape."
    num_samples = start_points.shape[0]

    if th.any(cuboid_dimensions > 50.0):
        log.warning(
            "WARNING: Trying to sample for a very large cuboid (at least one dimensions > 50). "
            "Terminating immediately, no hits will be registered."
        )
        return [(None, None, None, None, defaultdict(list)) for _ in range(num_samples)]

    assert cuboid_dimensions.ndim <= 2
    assert cuboid_dimensions.shape[-1] == 3, "Cuboid dimensions need to contain all three dimensions."
    if cuboid_dimensions.ndim == 2:
        assert cuboid_dimensions.shape[0] == num_samples, "Need as many offsets as samples requested."

    results = [(None, None, None, None, defaultdict(list)) for _ in range(num_samples)]
    rigid_bodies = None if obj is None else {link.prim_path for link in obj.links.values()}
    ignore_rigid_bodies = (
        None
        if ignore_objs is None
        else {link.prim_path for ignore_obj in ignore_objs for link in ignore_obj.links.values()}
    )

    for i in range(num_samples):
        refusal_reasons = results[i][4]
        # Try each sampled position in the AABB.
        for start_pos, end_pos in zip(start_points[i], end_points[i]):
            # If we have a list of cuboid dimensions, pick the one that corresponds to this particular sample.
            this_cuboid_dimensions = cuboid_dimensions if cuboid_dimensions.ndim == 1 else cuboid_dimensions[i]

            zero_cuboid_dimension = (this_cuboid_dimensions == 0.0).all()

            if not zero_cuboid_dimension:
                # Make sure we have valid (nonzero) x and y values
                assert (
                    this_cuboid_dimensions[:-1] > 0
                ).all(), f"Cuboid x and y dimensions must not be zero if z dimension is nonzero! Got: {this_cuboid_dimensions}"
                # Obtain the parallel rays using the direction sampling method.
                sources, destinations, grid = get_parallel_rays(
                    start_pos,
                    end_pos,
                    this_cuboid_dimensions[:2] / 2.0,
                    new_ray_per_horizontal_distance,
                )
            else:
                sources = [start_pos]
                destinations = [end_pos]

            # Time to cast the rays.
            cast_results = raytest_batch(
                start_points=sources, end_points=destinations, ignore_bodies=ignore_rigid_bodies
            )

            # Check whether sufficient number of rays hit the object
            hits = check_rays_hit_object(cast_results, hit_proportion, refusal_reasons["missed_object"], rigid_bodies)
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
            hit_positions = th.stack([ray_res["position"] for ray_res in filtered_cast_results])
            hit_normals = th.stack([ray_res["normal"] for ray_res in filtered_cast_results])
            hit_normals /= th.norm(hit_normals, dim=1, keepdim=True)

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
                if not check_normal_similarity(
                    center_hit_normal,
                    hit_normals,
                    parallel_ray_normal_angle_tolerance,
                    refusal_reasons["hit_normal_similarity"],
                ):
                    continue

                # Fit a plane to the points.
                plane_centroid, plane_normal = fit_plane(hit_positions, refusal_reasons["fit_plane"])
                if plane_centroid is None:
                    continue

                # The fit_plane normal can be facing either direction on the normal axis, but we want it to face away from
                # the object for purposes of normal checking and padding. To do this:
                # We get a vector from the centroid towards the center ray source, and flip the plane normal to match it.
                # The cosine has positive sign if the two vectors are similar and a negative one if not.
                plane_to_source = sources[center_idx] - plane_centroid
                plane_normal *= th.sign(th.dot(plane_to_source, plane_normal))

                # Check that the plane normal is similar to the hit normal
                if not check_normal_similarity(
                    center_hit_normal,
                    plane_normal[None, :],
                    parallel_ray_normal_angle_tolerance,
                    refusal_reasons["plane_normal_similarity"],
                ):
                    continue

                # Check that the points are all within some acceptable distance of the plane.
                if not check_distance_to_plane(
                    hit_positions,
                    plane_centroid,
                    plane_normal,
                    hit_to_plane_threshold,
                    refusal_reasons["dist_to_plane"],
                ):
                    continue

                # Get projection of the base onto the plane, fit a rotation, and compute the new center hit / corners.
                hit_positions = th.stack([ray_res.get("position", th.tensor([0.0] * 3)) for ray_res in cast_results])
                projected_hits = get_projection_onto_plane(hit_positions, plane_centroid, plane_normal)
                padding = cuboid_bottom_padding * plane_normal
                projected_hits += padding
                center_projected_hit = projected_hits[center_idx]
                cuboid_centroid = center_projected_hit + plane_normal * this_cuboid_dimensions[2] / 2.0

                rotation = compute_rotation_from_grid_sample(
                    grid,
                    projected_hits,
                    cuboid_centroid,
                    this_cuboid_dimensions,
                    hits,
                    refusal_reasons["rotation_not_computable"],
                )

                # Make sure there are enough hit points that can be used for alignment to find the rotation
                if rotation is None:
                    continue

                corner_vectors = (
                    0.5
                    * this_cuboid_dimensions
                    * th.tensor([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]], dtype=th.float32)
                )
                corner_positions = cuboid_centroid.unsqueeze(0) + T.quat_apply(rotation, corner_vectors)

                # Now we use the cuboid's diagonals to check that the cuboid is actually empty
                if verify_cuboid_empty and not check_cuboid_empty(
                    plane_normal,
                    corner_positions,
                    this_cuboid_dimensions,
                    refusal_reasons["cuboid_not_empty"],
                ):
                    continue

                if undo_cuboid_bottom_padding:
                    cuboid_centroid -= padding

            else:
                cuboid_centroid = center_hit_pos
                if not undo_cuboid_bottom_padding:
                    padding = cuboid_bottom_padding * center_hit_normal
                    cuboid_centroid += padding
                plane_normal = th.zeros(3)
                rotation = th.tensor([0, 0, 0, 1], dtype=th.float32)

            # We've found a nice attachment point. Continue onto next point to sample.
            results[i] = (cuboid_centroid, plane_normal, rotation, hit_link, refusal_reasons)
            break

    if m.DEBUG_SAMPLING:
        og.log.debug("Sampling rejection reasons:")
        counter = Counter()

        for instance in results:
            for reason, refusals in instance[-1].items():
                counter[reason] += len(refusals)

        og.log.debug("\n".join("%s: %d" % pair for pair in counter.items()))

    return results


def compute_rotation_from_grid_sample(
    two_d_grid, projected_hits, cuboid_centroid, this_cuboid_dimensions, hits, refusal_log
):
    """
    Computes

    Args:
        two_d_grid (n, 2): (x,y) raycast origin points in the local plane frame
        projected_hits ((k,3)-array): Points' positions projected onto the plane generated
        cuboid_centroid (3-array): (x,y,z) sampled position of the hit cuboid centroid in the global frame
        this_cuboid_dimensions (3-array): (x,y,z) size of cuboid being sampled from the grid
        hits (list of bool): whether each point from @two_d_grid is a valid hit or not
        refusal_log (dict): Dictionary to write debugging and log information to

    Returns:
        None or scipy.Rotation: If successfully hit, returns relative rotation from two_d_grid to
            generated hit plane. Otherwise, returns None
    """
    if sum(hits) < 3:
        if m.DEBUG_SAMPLING:
            refusal_log.append(f"insufficient hits to compute the rotation of the grid: needs 3, has {th.sum(hits)}")
        return None

    grid_in_planar_coordinates = two_d_grid.reshape(-1, 2)
    grid_in_planar_coordinates = grid_in_planar_coordinates[hits]
    grid_in_object_coordinates = th.zeros((len(grid_in_planar_coordinates), 3))
    grid_in_object_coordinates[:, :2] = grid_in_planar_coordinates
    grid_in_object_coordinates[:, 2] = -this_cuboid_dimensions[2] / 2.0

    projected_hits = projected_hits[hits]
    sampled_grid_relative_vectors = projected_hits - cuboid_centroid

    rotation = T.align_vector_sets(sampled_grid_relative_vectors, grid_in_object_coordinates)

    return rotation


def check_normal_similarity(center_hit_normal, hit_normals, tolerance, refusal_log):
    """
    Check whether the normals from @hit_normals are within some @tolerance of @center_hit_normal.

    Args:
        center_hit_normal (3-array): normal of the center hit point
        hit_normals ((n, 3)-array): normals of all the hit points
        tolerance (float): Acceptable deviation between the center hit normal and all normals
        refusal_log (dict): Dictionary to write debugging and log information to

    Returns:
        bool: Whether the normal similarity is acceptable or not
    """
    parallel_hit_main_hit_dot_products = th.clip(
        hit_normals @ center_hit_normal / (th.norm(hit_normals, dim=1) * th.norm(center_hit_normal)),
        -1.0,
        1.0,
    )
    parallel_hit_normal_angles_to_hit_normal = th.arccos(parallel_hit_main_hit_dot_products)
    all_rays_hit_with_similar_normal = th.all(parallel_hit_normal_angles_to_hit_normal < tolerance)
    if not all_rays_hit_with_similar_normal:
        if m.DEBUG_SAMPLING:
            refusal_log.append("angles %r" % (th.rad2deg(parallel_hit_normal_angles_to_hit_normal),))

        return False

    return True


def check_rays_hit_object(cast_results, threshold, refusal_log, body_names=None):
    """
    Checks whether rays hit a specific object, as specified by a list of @body_names

    Args:
        cast_results (list of dict): Output from raycast_batch.
        threshold (float): Relative ratio in [0, 1] specifying proportion of rays from @cast_results are
            required to hit @body_names to count as the object being hit
        refusal_log (list of str): Logging array for adding debug logs
        body_names (None or list or set of str): absolute USD paths to rigid bodies to check for hit. If not
            specified, then any valid hit will be accepted

    Returns:
        None or list of bool: Individual T/F for each ray -- whether it hit the object or not
    """
    body_names = None if body_names is None else set(body_names)
    ray_hits = [
        ray_res["hit"] and (body_names is None or ray_res["rigidBody"] in body_names) for ray_res in cast_results
    ]
    if sum(ray_hits) / len(cast_results) < threshold:
        if m.DEBUG_SAMPLING:
            refusal_log.append(
                f"{sum(ray_hits)} / {len(cast_results)} < {threshold} hits: {[ray_res['rigidBody'] for ray_res in cast_results if ray_res['hit']]}"
            )

        return None

    return ray_hits


def check_hit_max_angle_from_z_axis(hit_normal, max_angle_with_z_axis, refusal_log):
    """
    Check whether the normal @hit_normal deviates from the global z axis by more than @max_angle_with_z_axis

    Args:
        hit_normal (3-array): Normal vector to check with respect to global z-axis
        max_angle_with_z_axis (float): Maximum acceptable angle between the global z-axis and @hit_normal
        refusal_log (list of str): Logging array for adding debug logs

    Returns:
        bool: True if the angle between @hit_normal and the global z-axis is less than @max_angle_with_z_axis,
            otherwise False
    """
    hit_angle_with_z = th.arccos(th.clip(th.dot(hit_normal, th.tensor([0.0, 0.0, 1.0])), -1.0, 1.0))
    if hit_angle_with_z > max_angle_with_z_axis:
        if m.DEBUG_SAMPLING:
            refusal_log.append("normal %r" % hit_normal)

        return False

    return True


def compute_ray_destination(axis, is_top, start_pos, aabb_min, aabb_max):
    """
    Compute the point on the AABB defined by @aabb_min and @aabb_max from shooting a ray at @start_pos
    in the direction defined by global axis @axis and @is_top

    Args:
        axis (int): Which direction to compute the ray destination. Valid options are {0, 1, 2} -- the
            x, y, or z axes
        is_top (bool): Whether to shoot in the positive or negative @axis direction
        aabb_min (3-array): (x,y,z) position defining the lower corner of the AABB
        aabb_max (3-array): (x,y,z) position defining the upper corner of the AABB

    Returns:
        3-array: computed (x,y,z) point on the AABB surface
    """
    # Get the ray casting direction - we want to do it parallel to the sample axis.
    ray_direction = th.tensor([0, 0, 0])
    ray_direction[axis] = 1
    ray_direction *= -1 if is_top else 1

    # We want to extend our ray until it intersects one of the AABB's faces.
    # Start by getting the distances towards the min and max boundaries of the AABB on each axis.
    point_to_min = aabb_min - start_pos
    point_to_max = aabb_max - start_pos

    # Then choose the distance to the point in the correct direction on each axis.
    closer_point_on_each_axis = th.where(ray_direction < 0, point_to_min, point_to_max)

    # For each axis, find how many times the ray direction should be multiplied to reach the AABB's boundary.
    multiple_to_face_on_each_axis = closer_point_on_each_axis / ray_direction

    # Choose the minimum of these multiples, e.g. how many times the ray direction should be multiplied
    # to reach the nearest boundary.
    multiple_to_face = th.min(multiple_to_face_on_each_axis[th.isfinite(multiple_to_face_on_each_axis)]).item()

    # Finally, use the multiple we found to calculate the point on the AABB boundary that we want to cast our
    # ray until.
    point_on_face = start_pos + ray_direction * multiple_to_face

    # Make sure that we did not end up with all NaNs or infinities due to division issues.
    assert not th.any(th.isnan(point_on_face)) and not th.any(th.isinf(point_on_face))

    return point_on_face


def check_cuboid_empty(hit_normal, bottom_corner_positions, this_cuboid_dimensions, refusal_log):
    """
    Check whether the cuboid defined by @this_cuboid_dimensions and @bottom_corner_positions contains
    empty space or not

    Args:
        hit_normal (3-array): (x,y,z) normal
        bottom_corner_positions ((4, 3)-array): the positions defining the bottom corners of the cuboid
            being sampled
        this_cuboid_dimensions (3-array): (x,y,z) size of the sampled cuboid
        refusal_log (list of str): Logging array for adding debug logs

    Returns:
        bool: True if the cuboid is empty, else False
    """
    if m.DEBUG_SAMPLING:
        draw_debug_markers(bottom_corner_positions)

    # Compute top corners.
    top_corner_positions = bottom_corner_positions + hit_normal * this_cuboid_dimensions[2]

    # We only generate valid rays that have nonzero distances. If the inputted cuboid is flat (i.e.: one dimension
    # is zero, i.e.: it is in fact a rectangle), raise an error
    assert this_cuboid_dimensions[2] != 0, "Cannot check empty cuboid for cuboid with zero height!"

    # Get all the top-to-bottom corner pairs.
    # When we cast these rays, we check that the faces & volume of the cuboid are unoccupied.
    top_to_bottom_pairs = list(itertools.product(top_corner_positions, bottom_corner_positions))

    # Get all the same-height pairs. These also check that the surfaces areas are empty.
    bottom_pairs = list(itertools.combinations(bottom_corner_positions, 2))
    top_pairs = list(itertools.combinations(top_corner_positions, 2))

    # Combine all these pairs, cast the rays, and make sure the rays don't hit anything.
    pairs_list = top_to_bottom_pairs + bottom_pairs + top_pairs
    all_pairs = th.stack([th.cat([pair[0], pair[1]]) for pair in pairs_list])
    check_cast_results = raytest_batch(start_points=all_pairs[:, :3], end_points=all_pairs[:, 3:])
    if any(ray["hit"] for ray in check_cast_results):
        if m.DEBUG_SAMPLING:
            refusal_log.append("check ray info: %r" % (check_cast_results))

        return False

    return True
