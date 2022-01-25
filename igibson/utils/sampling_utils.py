import itertools
from collections import Counter, defaultdict

import numpy as np

import trimesh
from scipy.spatial.transform import Rotation
from scipy.stats import truncnorm

import igibson
from igibson.objects.visual_marker import VisualMarker
from igibson.utils import utils

_DEFAULT_AABB_OFFSET = 0.1
_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE = 1.0  # Around 60 degrees
_DEFAULT_HIT_TO_PLANE_THRESHOLD = 0.05
_DEFAULT_MAX_ANGLE_WITH_Z_AXIS = 3 * np.pi / 4
_DEFAULT_MAX_SAMPLING_ATTEMPTS = 10
_DEFAULT_CUBOID_BOTTOM_PADDING = 0.005
# We will cast an additional parallel ray for each additional this much distance.
_DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE = 0.1


def fit_plane(points):
    """
    Fits a plane to the given 3D points.
    Copied from https://stackoverflow.com/a/18968498

    :param points: np.array of shape (k, 3)
    :return Tuple[np.array, np.array] where first element is the points' centroid and the second is
    """
    assert points.shape[1] <= points.shape[0], "Cannot fit plane with only {} points in {} dimensions.".format(
        points.shape[0], points.shape[1]
    )
    ctr = points.mean(axis=0)
    x = points - ctr[np.newaxis, :]
    M = np.dot(x.T, x)
    normal = np.linalg.svd(M)[0][:, -1]
    return ctr, normal / np.linalg.norm(normal)


def get_distance_to_plane(points, plane_centroid, plane_normal):
    return np.abs(np.dot(points - plane_centroid, plane_normal))


def get_projection_onto_plane(points, plane_centroid, plane_normal):
    distances_to_plane = get_distance_to_plane(points, plane_centroid, plane_normal)
    return points - np.outer(distances_to_plane, plane_normal)


def draw_debug_markers(hit_positions):
    color = np.concatenate([np.random.rand(3), [1]])
    for vec in hit_positions:
        m = VisualMarker(rgba_color=color, radius=0.001, initial_offset=vec)
        # This feature is not usable out-of-the-box since this function (or its callers) don't have access to a
        # simulator instance to be able to load this marker into. You can work around this by creating a variable
        # inside the simulator.py file called SIMULATOR and making it refer to the global simulator instance.
        # simulator.SIMULATOR.import_object(m)


def get_parallel_rays(
    source, destination, offset, new_ray_per_horizontal_distance=_DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE
):
    """Given a ray described by a source and a destination, sample parallel rays and return together with input ray.

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

    # Apply the grid onto the orthogonal vectors to obtain the rays.
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


def sample_cuboid_on_object(
    obj,
    num_samples,
    cuboid_dimensions,
    bimodal_mean_fraction,
    bimodal_stdev_fraction,
    axis_probabilities,
    undo_padding=False,
    aabb_offset=_DEFAULT_AABB_OFFSET,
    max_sampling_attempts=_DEFAULT_MAX_SAMPLING_ATTEMPTS,
    max_angle_with_z_axis=_DEFAULT_MAX_ANGLE_WITH_Z_AXIS,
    hit_to_plane_threshold=_DEFAULT_HIT_TO_PLANE_THRESHOLD,
    refuse_downwards=False,
):
    """
    Samples points on an object's surface using ray casting.

    :param obj: The object to sample points on.
    :param num_samples: int, the number of points to try to sample.
    :param cuboid_dimensions: Float sequence of len 3, the size of the empty cuboid we are trying to sample. Can also
        provice list of cuboid dimension triplets in which case each i'th sample will be sampled using the i'th triplet.
    :param bimodal_mean_fraction: float, the mean of one side of the symmetric bimodal distribution as a fraction of the
        min-max range.
    :param bimodal_stdev_fraction: float, the standard deviation of one side of the symmetric bimodal distribution as a
        fraction of the min-max range.
    :param axis_probabilities: Array of shape (3, ), the probability of ray casting along each axis.
    :param undo_padding: bool. Whether the bottom padding that's applied to the cuboid should be removed before return.
        Useful when the cuboid needs to be flush with the surface for whatever reason. Note that the padding will still
        be applied initially (since it's not possible to do the cuboid emptiness check without doing this - otherwise
        the rays will hit the sampled-on object), so the emptiness check still checks a padded cuboid. This flag will
        simply make the sampler undo the padding prior to returning.
    :param aabb_offset: float, padding for AABB to make sure rays start outside the actual object.
    :param max_sampling_attempts: int, how many times sampling will be attempted for each requested point.
    :param max_angle_with_z_axis: float, maximum angle between hit normal and positive Z axis allowed. Can be used to
        disallow downward-facing hits when refuse_downwards=True.
    :param hit_to_plane_threshold: float, how far any given hit position can be from the least-squares fit plane to
        all of the hit positions before the sample is rejected.
    :param refuse_downwards: bool, whether downward-facing hits (as defined by max_angle_with_z_axis) are allowed.
    :return: List of num_samples elements where each element is a tuple in the form of
        (cuboid_centroid, cuboid_up_vector, cuboid_rotation, {refusal_reason: [refusal_details...]}). Cuboid positions
        are set to None when no successful sampling happens within the max number of attempts. Refusal details are only
        filled if the debug_sampling flag is globally set to True.
    """
    bbox_center, bbox_orn, bbox_bf_extent, _ = obj.get_base_aligned_bounding_box(xy_aligned=True, fallback_to_aabb=True)
    half_extent_with_offset = (bbox_bf_extent / 2) + aabb_offset

    body_ids = obj.get_body_ids()

    cuboid_dimensions = np.array(cuboid_dimensions)
    assert cuboid_dimensions.ndim <= 2
    assert cuboid_dimensions.shape[-1] == 3, "Cuboid dimensions need to contain all three dimensions."
    if cuboid_dimensions.ndim == 2:
        assert cuboid_dimensions.shape[0] == num_samples, "Need as many offsets as samples requested."

    results = [(None, None, None, None, defaultdict(list)) for _ in range(num_samples)]

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

        refusal_reasons = results[i][4]

        # Try each sampled position in the AABB.
        for axis, is_top, start_pos in samples:
            # Compute the ray's destination using the sampling & AABB information.
            point_on_face = compute_ray_destination(
                axis, is_top, start_pos, -half_extent_with_offset, half_extent_with_offset
            )

            # If we have a list of offset distances, pick the distance for this particular sample we're getting.
            this_cuboid_dimensions = cuboid_dimensions if cuboid_dimensions.ndim == 1 else cuboid_dimensions[i]

            # Obtain the parallel rays using the direction sampling method.
            bbf_sources, bbf_destinations, grid = get_parallel_rays(
                start_pos, point_on_face, this_cuboid_dimensions[:2] / 2.0
            )

            # Transform the sources, destinations and grid to the world frame coordinates.
            to_wf_transform = utils.quat_pos_to_mat(bbox_center, bbox_orn)
            sources = trimesh.transformations.transform_points(bbf_sources, to_wf_transform)
            destinations = trimesh.transformations.transform_points(bbf_destinations, to_wf_transform)

            # Time to cast the rays.
            cast_results = p.rayTestBatch(rayFromPositions=sources, rayToPositions=destinations, numThreads=0)

            # for ray_start, ray_end in zip(sources, destinations):
            #     p.addUserDebugLine(ray_start, ray_end, lineWidth=4)

            threshold, hits = check_rays_hit_object(cast_results, body_ids, refusal_reasons["missed_object"], 0.6)
            if not threshold:
                continue

            filtered_cast_results = []
            center_idx = int(len(cast_results) / 2)
            filtered_center_idx = None
            center_hit = False
            for idx, hit in enumerate(hits):
                if hit:
                    filtered_cast_results.append(cast_results[idx])
                    if idx == center_idx:
                        center_hit = True
                        filtered_center_idx = len(filtered_cast_results) - 1

            # Only consider objects whose center idx has a ray hit
            if not center_hit:
                continue

            # Process the hit positions and normals.
            hit_positions = np.array([ray_res[3] for ray_res in filtered_cast_results])
            hit_normals = np.array([ray_res[4] for ray_res in filtered_cast_results])
            hit_normals /= np.linalg.norm(hit_normals, axis=1)[:, np.newaxis]

            assert filtered_center_idx
            hit_link = filtered_cast_results[filtered_center_idx][1]
            center_hit_normal = hit_normals[filtered_center_idx]

            # Reject anything facing more than 45deg downwards if requested.
            if refuse_downwards:
                if not check_hit_max_angle_from_z_axis(
                    center_hit_normal, max_angle_with_z_axis, refusal_reasons["downward_normal"]
                ):
                    continue

            # Check that none of the parallel rays' hit normal differs from center ray by more than threshold.
            if not check_normal_similarity(center_hit_normal, hit_normals, refusal_reasons["hit_normal_similarity"]):
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
                center_hit_normal, plane_normal[None, :], refusal_reasons["plane_normal_similarity"]
            ):
                continue

            # Check that the points are all within some acceptable distance of the plane.
            distances = get_distance_to_plane(hit_positions, plane_centroid, plane_normal)
            if np.any(distances > hit_to_plane_threshold):
                if igibson.debug_sampling:
                    refusal_reasons["dist_to_plane"].append("distances to plane: %r" % (distances,))
                continue

            # Get projection of the base onto the plane, fit a rotation, and compute the new center hit / corners.
            hit_positions = np.array([ray_res[3] for ray_res in cast_results])
            projected_hits = get_projection_onto_plane(hit_positions, plane_centroid, plane_normal)
            padding = _DEFAULT_CUBOID_BOTTOM_PADDING * plane_normal
            projected_hits += padding
            center_projected_hit = projected_hits[center_idx]
            cuboid_centroid = center_projected_hit + plane_normal * this_cuboid_dimensions[2] / 2.0
            rotation = compute_rotation_from_grid_sample(grid, projected_hits, cuboid_centroid, this_cuboid_dimensions)
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

            # Now we use the cuboid's diagonals to check that the cuboid is actually empty.
            if not check_cuboid_empty(
                plane_normal, corner_positions, refusal_reasons["cuboid_not_empty"], this_cuboid_dimensions
            ):
                continue

            if undo_padding:
                cuboid_centroid -= padding

            # We've found a nice attachment point. Continue onto next point to sample.
            results[i] = (cuboid_centroid, plane_normal, rotation.as_quat(), hit_link, refusal_reasons)
            break

    if igibson.debug_sampling:
        print("Sampling rejection reasons:")
        counter = Counter()

        for instance in results:
            for reason, refusals in instance[-1].items():
                counter[reason] += len(refusals)

        print("\n".join("%s: %d" % pair for pair in counter.items()))

    return results


def compute_rotation_from_grid_sample(two_d_grid, hit_positions, cuboid_centroid, this_cuboid_dimensions):
    # TODO: Figure out if the normalization has any advantages.
    grid_in_planar_coordinates = two_d_grid.reshape(-1, 2)
    grid_in_object_coordinates = np.zeros((len(grid_in_planar_coordinates), 3))
    grid_in_object_coordinates[:, :2] = grid_in_planar_coordinates
    grid_in_object_coordinates[:, 2] = -this_cuboid_dimensions[2] / 2.0
    grid_in_object_coordinates /= np.linalg.norm(grid_in_object_coordinates, axis=1)[:, None]

    sampled_grid_relative_vectors = hit_positions - cuboid_centroid
    sampled_grid_relative_vectors /= np.linalg.norm(sampled_grid_relative_vectors, axis=1)[:, None]

    # Grab the vectors that are nonzero in both
    nonzero_indices = np.logical_and(
        np.any(np.isfinite(grid_in_object_coordinates), axis=1),
        np.any(np.isfinite(sampled_grid_relative_vectors), axis=1),
    )
    grid_in_object_coordinates = grid_in_object_coordinates[nonzero_indices]
    sampled_grid_relative_vectors = sampled_grid_relative_vectors[nonzero_indices]

    rotation, _ = Rotation.align_vectors(sampled_grid_relative_vectors, grid_in_object_coordinates)
    return rotation


def check_normal_similarity(center_hit_normal, hit_normals, refusal_log):
    parallel_hit_main_hit_dot_products = np.clip(
        np.dot(hit_normals, center_hit_normal)
        / (np.linalg.norm(hit_normals, axis=1) * np.linalg.norm(center_hit_normal)),
        -1.0,
        1.0,
    )
    parallel_hit_normal_angles_to_hit_normal = np.arccos(parallel_hit_main_hit_dot_products)
    all_rays_hit_with_similar_normal = np.all(
        parallel_hit_normal_angles_to_hit_normal < _PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE
    )
    if not all_rays_hit_with_similar_normal:
        if igibson.debug_sampling:
            refusal_log.append("angles %r" % (np.rad2deg(parallel_hit_normal_angles_to_hit_normal),))

        return False

    return True


def check_rays_hit_object(cast_results, body_ids, refusal_log, threshold=1.0):
    hit_body_ids = [ray_res[0] for ray_res in cast_results]
    ray_hits = list(hit_body_id in body_ids for hit_body_id in hit_body_ids)
    if not (sum(ray_hits) / len(hit_body_ids)) >= threshold:
        if igibson.debug_sampling:
            refusal_log.append("hits %r" % hit_body_ids)

        return False, ray_hits

    return True, ray_hits


def check_hit_max_angle_from_z_axis(hit_normal, max_angle_with_z_axis, refusal_log):
    hit_angle_with_z = np.arccos(np.clip(np.dot(hit_normal, np.array([0, 0, 1])), -1.0, 1.0))
    if hit_angle_with_z > max_angle_with_z_axis:
        if igibson.debug_sampling:
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


def check_cuboid_empty(hit_normal, bottom_corner_positions, refusal_log, this_cuboid_dimensions):
    if igibson.debug_sampling:
        draw_debug_markers(bottom_corner_positions)

    # Compute top corners.
    top_corner_positions = bottom_corner_positions + hit_normal * this_cuboid_dimensions[2]

    # Get all the top-to-bottom corner pairs. When we cast these rays, we check for two things: that the cuboid
    # height is actually available, and the faces & volume of the cuboid are unoccupied.
    top_to_bottom_pairs = list(itertools.product(top_corner_positions, bottom_corner_positions))

    # Get all the same-height pairs. These also check that the surfaces areas are empty.
    bottom_pairs = list(itertools.combinations(bottom_corner_positions, 2))
    top_pairs = list(itertools.combinations(top_corner_positions, 2))

    # Combine all these pairs, cast the rays, and make sure the rays don't hit anything.
    all_pairs = np.array(top_to_bottom_pairs + bottom_pairs + top_pairs)
    check_cast_results = p.rayTestBatch(
        rayFromPositions=all_pairs[:, 0, :], rayToPositions=all_pairs[:, 1, :], numThreads=0
    )
    if not all(ray[0] == -1 for ray in check_cast_results):
        if igibson.debug_sampling:
            refusal_log.append("check ray info: %r" % (check_cast_results,))

        return False

    return True
