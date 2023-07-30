from collections import namedtuple

import numpy as np

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.utils.sampling_utils import raytest_batch


# Create settings for this module
m = create_module_macros(module_path=__file__)
m.MAX_DISTANCE_VERTICAL = 5.0
m.MAX_DISTANCE_HORIZONTAL = 5.0

# How many 2-D bases to try during horizontal adjacency check. When 1, only the standard axes will be considered.
# When 2, standard axes + 45 degree rotated will be considered. The tried axes will be equally spaced. The higher
# this number, the lower the possibility of false negatives in Inside and NextTo.
m.HORIZONTAL_AXIS_COUNT = 5

AxisAdjacencyList = namedtuple("AxisAdjacencyList", ("positive_neighbors", "negative_neighbors"))


def flatten_planes(planes):
    # Converts the body-by-plane logic to a flat body-by-axis setup,
    # for when we don't care about the axes' relationship with each other.
    return (axis for axes_by_plane in planes for axis in axes_by_plane)


def get_equidistant_coordinate_planes(n_planes):
    """Given a number, sample that many equally spaced coordinate planes.

    The samples will cover all 360 degrees (although rotational symmetry
    is assumed, e.g. if you take into account the axis index and the
    positive/negative directions, only 1/4 of the possible coordinate (1 quadrant, np.pi / 2.0)
    planes will be sampled: the ones where the first axis' positive direction
    is in the first quadrant).

    Args:
        n_planes (int): number of planes to sample

    Returns:
        3D-array: (n_planes, 2, 3) array where the first dimension
            is the sampled plane index, the second dimension is the axis index
            (0/1), and the third dimension is the 3-D world-coordinate vector
            corresponding to the axis.
    """
    # Compute the positive directions of the 1st axis of each plane.
    first_axis_angles = np.linspace(0, np.pi / 2, n_planes)
    first_axes = np.stack(
        [np.cos(first_axis_angles), np.sin(first_axis_angles), np.zeros_like(first_axis_angles)], axis=1
    )

    # Compute the positive directions of the 2nd axes. These axes are
    # orthogonal to both their corresponding first axes and to the Z axis.
    second_axes = np.cross([0, 0, 1], first_axes)

    # Return the axes in the shape (n_planes, 2, 3)
    return np.stack([first_axes[:, None, :], second_axes[:, None, :]], axis=1)


def compute_adjacencies(obj, axes, max_distance):
    """
    Given an object and a list of axes, find the adjacent objects in the axes'
    positive and negative directions.

    Args:
        obj (StatefulObject): The object to check adjacencies of.
        axes (2D-array): (n_axes, 3) array defining the axes to check in.
            Note that each axis will be checked in both its positive and negative direction.

    Returns:
        list of AxisAdjacencyList: List of length len(axes) containing the adjacencies.
    """
    # Get vectors for each of the axes' directions.
    # The ordering is axes1+, axis1-, axis2+, axis2- etc.
    directions = np.empty((len(axes) * 2, 3))
    directions[0::2] = axes
    directions[1::2] = -axes

    # Prepare this object's info for ray casting.
    # Use AABB center instead of position because we cannot get valid position
    # for fixed objects if fixed links are merged.
    aabb_lower, aabb_higher = obj.states[AABB].get_value()
    object_position = (aabb_lower + aabb_higher) / 2.0
    prim_paths = obj.link_prim_paths

    # Prepare the rays to cast.
    ray_starts = np.tile(object_position, (len(directions), 1))
    ray_endpoints = ray_starts + (directions * max_distance)

    # Cast time.
    ray_results = raytest_batch(
        ray_starts,
        ray_endpoints,
        only_closest=False,
        ignore_bodies=prim_paths,
        ignore_collisions=prim_paths
    )

    # Add the results to the appropriate lists
    # For now, we keep our result in the dimensionality of (direction, hit_object_order).
    # We convert the hit link into unique objects encountered
    objs_by_direction = []
    for results in ray_results:
        unique_objs = set()
        for result in results:
            # Check if the inferred hit object is not None, we add it to our set
            obj_prim_path = "/".join(result["rigidBody"].split("/")[:-1])
            obj = og.sim.scene.object_registry("prim_path", obj_prim_path, None)
            if obj is not None:
                unique_objs.add(obj)
        objs_by_direction.append(unique_objs)

    # Reshape so that these have the following indices:
    # (axis_idx, direction-one-or-zero, hit_idx)
    objs_by_axis = [
        AxisAdjacencyList(positive_neighbors, negative_neighbors)
        for positive_neighbors, negative_neighbors in zip(objs_by_direction[::2], objs_by_direction[1::2])
    ]
    return objs_by_axis


class VerticalAdjacency(AbsoluteObjectState):
    """
    State representing the object's vertical adjacencies.
    Value is a AxisAdjacencyList object.
    """

    def _get_value(self):
        # Call the adjacency computation with th Z axis.
        bodies_by_axis = compute_adjacencies(self.obj, np.array([[0, 0, 1]]), m.MAX_DISTANCE_VERTICAL)

        # Return the adjacencies from the only axis we passed in.
        return bodies_by_axis[0]

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)
        return deps

    # Nothing needs to be done to save/load adjacency since it will happen due to pose caching.


class HorizontalAdjacency(AbsoluteObjectState):
    """
    State representing the object's horizontal adjacencies in a preset number of directions.

    The HorizontalAdjacency state returns adjacency lists for equally spaced coordinate planes.
    Each plane consists of 2 orthogonal axes, and adjacencies are checked for both the positive
    and negative directions of each axis.

    The value of the state is List[List[AxisAdjacencyList]], where the list dimensions are
    m.HORIZONTAL_AXIS_COUNT and 2. The first index is used to choose between the different planes,
    the second index to choose between the orthogonal axes of that plane. Given a plane/axis combo,
    the item in the list is a AxisAdjacencyList containing adjacencies in both directions of the
    axis.

    If the idea of orthogonal bases is not relevant (and your use case simply requires checking
    adjacencies in each direction), the flatten_planes() function can be used on the state value
    to reduce the output to List[AxisAdjacencyList], a list of adjacency lists for all
    2 * m.HORIZONTAL_AXIS_COUNT directions.
    """

    def _get_value(self):
        coordinate_planes = get_equidistant_coordinate_planes(m.HORIZONTAL_AXIS_COUNT)

        # Flatten the axis dimension and input into compute_adjacencies.
        bodies_by_axis = compute_adjacencies(self.obj, coordinate_planes.reshape(-1, 3), m.MAX_DISTANCE_HORIZONTAL)

        # Now reshape the bodies_by_axis to group by coordinate planes.
        bodies_by_plane = list(zip(bodies_by_axis[::2], bodies_by_axis[1::2]))

        # Return the adjacencies.
        return bodies_by_plane

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)
        return deps

    # Nothing needs to be done to save/load adjacency since it will happen due to pose caching.
