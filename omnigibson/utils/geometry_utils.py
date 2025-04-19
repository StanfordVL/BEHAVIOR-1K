"""
A set of helper utility functions for dealing with 3D geometry
"""

import math

import torch as th

import omnigibson.utils.transform_utils as T


def wrap_angle(theta):
    """ "
    Converts an angle to the range [-pi, pi).

    Args:
        theta (float): angle in radians

    Returns:
        float: angle in radians in range [-pi, pi)
    """
    return (theta + math.pi) % (2 * math.pi) - math.pi


def get_particle_positions_in_frame(pos, quat, scale, particle_positions):
    """
    Transforms particle positions @positions into the frame specified by @pos and @quat with new scale @scale,
    where @pos and @quat are assumed to be specified in the same coordinate frame that @particle_positions is specified

    Args:
        pos (3-array): (x,y,z) pos of the new frame
        quat (4-array): (x,y,z,w) quaternion orientation of the new frame
        scale (3-array): (x,y,z) local scale of the new frame
        particle_positions ((N, 3) array): positions

    Returns:
        (N,) array: updated particle positions in the new coordinate frame
    """

    # Get pose of origin (global frame) in new_frame
    origin_in_new_frame = T.pose_inv(T.pose2mat((pos, quat)))
    # Batch the transforms to get all particle points in the local link frame
    positions_tensor = th.tile(th.eye(4).reshape(1, 4, 4), (len(particle_positions), 1, 1))  # (N, 4, 4)
    # Scale by the new scale#
    positions_tensor[:, :3, 3] = particle_positions
    particle_positions = (origin_in_new_frame @ positions_tensor)[:, :3, 3]  # (N, 3)
    # Scale by the new scale
    return particle_positions / scale.reshape(1, 3)


def get_particle_positions_from_frame(pos, quat, scale, particle_positions):
    """
    Transforms particle positions @positions from the frame specified by @pos and @quat with new scale @scale.

    This is similar to @get_particle_positions_in_frame, but does the reverse operation, inverting @pos and @quat

    Args:
        pos (3-array): (x,y,z) pos of the local frame
        quat (4-array): (x,y,z,w) quaternion orientation of the local frame
        scale (3-array): (x,y,z) local scale of the local frame
        particle_positions ((N, 3) array): positions

    Returns:
        (N,) array: updated particle positions in the parent coordinate frame
    """
    # Scale by the new scale
    particle_positions = particle_positions * scale.reshape(1, 3)

    # Get pose of origin (global frame) in new_frame
    origin_in_new_frame = T.pose2mat((pos, quat))
    # Batch the transforms to get all particle points in the local link frame
    positions_tensor = th.tile(th.eye(4).reshape(1, 4, 4), (len(particle_positions), 1, 1))  # (N, 4, 4)
    # Scale by the new scale#
    positions_tensor[:, :3, 3] = particle_positions
    return (origin_in_new_frame @ positions_tensor)[:, :3, 3]  # (N, 3)


def check_points_in_cube(size, particle_positions):
    """
    Checks which points are within a cube with specified size @size.

    NOTE: Assumes the cube and positions are expressed
    in the same coordinate frame such that the cube's dimensions are axis-aligned with (x,y,z)

    Args:
        size float: length of each side of the cube, specified in its local frame
        particle_positions ((N, 3) array): positions to check for whether it is in the cube

    Returns:
        (N,) array: boolean numpy array specifying whether each point lies in the cube.
    """
    return ((-size / 2.0 < particle_positions) & (particle_positions < size / 2.0)).sum(dim=-1) == 3


def check_points_in_cone(size, particle_positions):
    """
    Checks which points are within a cone with specified size @size.

    NOTE: Assumes the cone and positions are
    expressed in the same coordinate frame such that the cone's height is aligned with the z-axis

    Args:
        size (2-array): (radius, height) dimensions of the cone, specified in its local frame
        particle_positions ((N, 3) array): positions to check for whether it is in the cone

    Returns:
        (N,) array: boolean numpy array specifying whether each point lies in the cone.
    """
    radius, height = size
    in_height = (-height / 2.0 < particle_positions[:, -1]) & (particle_positions[:, -1] < height / 2.0)
    in_radius = th.norm(particle_positions[:, :-1], dim=-1) < (
        radius * (1 - (particle_positions[:, -1] + height / 2.0) / height)
    )
    return in_height & in_radius


def check_points_in_cylinder(size, particle_positions):
    """
    Checks which points are within a cylinder with specified size @size.

    NOTE: Assumes the cylinder and positions are
    expressed in the same coordinate frame such that the cylinder's height is aligned with the z-axis

    Args:
        size (2-array): (radius, height) dimensions of the cylinder, specified in its local frame
        particle_positions ((N, 3) array): positions to check for whether it is in the cylinder

    Returns:
        (N,) array: boolean numpy array specifying whether each point lies in the cylinder.
    """
    radius, height = size
    in_height = (-height / 2.0 < particle_positions[:, -1]) & (particle_positions[:, -1] < height / 2.0)
    in_radius = th.norm(particle_positions[:, :-1], dim=-1) < radius
    return in_height & in_radius


def check_points_in_sphere(size, particle_positions):
    """
    Checks which points are within a sphere with specified size @size.

    NOTE: Assumes the sphere and positions are expressed in the same coordinate frame

    Args:
        size (float): radius dimensions of the sphere
        particle_positions ((N, 3) array): positions to check for whether it is in the sphere

    Returns:
        (N,) array: boolean numpy array specifying whether each point lies in the sphere
    """
    return th.norm(particle_positions, dim=-1) < size


def check_points_in_convex_hull_mesh(mesh_face_centroids, mesh_face_normals, particle_positions):
    """
    Checks which points are within a sphere with specified size @size.

    NOTE: Assumes the mesh and positions are expressed in the same coordinate frame

    Args:
        mesh_face_centroids (D, 3): (x,y,z) location of the centroid of each mesh face, expressed in its local frame
        mesh_face_normals (D, 3): (x,y,z) normalized direction vector of each mesh face, expressed in its local frame
        particle_positions ((N, 3) array): positions to check for whether it is in the mesh

    Returns:
        (N,) array: boolean numpy array specifying whether each point lies in the mesh
    """
    # For every mesh point / normal and particle position pair, we check whether it is "inside" (i.e.: the point lies
    # BEHIND the normal plane -- this is easily done by taking the dot product with the vector from the point to the
    # particle position with the normal, and validating that the value is < 0)
    D, _ = mesh_face_centroids.shape
    N, _ = particle_positions.shape
    mesh_points = th.tile(mesh_face_centroids.reshape(1, D, 3), (N, 1, 1))
    mesh_normals = th.tile(mesh_face_normals.reshape(1, D, 3), (N, 1, 1))
    particle_positions = th.tile(particle_positions.reshape(N, 1, 3), (1, D, 1))
    # All arrays are now (N, D, 3) shape -- efficient for batching
    in_range = ((particle_positions - mesh_points) * mesh_normals).sum(dim=-1) < 0  # shape (N, D)
    # All D normals must be satisfied for a single point to be considered inside the hull
    in_range = in_range.sum(dim=-1) == D
    return in_range
