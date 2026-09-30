import numpy as np
import pytest
import torch as th
import trimesh

from omnigibson.utils.usd_utils import _min_distance_to_mesh_surface


def _to_tensors(mesh):
    return (
        th.tensor(mesh.vertices, dtype=th.float32),
        th.tensor(mesh.faces, dtype=th.int64),
    )


# A representative spread of primitive shapes, including thin / oblong ones where the closest surface point is
# expected to lie well away from the nearest vertex.
MESHES = {
    "box": trimesh.creation.box(extents=[1.0, 0.5, 2.0]),
    "icosphere": trimesh.creation.icosphere(subdivisions=3, radius=1.0),
    "cylinder": trimesh.creation.cylinder(radius=0.3, height=1.0, sections=64),
    "cone": trimesh.creation.cone(radius=0.5, height=1.2, sections=48),
    "torus": trimesh.creation.torus(major_radius=1.0, minor_radius=0.25),
    "thin_plate": trimesh.creation.box(extents=[2.0, 0.02, 1.0]),
    "long_rod": trimesh.creation.box(extents=[0.5, 0.5, 5.0]),
}


@pytest.mark.parametrize("mesh_name", list(MESHES))
def test_min_distance_to_mesh_surface_matches_trimesh(mesh_name):
    """The computed distance should match trimesh's closest-point-on-surface distance."""
    mesh = MESHES[mesh_name]
    points, faces = _to_tensors(mesh)

    # Query from the mesh's center of mass, its bounding box center, and a vertex
    query_points = [
        th.tensor(mesh.center_mass, dtype=th.float32),
        th.tensor(mesh.bounds.mean(axis=0), dtype=th.float32),
        th.tensor(mesh.vertices[0], dtype=th.float32),
    ]
    for point in query_points:
        distance = _min_distance_to_mesh_surface(points, faces, point).item()
        expected = trimesh.proximity.closest_point(mesh, np.array([point.numpy()]))[1][0]
        assert np.isclose(distance, expected, atol=1e-3), f"{mesh_name}: got {distance}, expected {expected}"


def test_min_distance_to_mesh_surface_is_not_just_vertices():
    """Regression test for issue #773: the closest surface point need not be a vertex."""
    mesh = MESHES["thin_plate"]
    points, faces = _to_tensors(mesh)
    point = th.tensor(mesh.center_mass, dtype=th.float32)

    surface_distance = _min_distance_to_mesh_surface(points, faces, point).item()
    vertex_distance = th.norm(points - point, dim=-1).min().item()

    # For a thin plate, the nearest vertex is far but the surface is just half the thickness away
    assert surface_distance < vertex_distance
    assert np.isclose(surface_distance, mesh.extents.min() / 2.0, atol=1e-4)
