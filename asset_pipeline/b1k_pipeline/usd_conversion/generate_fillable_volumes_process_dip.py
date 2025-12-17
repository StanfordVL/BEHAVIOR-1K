import traceback
import torch as th
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.asset_utils import decrypted
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
import trimesh
import tqdm

gm.HEADLESS = True
gm.USE_ENCRYPTED_ASSETS = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False

MAX_BBOX = 0.3


def find_largest_connected_component(points, d):
    # Create a KDTree for efficient nearest neighbor search
    points_np = points.numpy().copy()
    tree = KDTree(points_np)

    # Find pairs of points within distance d
    pairs = tree.query_pairs(r=d, output_type="ndarray")

    # Create an adjacency matrix for the graph
    n_points = points.shape[0]
    adjacency_matrix = csr_matrix(
        (np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])),
        shape=(n_points, n_points),
    )

    # Make the matrix symmetric since the graph is undirected
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T

    # Find connected components
    _, labels = connected_components(
        csgraph=adjacency_matrix, directed=False, return_labels=True
    )

    # Find the largest connected component
    largest_component_label = th.argmax(th.bincount(th.tensor(labels)))
    largest_component_indices = th.where(th.tensor(labels) == largest_component_label)[
        0
    ]

    # Return the points belonging to the largest connected component
    return points[largest_component_indices]


def generate_box(box_half_extent):
    # The floor plane already exists
    # We just need to generate the side planes
    plane_centers = (
        th.tensor(
            [
                [1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
            ]
        )
        * box_half_extent
    )
    for i, pc in enumerate(plane_centers):
        plane = lazy.omni.isaac.core.objects.ground_plane.GroundPlane(
            prim_path=f"/World/plane_{i}",
            name=f"plane_{i}",
            z_position=0,
            size=box_half_extent[2].item(),
            color=None,
            visible=False,
            # TODO: update with new PhysicsMaterial API
            # static_friction=static_friction,
            # dynamic_friction=dynamic_friction,
            # restitution=restitution,
        )

        plane_as_prim = XFormPrim(
            relative_prim_path=f"/plane_{i}",
            name=plane.name,
        )
        plane_as_prim.load(None)

        # Build the plane orientation from the plane normal
        horiz_dir = pc - th.tensor([0, 0, box_half_extent[2]])
        plane_z = -1 * horiz_dir / th.norm(horiz_dir)
        plane_x = th.tensor([0, 0, 1], dtype=th.float32)
        plane_y = th.cross(plane_z, plane_x)
        plane_mat = th.stack([plane_x, plane_y, plane_z], dim=1)
        plane_quat = T.mat2quat(plane_mat)
        plane_as_prim.set_position_orientation(pc, plane_quat)


def generate_particles_in_box(water, box_half_extent):
    particle_radius = water.particle_radius

    # Grab the link's AABB (or fallback to obj AABB if link does not have a valid AABB),
    # and generate a grid of points based on the sampling distance
    low = th.tensor([-1, -1, 0]) * box_half_extent
    high = th.tensor([1, 1, 4]) * box_half_extent + th.tensor([0, 0, 0.05])
    extent = high - low
    # We sample the range of each extent minus
    sampling_distance = 2 * particle_radius
    n_particles_per_axis = (extent / sampling_distance).long()
    assert th.all(
        n_particles_per_axis > 0
    ), f"box is too small to sample any particle of radius {particle_radius}."

    # 1e-10 is added because the extent might be an exact multiple of particle radius
    arrs = [
        th.arange(l + particle_radius, h - particle_radius + 1e-10, particle_radius * 2)
        for l, h, n in zip(low, high, n_particles_per_axis)
    ]
    # Generate 3D-rectangular grid of points
    particle_positions = th.stack(th.meshgrid(*arrs, indexing="ij")).view(3, -1).t()

    water.generate_particles(
        positions=particle_positions,
    )


def draw_mesh(mesh, parent_pos):
    draw = lazy.omni.isaac.debug_draw._debug_draw.acquire_debug_draw_interface()
    edge_vert_idxes = mesh.edges_unique
    N = len(edge_vert_idxes)
    colors = [(1.0, 0.0, 0.0, 1.0) for _ in range(N)]
    sizes = [1.0 for _ in range(N)]
    points1 = [
        tuple(x)
        for x in (th.tensor(mesh.vertices[edge_vert_idxes[:, 0]]) + parent_pos).tolist()
    ]
    points2 = [
        tuple(x)
        for x in (th.tensor(mesh.vertices[edge_vert_idxes[:, 1]]) + parent_pos).tolist()
    ]
    draw.draw_lines(points1, points2, colors, sizes)


def check_in_contact(system, positions):
    """
    Checks whether each particle specified by @particle_positions are in contact with any rigid body.

    Args:
        positions (th.Tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions

    Returns:
        n-array: (n_particles,) boolean array, True if in contact, otherwise False
    """
    in_contact = th.zeros(len(positions), dtype=bool)
    for idx, pos in enumerate(positions):
        in_contact[idx] = og.sim.psqi.overlap_sphere_any(
            system.particle_contact_radius * 0.8, pos.numpy().copy()
        )
    return in_contact


def process_object(cat, mdl, out_path):
    if og.sim:
        og.clear()
    else:
        og.launch()

    if og.sim.is_playing():
        og.sim.stop()

    # First get the native bounding box of the object
    usd_path = DatasetObject.get_usd_path(category=cat, model=mdl)
    usd_path = usd_path.replace(".usd", ".encrypted.usd")
    with decrypted(usd_path) as fpath:
        stage = lazy.pxr.Usd.Stage.Open(fpath)
        prim = stage.GetDefaultPrim()
        bounding_box = th.tensor(prim.GetAttribute("ig:nativeBB").Get())

    # Then get the appropriate bounding box such that the object fits in a
    # MAX_BBOX cube.
    scale = MAX_BBOX / th.max(bounding_box)

    if scale > 1:
        print(
            "The object won't be scaled because it's smaller than the requested bounding box."
        )
        scale = 1

    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "fillable",
                "category": cat,
                "model": mdl,
                "kinematic_only": False,
                "fixed_base": True,
                "scale": [scale, scale, scale],
            },
        ],
    }

    env = og.Environment(configs=cfg)
    og.sim.step()

    fillable = env.scene.object_registry("name", "fillable")

    # Fix all joints to upper position
    joint_limits = {}
    for joint_name, joint in fillable.joints.items():
        if joint.has_limit:
            joint_limits[joint_name] = (joint.lower_limit, joint.upper_limit)
            joint.set_pos(joint.upper_limit)
            joint.lower_limit = joint.upper_limit - 0.001
    og.sim.update_handles()

    # Now move it into position
    aabb_extent = th.tensor(fillable.aabb_extent)
    obj_bbox_height = aabb_extent[2]
    obj_bbox_center = th.tensor(fillable.aabb_center)
    obj_bbox_bottom = obj_bbox_center - th.tensor([0, 0, obj_bbox_height / 2])
    obj_current_pos = th.tensor(fillable.get_position_orientation()[0])
    obj_pos_wrt_bbox = obj_current_pos - obj_bbox_bottom
    obj_dipped_pos = obj_pos_wrt_bbox
    obj_free_pos = obj_pos_wrt_bbox + th.tensor([0, 0, 2 * aabb_extent[2] + 0.2])
    fillable.set_position_orientation(obj_free_pos, th.tensor([0, 0, 0, 1]))
    og.sim.step()

    # Now generate the box and the particles
    water = env.scene.get_system("water")
    box_half_extent = aabb_extent * 0.55
    box_half_extent = th.maximum(
        box_half_extent, aabb_extent / 2 + 2.1 * water.particle_radius
    )
    generate_box(box_half_extent)
    og.sim.step()
    generate_particles_in_box(water, box_half_extent)
    for _ in range(100):
        og.sim.step()

    # Move the object down into the box slowly
    # lin_vel = 0.2
    # while True:
    #     delta_z = -lin_vel * og.sim.get_rendering_dt()
    #     cur_pos = fillable.get_position_orientation()[0]
    #     new_pos = cur_pos + np.array([0, 0, delta_z])
    #     fillable.set_position_orientation(position=new_pos)
    #     og.sim.step()
    #     if fillable.get_position_orientation()[0][2] < obj_dipped_pos[2]:
    #         break
    fillable.set_position_orientation(position=obj_dipped_pos)

    # Let the particles settle
    for _ in range(100):
        og.sim.step()

    # Slowly close the doors linearly
    n_steps_for_close = 100
    for i in range(n_steps_for_close):
        for joint_name, joint in fillable.joints.items():
            openness_ratio = i / n_steps_for_close
            if joint_name in joint_limits:
                lower, upper = joint_limits[joint_name]
                interpolated_pos = upper - openness_ratio * (upper - lower)
                joint.lower_limit = interpolated_pos - 0.001
                joint.upper_limit = interpolated_pos
                joint.set_pos(interpolated_pos)
        og.sim.step()

    # Wait here a bit
    for _ in range(100):
        og.sim.step()

    # Now move the object out of the water
    lin_vel = 0.01
    while True:
        delta_z = lin_vel * og.sim.get_rendering_dt()
        cur_pos = th.tensor(fillable.get_position_orientation()[0])
        new_pos = cur_pos + th.tensor([0, 0, delta_z])
        fillable.set_position_orientation(position=new_pos)
        og.sim.step()
        if fillable.get_position_orientation()[0][2] > obj_free_pos[2]:
            break

    for _ in range(180):
        og.sim.step()

    # # Temporarily use a fixed shakeoff. TODO: Fix the math below.
    # # Gentle side-by-side shakeoff
    # spill_fraction = 0.05
    # extents = aabb_extent[:2]
    # # formula for how much to rotate for total spill to be spill_fraction of the volume
    # angles = np.arctan(extents / (2 * aabb_extent[2] * spill_fraction))
    # angles = np.flip(angles)
    # angles = np.full((2,), np.deg2rad(10))

    # print("Rotation amounts (degrees): ", np.rad2deg(angles))

    # rotations = np.array([np.eye(3)[i] * angle * side for i, angle in enumerate(angles) for side in [-1, 1]])
    # for r in rotations:
    #     total_steps = 60
    #     for _ in range(total_steps):
    #         delta_orn = R.from_euler("xyz", r / total_steps)
    #         cur_rot = R.from_quat(fillable.get_orientation())
    #         new_rot = delta_orn * cur_rot
    #         fillable.set_orientation(new_rot.as_quat())
    #         og.sim.step()
    #     for _ in range(90):
    #         og.sim.step()
    #     for _ in range(total_steps):
    #         delta_orn = R.from_euler("xyz", -r / total_steps)
    #         cur_rot = R.from_quat(fillable.get_orientation())
    #         new_rot = delta_orn * cur_rot
    #         fillable.set_orientation(new_rot.as_quat())
    #         og.sim.step()
    #     for _ in range(90):
    #         og.sim.step()

    # Get the particles whose center is within the object's AABB
    aabb_min, aabb_max = fillable.aabb
    particles = th.tensor(water.get_particles_position_orientation()[0])
    particles = particles[th.where(check_in_contact(water, particles) == 0)[0]]
    particle_point_offsets = th.stack(
        [e * side * water.particle_radius for e in th.eye(3) for side in [-1, 1]]
        + [th.zeros(3)]
    )
    points = particles.unsqueeze(1).repeat(
        1, len(particle_point_offsets), 1
    ) + particle_point_offsets.unsqueeze(0)
    points = points.reshape(-1, 3)
    points = points[
        th.all(points <= (aabb_max + th.tensor([0, 0, water.particle_radius])), dim=1)
    ]
    points = points[th.all(points >= aabb_min, dim=1)]
    assert len(points) > 0, "No points found in the AABB of the object."

    # Get the points that belong to the largest connected component
    points = find_largest_connected_component(points, 2 * water.particle_radius)

    # Get the particles in the frame of the object
    points -= fillable.get_position_orientation()[0]

    # Get the convex hull of the particles
    hull = trimesh.convex.convex_hull(points.numpy().copy())

    # Save it somewhere
    hull.export(
        out_path,
        file_type="obj",
        include_normals=False,
        include_color=False,
        include_texture=False,
    )

    # draw_mesh(hull, fillable.get_position_orientation()[0])
    # while True:
    #     og.sim.render()

    # from omnigibson.utils.deprecated_utils import CreateMeshPrimWithDefaultXformCommand
    # container_prim_path = fillable.root_link.prim_path + "/container"
    # CreateMeshPrimWithDefaultXformCommand(prim_path=container_prim_path, prim_type="Mesh", trimesh_mesh=hull).do()
    # mesh_prim = XFormPrim(name="container", prim_path=container_prim_path)


def main():
    import sys, pathlib

    dataset_root = str(pathlib.Path(sys.argv[1]))
    gm.DATASET_PATH = str(dataset_root)

    batch = sys.argv[2:]
    for path in batch:
        obj_category, obj_model = pathlib.Path(path).parts[-2:]
        obj_dir = pathlib.Path(dataset_root) / "objects" / obj_category / obj_model
        assert obj_dir.exists()
        print(f"Processing {path}")
        out_path = obj_dir / "fillable_dip.obj"
        try:
            process_object(obj_category, obj_model, out_path)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
