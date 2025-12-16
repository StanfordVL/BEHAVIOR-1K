import traceback
import torch as th
import omnigibson as og
from omnigibson.macros import gm, macros
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.asset_utils import decrypted
from omnigibson.utils.sampling_utils import raytest_batch, raytest
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
import trimesh
from scipy.spatial import ConvexHull

gm.HEADLESS = True
gm.USE_ENCRYPTED_ASSETS = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

MAX_BBOX = 0.3


def draw_mesh(mesh, parent_pos):
    draw = lazy.omni.isaac.debug_draw._debug_draw.acquire_debug_draw_interface()
    edge_vert_idxes = mesh.edges_unique
    N = len(edge_vert_idxes)
    colors = [(1.0, 0.0, 0.0, 1.0) for _ in range(N)]
    sizes = [1.0 for _ in range(N)]
    points1 = [
        tuple(x) for x in (mesh.vertices[edge_vert_idxes[:, 0]] + parent_pos).tolist()
    ]
    points2 = [
        tuple(x) for x in (mesh.vertices[edge_vert_idxes[:, 1]] + parent_pos).tolist()
    ]
    draw.draw_lines(points1, points2, colors, sizes)


def sample_radial_rays(point, n=40, dist=4.0):
    angles = th.arange(n) * 2 * th.pi / n
    x = th.cos(angles)
    y = th.sin(angles)
    start_points = th.ones((n, 3)) * point.reshape(1, 3)
    end_points = th.stack([x, y, th.zeros(n)], dim=1) * dist + point.reshape(1, 3)
    return raytest_batch(start_points, end_points)


def sample_raytest_start_end_full_grid_topdown(
    obj,
    ray_spacing,
    aabb_offset=None,
    aabb_offset_fraction=macros.utils.sampling_utils.DEFAULT_AABB_OFFSET_FRACTION,
):
    bbox_center = obj.aabb_center
    bbox_orn = th.tensor([0, 0, 0, 1.0])
    bbox_bf_extent = obj.aabb_extent
    aabb_offset = (
        aabb_offset_fraction * bbox_bf_extent if aabb_offset is None else aabb_offset
    )
    # bbox_center, bbox_orn, bbox_bf_extent, _ = obj.get_base_aligned_bbox(xy_aligned=True, fallback_to_aabb=True)

    half_extent_with_offset = (bbox_bf_extent / 2) + aabb_offset
    x = th.linspace(
        -half_extent_with_offset[0],
        half_extent_with_offset[0],
        int(half_extent_with_offset[0] * 2 / ray_spacing) + 1,
    )
    y = th.linspace(
        -half_extent_with_offset[1],
        half_extent_with_offset[1],
        int(half_extent_with_offset[1] * 2 / ray_spacing) + 1,
    )
    n_rays = len(x) * len(y)

    start_points = th.stack(
        [
            x.repeat(len(y)),
            y.repeat_interleave(len(x)),
            th.ones(n_rays) * half_extent_with_offset[2],
        ]
    ).T

    end_points = start_points.clone()
    end_points[:, 2] = -half_extent_with_offset[2]

    # Convert the points into the world frame
    to_wf_transform = T.pose2mat((bbox_center, bbox_orn))
    start_points = th.tensor(
        trimesh.transformations.transform_points(start_points, to_wf_transform)
    )
    end_points = th.tensor(
        trimesh.transformations.transform_points(end_points, to_wf_transform)
    )

    start_points = start_points.unsqueeze(1)
    end_points = end_points.unsqueeze(1)

    return start_points, end_points


def sample_fillable_point_from_top(obj, n_rays, prop_successful):
    aabb_low, aabb_high = obj.aabb
    # Sample uniformly from the top of the AABB, and take the lowest z and randomly sample again
    start_rays, end_rays = sample_raytest_start_end_full_grid_topdown(
        obj, th.mean((aabb_high[:2] - aabb_low[:2]) / 10.0)
    )
    down_results = raytest_batch(start_rays.reshape(-1, 3), end_rays.reshape(-1, 3))
    down_hit_results = th.stack(
        [th.tensor(result["position"]) for result in down_results if result["hit"]]
    )

    z = down_hit_results[:, 2].min() + 0.01
    point = down_hit_results[th.argmin(down_hit_results[:, 2])]
    point[2] += 0.01
    results = sample_radial_rays(point, n=n_rays)

    if th.mean(
        th.tensor([result["hit"] for result in results], dtype=th.float32)
    ) >= prop_successful and th.all(
        th.tensor([result["distance"] > 0 for result in results if result["hit"]])
    ):
        xs = th.tensor([result["position"][0] for result in results if result["hit"]])
        ys = th.tensor([result["position"][1] for result in results if result["hit"]])
        center = th.tensor([xs.mean(), ys.mean(), z])

        return True, center

    return False, th.zeros(3)


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
                "kinematic_only": False,
                "fixed_base": True,
            },
        ],
    }

    env = og.Environment(configs=cfg)
    og.sim.step()

    fillable = env.scene.object_registry("name", "fillable")
    fillable.set_position([0, 0, 0])
    og.sim.step()

    # 1. - 2b. Start at center of AABB
    point = fillable.aabb_center
    aabb_center = fillable.aabb_center
    aabb_low, aabb_high = fillable.aabb
    avg_diameter = th.mean(fillable.aabb_extent[:2])
    success = False
    results = None
    n_rays = 40
    n_rays_half = int(n_rays / 2)
    n_rays_quarter = int(n_rays / 4)
    assert n_rays_half == n_rays / 2.0
    assert n_rays_quarter == n_rays / 4.0
    prop_successful = 0.75

    for i in range(200):
        results = sample_radial_rays(point, n=n_rays)
        if th.mean(
            th.tensor([result["hit"] for result in results], dtype=th.float32)
        ) >= prop_successful and th.all(
            th.tensor([result["distance"] > 0 for result in results if result["hit"]])
        ):
            success = True
            break
        point = th.rand(3) * (aabb_high - aabb_low) + aabb_low
        point[2] = aabb_center[2]

    if not success:
        # Try sampling from the top
        success, point = sample_fillable_point_from_top(
            fillable, n_rays, prop_successful
        )

    if success:
        # 2c. Move point to center of circle
        xs = th.tensor([result["position"][0] for result in results if result["hit"]])
        ys = th.tensor([result["position"][1] for result in results if result["hit"]])
        center = th.tensor([xs.mean(), ys.mean(), point[2]])

        # 3. Shoot ray up and down to hit boundaries
        down_ray = {"hit": False}

        while (not down_ray["hit"] or down_ray["distance"] == 0.0) and center[2] <= (
            aabb_high[2] + 0.01
        ):
            center = center + th.tensor([0, 0, 0.01])
            down_ray = raytest(
                start_point=center,
                end_point=center - th.tensor([0, 0, 4.0]),
            )
        if not down_ray["hit"] or down_ray["distance"] == 0.0:
            success = False

    if success:
        results = sample_radial_rays(center, n=n_rays)
        if not (
            th.mean(th.tensor([result["hit"] for result in results], dtype=th.float32))
            >= prop_successful
            and th.all(
                th.tensor(
                    [result["distance"] > 0 for result in results if result["hit"]]
                )
            )
        ):
            success, center = sample_fillable_point_from_top(
                fillable, n_rays, prop_successful
            )
            assert success

        down_ray = raytest(
            start_point=center,
            end_point=center - th.tensor([0, 0, 4.0]),
        )

        assert down_ray["hit"]
        z_low = down_ray["position"][2]
        up_ray = raytest(
            start_point=center,
            end_point=center + th.tensor([0, 0, 4.0]),
        )
        z_high = up_ray["position"][2] if up_ray["hit"] else aabb_high[2]

        # 4. Sample points and record positions
        mesh_points = th.zeros((0, 3))
        z_dist = th.clamp((aabb_high[2] - aabb_low[2]) / 8.0, 0.001, 0.05)
        offset = avg_diameter / 20.0
        min_diameter = avg_diameter / 5.0
        z = z_low + offset / 2.0
        layer_points = []
        layer_hit_positions = None
        i = 0
        while z <= z_high:
            layer_center = th.tensor([center[0], center[1], z])
            layer_results = sample_radial_rays(point=layer_center, n=n_rays)
            layer_hit_positions = th.stack(
                [
                    th.tensor(result["position"])
                    for result in layer_results
                    if result["hit"]
                ]
            )
            if not len(layer_hit_positions) / len(layer_results) >= prop_successful:
                if i <= 1:
                    # We failed to sample, probably a degenerate mesh, fail
                    success = False
                break
            if len(mesh_points) > 0 and th.any(
                (
                    layer_hit_positions.max(dim=0)[0][:2]
                    - layer_hit_positions.min(dim=0)[0][:2]
                )
                < min_diameter
            ):
                break
            layer_rays = layer_hit_positions - layer_center.unsqueeze(0)
            layer_points = layer_hit_positions - layer_rays * offset / th.norm(
                layer_rays, dim=1, keepdim=True
            )
            mesh_points = th.cat([mesh_points, layer_points], dim=0)
            z += z_dist
            i += 1

        # Prune previous layer
        mesh_points = mesh_points[: -len(layer_points)]

    if not success:
        print("Failed to find sufficient cavity to generate fillable, skipping.")
        og.sim.stop()
        return

    # 5. Create trimesh
    tm = trimesh.Trimesh(vertices=mesh_points.numpy().copy())

    # 6. Create convex hull
    hull = tm.convex_hull
    hull.unmerge_vertices()

    # 7. Take the xy-plane projection of the convex hull, and then sample rays shooting downwards to compensate for the
    # original z offset
    # We know the bounding box is [1, 1, 1], so sample points uniformly in XY plane

    # Generate the convex hull
    proj = ConvexHull(hull.vertices[:, :2].copy())
    equations = th.tensor(proj.equations, dtype=th.float32)

    n_dim_samples = 10
    x_range = th.linspace(aabb_low[0], aabb_high[0], n_dim_samples)
    y_range = th.linspace(aabb_low[1], aabb_high[1], n_dim_samples)
    ray_grid = th.stack(th.meshgrid(x_range, y_range, indexing="ij"), dim=-1)
    ray_grid_flattened = ray_grid.reshape(-1, 2)

    # Check which rays are within the polygon
    # Each inequality is of the form Ax + By + C <= 0
    # We need to check if the point satisfies all inequalities
    is_within = th.all(
        (ray_grid_flattened @ equations[:, :-1].T) + equations[:, -1] <= 0, dim=1
    )
    xy_ray_positions = ray_grid_flattened[is_within]

    # Shoot these rays downwards and record their poses -- add them to the point set
    z_range = th.linspace(aabb_low[2], aabb_high[2], n_dim_samples)
    additional_points = []
    for xy_ray_pos in xy_ray_positions:
        # Find a corresponding z value within the convex hull to use as the start raycasting point
        start_samples = th.zeros((n_dim_samples, 3))
        start_samples[:, :2] = xy_ray_pos
        start_samples[:, 2] = z_range
        is_contained = hull.contains(start_samples.numpy().copy())
        if not th.any(th.tensor(is_contained)):
            # Just skip this sample
            continue
        # Use the lowest point (i.e.: the first idx that is True) as the raycasting start point
        z = z_range[th.where(th.tensor(is_contained))[0][0]]

        # Raycast downwards and record the hit point
        start = th.tensor([*xy_ray_pos, z])
        end = start.clone()
        end[2] = aabb_low[2]
        down_ray = raytest(
            start_point=start,
            end_point=end,
        )
        # If we have a valid hit with nonzero distance, record this point
        if down_ray["hit"] and down_ray["distance"] > 0:
            additional_points.append(th.tensor(down_ray["position"]))

    # Append all additional points to our existing set of points
    if additional_points:
        mesh_points = th.cat([mesh_points, th.stack(additional_points)], dim=0)

    # Denormalize the mesh points based on the objects' scale
    scale = 1.0 / th.tensor(fillable.scale)
    mesh_points = mesh_points * scale.reshape(1, 3)

    # Re-write to trimesh and take the finalized convex hull
    tm = trimesh.Trimesh(vertices=mesh_points.numpy().copy())

    # 6. Create convex hull
    hull = tm.convex_hull
    hull.unmerge_vertices()

    # Save it somewhere
    hull.export(
        out_path,
        file_type="obj",
        include_normals=False,
        include_color=False,
        include_texture=False,
    )


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
        out_path = obj_dir / "fillable_ray.obj"
        try:
            process_object(obj_category, obj_model, out_path)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
