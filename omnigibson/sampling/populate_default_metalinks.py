import omnigibson as og
from omnigibson.macros import gm, macros
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject, USDObject
from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh
import numpy as np
from pxr import Vt, UsdGeom, Gf, UsdPhysics, PhysxSchema, Sdf, Usd
from omnigibson.utils.sampling_utils import raytest_batch, raytest #, sample_raytest_start_end_full_grid_topdown
from omnigibson.utils.asset_utils import decrypt_file, encrypt_file, get_all_object_category_models, get_all_object_categories
import omnigibson.utils.transform_utils as T
import trimesh
from omni.kit.primitive.mesh.evaluators.cube import CubeEvaluator
# from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
import carb
import omni
import os
import bddl
from bddl.object_taxonomy import ObjectTaxonomy

"""
Algorithm:

1. Start at center of AABB
2a. Sample random points in 2D x-y grid and shoot rays in a circle
2b. The starting point is the first point where all rays hit the object -- implies the object is in a cavity
2c. Move the starting point until it's at the center of the circle
3. Sample ray downwards until hit obj (assert this, should not hit bottom of AABB), 
    and upwards until hit obj or hit top of AABB. Moving downwards, sample points and record their positions
4. Now, we have (potentially non-convex) hull
5. Create mesh in trimesh from vertices
6. Convex decomp it so we have N convex subhulls
7. Programmatically create meshes in USD, and also create fixed joint for metalink
8. Save the USD
"""
def create_mesh(prim_path, stage):
    stage.DefinePrim(prim_path, "Mesh")
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    return mesh

def create_fixed_joint(prim_path, stage, body0, body1):
    # Create the joint
    joint = UsdPhysics.__dict__["FixedJoint"].Define(stage, prim_path)

    # Possibly add body0, body1 targets
    if body0 is not None:
        assert stage.GetPrimAtPath(body0).IsValid(), f"Invalid body0 path specified: {body0}"
        joint.GetBody0Rel().SetTargets([Sdf.Path(body0)])
    if body1 is not None:
        assert stage.GetPrimAtPath(body1).IsValid(), f"Invalid body1 path specified: {body1}"
        joint.GetBody1Rel().SetTargets([Sdf.Path(body1)])

    # Get the prim pointed to at this path
    joint_prim = stage.GetPrimAtPath(prim_path)

    # Apply joint API interface
    PhysxSchema.PhysxJointAPI.Apply(joint_prim)

    # Possibly (un-/)enable this joint
    joint_prim.GetAttribute("physics:jointEnabled").Set(True)

    # Return this joint
    return joint_prim


def sample_radial_rays(point, n=40, dist=4.0):
    angles = np.arange(n) * 2 * np.pi / n
    x = np.cos(angles)
    y = np.sin(angles)
    start_points = np.ones((n, 3)) * point.reshape(1, 3)
    end_points = np.array([x, y, np.zeros(n)]).T * dist + point.reshape(1, 3)
    return raytest_batch(start_points, end_points)


def point_inside_aabb(point, aabb_low, aabb_high):
    return np.all(point >= aabb_low) and np.all(point <= aabb_high)


def sample_raytest_start_end_full_grid_topdown(
    obj,
    ray_spacing,
    aabb_offset=macros.utils.sampling_utils.DEFAULT_AABB_OFFSET,
):
    bbox_center = obj.aabb_center
    bbox_orn = np.array([0, 0, 0, 1.0])
    bbox_bf_extent = obj.aabb_extent
    # bbox_center, bbox_orn, bbox_bf_extent, _ = obj.get_base_aligned_bbox(xy_aligned=True, fallback_to_aabb=True)

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


def sample_fillable_point_from_top(obj, n_rays, prop_successful):
    aabb_low, aabb_high = obj.aabb
    # Sample uniformly from the top of the AABB, and take the lowest z and randomly sample again
    start_rays, end_rays = sample_raytest_start_end_full_grid_topdown(obj,
                                                                      np.mean((aabb_high[:2] - aabb_low[:2]) / 10.0))
    down_results = raytest_batch(start_rays.reshape(-1, 3), end_rays.reshape(-1, 3))
    down_hit_results = np.array([result["position"] for result in down_results if result["hit"]])

    z = down_hit_results[:, 2].min() + 0.01
    point = down_hit_results[np.argmin(down_hit_results[:, 2])]
    point[2] += 0.01
    results = sample_radial_rays(point, n=n_rays)

    if np.mean([result["hit"] for result in results]) >= prop_successful and np.all(
        [result["distance"] > 0 for result in results if result["hit"]]):

        xs = np.array([result["position"][0] for result in results if result["hit"]])
        ys = np.array([result["position"][1] for result in results if result["hit"]])
        center = np.array([xs.mean(), ys.mean(), z])

        return True, center

    return False, np.zeros(3)


def generate_fillable_volume(obj):
    container_link_prefix = macros.object_states.contains.CONTAINER_LINK_PREFIX
    # Check if it has any fillable links
    for link_name in obj.links.keys():
        if container_link_prefix in link_name:
            return

    # Make sure sim is playing
    og.sim.play()
    assert obj.fixed_base
    assert np.all(np.isclose(obj.get_position(), 0.0, rtol=1e-3, atol=1e-3))

    # 1. - 2b. Start at center of AABB
    point = obj.aabb_center
    aabb_center = obj.aabb_center
    aabb_low, aabb_high = obj.aabb
    avg_diameter = np.mean(obj.aabb_extent[:2])
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
        if np.mean([result["hit"] for result in results]) >= prop_successful and np.all([result["distance"] > 0 for result in results if result["hit"]]):
            success = True
            break
        point = np.random.uniform(low=aabb_low, high=aabb_high)
        point[2] = aabb_center[2]

    if not success:
        # Try sampling from the top
        success, point = sample_fillable_point_from_top(obj, n_rays, prop_successful)

    if success:
        # 2c. Move point to center of circle
        xs = np.array([result["position"][0] for result in results if result["hit"]])
        ys = np.array([result["position"][1] for result in results if result["hit"]])
        center = np.array([xs.mean(), ys.mean(), point[2]])
        # delta_x = (results[0]["distance"] - results[n_rays_half]["distance"]) / 2.0
        # delta_y = (results[n_rays_quarter]["distance"] - results[n_rays_half + n_rays_quarter]["distance"]) / 2.0
        # center = point + np.array([delta_x, delta_y, 0.0])

        # 3. Shoot ray up and down to hit boundaries
        down_ray = {"hit": False}

        while (not down_ray["hit"] or down_ray["distance"] == 0.0) and center[2] <= (aabb_high[2] + 0.01):
            center = center + np.array([0, 0, 0.01])
            down_ray = raytest(
                start_point=center,
                end_point=center - np.array([0, 0, 4.0]),
            )
        if not down_ray["hit"] or down_ray["distance"] == 0.0:
            success = False

    if success:
        results = sample_radial_rays(center, n=n_rays)
        if not (np.mean([result["hit"] for result in results]) >= prop_successful and np.all(
            [result["distance"] > 0 for result in results if result["hit"]])):

            success, center = sample_fillable_point_from_top(obj, n_rays, prop_successful)
            assert success

        down_ray = raytest(
            start_point=center,
            end_point=center - np.array([0, 0, 4.0]),
        )

        assert down_ray["hit"]
        z_low = down_ray["position"][2]
        up_ray = raytest(
            start_point=center,
            end_point=center + np.array([0, 0, 4.0]),
        )
        z_high = up_ray["position"][2] if up_ray["hit"] else aabb_high[2]

        # 4. Sample points and record positions
        mesh_points = np.array([]).reshape(0, 3)
        z_dist = np.clip((aabb_high[2] - aabb_low[2]) / 8.0, 0.001, 0.05)
        offset = avg_diameter / 20.0
        min_diameter = avg_diameter / 5.0
        z = z_low + offset / 2.0
        layer_points = []
        layer_hit_positions = None
        i = 0
        while z <= z_high:
            layer_center = np.array([center[0], center[1], z])
            layer_results = sample_radial_rays(point=layer_center, n=n_rays)
            layer_hit_positions = np.array([result["position"] for result in layer_results if result["hit"]])
            if not len(layer_hit_positions) / len(layer_results) >= prop_successful:
                if i <= 1:
                    # We failed to sample, probably a degenerate mesh, fail
                    success = False
                break
            # If the distance between points is less than a threshold, also break
            # assert layer_results[0]["distance"] != layer_results[n_rays_half]["distance"]
            # if (layer_results[0]["distance"] + layer_results[n_rays_half]["distance"]) < min_diameter:
            if len(mesh_points) > 0 and np.any((layer_hit_positions.max(axis=0)[:2] - layer_hit_positions.min(axis=0)[:2]) < min_diameter):
                break
            layer_rays = (layer_hit_positions - layer_center.reshape(1, 3))
            layer_points = layer_hit_positions - layer_rays * offset / np.linalg.norm(layer_rays)
            # layer_points = [np.array(result["position"]) - (result["position"] - layer_center) * offset / np.linalg.norm(
            #     (result["position"] - layer_center)) for result in layer_results]
            mesh_points = np.concatenate([mesh_points, layer_points], axis=0)
            z += z_dist
            i += 1

        # Prune previous layer
        mesh_points = mesh_points[:-len(layer_points)]

    if not success:
        print("Failed to find sufficient cavity to generate fillable, skipping.")
        og.sim.stop()
        return
        # print("Failed to find sufficient cavity to generate fillable, creating dummy volume instead.")
        # extents = np.array([aabb_low, aabb_high]).T * 0.01
        # mesh_points = np.stack([arr.flatten() for arr in np.meshgrid(extents[0], extents[1], extents[2])]).T

    # 5. Create trimesh
    # print(f"n mesh points: {len(mesh_points)}")
    tm = trimesh.Trimesh(vertices=np.array(mesh_points))

    # 6. Create convex hull
    # c_tms = trimesh.decomposition.convex_decomposition(tm)
    ctm = tm.convex_hull
    ctm.unmerge_vertices()

    # Create mesh in USD
    og.sim.stop()

    # Open original USD
    usd_path = obj.usd_path
    encrypted = isinstance(obj, DatasetObject)
    if encrypted:
        # Decrypt first
        # filename = usd_path.split("/")[-1].split(".usd")[0]
        dirpath = os.path.dirname(usd_path)
        encrypted_usd_path = usd_path.replace(".usd", ".encrypted.usd")
        # decrypted_usd_path = os.path.join(dirpath, f"{filename}.usd")
        decrypt_file(encrypted_usd_path, usd_path)

    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetDefaultPrim()

    obj_prim_path = prim.GetPath().pathString
    container_link_path = f"{obj_prim_path}/{container_link_prefix}_0_0_link"
    container_link = stage.DefinePrim(container_link_path, "Xform")
    mesh_prim = create_mesh(prim_path=f"{container_link_path}/{container_link_prefix}_mesh_0", stage=stage).GetPrim()

    # Write mesh data
    mesh_prim.GetAttribute("faceVertexCounts").Set(np.ones(len(ctm.faces), dtype=int) * 3)
    mesh_prim.GetAttribute("points").Set(Vt.Vec3fArray.FromNumpy(ctm.vertices))
    mesh_prim.GetAttribute("faceVertexIndices").Set(np.arange(len(ctm.vertices)))
    mesh_prim.GetAttribute("normals").Set(Vt.Vec3fArray.FromNumpy(ctm.vertex_normals))
    # mesh_prim.GetAttribute("primvars:st").Set(Vt.Vec2fArray.FromNumpy(np.zeros((len(ctm.vertices), 2))))

    # Make invisible
    UsdGeom.Imageable(mesh_prim).MakeInvisible()

    # Create fixed joint
    obj_root_path = f"{obj_prim_path}/base_link"
    create_fixed_joint(
        prim_path=f"{obj_root_path}/{container_link_prefix}_0_0_joint",
        stage=stage,
        body0=f"{obj_root_path}",
        body1=f"{container_link_path}",
    )

    # Save the USD
    stage.Save()

    if encrypted:
        encrypted_usd_path = usd_path.replace(".usd", ".encrypted.usd")
        encrypt_file(usd_path, encrypted_usd_path)
        os.remove(usd_path)


def generate_metalink(obj, metalink_prefix):
    # Check if it has any fillable links
    for link_name in obj.links.keys():
        if metalink_prefix in link_name:
            return

    # Create link in USD
    og.sim.stop()

    # Open original USD
    usd_path = obj.usd_path
    encrypted = isinstance(obj, DatasetObject)
    if encrypted:
        # Decrypt first
        encrypted_usd_path = usd_path.replace(".usd", ".encrypted.usd")
        decrypt_file(encrypted_usd_path, usd_path)

    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetDefaultPrim()

    obj_prim_path = prim.GetPath().pathString
    metalink_path = f"{obj_prim_path}/{metalink_prefix}_0_0_link"
    metalink = stage.DefinePrim(metalink_path, "Xform")

    # Create fixed joint
    obj_root_path = f"{obj_prim_path}/base_link"
    create_fixed_joint(
        prim_path=f"{obj_root_path}/{metalink_prefix}_0_0_joint",
        stage=stage,
        body0=f"{obj_root_path}",
        body1=f"{metalink_path}",
    )

    # Save the USD
    stage.Save()

    if encrypted:
        encrypted_usd_path = usd_path.replace(".usd", ".encrypted.usd")
        encrypt_file(usd_path, encrypted_usd_path)
        os.remove(usd_path)


def generate_toggle_button_link(obj):
    return generate_metalink(obj=obj, metalink_prefix=macros.object_states.toggle.TOGGLE_LINK_PREFIX)


def generate_particlesource_link(obj):
    return generate_metalink(obj=obj, metalink_prefix=macros.object_states.particle_source_or_sink.SOURCE_LINK_PREFIX)


def generate_particlesink_link(obj):
    return generate_metalink(obj=obj, metalink_prefix=macros.object_states.particle_source_or_sink.SINK_LINK_PREFIX)


ABILITY_TO_METALINK_FCN = {
    "particleSource": generate_particlesource_link,
    "particleSink": generate_particlesink_link,
    # "toggleable": generate_toggle_button_link,
    # "fillable": generate_fillable_volume,
}

ot = ObjectTaxonomy()

SIDEWAYS_MODELS = {
    "fedafr",
    "fdqxyk",
    "cvdbum",
    "hacehh",
    "jwxbpa",
    "lfxtqa",
    "vlikrk",
    "udfxuu",
    "wqgndf",
    "nbuspz",
    "spopfj",
}

UPSIDE_DOWN_MODELS = {
    "damllm",
    "fgizgn",
    "cprjvq",
    "aewpzn",
    "snvhlz",
    "lkomhp",
    "odmjdd",
    "vitdwc",
    "fbfmwt",
    "kvgaar",
    "obuxbe",
    "ozrwwk",
    "tfzijn",
    "lageli",
    "rusmlm",
    "twknia",
    "ptciim",
    "szjfpb",
    "vxmzmq",
}

UNFILLABLE_MODELS = {
    "waousd",
    "vbiqcq",
}

SHALLOW_MODELS = {
    "pjinwe",
    "bgxzec",
    "xtdcau",
    "qtfzeq",
    "uakqei",
    "iawoof",
    "iaaiyi",
    "tkgsho",
}

WIREMESH_MODELS = {
    "mmegts",
}

BAD_MESH_MODELS = {
    "kdiwzf",
}

INVALID_MODELS = set.union(
    SIDEWAYS_MODELS,
    UPSIDE_DOWN_MODELS,
    UNFILLABLE_MODELS,
    SHALLOW_MODELS,
    WIREMESH_MODELS,
    BAD_MESH_MODELS,
)

def populate_metalinks(start_at=None):
    og.sim.clear()
    scene = Scene(use_floor_plane=False)
    og.sim.import_scene(scene)

    for category in get_all_object_categories():
        if start_at is not None:
            if category == start_at:
                start_at = None
            else:
                continue
        synset = ot.get_synset_from_category(category)
        if synset is None:
            print(f"No valid synset found for category: {category}. Skipping")
            continue
        abilities = ot.get_abilities(synset)
        metalink_abilities = set.intersection(set(abilities.keys()), set(ABILITY_TO_METALINK_FCN.keys()))
        print(f"{category} metalink abilities: {metalink_abilities}")
        if len(metalink_abilities) > 0:
            for model in get_all_object_category_models(category):
                if model in INVALID_MODELS:
                    continue
                print(f"Creating metalinks for obj: {category}, {model}...")
                obj = DatasetObject(
                    name="obj",
                    category=category,
                    model=model,
                    fixed_base=True,
                    abilities={},
                    include_default_states=False,
                )
                og.sim.import_object(obj)
                for metalink_ability in metalink_abilities:
                    print(f"Creating metalink {metalink_ability}...")
                    ABILITY_TO_METALINK_FCN[metalink_ability](obj)
                og.sim.stop()
                og.sim.remove_object(obj)


# scene = Scene(use_floor_plane=False)
# og.sim.import_scene(scene)

# obj = DatasetObject(
#     name="obj",
#     category="saucepan",
#     model="fsinsu",
#     # usd_path=f"{gm.DATASET_PATH}/objects/saucepan/fsinsu/usd/fsinsu.usd",
#     fixed_base=True,
#     abilities={},
#     include_default_states=False,
# )
# og.sim.import_object(obj)

# populate_metalinks("wineglass")
