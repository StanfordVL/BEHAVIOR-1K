import os
import traceback

# Must be set before cv2 is imported for EXR (IOR map) support.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import json
import numpy as np
from PIL import Image
from b1k_pipeline.urdfpy import URDF
import b1k_pipeline.utils
import trimesh
import fs.copy
from fs.tempfs import TempFS
from dask.distributed import LocalCluster, as_completed
import tqdm
import glob

# glTF's fixed normal-incidence reflectance for dielectrics (IOR 1.5).
GLTF_DIELECTRIC_SPECULAR = 0.04
EPSILON = 1e-6

# Map keys emitted into the per-link MTL files by export_objs_global.py.
MTL_CHANNELS = {
    "map_Kd": "diffuse",
    "map_bump": "normal",
    "map_Pr": "glossiness",  # VRay reflection glossiness bake (NOT roughness)
    "map_Pm": "metalness",
    "map_Tf": "refraction",  # VRay refraction filter color
    "map_Ks": "reflection",  # VRay reflection filter color
    "map_Ns": "ior",  # VRay Fresnel IOR bake, stored as EXR
}

# Fixed material used for links carrying the Tglass tag, mirroring the
# OmniGlass override applied during USD import in OmniGibson.
GLASS_MATERIAL_KWARGS = dict(
    name="glass",
    baseColorFactor=(0.8, 0.9, 0.95, 0.3),
    metallicFactor=0.0,
    roughnessFactor=0.05,
    alphaMode="BLEND",
    doubleSided=True,
)


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _load_image(path):
    """Load an image as a float32 HxWx3 array of raw (undecoded) values in [0, 1]-ish range.

    EXR files are loaded via OpenCV and contain linear float data (values may
    exceed 1, e.g. IOR). Everything else is loaded via PIL as 8-bit sRGB-encoded.
    """
    if os.path.splitext(path)[1].lower() == ".exr":
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Could not load EXR file {path}"
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        else:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        return img
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _resize(img, shape):
    if img.shape[:2] == shape:
        return img
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _get_material_maps(obj_path):
    """Parse the MTL referenced by an OBJ file into a {channel: absolute path} dict."""
    obj_dir = os.path.dirname(obj_path)
    mtl_files = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("mtllib "):
                mtl_files.append(line.split("mtllib ", 1)[1].strip())
    assert len(mtl_files) <= 1, f"Expected at most one mtllib in {obj_path}, found {mtl_files}"
    if not mtl_files:
        return {}

    maps = {}
    mtl_path = os.path.join(obj_dir, mtl_files[0])
    with open(mtl_path, "r") as f:
        for line in f:
            tokens = line.strip().split(None, 1)
            if len(tokens) == 2 and tokens[0] in MTL_CHANNELS:
                map_path = os.path.normpath(os.path.join(obj_dir, tokens[1].strip()))
                assert os.path.exists(map_path), f"Texture {map_path} referenced by {mtl_path} does not exist"
                maps[MTL_CHANNELS[tokens[0]]] = map_path
    return maps


def convert_vray_to_pbr_material(name, maps):
    """Convert a set of baked VRay channel maps into a glTF metallic-roughness PBRMaterial.

    The math replicates OmniGibson's omnigibson_vray_mtl.mdl / vray_materials.mdl:
      - roughness = (1 - glossiness)^2
      - R0 = ((1 - ior) / (1 + ior))^2, per-texel from the EXR IOR bake
      - dielectric specular F0 = R0 * reflection filter color
    The per-texel F0 is folded into the metallic-roughness model using the Khronos
    specular-glossiness -> metallic-roughness solve, then blended with VRay's
    explicit metalness channel. The refraction filter becomes base color alpha.
    8-bit PNG channels are sRGB-decoded before the math, matching how the MDL
    samples them (auto colorspace); the EXR IOR map is already linear.
    """
    assert "diffuse" in maps, f"Material {name} has no diffuse map: {maps}"
    diffuse_srgb = _load_image(maps["diffuse"])
    shape = diffuse_srgb.shape[:2]

    def load_channel(channel, default):
        if channel not in maps:
            return np.full((*shape, 3), default, dtype=np.float32)
        return _resize(_load_image(maps[channel]), shape)

    diffuse_lin = _srgb_to_linear(diffuse_srgb)
    # Defaults match the VRayMtl MDL parameter defaults.
    gloss = _srgb_to_linear(load_channel("glossiness", 1.0)).mean(axis=-1)
    metal_vray = _srgb_to_linear(load_channel("metalness", 0.0)).mean(axis=-1)
    refl_lin = _srgb_to_linear(load_channel("reflection", 0.0))
    refr_lin = _srgb_to_linear(load_channel("refraction", 0.0))
    ior = load_channel("ior", 1.6).mean(axis=-1)  # EXR: linear, no decode

    roughness = (1.0 - gloss) ** 2
    r0 = ((1.0 - ior) / np.maximum(1.0 + ior, EPSILON)) ** 2
    f0 = r0[..., None] * refl_lin

    # Khronos KHR_materials_pbrSpecularGlossiness -> metallic-roughness solve.
    spec_max = f0.max(axis=-1)
    one_minus_spec = 1.0 - spec_max
    diff_lum = diffuse_lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    a = GLTF_DIELECTRIC_SPECULAR
    b = diff_lum * one_minus_spec / (1.0 - a) + spec_max - 2.0 * a
    c = a - spec_max
    d = np.maximum(b * b - 4.0 * a * c, 0.0)
    metal_solved = np.clip((-b + np.sqrt(d)) / (2.0 * a), 0.0, 1.0)
    metal_solved = np.where(spec_max < a, 0.0, metal_solved)

    base_from_diff = diffuse_lin * (one_minus_spec / (1.0 - a) / np.maximum(1.0 - metal_solved, EPSILON))[..., None]
    base_from_spec = (f0 - a * (1.0 - metal_solved)[..., None]) / np.maximum(metal_solved, EPSILON)[..., None]
    t = (metal_solved**2)[..., None]
    base_solved = np.clip(base_from_diff * (1.0 - t) + base_from_spec * t, 0.0, 1.0)

    # VRay's explicit metalness reflects with the Diffuse color as tint, which is
    # exactly the glTF metal model, so blend the solved dielectric toward that.
    metallic = np.clip(metal_vray + (1.0 - metal_vray) * metal_solved, 0.0, 1.0)
    base_lin = base_solved * (1.0 - metal_vray)[..., None] + diffuse_lin * metal_vray[..., None]

    # glTF core has no transmission; approximate the refraction filter with alpha.
    alpha = np.clip(1.0 - refr_lin.mean(axis=-1), 0.0, 1.0)
    transparent = alpha.min() < 1.0 - 2.0 / 255.0

    base_srgb = (_linear_to_srgb(base_lin) * 255.0).round().astype(np.uint8)
    if transparent:
        alpha_u8 = (alpha * 255.0).round().astype(np.uint8)
        base_color_texture = Image.fromarray(np.dstack([base_srgb, alpha_u8]), mode="RGBA")
    else:
        base_color_texture = Image.fromarray(base_srgb, mode="RGB")

    # Linear-encoded per glTF spec: G = roughness, B = metallic (R unused).
    mr = np.dstack(
        [
            np.full(shape, 255, dtype=np.uint8),
            (np.clip(roughness, 0.0, 1.0) * 255.0).round().astype(np.uint8),
            (metallic * 255.0).round().astype(np.uint8),
        ]
    )
    metallic_roughness_texture = Image.fromarray(mr, mode="RGB")

    normal_texture = Image.open(maps["normal"]).convert("RGB") if "normal" in maps else None

    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorTexture=base_color_texture,
        metallicRoughnessTexture=metallic_roughness_texture,
        normalTexture=normal_texture,
        alphaMode="BLEND" if transparent else None,
    )


def _material_to_factors(material):
    """Reduce a textured PBRMaterial to factor-only for meshes without UVs."""
    base = np.asarray(material.baseColorTexture.convert("RGBA"), dtype=np.float32) / 255.0
    mean_rgba = base.reshape(-1, 4).mean(axis=0)
    mean_lin = _srgb_to_linear(mean_rgba[:3])
    mr = np.asarray(material.metallicRoughnessTexture, dtype=np.float32) / 255.0
    return trimesh.visual.material.PBRMaterial(
        name=material.name,
        baseColorFactor=(*mean_lin, mean_rgba[3]),
        roughnessFactor=float(mr[..., 1].mean()),
        metallicFactor=float(mr[..., 2].mean()),
        alphaMode=material.alphaMode,
    )


def _apply_material(mesh, material):
    if isinstance(mesh.visual, trimesh.visual.TextureVisuals) and mesh.visual.uv is not None and len(mesh.visual.uv):
        mesh.visual.material = material
    else:
        factors = material if material.baseColorTexture is None else _material_to_factors(material)
        mesh.visual = trimesh.visual.TextureVisuals(material=factors)


def urdf_to_glb(in_obj_dir, out_obj_dir):
    # Get the URDF file
    urdf_files = glob.glob(in_obj_dir + "/urdf/*.urdf")
    assert (
        len(urdf_files) == 1
    ), f"Expected exactly one URDF file in {in_obj_dir}, found {len(urdf_files)}"

    urdf_file = urdf_files[0]
    urdf_dir = os.path.dirname(urdf_file)

    # Load the link tags so that Tglass-tagged links can be overridden.
    link_tags = {}
    metadata_file = os.path.join(in_obj_dir, "misc", "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            link_tags = json.load(f).get("link_tags") or {}

    robot = URDF.load(urdf_file)
    links = [l for l in robot.links if "meta__" not in l.name]
    link_fk = robot.link_fk(links=links)

    material_cache = {}
    scene = trimesh.Scene()
    for link in links:
        is_glass = "glass" in (link_tags.get(link.name) or [])
        for visual in link.visuals:
            pose = link_fk[link].dot(visual.origin)
            if visual.geometry.mesh is not None and visual.geometry.mesh.scale is not None:
                S = np.eye(4, dtype=np.float64)
                S[:3, :3] = np.diag(visual.geometry.mesh.scale)
                pose = pose.dot(S)

            material = None
            if is_glass:
                material = trimesh.visual.material.PBRMaterial(**GLASS_MATERIAL_KWARGS)
            elif visual.geometry.mesh is not None:
                obj_path = os.path.normpath(os.path.join(urdf_dir, visual.geometry.mesh.filename))
                if obj_path not in material_cache:
                    maps = _get_material_maps(obj_path)
                    name = os.path.splitext(os.path.basename(obj_path))[0]
                    material_cache[obj_path] = convert_vray_to_pbr_material(name, maps) if maps else None
                material = material_cache[obj_path]

            for mesh in visual.geometry.meshes:
                m = mesh.copy()
                m.apply_transform(pose)
                if material is not None:
                    _apply_material(m, material)
                scene.add_geometry(m)

    model_id = os.path.splitext(os.path.basename(urdf_file))[0]
    out_file = os.path.join(out_obj_dir, f"{model_id}.glb")
    scene.export(out_file)


def main():
    with (
        b1k_pipeline.utils.ParallelZipFS("objects.zip") as source_fs,
        TempFS(temp_dir=str(b1k_pipeline.utils.TMP_DIR)) as temp_fs,
        b1k_pipeline.utils.ParallelZipFS("objects_glb.zip", write=True) as out_fs,
    ):
        # Copy everything over to the temp FS
        print("Copying input to temp fs...")
        objdir_glob = [item.path for item in source_fs.glob("objects/*/*/")]
        for item in tqdm.tqdm(objdir_glob):
            if (
                source_fs.opendir(item).opendir("urdf").glob("*.urdf").count().files
                == 0
            ):
                continue
            fs.copy.copy_fs(
                source_fs.opendir(item), temp_fs.makedirs(item, recreate=True)
            )

        cluster = LocalCluster()
        dask_client = cluster.get_client()

        obj_futures = {}

        for objdir in tqdm.tqdm(
            objdir_glob, desc="Processing targets to queue objects"
        ):
            obj_futures[
                dask_client.submit(
                    urdf_to_glb,
                    temp_fs.opendir(objdir).getsyspath("/"),
                    out_fs.makedirs(objdir).getsyspath("/"),
                    pure=False,
                )
            ] = objdir

        for future in tqdm.tqdm(
            as_completed(obj_futures.keys()),
            total=len(obj_futures),
            desc="Processing objects",
        ):
            try:
                future.result()
            except:
                traceback.print_exc()

        print("Finished processing")


if __name__ == "__main__":
    main()
