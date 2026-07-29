import os
import traceback

# Must be set before cv2 is imported for EXR (IOR map) support.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import io
import json
import multiprocessing
import xml.etree.ElementTree as ET
from concurrent import futures

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from b1k_pipeline.urdfpy import URDF
import b1k_pipeline.utils
import trimesh
import fs.copy
from fs.tempfs import TempFS
import tqdm
import glob

# glTF's fixed normal-incidence reflectance for dielectrics (IOR 1.5).
GLTF_DIELECTRIC_SPECULAR = 0.04
EPSILON = 1e-6

# --- Lighting -----------------------------------------------------------------
# glTF punctual lights carry *physical photometric* units (candela for point/
# spot, lux for directional) and emissive is a surface's own emitted luminance.
# This is the portable representation: physically-based renderers (Mitsuba,
# Filament, Blender) agree on it, and real-time viewers (three.js, <model-viewer>)
# match once their exposure/tone-mapping is set. The large apparent brightness
# differences between renderers are an *exposure* (camera) setting, not something
# to bake into the asset -- so we author real-world-ish values for a well-lit
# daytime interior and expect ONE exposure setting per viewer.
#
# What is exposure-invariant (and therefore what we actually tune here) is the
# RATIO between emitters and lit surfaces. With the values below, a wall lit by
# key+fill sits around ~1000 cd/m^2 and the sky/fixtures read a few times
# brighter -- a believable balance under any exposure.
#
# IMPORTANT: emissive surfaces do NOT illuminate anything in glTF; the skybox and
# the per-light marker meshes only glow. All real lighting comes from the
# punctual lights below. Because those are physical (thousands of lux), the
# emissives use KHR_materials_emissive_strength so they read as bright emitters
# (not near-black) under a physical exposure; renderers lacking that extension
# just clamp emissive to 1.0 (dimmer backdrop, still valid).

# Base key/fill/up directional lights (lux) give even illumination independent of
# which objects happen to carry fixtures. Directions are the world-space vectors
# the light travels along (scenes are +Z up). Up-fill lifts undersides for
# real-time viewers that have no global illumination; GI renderers add their own.
KEY_LIGHT = {"intensity": 5000.0, "direction": (0.3, 0.4, -1.0), "color": (1.0, 0.97, 0.92)}
FILL_LIGHT = {"intensity": 2000.0, "direction": (-0.4, -0.3, -0.7), "color": (0.9, 0.94, 1.0)}
UP_FILL_LIGHT = {"intensity": 800.0, "direction": (0.0, 0.0, 1.0), "color": (0.95, 0.95, 0.95)}
SCENE_DIRECTIONAL_LIGHTS = [KEY_LIGHT, FILL_LIGHT, UP_FILL_LIGHT]

# Per-fixture point lights (object "lights" meta links), candela. A physical
# ceiling fixture is ~100-300 cd; these are local accents on top of the base.
# The annotated per-light intensity is ignored (fixed value), mirroring
# OmniGibson forcing gm.FORCE_LIGHT_INTENSITY on every dataset-object light.
POINT_LIGHT_INTENSITY = 200.0

# The punctual light is nudged this far (meters) out of its opaque emissive marker
# mesh, along the emitter's emission direction (local -Z), so the mesh doesn't
# self-occlude it in shadow-computing renderers (Blender, Filament, three.js).
POINT_LIGHT_OFFSET = 0.05

# KHR_materials_emissive_strength applied to the emissive skybox and light-marker
# meshes so they read as bright emitters (cd/m^2) against the physically-lit
# surfaces instead of being crushed to ~1 nit.
EMISSIVE_STRENGTH = 5000.0
EMISSIVE_MATERIAL_NAMES = {"skybox", "light_emissive"}

# Scene cameras carry no field-of-view in the scene URDF, so we emit a sensible
# fixed perspective. yfov is in radians; znear in meters (scene GLBs are metric).
CAMERA_YFOV = float(np.radians(45.0))
CAMERA_ZNEAR = 0.01

# A KHR_lights_punctual light is invisible geometry, so we also add an emissive
# mesh at each light, shaped and sized to match the annotated emitter. The light
# "type" field maps as in OmniGibson's _LIGHT_MAPPING (0=Rect, 2=Sphere, 4=Disk);
# for Rect the emitter is length x width, for Disk/Sphere the radius is length
# (see _generate_meshes_for_primitive_meta_links in asset_conversion_utils.py).
_LIGHT_TYPE_RECT = 0
_LIGHT_TYPE_SPHERE = 2
_LIGHT_TYPE_DISK = 4
# Thickness of the flat (rect/disk) emitter marker meshes, and fallback size for
# lights missing dimensions or of an unknown type.
LIGHT_MESH_THICKNESS = 0.01
LIGHT_MESH_FALLBACK_RADIUS = 0.05

# TODO: Temporary filter for a quick test run -- set to an empty set to export
# every scene. These two scenes have cameras in the current scenes.zip.
SCENES_TO_EXPORT = {"house_double_floor_lower", "restaurant_diner"}

# glTF has no native skybox, so scenes get a large inward-facing emissive cube
# textured with the same cube-cross image the MJCF skybox uses. The grid layout
# mirrors the MJCF gridsize "3 4" / gridlayout ".U..LFRB.D..".
SKYBOX_SOURCE = b1k_pipeline.utils.PIPELINE_ROOT / "b1k_pipeline" / "assets" / "skybox.png"
SKYBOX_GRID = (4, 3)  # (cols, rows)
SKYBOX_CELLS = {"U": (1, 0), "D": (1, 2), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1)}
# Skybox half-size as a multiple of the scene's half-extent, with a floor (m).
SKYBOX_MARGIN = 10.0
SKYBOX_MIN_HALF_SIZE = 50.0


def _skybox_mesh(center, half_size, image):
    """A large inward-facing emissive cube (skybox) centered at the scene.

    Each cube face maps to a cell of the cube-cross image with +Z as world up.
    The image top (sky) maps to the higher v (trimesh/glTF v origin is bottom),
    so world-up corners get v_up. Side azimuth/rotation is unconstrained since
    the sky faces are near-uniform. The material is emissive (unlit-looking) and
    double-sided so it renders from inside.
    """
    cols, rows = SKYBOX_GRID
    vertices, faces, uvs = [], [], []

    def add_face(corners, letter):
        # corners = (bottom-left, bottom-right, top-right, top-left); "top" is +Z.
        col, row = SKYBOX_CELLS[letter]
        u0, u1 = col / cols, (col + 1) / cols
        v_up, v_down = 1 - row / rows, 1 - (row + 1) / rows
        base = len(vertices)
        vertices.extend(corners)
        uvs.extend([(u0, v_down), (u1, v_down), (u1, v_up), (u0, v_up)])
        faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

    h = half_size
    add_face([(-h, h, -h), (h, h, -h), (h, h, h), (-h, h, h)], "F")  # +Y
    add_face([(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)], "R")  # +X
    add_face([(h, -h, -h), (-h, -h, -h), (-h, -h, h), (h, -h, h)], "B")  # -Y
    add_face([(-h, h, -h), (-h, -h, -h), (-h, -h, h), (-h, h, h)], "L")  # -X
    add_face([(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)], "U")  # +Z (top)
    add_face([(-h, h, -h), (h, h, -h), (h, -h, -h), (-h, -h, -h)], "D")  # -Z (bottom)

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices, dtype=np.float64) + np.asarray(center, dtype=np.float64),
        faces=np.array(faces),
        process=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array(uvs, dtype=np.float64),
        material=trimesh.visual.material.PBRMaterial(
            name="skybox",
            baseColorFactor=(0, 0, 0, 255),
            emissiveFactor=(1.0, 1.0, 1.0),
            emissiveTexture=image,
            doubleSided=True,
        ),
    )
    return mesh


def _emissive_light_mesh(light):
    """An emissive mesh marking a light source, shaped/sized/oriented to match it.

    Rect lights become a thin box (length x width), disk lights a thin cylinder of
    radius=length, sphere lights a sphere of radius=length. The mesh is placed at
    the light's base-frame pose. Flat emitters are double-sided so they read from
    either side.
    """
    light_type = light["type"]
    length = light["length"] if light["length"] > 1e-4 else LIGHT_MESH_FALLBACK_RADIUS
    width = light["width"] if light["width"] > 1e-4 else LIGHT_MESH_FALLBACK_RADIUS

    if light_type == _LIGHT_TYPE_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=length)
    elif light_type == _LIGHT_TYPE_DISK:
        mesh = trimesh.creation.cylinder(radius=length, height=LIGHT_MESH_THICKNESS, sections=24)
    elif light_type == _LIGHT_TYPE_RECT:
        mesh = trimesh.creation.box(extents=(length, width, LIGHT_MESH_THICKNESS))
    else:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=LIGHT_MESH_FALLBACK_RADIUS)

    mesh.apply_transform(np.array(light["pose"], dtype=np.float64))
    emissive = tuple(float(min(max(c, 0.0), 1.0)) for c in light["color"])
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="light_emissive",
            baseColorFactor=(0, 0, 0, 255),
            emissiveFactor=emissive,
            doubleSided=True,
        )
    )
    return mesh

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
    In the MDL, the specular lobe is *scaled by the reflection filter*: a texel
    with reflection = 0 is fully matte no matter what the glossiness bake says
    (VRay bakes glossiness = 1.0, its parameter default, on non-reflective
    materials). Core glTF cannot lower dielectric F0 below its fixed 0.04, so we
    express weak-or-absent specular by pushing roughness toward 1 with
    spec_scale = clamp(F0 / 0.04, 0, 1). F0 here never exceeds ~0.07, so the
    small excess above 0.04 is simply accepted (no fake metallic: the Khronos
    specular->metallic solve turns dark low-F0 texels into visibly reflective
    metal, which is what made walls/floors mirror-like). Explicit metals come
    only from the metalness bake and keep their full glossiness.

    The refraction filter becomes base color alpha. 8-bit PNG channels are
    sRGB-decoded before the math, matching how the MDL samples them (auto
    colorspace); the EXR IOR map is already linear.
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
    spec_max = f0.max(axis=-1)

    # Fold the specular strength into roughness: glTF dielectrics always reflect
    # at F0 >= 0.04, so texels whose VRay specular is weaker than that (down to
    # reflection = 0, i.e. fully matte) are expressed by pushing roughness to 1.
    # Explicit metals (metalness bake) are not scaled by the reflection filter in
    # the MDL, so they keep their baked glossiness.
    spec_scale = np.clip(spec_max / GLTF_DIELECTRIC_SPECULAR, 0.0, 1.0)
    spec_scale = spec_scale + (1.0 - spec_scale) * metal_vray
    roughness = 1.0 - (1.0 - roughness) * spec_scale

    metallic = np.clip(metal_vray, 0.0, 1.0)
    base_lin = diffuse_lin

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


def _direction_to_quat_xyzw(direction):
    """Rotation (xyzw) orienting a glTF light so its local -Z travels along `direction`."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    rotation, _ = R.align_vectors([d], [[0.0, 0.0, -1.0]])
    return [float(v) for v in rotation.as_quat()]


def _make_lights_cameras_postprocessor(lights, cameras):
    """Build a trimesh export tree_postprocessor that injects KHR_lights_punctual
    lights + perspective cameras into the glTF header, and lifts the emissive
    skybox/light-marker materials with KHR_materials_emissive_strength.

    trimesh's glTF exporter cannot write lights (no KHR_lights_punctual support),
    writes at most one camera, and cannot set emissive strength, so we mutate the
    raw header dict on the way out. These are pure-JSON additions (no buffer/
    accessor data), so this needs no extra dependency and produces spec-valid glTF.

    Args:
        lights (list): light dicts, each with "kind" ("point" or "directional"),
            "color" (linear rgb 0-1), "intensity" (candela for point, lux for
            directional), "name", and either "position" (point) or "direction"
            (directional; world vector the light travels along).
        cameras (list): dicts with "position" (xyz), "quat_xyzw", and "name".
            glTF cameras look down -Z with +Y up, matching the scene camera
            convention used elsewhere in the pipeline.
    """

    def postprocessor(tree):
        nodes = tree["nodes"]
        scene_node_indices = tree["scenes"][0].setdefault("nodes", [])
        added_node_indices = []
        extensions_used = set(tree.get("extensionsUsed", []))

        for camera in cameras:
            camera_array = tree.setdefault("cameras", [])
            camera_index = len(camera_array)
            camera_array.append(
                {
                    "type": "perspective",
                    "name": camera["name"],
                    "perspective": {"yfov": CAMERA_YFOV, "znear": CAMERA_ZNEAR},
                }
            )
            nodes.append(
                {
                    "name": camera["name"],
                    "translation": [float(v) for v in camera["position"]],
                    "rotation": [float(v) for v in camera["quat_xyzw"]],
                    "camera": camera_index,
                }
            )
            added_node_indices.append(len(nodes) - 1)

        if lights:
            punctual = tree.setdefault("extensions", {}).setdefault("KHR_lights_punctual", {})
            light_array = punctual.setdefault("lights", [])
            for light in lights:
                light_index = len(light_array)
                name = light.get("name", f"light_{light_index}")
                light_array.append(
                    {
                        "type": light["kind"],
                        "name": name,
                        "color": [float(v) for v in light["color"]],
                        "intensity": float(light["intensity"]),
                    }
                )
                node = {"name": name, "extensions": {"KHR_lights_punctual": {"light": light_index}}}
                if light["kind"] == "directional":
                    node["rotation"] = _direction_to_quat_xyzw(light["direction"])
                else:
                    node["translation"] = [float(v) for v in light["position"]]
                nodes.append(node)
                added_node_indices.append(len(nodes) - 1)
            extensions_used.add("KHR_lights_punctual")

        # Lift emissive backdrops/markers so they read as bright emitters against
        # the physically-lit surfaces (renderers without the extension clamp to 1).
        for material in tree.get("materials", []):
            if material.get("name") in EMISSIVE_MATERIAL_NAMES:
                material.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": EMISSIVE_STRENGTH
                }
                extensions_used.add("KHR_materials_emissive_strength")

        if extensions_used:
            tree["extensionsUsed"] = sorted(extensions_used)
        scene_node_indices.extend(added_node_indices)

    return postprocessor


def _object_lights_in_base_frame(links, link_fk, meta_links):
    """Collect an object's light meta links, expressed in its base link frame.

    metadata.json stores lights per parent link as
    meta_links[link_name]["lights"][light_id] = [ {position, orientation, color,
    type, length, width, ...}, ... ] with poses in that parent link's frame. We
    fold in link_fk (rest pose) so the result (full pose plus emitter dimensions)
    is in the object's base frame, matching the object GLB geometry.
    """
    name_to_link = {link.name: link for link in links}
    lights = []
    for link_name, meta_types in meta_links.items():
        if "lights" not in meta_types:
            continue
        link = name_to_link.get(link_name)
        if link is None:
            continue
        link_transform = link_fk[link]
        for light_list in meta_types["lights"].values():
            for light in light_list:
                pose = np.eye(4)
                pose[:3, :3] = R.from_quat(light.get("orientation", [0, 0, 0, 1])).as_matrix()
                pose[:3, 3] = light["position"]
                pose_base = link_transform @ pose
                color = (np.array(light["color"], dtype=np.float64) / 255.0).tolist()
                light_type = int(light.get("type", _LIGHT_TYPE_SPHERE))
                length = float(light.get("length", 0.0))

                # Push the punctual light out of its (opaque) marker mesh along the
                # emitter's -Z so the mesh can't self-occlude it. The clearance is
                # the mesh's half-extent along that axis, matching _emissive_light_mesh.
                if light_type == _LIGHT_TYPE_SPHERE:
                    clearance = length if length > 1e-4 else LIGHT_MESH_FALLBACK_RADIUS
                elif light_type in (_LIGHT_TYPE_DISK, _LIGHT_TYPE_RECT):
                    clearance = LIGHT_MESH_THICKNESS / 2.0
                else:
                    clearance = LIGHT_MESH_FALLBACK_RADIUS
                emit_dir = pose_base[:3, :3] @ np.array([0.0, 0.0, -1.0])
                emit_position = pose_base[:3, 3] + emit_dir * (clearance + POINT_LIGHT_OFFSET)

                lights.append(
                    {
                        "pose": pose_base.tolist(),
                        "position": pose_base[:3, 3].tolist(),  # marker-mesh location
                        "emit_position": emit_position.tolist(),  # offset punctual-light location
                        "color": color,
                        "type": light_type,
                        "length": length,
                        "width": float(light.get("width", 0.0)),
                    }
                )
    return lights


def urdf_to_glb(in_obj_dir, out_obj_dir):
    # Get the URDF file
    urdf_files = glob.glob(in_obj_dir + "/urdf/*.urdf")
    assert (
        len(urdf_files) == 1
    ), f"Expected exactly one URDF file in {in_obj_dir}, found {len(urdf_files)}"

    urdf_file = urdf_files[0]
    urdf_dir = os.path.dirname(urdf_file)

    # Load the link tags (for Tglass overrides) and light meta links.
    link_tags = {}
    meta_links = {}
    metadata_file = os.path.join(in_obj_dir, "misc", "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        link_tags = metadata.get("link_tags") or {}
        meta_links = metadata.get("meta_links") or {}

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

    # Object lights, in the base frame; also baked into the per-object GLB. Each
    # gets an emissive marker sphere (part of the geometry, so it rides along when
    # the object is instanced into a scene) plus a punctual light for illumination.
    lights = _object_lights_in_base_frame(links, link_fk, meta_links)
    for light in lights:
        scene.add_geometry(_emissive_light_mesh(light))
    point_specs = [
        {"kind": "point", "position": light["emit_position"], "color": light["color"], "intensity": POINT_LIGHT_INTENSITY}
        for light in lights
    ]

    model_id = os.path.splitext(os.path.basename(urdf_file))[0]
    out_file = os.path.join(out_obj_dir, f"{model_id}.glb")
    data = scene.export(file_type="glb", tree_postprocessor=_make_lights_cameras_postprocessor(point_specs, []))
    with open(out_file, "wb") as f:
        f.write(data)

    # Returned to the scene phase so it can place these lights per instance.
    return lights


def _instance_transform(bbox_center, bbox_rot, scale):
    """Transform placing an object GLB (in its unscaled base frame) into the scene.

    Mirrors DatasetObject: the object is scaled in its local frame, rotated, then
    positioned so its bounding-box center lands at the scene pose.
    """
    transform = np.eye(4)
    transform[:3, :3] = bbox_rot.as_matrix() @ np.diag(scale)
    transform[:3, 3] = bbox_center
    return transform


def scene_urdf_to_glb(urdf_str, obj_glb_root, obj_src_root, lights_by_model, out_file):
    """Compose a full scene GLB from the per-object GLBs referenced by a scene URDF.

    Objects are instanced (shared mesh, one node per instance) at their per-instance
    scaled pose; each object's lights are placed per instance, and the scene cameras
    are emitted as glTF cameras.

    Args:
        urdf_str (str): The scene URDF (as written by export_scenes_global.py).
        obj_glb_root (str): Directory holding objects/{cat}/{model}/{model}.glb.
        obj_src_root (str): Directory holding objects/{cat}/{model}/misc/metadata.json.
        lights_by_model (dict): {"{cat}/{model}": [ {position, color}, ... ]} in base frame.
        out_file (str): Output scene GLB path.
    """
    root = ET.parse(io.StringIO(urdf_str)).getroot()
    joints_by_child = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}

    scene = trimesh.Scene()
    model_geom_names = {}  # (cat, model) -> [shared geometry name, ...] registered in the scene
    metadata_cache = {}  # (cat, model) -> (native_bbox, base_link_offset)
    scene_lights = []

    for link in root.findall("link"):
        if link.attrib.get("name") == "world":
            continue
        obj_category = link.attrib["category"]
        obj_model = link.attrib["model"]
        obj_name = link.attrib["name"]
        model_key = (obj_category, obj_model)

        try:
            glb_path = os.path.join(obj_glb_root, "objects", obj_category, obj_model, f"{obj_model}.glb")
            if not os.path.exists(glb_path):
                print(f"Skipping {obj_name}: no object GLB at {glb_path}")
                continue

            bbox_size = np.fromstring(link.attrib["bounding_box"], sep=" ")
            joint = joints_by_child[obj_name]
            origin = joint.find("origin")
            bbox_center = np.fromstring(origin.attrib["xyz"], sep=" ")
            bbox_rot = R.from_euler("xyz", np.fromstring(origin.attrib["rpy"], sep=" "))

            # Per-instance scale relative to the object's native bbox (as DatasetObject).
            if model_key not in metadata_cache:
                metadata_path = os.path.join(obj_src_root, "objects", obj_category, obj_model, "misc", "metadata.json")
                with open(metadata_path, "r") as f:
                    obj_metadata = json.load(f)
                metadata_cache[model_key] = (
                    np.array(obj_metadata["bbox_size"], dtype=np.float64),
                    np.array(obj_metadata["base_link_offset"], dtype=np.float64),
                )
            native_bbox, base_link_offset = metadata_cache[model_key]
            scale = np.ones(3)
            valid = native_bbox > 1e-4
            scale[valid] = bbox_size[valid] / native_bbox[valid]

            # The scene URDF places the bbox center; the object GLB is in the base
            # frame, so offset by the scaled base link offset.
            base_link_pos = bbox_center - bbox_rot.apply(scale * base_link_offset)
            transform = _instance_transform(base_link_pos, bbox_rot, scale)

            # Register the object's geometry once, then instance it per occurrence.
            if model_key not in model_geom_names:
                template = trimesh.load(glb_path)
                names = []
                for geom_name, geometry in template.geometry.items():
                    shared_name = f"{obj_category}-{obj_model}::{geom_name}"
                    scene.geometry[shared_name] = geometry
                    names.append(shared_name)
                model_geom_names[model_key] = names
            for shared_name in model_geom_names[model_key]:
                scene.graph.update(
                    frame_from="world",
                    frame_to=f"{obj_name}::{shared_name}",
                    matrix=transform,
                    geometry=shared_name,
                )

            # Place this object's fixtures (base frame) into the world frame. Use
            # the offset emit_position so the punctual light clears its marker mesh.
            for i, light in enumerate(lights_by_model.get(f"{obj_category}/{obj_model}", [])):
                position = np.array(light["emit_position"], dtype=np.float64)
                world_position = (transform @ np.append(position, 1.0))[:3]
                scene_lights.append(
                    {
                        "kind": "point",
                        "position": world_position.tolist(),
                        "color": light["color"],
                        "intensity": POINT_LIGHT_INTENSITY,
                        "name": f"{obj_name}_light_{i}",
                    }
                )
        except Exception:
            print(f"Failed to place {obj_name}:")
            traceback.print_exc()

    # Skybox: a large emissive cube enclosing the placed geometry (glTF has no
    # native skybox), reusing the same image as the MJCF skybox.
    if scene.geometry and os.path.exists(SKYBOX_SOURCE):
        bounds = scene.bounds
        center = bounds.mean(axis=0)
        half_size = max((bounds[1] - bounds[0]).max() / 2.0 * SKYBOX_MARGIN, SKYBOX_MIN_HALF_SIZE)
        scene.add_geometry(
            _skybox_mesh(center, half_size, Image.open(SKYBOX_SOURCE).convert("RGB")),
            geom_name="skybox",
        )

    # Scene cameras (stored as <camera> elements by export_scenes_global.py). The
    # stored quaternion is xyzw in the -Z-forward/+Y-up convention glTF expects.
    cameras = []
    for camera in root.findall("camera"):
        cameras.append(
            {
                "name": f"camera_{camera.attrib['name']}",
                "position": np.fromstring(camera.attrib["xyz"], sep=" ").tolist(),
                "quat_xyzw": np.fromstring(camera.attrib["quat"], sep=" ").tolist(),
            }
        )

    # Base key/fill/up directional lights give even illumination regardless of
    # which objects carry fixtures; the per-fixture point lights are accents.
    base_lights = [
        {
            "kind": "directional",
            "direction": spec["direction"],
            "color": spec["color"],
            "intensity": spec["intensity"],
            "name": name,
        }
        for name, spec in zip(("key_light", "fill_light", "up_fill_light"), SCENE_DIRECTIONAL_LIGHTS)
    ]

    data = scene.export(
        file_type="glb",
        tree_postprocessor=_make_lights_cameras_postprocessor(base_lights + scene_lights, cameras),
    )
    with open(out_file, "wb") as f:
        f.write(data)


def main():
    with (
        b1k_pipeline.utils.ParallelZipFS("objects.zip") as source_fs,
        b1k_pipeline.utils.ParallelZipFS("scenes.zip") as scenes_fs,
        TempFS(temp_dir=str(b1k_pipeline.utils.TMP_DIR)) as temp_fs,
        b1k_pipeline.utils.ParallelZipFS("objects_glb.zip", write=True) as out_fs,
    ):
        # Read the target scene URDFs up front, collecting the models they use.
        scene_urdfs = {}  # (scene_name, suffix) -> urdf string
        needed_models = set()  # (cat, model) referenced by the target scenes
        for target in b1k_pipeline.utils.get_targets("final_scenes"):
            scene_name = target.split("/")[-1]
            if SCENES_TO_EXPORT and scene_name not in SCENES_TO_EXPORT:
                continue
            # TODO: Temporarily exporting only the "best" (uncluttered) version.
            for suffix in ["best"]:
                urdf_path = f"scenes/{scene_name}/urdf/{scene_name}_{suffix}.urdf"
                if not scenes_fs.exists(urdf_path):
                    continue
                urdf_str = scenes_fs.readtext(urdf_path)
                scene_urdfs[(scene_name, suffix)] = urdf_str
                for link in ET.fromstring(urdf_str).findall("link"):
                    if link.attrib.get("name") != "world":
                        needed_models.add((link.attrib["category"], link.attrib["model"]))

        # When exporting only a subset of scenes, build just the object GLBs those
        # scenes need. A full run (empty filter) builds every object.
        restrict_objects = bool(SCENES_TO_EXPORT)

        # Copy the object sources over to the temp FS (also used by the scene phase
        # to read object metadata).
        print("Copying input to temp fs...")
        objdir_glob = [item.path for item in source_fs.glob("objects/*/*/")]
        objdirs_to_build = []
        for item in tqdm.tqdm(objdir_glob):
            cat, model = item.strip("/").split("/")[-2:]
            if restrict_objects and (cat, model) not in needed_models:
                continue
            if source_fs.opendir(item).opendir("urdf").glob("*.urdf").count().files == 0:
                continue
            fs.copy.copy_fs(source_fs.opendir(item), temp_fs.makedirs(item, recreate=True))
            objdirs_to_build.append(item)

        with futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Phase 1: export a GLB per object and collect each object's lights.
            obj_futures = {
                executor.submit(
                    urdf_to_glb,
                    temp_fs.opendir(objdir).getsyspath("/"),
                    out_fs.makedirs(objdir, recreate=True).getsyspath("/"),
                ): objdir
                for objdir in objdirs_to_build
            }

            lights_by_model = {}  # "{cat}/{model}" -> lights in base frame
            for future in tqdm.tqdm(
                futures.as_completed(obj_futures), total=len(obj_futures), desc="Processing objects"
            ):
                objdir = obj_futures[future]
                try:
                    lights = future.result()
                    if lights:
                        # objdir is /objects/{cat}/{model}/
                        cat, model = objdir.strip("/").split("/")[-2:]
                        lights_by_model[f"{cat}/{model}"] = lights
                except:
                    traceback.print_exc()

            # Phase 2: compose a GLB per scene from the object GLBs, adding lights/cameras.
            obj_glb_root = out_fs.getsyspath("/")
            obj_src_root = temp_fs.getsyspath("/")
            scene_futures = {}
            for (scene_name, suffix), urdf_str in scene_urdfs.items():
                out_file = os.path.join(
                    out_fs.makedirs(f"scenes/{scene_name}", recreate=True).getsyspath("/"),
                    f"{scene_name}_{suffix}.glb",
                )
                scene_futures[
                    executor.submit(
                        scene_urdf_to_glb,
                        urdf_str,
                        obj_glb_root,
                        obj_src_root,
                        lights_by_model,
                        out_file,
                    )
                ] = f"{scene_name}_{suffix}"

            for future in tqdm.tqdm(
                futures.as_completed(scene_futures), total=len(scene_futures), desc="Processing scenes"
            ):
                try:
                    future.result()
                except:
                    traceback.print_exc()

        print("Finished processing")


if __name__ == "__main__":
    main()
