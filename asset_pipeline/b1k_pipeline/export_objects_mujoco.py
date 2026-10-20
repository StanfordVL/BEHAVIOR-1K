import argparse
import dataclasses
import glob
import json
import multiprocessing
import os
import pathlib
import re
import shutil
import tempfile
import traceback
import xml.etree.ElementTree as ET
import zipfile
from concurrent import futures
from xml.dom import minidom

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

import tqdm

import b1k_pipeline.utils
from b1k_pipeline import mujoco_common
from b1k_pipeline.export_glbs import (
    GLASS_MATERIAL_KWARGS,
    _apply_material,
    _get_material_maps,
    convert_vray_to_pbr_material,
)
from b1k_pipeline.urdfpy import URDF

# Average category specs compiled by aggregate_metadata.py into metadata.zip
# (the source of the copy shipped with OmniGibson): used to replicate the
# category-mass override that DatasetObject applies at load time.
AVG_CATEGORY_SPECS_ZIP = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "parallels" / "metadata.zip"
_AVG_CATEGORY_SPECS = None


def _get_avg_category_specs():
    global _AVG_CATEGORY_SPECS
    if _AVG_CATEGORY_SPECS is None:
        with zipfile.ZipFile(AVG_CATEGORY_SPECS_ZIP) as zf, zf.open("metadata/avg_category_specs.json") as f:
            _AVG_CATEGORY_SPECS = json.load(f)
    return _AVG_CATEGORY_SPECS

# Matches OmniGibson's minimum per-link mass in the FORCE_CATEGORY_MASS path.
MIN_LINK_MASS = 1e-6

META_LINK_NAME_PATTERN = re.compile(
    r"^meta__(?P<parent>.+)_(?P<type>[a-z]+)_(?P<id>[A-Za-z0-9]+)_(?P<subid>[0-9]+)_link$"
)

# Same taxonomy as OmniGibson's asset_conversion_utils._ALLOWED_META_TYPES.
META_TYPE_REPRESENTATIONS = {
    "particlesource": "dimensionless",
    "togglebutton": "primitive",
    "attachment": "dimensionless",
    "heatsource": "dimensionless",
    "particleapplier": "primitive",
    "particleremover": "primitive",
    "particlesink": "primitive",
    "slicer": "primitive",
    "container": "primitive",
    "lights": "light",
}

DIMENSIONLESS_SITE_RADIUS = 0.005

# Geom/site groups (see mujoco_common): 2 = visual, 3 = collision, 4 = meta
# volumes/sites. Groups above 2 are hidden by default in the MuJoCo visualizer.
VISUAL_GROUP = str(mujoco_common.VISUAL_GROUP)
COLLISION_GROUP = str(mujoco_common.COLLISION_GROUP)
META_GROUP = str(mujoco_common.META_GROUP)

# Collision pieces thinner than this along an axis, in meters, are dropped:
# MuJoCo can't compute their convex hull.
DEGENERATE_EXTENT = 1e-5

# Volume meta link types: their meshes become geoms rather than sites.
VOLUME_META_TYPES = ("fillable", "openfillable")


def _fmt(values):
    return " ".join(f"{v:.10g}" for v in np.asarray(values).flatten())


def _pose_attrs(transform):
    pos = transform[:3, 3]
    quat = trimesh.transformations.quaternion_from_matrix(transform)  # wxyz
    attrs = {}
    if np.linalg.norm(pos) > 0:
        attrs["pos"] = _fmt(pos)
    if not np.allclose(quat, [1, 0, 0, 0], atol=1e-12):
        attrs["quat"] = _fmt(quat)
    return attrs


def _xyzw_to_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def _get_category_mass(in_obj_dir):
    """Compute the object's target total mass following DatasetObject's
    category-mass override (gm.FORCE_CATEGORY_MASS=True semantics)."""
    category = pathlib.Path(in_obj_dir).parts[-2]
    avg_category_specs = _get_avg_category_specs()
    if category in avg_category_specs:
        return avg_category_specs[category]["mass"]
    print(f"Category {category} not found in average object specs! Defaulting to unit mass.")
    return 1.0


def _strip_obj_materials(src_path, dst_path):
    """Copy an OBJ file, dropping MTL references (MuJoCo materials come from the MJCF)."""
    with open(src_path, "r") as src, open(dst_path, "w") as dst:
        for line in src:
            if line.startswith("mtllib ") or line.startswith("usemtl "):
                continue
            dst.write(line)


def _link_visual_meshes(link, urdf_dir):
    """Yield (mesh, transform, obj_path) for each visual mesh of a link, in link frame."""
    for visual in link.visuals:
        if visual.geometry.mesh is None:
            continue
        pose = visual.origin.copy()
        if visual.geometry.mesh.scale is not None:
            S = np.eye(4)
            S[:3, :3] = np.diag(visual.geometry.mesh.scale)
            pose = pose.dot(S)
        obj_path = os.path.normpath(os.path.join(urdf_dir, visual.geometry.mesh.filename))
        for mesh in visual.geometry.meshes:
            yield mesh, pose, obj_path


def _export_link_glb(link, urdf_dir, is_glass, out_path):
    """Export a link's visual meshes as a GLB in the link frame, with the same
    PBR conversion as export_glbs. Returns whether a GLB was written."""
    scene = trimesh.Scene()
    material = None
    for mesh, pose, obj_path in _link_visual_meshes(link, urdf_dir):
        m = mesh.copy()
        m.apply_transform(pose)
        if is_glass:
            if material is None:
                material = trimesh.visual.material.PBRMaterial(**GLASS_MATERIAL_KWARGS)
        elif material is None:
            maps = _get_material_maps(obj_path)
            if maps:
                name = os.path.splitext(os.path.basename(obj_path))[0]
                material = convert_vray_to_pbr_material(name, maps)
        if material is not None:
            _apply_material(m, material)
        scene.add_geometry(m)
    if not scene.geometry:
        return False
    scene.export(out_path)
    return True


def _record_meta_element(meta_records, meta_type, meta_id, link, body, element_xml):
    """Record a meta element in b1k_metadata's `meta_links` table.

    Positions, orientations and sizes are in the frame of the body that holds
    the element, unscaled, and names are those written into this MJCF (attach
    prefixes them in scenes)."""
    attrib = element_xml.attrib
    record = {"link": link, "body": body, "element": element_xml.tag, "name": attrib["name"]}
    if element_xml.tag in ("site", "geom"):
        record["shape"] = attrib.get("type", "sphere")
    for key in ("pos", "quat", "size", "dir"):
        if key in attrib:
            record[key] = [float(x) for x in attrib[key].split()]
    if "mesh" in attrib:
        record["mesh"] = attrib["mesh"]
    meta_records.setdefault(meta_type, {}).setdefault(str(meta_id), []).append(record)


def _add_meta_sites_from_metadata(body_xml, link_name, meta_links, meta_records):
    """Add sites/lights for the metadata-defined meta links of a link.

    Mirrors OmniGibson's _generate_meshes_for_primitive_meta_links conventions:
    primitives use size[0] as radius and size[2] as height with the shape origin
    at the base (offset by half height along local z); particleapplier cones are
    flipped so the site's -z is the emission direction. Each element is also
    recorded in `meta_records` (see _record_meta_element).
    """
    for meta_type, meta_link_infos in meta_links.items():
        representation = META_TYPE_REPRESENTATIONS.get(meta_type)
        if representation is None:
            continue
        for link_id, mesh_info_list in meta_link_infos.items():
            if isinstance(mesh_info_list, dict):
                keys = [str(x) for x in range(len(mesh_info_list))]
                mesh_info_list = [mesh_info_list[k] for k in keys]
            for i, mesh_info in enumerate(mesh_info_list):
                name = f"meta__{link_name}_{meta_type}_{link_id}_{i}"
                pos = np.asarray(mesh_info["position"], dtype=np.float64)
                orn = R.from_quat(mesh_info["orientation"])

                if representation == "light":
                    light_xml = ET.SubElement(body_xml, "light")
                    light_xml.attrib = {
                        "name": name,
                        "pos": _fmt(pos),
                        "dir": _fmt(orn.apply([0, 0, -1])),
                        "diffuse": _fmt(np.asarray(mesh_info["color"], dtype=np.float64) / 255.0),
                        "castshadow": "false",
                    }
                    _record_meta_element(meta_records, meta_type, link_id, link_name, link_name, light_xml)
                    continue

                site_attrs = {"name": name, "class": "meta_site"}
                if representation == "dimensionless":
                    if meta_type == "particlesource":
                        radius, height = 0.0125, 0.05
                        center = pos + orn.apply([0, 0, -height / 2.0])
                        site_attrs.update(
                            type="cylinder", size=_fmt([radius, height / 2.0]), pos=_fmt(center)
                        )
                    else:
                        site_attrs.update(
                            type="ellipsoid",
                            size=_fmt([DIMENSIONLESS_SITE_RADIUS] * 3),
                            pos=_fmt(pos),
                        )
                    site_attrs["quat"] = _fmt(_xyzw_to_wxyz(orn.as_quat()))
                else:  # primitive
                    shape_type = mesh_info["type"]
                    size = np.asarray(mesh_info.get("size", [0.01, 0.01, 0.01]), dtype=np.float64)
                    if shape_type == "sphere":
                        site_attrs.update(
                            type="ellipsoid", size=_fmt([size[0]] * 3), pos=_fmt(pos)
                        )
                        site_attrs["quat"] = _fmt(_xyzw_to_wxyz(orn.as_quat()))
                    elif shape_type == "box":
                        center = pos + orn.apply([0, 0, size[2] / 2.0])
                        site_attrs.update(type="box", size=_fmt(size / 2.0), pos=_fmt(center))
                        site_attrs["quat"] = _fmt(_xyzw_to_wxyz(orn.as_quat()))
                    elif shape_type in ("cylinder", "cone"):
                        # MuJoCo has no cone sites; approximate cones with their
                        # bounding cylinder. Cones (particleapplier) are flipped
                        # 180 degrees about x so the site's -z is the emission
                        # direction, matching OmniGibson.
                        radius, height = size[0], size[2]
                        if shape_type == "cone":
                            orn = orn * R.from_euler("x", np.pi)
                            center = pos + orn.apply([0, 0, -height / 2.0])
                        else:
                            center = pos + orn.apply([0, 0, height / 2.0])
                        site_attrs.update(
                            type="cylinder", size=_fmt([radius, height / 2.0]), pos=_fmt(center)
                        )
                        site_attrs["quat"] = _fmt(_xyzw_to_wxyz(orn.as_quat()))
                    else:
                        raise ValueError(f"Unknown meta link primitive type {shape_type} for {name}")
                site_xml = ET.SubElement(body_xml, "site", site_attrs)
                _record_meta_element(meta_records, meta_type, link_id, link_name, link_name, site_xml)


def _xml_rotation(element_xml):
    """Rotation matrix of an MJCF element's `quat` (the only orientation
    attribute this exporter writes)."""
    for attr in ("euler", "axisangle", "xyaxes", "zaxis"):
        assert attr not in element_xml.attrib, f"Unexpected {attr} on {element_xml.attrib.get('name')}"
    if "quat" not in element_xml.attrib:
        return np.eye(3)
    w, x, y, z = (float(v) for v in element_xml.attrib["quat"].split())
    return R.from_quat([x, y, z, w]).as_matrix()


def anisotropic_scaling_problems(root):
    """Why a frame scale `sx sy sz` with `scaling="geometry"` wouldn't compile.

    MuJoCo stretches each shape in its own coordinates, so a non-uniform frame
    scale requires every hinge and slide joint, and every box, ellipsoid or
    plane geom, to be aligned with the frame's axes; meshes always work, and
    other primitives can't be stretched arbitrarily. Objects are scaled in
    their base link's frame, so alignment is checked in that frame.

    Returns:
        A list of problems, empty if any anisotropic scale works.
    """
    problems = []

    def visit(body_xml, rotation):
        for joint_xml in body_xml.findall("joint"):
            joint_type = joint_xml.attrib.get("type", "hinge")
            if joint_type not in ("hinge", "slide"):
                problems.append(f"joint {joint_xml.attrib.get('name')} has type {joint_type}")
                continue
            axis = np.array([float(v) for v in joint_xml.attrib.get("axis", "0 0 1").split()])
            if not mujoco_common.is_axis_aligned(rotation @ axis):
                problems.append(
                    f"joint {joint_xml.attrib.get('name')} axis {_fmt(rotation @ axis)} is not aligned"
                )
        for geom_xml in body_xml.findall("geom"):
            geom_type = geom_xml.attrib.get("type", "sphere")
            name = geom_xml.attrib.get("name")
            if "fromto" in geom_xml.attrib:
                problems.append(f"geom {name} uses fromto")
            elif geom_type == "mesh":
                continue
            elif geom_type in ("box", "ellipsoid", "plane"):
                if not mujoco_common.is_frame_aligned(rotation @ _xml_rotation(geom_xml)):
                    problems.append(f"{geom_type} geom {name} is not aligned")
            else:
                problems.append(f"{geom_type} geom {name} can't be stretched anisotropically")
        for child_xml in body_xml.findall("body"):
            visit(child_xml, rotation @ _xml_rotation(child_xml))
        for frame_xml in body_xml.findall("frame"):
            visit(frame_xml, rotation @ _xml_rotation(frame_xml))

    for base_body_xml in root.find("worldbody").findall("body"):
        # The base link's own orientation is the frame the scale applies in.
        visit(base_body_xml, np.eye(3))

    for tag in ("flexcomp", "flex", "skin", "plugin"):
        if root.find(f".//{tag}") is not None:
            problems.append(f"contains <{tag}>")

    return problems


def assert_anisotropically_scalable(root):
    """Raise ValueError, listing every problem, unless any anisotropic frame
    scale of the object compiles."""
    problems = anisotropic_scaling_problems(root)
    if problems:
        model_id = root.attrib.get("model", "<unknown>")
        raise ValueError(f"[{model_id}] not anisotropically scalable: " + "; ".join(problems))


def _openable_metadata(metadata, joint_names):
    """The joints OmniGibson's Open state checks, as _get_relevant_joints sees them.

    OmniGibson's USD conversion turns metadata.json's `[[id, name], ...]` into a
    `{str(id): name}` dict, and _get_relevant_joints reads `.items()` of it, so
    every relevant joint has direction 1. Without annotations, every joint is
    relevant, with direction 1.
    """
    raw_ids = metadata.get("openable_joint_ids")
    both_sides = bool(metadata.get("openable_both_sides", False))
    if raw_ids:
        as_seen = {str(pair[0]): pair[1] for pair in raw_ids}
        names = list(as_seen.values())
        for name in names:
            assert name in joint_names, f"Openable joint {name} is not a joint of the object"
        joints = [{"joint": name, "direction": 1} for name in names]
        both_sides_seen = both_sides
    else:
        joints = [{"joint": name, "direction": 1} for name in joint_names]
        both_sides_seen = False
    return {
        "openable_joint_ids": raw_ids,
        "openable_both_sides": metadata.get("openable_both_sides"),
        "both_sides": both_sides_seen,
        "joints": joints,
    }


JOINT_FIXES_FILE = b1k_pipeline.utils.PIPELINE_ROOT / "metadata" / "mujoco_joint_fixes.json"
_JOINT_FIXES = None


def _joint_fixes():
    """Per-object joint fixes, from metadata/mujoco_joint_fixes.json."""
    global _JOINT_FIXES
    if _JOINT_FIXES is None:
        _JOINT_FIXES = json.loads(JOINT_FIXES_FILE.read_text())["objects"] if JOINT_FIXES_FILE.exists() else {}
    return _JOINT_FIXES


_TAXONOMY = None


def _taxonomy_info(category):
    """(synset, abilities) of a category from bddl's object taxonomy."""
    global _TAXONOMY
    if _TAXONOMY is None:
        from bddl.object_taxonomy import ObjectTaxonomy

        _TAXONOMY = ObjectTaxonomy()
    synset = _TAXONOMY.get_synset_from_category(category)
    if synset is None:
        return None, {}
    return synset, _TAXONOMY.get_abilities(synset)


@dataclasses.dataclass
class ObjectMjcf:
    """An object's MJCF and the files its archive needs.

    Attributes:
        root: The <mujoco> element.
        files: Archive name -> (mode, source path); mode "copy" copies the
            file, "strip_obj" copies an OBJ without its material references.
        images: Archive name -> PIL image, written as PNG.
        metadata: The b1k_metadata dict embedded in `root`.
        tmp_dir: Scratch directory holding generated files; the caller removes it.
    """

    root: ET.Element
    files: dict
    images: dict
    metadata: dict
    tmp_dir: str


def mjcf_to_string(root):
    return minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")


def write_mjz(obj_mjcf, out_file):
    """Write an object's MJZ: model.xml plus every referenced file."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(mujoco_common.ARCHIVE_ROOT_FILE, mjcf_to_string(obj_mjcf.root))
        for archive_name, (mode, src_path) in obj_mjcf.files.items():
            if mode == "strip_obj":
                stripped_path = os.path.join(obj_mjcf.tmp_dir, "stripped_" + os.path.basename(archive_name))
                _strip_obj_materials(src_path, stripped_path)
                zf.write(stripped_path, archive_name)
            else:
                zf.write(src_path, archive_name)
        for archive_name, image in obj_mjcf.images.items():
            with zf.open(archive_name, "w") as f:
                image.save(f, format="png")


def validate_mjz(mjz_path):
    """Compile an archive with MuJoCo, step it forward once, and check that its
    bounds are finite. Imports MuJoCo lazily, since only --validate needs it."""
    import mujoco

    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(mjz_path) as zf:
        zf.extractall(tmp)
        model = mujoco.MjModel.from_xml_path(os.path.join(tmp, mujoco_common.ARCHIVE_ROOT_FILE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if not (np.all(np.isfinite(data.geom_xpos)) and np.all(np.isfinite(model.geom_rbound))):
        raise ValueError(f"{mjz_path}: non-finite geom positions or bounds")
    if model.ngeom == 0:
        raise ValueError(f"{mjz_path}: no geoms")


def build_object_mjcf(in_obj_dir):
    """Build an object's MJCF from its URDF directory (objects/{category}/{model}).

    Returns:
        ObjectMjcf. The caller removes its tmp_dir.
    """
    urdf_files = glob.glob(in_obj_dir + "/urdf/*.urdf")
    assert (
        len(urdf_files) == 1
    ), f"Expected exactly one URDF file in {in_obj_dir}, found {len(urdf_files)}"
    urdf_file = urdf_files[0]
    urdf_dir = os.path.dirname(urdf_file)
    model_id = os.path.splitext(os.path.basename(urdf_file))[0]
    category = pathlib.Path(in_obj_dir).parts[-2]

    metadata = {}
    metadata_file = os.path.join(in_obj_dir, "misc", "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    link_tags = metadata.get("link_tags") or {}
    meta_links = metadata.get("meta_links") or {}
    # Every meta element written, for b1k_metadata (see _record_meta_element).
    meta_records = {}
    # Collision pieces too flat for MuJoCo's convex hull.
    dropped_collisions = []

    robot = URDF.load(urdf_file)
    tmp_dir = tempfile.mkdtemp(prefix=f"mjcf_{model_id}_")

    # The category mass is distributed across the links' collision volumes,
    # replicating DatasetObject._post_load with gm.FORCE_CATEGORY_MASS=True.
    category_mass = _get_category_mass(in_obj_dir)
    link_volumes = {}
    for link in robot.links:
        volume = 0.0
        for collision in link.collisions:
            if collision.geometry.mesh is None:
                continue
            scale = collision.geometry.mesh.scale if collision.geometry.mesh.scale is not None else np.ones(3)
            for mesh in collision.geometry.meshes:
                volume += abs(mesh.volume) * abs(np.prod(scale))
        link_volumes[link.name] = volume
    total_volume = sum(link_volumes.values())
    assert total_volume > 0, f"Object {model_id} has no collision volume"

    # Assemble the MJCF
    root = ET.Element("mujoco", {"model": model_id})
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "meshdir": ".",
            "texturedir": ".",
            "inertiagrouprange": f"{COLLISION_GROUP} {COLLISION_GROUP}",
        },
    )
    default_xml = ET.SubElement(root, "default")
    visual_default = ET.SubElement(default_xml, "default", {"class": "visual"})
    ET.SubElement(visual_default, "geom", {"group": VISUAL_GROUP, "contype": "0", "conaffinity": "0", "mass": "0"})
    collision_default = ET.SubElement(default_xml, "default", {"class": "collision"})
    ET.SubElement(collision_default, "geom", {"group": COLLISION_GROUP})
    volume_default = ET.SubElement(default_xml, "default", {"class": "meta_volume"})
    ET.SubElement(
        volume_default,
        "geom",
        {"group": META_GROUP, "contype": "0", "conaffinity": "0", "mass": "0", "rgba": "0 0 1 0.15"},
    )
    site_default = ET.SubElement(default_xml, "default", {"class": "meta_site"})
    ET.SubElement(site_default, "site", {"group": META_GROUP, "rgba": "1 0 0 0.3"})

    asset_xml = ET.SubElement(root, "asset")
    worldbody_xml = ET.SubElement(root, "worldbody")

    # Files to pack into the MJZ: {archive name: source path on disk}
    archive_files = {}
    # Files generated in-memory as PIL images: {archive name: image}
    archive_images = {}

    def add_visual_assets(link, is_glass):
        """Create the visual mesh asset + PBR material for a link. Returns
        (mesh_asset_name or None, material_name or None, visual origin)."""
        mesh_visuals = [v for v in link.visuals if v.geometry.mesh is not None]
        if not mesh_visuals:
            return None, None, None
        assert len(mesh_visuals) == 1, f"Expected one visual mesh per link, got {len(mesh_visuals)}"
        (visual,) = mesh_visuals
        obj_path = os.path.normpath(os.path.join(urdf_dir, visual.geometry.mesh.filename))

        mesh_name = f"{link.name}_visual"
        archive_files[f"visual/{link.name}.obj"] = ("strip_obj", obj_path)
        # Shell inertia so that thin/non-watertight visual meshes compile; the
        # visual geoms are massless so the inertia is never used.
        mesh_attrs = {"name": mesh_name, "file": f"visual/{link.name}.obj", "inertia": "shell"}
        if visual.geometry.mesh.scale is not None and not np.allclose(visual.geometry.mesh.scale, 1.0):
            mesh_attrs["scale"] = _fmt(visual.geometry.mesh.scale)
        ET.SubElement(asset_xml, "mesh", mesh_attrs)

        material_name = f"{link.name}_material"
        if is_glass:
            ET.SubElement(
                asset_xml,
                "material",
                {
                    "name": material_name,
                    "rgba": _fmt(GLASS_MATERIAL_KWARGS["baseColorFactor"]),
                    "metallic": str(GLASS_MATERIAL_KWARGS["metallicFactor"]),
                    "roughness": str(GLASS_MATERIAL_KWARGS["roughnessFactor"]),
                },
            )
            return mesh_name, material_name, visual.origin

        maps = _get_material_maps(obj_path)
        if not maps:
            return mesh_name, None, visual.origin
        pbr = convert_vray_to_pbr_material(os.path.splitext(os.path.basename(obj_path))[0], maps)
        material_xml = ET.SubElement(asset_xml, "material", {"name": material_name})
        rgb_role = "rgba" if pbr.baseColorTexture.mode == "RGBA" else "rgb"
        layers = [(rgb_role, "albedo", pbr.baseColorTexture), ("orm", "orm", pbr.metallicRoughnessTexture)]
        if pbr.normalTexture is not None:
            layers.append(("normal", "normal", pbr.normalTexture))
        for role, suffix, image in layers:
            texture_name = f"{link.name}_{suffix}"
            texture_file = f"textures/{texture_name}.png"
            archive_images[texture_file] = image
            ET.SubElement(asset_xml, "texture", {"name": texture_name, "type": "2d", "file": texture_file})
            ET.SubElement(material_xml, "layer", {"texture": texture_name, "role": role})
        return mesh_name, material_name, visual.origin

    def add_regular_link(body_xml, link):
        link_mass = max(category_mass * (link_volumes[link.name] / total_volume), MIN_LINK_MASS)
        is_glass = "glass" in (link_tags.get(link.name) or [])

        # Visual geom (MuJoCo cannot load GLB, so the MJCF references the OBJ
        # with the converted PBR textures; the equivalent GLB is packed alongside).
        mesh_name, material_name, visual_origin = add_visual_assets(link, is_glass)
        if mesh_name is not None:
            geom_attrs = {"class": "visual", "type": "mesh", "mesh": mesh_name, "name": f"{link.name}_visual"}
            # The URDF places articulated links' frames at the joint and offsets
            # the meshes via the visual origin (mesh_offset in export_objs_global),
            # so the visual geom must carry that origin.
            geom_attrs.update(_pose_attrs(visual_origin))
            if material_name is not None:
                geom_attrs["material"] = material_name
            ET.SubElement(body_xml, "geom", geom_attrs)

            glb_path = os.path.join(tmp_dir, f"{link.name}.glb")
            if _export_link_glb(link, urdf_dir, is_glass, glb_path):
                archive_files[f"visual/{link.name}.glb"] = ("copy", glb_path)

        # Collision geoms carry the link's share of the category mass,
        # distributed across the CoACD pieces by volume.
        collision_specs = []
        for collision in link.collisions:
            if collision.geometry.mesh is None:
                continue
            scale = collision.geometry.mesh.scale if collision.geometry.mesh.scale is not None else np.ones(3)
            obj_path = os.path.normpath(os.path.join(urdf_dir, collision.geometry.mesh.filename))
            # MuJoCo can't take the convex hull of a flat piece (some CoACD
            # pieces are scaled to nothing along an axis): drop it.
            extents = np.max([np.ptp(m.vertices, axis=0) for m in collision.geometry.meshes], axis=0)
            extents = extents * np.abs(scale)
            if np.min(extents) < DEGENERATE_EXTENT:
                dropped_collisions.append(os.path.basename(obj_path))
                continue
            volume = sum(abs(mesh.volume) for mesh in collision.geometry.meshes) * abs(np.prod(scale))
            collision_specs.append((obj_path, scale, collision.origin, volume))

        for i, (obj_path, scale, origin, volume) in enumerate(collision_specs):
            geom_mass = max(link_mass * (volume / link_volumes[link.name]), MIN_LINK_MASS)
            mesh_name = f"{link.name}_collision_{i}"
            archive_name = f"collision/{os.path.basename(obj_path)}"
            archive_files[archive_name] = ("copy", obj_path)
            ET.SubElement(
                asset_xml, "mesh", {"name": mesh_name, "file": archive_name, "scale": _fmt(scale)}
            )
            geom_attrs = {
                "class": "collision",
                "type": "mesh",
                "mesh": mesh_name,
                "name": mesh_name,
                "mass": f"{geom_mass:.10g}",
            }
            geom_attrs.update(_pose_attrs(origin))
            ET.SubElement(body_xml, "geom", geom_attrs)

        # Meta links stored in metadata.json become sites/lights.
        _add_meta_sites_from_metadata(body_xml, link.name, meta_links.get(link.name) or {}, meta_records)

    def add_volume_meta_link(body_xml, link):
        """Fillable/openfillable meta links: the convex volume meshes become
        non-colliding massless geoms, plus a box site fitted to their AABB so
        MuJoCo's insidesite sensor can be used for containment checks."""
        meta_match = META_LINK_NAME_PATTERN.fullmatch(link.name)
        meta_type, meta_id, parent = meta_match["type"], meta_match["id"], meta_match["parent"]
        bounds = []
        for i, visual in enumerate(link.visuals):
            if visual.geometry.mesh is None:
                continue
            scale = visual.geometry.mesh.scale if visual.geometry.mesh.scale is not None else np.ones(3)
            obj_path = os.path.normpath(os.path.join(urdf_dir, visual.geometry.mesh.filename))
            mesh_name = f"{link.name}_{i}"
            archive_name = f"volumes/{os.path.basename(obj_path)}"
            archive_files[archive_name] = ("copy", obj_path)
            ET.SubElement(
                asset_xml,
                "mesh",
                {"name": mesh_name, "file": archive_name, "scale": _fmt(scale), "inertia": "shell"},
            )
            geom_attrs = {"class": "meta_volume", "type": "mesh", "mesh": mesh_name, "name": mesh_name}
            geom_attrs.update(_pose_attrs(visual.origin))
            geom_xml = ET.SubElement(body_xml, "geom", geom_attrs)
            _record_meta_element(meta_records, meta_type, meta_id, parent, link.name, geom_xml)
            pose = visual.origin.copy()
            S = np.eye(4)
            S[:3, :3] = np.diag(scale)
            pose = pose.dot(S)
            for mesh in visual.geometry.meshes:
                m = mesh.copy()
                m.apply_transform(pose)
                bounds.append(m.bounds)
        if bounds:
            bounds = np.array(bounds)
            lo, hi = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
            site_xml = ET.SubElement(
                body_xml,
                "site",
                {
                    "name": f"{link.name}_site",
                    "class": "meta_site",
                    "type": "box",
                    "pos": _fmt((lo + hi) / 2.0),
                    "size": _fmt(np.maximum((hi - lo) / 2.0, 1e-4)),
                },
            )
            _record_meta_element(meta_records, meta_type, meta_id, parent, link.name, site_xml)

    # Recursively build the body tree following the URDF articulation.
    children = {}  # parent link name -> list of (joint, child link)
    for joint in robot.joints:
        children.setdefault(joint.parent, []).append(joint)
    link_map = {link.name: link for link in robot.links}

    def add_body(parent_xml, link, joint, parent_rotation):
        body_xml = ET.SubElement(parent_xml, "body", {"name": link.name})
        rotation = parent_rotation
        if joint is not None:
            body_xml.attrib.update(_pose_attrs(joint.origin))
            rotation = parent_rotation @ joint.origin[:3, :3]
            fix = joint_fixes.get(joint.name)
            if fix is not None and fix["action"] == "fix":
                applied_fixes[joint.name] = fix
            elif joint.joint_type in ("revolute", "continuous", "prismatic"):
                axis = np.asarray(joint.axis, dtype=np.float64)
                if fix is not None:
                    # Snap the axis onto the nearest base-link axis, expressed
                    # in the joint frame.
                    assert fix["action"] == "snap", fix
                    k = "xyz".index(fix["nearest_base_axis"][1])
                    snapped = np.zeros(3)
                    snapped[k] = 1.0 if fix["nearest_base_axis"][0] == "+" else -1.0
                    axis = rotation.T @ snapped
                    applied_fixes[joint.name] = fix
                joint_attrs = {
                    "name": joint.name,
                    "type": "slide" if joint.joint_type == "prismatic" else "hinge",
                    "axis": _fmt(axis),
                }
                if joint.joint_type != "continuous" and joint.limit is not None:
                    joint_attrs["range"] = _fmt([joint.limit.lower, joint.limit.upper])
                    joint_attrs["limited"] = "true"
                ET.SubElement(body_xml, "joint", joint_attrs)
            elif joint.joint_type != "fixed":
                raise ValueError(f"Unsupported joint type {joint.joint_type} for {joint.name}")

        meta_match = META_LINK_NAME_PATTERN.fullmatch(link.name)
        if meta_match:
            add_volume_meta_link(body_xml, link)
        else:
            add_regular_link(body_xml, link)

        for child_joint in children.get(link.name, []):
            add_body(body_xml, link_map[child_joint.child], child_joint, rotation)

    # Joint fixes for anisotropic scaling (see find_mujoco_joint_fixes.py).
    joint_fixes = _joint_fixes().get(f"{category}/{model_id}", {}).get("joints", {})
    applied_fixes = {}

    # No freejoint on the base body: scene MJCFs (export_scenes_mujoco.py) add
    # one at attach time for loose objects, while fixed objects are attached
    # statically. A soft weld on a freejointed base would drift; a static
    # attach is exact. Standalone, the object loads as a fixed-base model.
    base_body_xml = ET.SubElement(worldbody_xml, "body", {"name": robot.base_link.name})
    meta_match = META_LINK_NAME_PATTERN.fullmatch(robot.base_link.name)
    assert not meta_match, f"Base link {robot.base_link.name} should not be a meta link"
    add_regular_link(base_body_xml, robot.base_link)
    for child_joint in children.get(robot.base_link.name, []):
        add_body(base_body_xml, link_map[child_joint.child], child_joint, np.eye(3))
    missing = set(joint_fixes) - set(applied_fixes)
    assert not missing, f"Joint fixes for unknown joints of {category}/{model_id}: {sorted(missing)}"

    # Scenes scale objects with frame scales. Uniform scales always work, but
    # anisotropic ones need the object to be stretchable along its base link's
    # axes: record whether it is, so that scene exporters can assert it for
    # the objects they scale anisotropically.
    scaling_problems = anisotropic_scaling_problems(root)

    # Contact bitmasks of structure and of the moving parts of structural doors
    # (see mujoco_common.STRUCTURE_CONTACT).
    if category in mujoco_common.STRUCTURE_CATEGORIES:
        for geom in worldbody_xml.iter("geom"):
            if geom.attrib.get("class") == "collision":
                geom.attrib.update(mujoco_common.STRUCTURE_CONTACT)
    elif category in mujoco_common.STRUCTURAL_DOOR_CATEGORIES:

        def mark_moving(body_xml, moving):
            moving = moving or body_xml.find("joint") is not None
            if moving:
                for geom in body_xml.findall("geom"):
                    if geom.attrib.get("class") == "collision":
                        geom.attrib.update(mujoco_common.STRUCTURAL_DOOR_CONTACT)
            for child in body_xml.findall("body"):
                mark_moving(child, moving)

        mark_moving(base_body_xml, False)

    # OmniGibson disables self-collisions of dataset objects. MuJoCo already
    # skips a body's contacts with its parent unless the parent is static,
    # which a fixed object's base link is, so exclude every pair of links that
    # carry collision geoms.
    colliding_bodies = [
        body.attrib["name"]
        for body in worldbody_xml.iter("body")
        if any(g.attrib.get("class") == "collision" for g in body.findall("geom"))
    ]
    if len(colliding_bodies) > 1:
        contact_xml = ET.SubElement(root, "contact")
        for i, body1 in enumerate(colliding_bodies):
            for body2 in colliding_bodies[i + 1 :]:
                ET.SubElement(contact_xml, "exclude", {"body1": body1, "body2": body2})

    # BEHAVIOR metadata that MJCF can't express. All lengths are unscaled: in a
    # scene, multiply them by the instance's scale.
    joint_names = [
        joint.name
        for joint in robot.joints
        if joint.joint_type in ("revolute", "continuous", "prismatic")
        and applied_fixes.get(joint.name, {}).get("action") != "fix"
    ]
    synset, abilities = _taxonomy_info(category)
    b1k_metadata = {
        "format": mujoco_common.METADATA_FORMAT,
        "dataset_version": mujoco_common.DATASET_VERSION,
        "category": category,
        "model": model_id,
        "synset": synset,
        "abilities": abilities,
        "structure": category in mujoco_common.STRUCTURE_CATEGORIES,
        "bbox_size": metadata.get("bbox_size"),
        "base_link_offset": metadata.get("base_link_offset"),
        "mass": category_mass,
        "joints": joint_names,
        "anisotropic_scaling": {"ok": not scaling_problems, "problems": scaling_problems},
        "dropped_collisions": dropped_collisions,
        "joint_fixes": {
            name: {"action": fix["action"], "off_axis_deg": fix["off_axis_deg"]}
            for name, fix in sorted(applied_fixes.items())
        },
        "openable": _openable_metadata(metadata, joint_names),
        "meta_links": meta_records,
        "link_tags": link_tags,
        "object_parts": metadata.get("object_parts") or [],
        "groups": mujoco_common.GROUPS,
    }
    custom_xml = ET.SubElement(root, "custom")
    ET.SubElement(
        custom_xml,
        "text",
        {"name": mujoco_common.METADATA_TEXT, "data": mujoco_common.metadata_json(b1k_metadata)},
    )

    return ObjectMjcf(root=root, files=archive_files, images=archive_images, metadata=b1k_metadata, tmp_dir=tmp_dir)


def urdf_to_mjz(in_obj_dir, out_obj_dir, validate=False):
    """Export an object's MJZ to {out_obj_dir}/mjz/{model}.mjz.

    The mjz subdirectory mirrors the usd subdirectory of the USD dataset
    layout. The archive holds model.xml plus all referenced assets and the
    per-link GLBs."""
    obj_mjcf = build_object_mjcf(in_obj_dir)
    model_id = obj_mjcf.root.attrib["model"]
    out_file = os.path.join(out_obj_dir, "mjz", f"{model_id}.mjz")
    try:
        write_mjz(obj_mjcf, out_file)
    finally:
        shutil.rmtree(obj_mjcf.tmp_dir, ignore_errors=True)
    if validate:
        validate_mjz(out_file)
    return obj_mjcf.metadata["anisotropic_scaling"]["problems"]


OBJECTS_ZIP_PATH = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "parallels" / "objects.zip"
_SOURCE_ZIP = None


def _get_source_zip():
    # One lazily opened handle per worker process.
    global _SOURCE_ZIP
    if _SOURCE_ZIP is None:
        _SOURCE_ZIP = zipfile.ZipFile(OBJECTS_ZIP_PATH)
    return _SOURCE_ZIP


def process_object(obj_prefix, members, out_obj_dir, validate=False):
    """Extract one object's files straight from objects.zip into a scratch
    directory and export its MJZ."""
    b1k_pipeline.utils.TMP_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(b1k_pipeline.utils.TMP_DIR)) as scratch_dir:
        _get_source_zip().extractall(scratch_dir, members)
        return urdf_to_mjz(os.path.join(scratch_dir, obj_prefix), out_obj_dir, validate=validate)


def main():
    parser = argparse.ArgumentParser(description="Export every object as a MuJoCo MJZ archive.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compile each archive with MuJoCo, run mj_forward, and check that its bounds are finite.",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        help="Only export these objects, given as category/model, into objects_mujoco_debug.zip.",
    )
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    out_name = "objects_mujoco_debug.zip" if args.objects else "objects_mujoco.zip"
    with b1k_pipeline.utils.ParallelZipFS(out_name, write=True) as out_fs:
        # Group the zip members by object directory (objects/{category}/{model}).
        members_by_object = {}
        with zipfile.ZipFile(OBJECTS_ZIP_PATH) as zf:
            for name in zf.namelist():
                parts = name.split("/")
                if len(parts) < 4 or parts[0] != "objects":
                    continue
                members_by_object.setdefault("/".join(parts[:3]), []).append(name)

        # Skip objects that don't have a URDF.
        members_by_object = {
            prefix: members
            for prefix, members in members_by_object.items()
            if any(m.startswith(f"{prefix}/urdf/") and m.endswith(".urdf") for m in members)
        }
        if args.objects:
            wanted = {f"objects/{o}" for o in args.objects}
            members_by_object = {p: m for p, m in members_by_object.items() if p in wanted}

        errors, warnings = {}, {}
        with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            obj_futures = {
                executor.submit(
                    process_object,
                    prefix,
                    members,
                    out_fs.makedirs(prefix).getsyspath("/"),
                    args.validate,
                ): prefix
                for prefix, members in members_by_object.items()
            }

            for future in tqdm.tqdm(
                futures.as_completed(obj_futures),
                total=len(obj_futures),
                desc="Processing objects",
            ):
                try:
                    scaling_problems = future.result()
                    if scaling_problems:
                        warnings[obj_futures[future]] = "not anisotropically scalable: " + "; ".join(
                            scaling_problems
                        )
                except:
                    errors[obj_futures[future]] = traceback.format_exc()
                    traceback.print_exc()

        print("Finished processing")

    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    out_json = "export_objects_mujoco_debug.json" if args.objects else "export_objects_mujoco.json"
    with pipeline_fs.pipeline_output().open(out_json, "w") as f:
        json.dump({"success": not errors, "errors": errors, "warnings": warnings}, f, indent=1)


if __name__ == "__main__":
    main()
