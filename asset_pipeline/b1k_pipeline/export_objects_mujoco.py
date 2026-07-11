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

# Geom/site groups: 2 = visual, 3 = collision, 4 = meta volumes/sites. Groups
# above 2 are hidden by default in the MuJoCo visualizer.
VISUAL_GROUP = "2"
COLLISION_GROUP = "3"
META_GROUP = "4"


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


def _add_meta_sites_from_metadata(body_xml, link_name, meta_links):
    """Add sites/lights for the metadata-defined meta links of a link.

    Mirrors OmniGibson's _generate_meshes_for_primitive_meta_links conventions:
    primitives use size[0] as radius and size[2] as height with the shape origin
    at the base (offset by half height along local z); particleapplier cones are
    flipped so the site's -z is the emission direction.
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
                    }
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
                            type="sphere", size=_fmt([DIMENSIONLESS_SITE_RADIUS]), pos=_fmt(pos)
                        )
                    site_attrs["quat"] = _fmt(_xyzw_to_wxyz(orn.as_quat()))
                else:  # primitive
                    shape_type = mesh_info["type"]
                    size = np.asarray(mesh_info.get("size", [0.01, 0.01, 0.01]), dtype=np.float64)
                    if shape_type == "sphere":
                        site_attrs.update(type="sphere", size=_fmt([size[0]]), pos=_fmt(pos))
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
                ET.SubElement(body_xml, "site").attrib = site_attrs


def urdf_to_mjz(in_obj_dir, out_obj_dir):
    urdf_files = glob.glob(in_obj_dir + "/urdf/*.urdf")
    assert (
        len(urdf_files) == 1
    ), f"Expected exactly one URDF file in {in_obj_dir}, found {len(urdf_files)}"
    urdf_file = urdf_files[0]
    urdf_dir = os.path.dirname(urdf_file)
    model_id = os.path.splitext(os.path.basename(urdf_file))[0]

    metadata = {}
    metadata_file = os.path.join(in_obj_dir, "misc", "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    link_tags = metadata.get("link_tags") or {}
    meta_links = metadata.get("meta_links") or {}

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
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": ".", "texturedir": "."})
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
        _add_meta_sites_from_metadata(body_xml, link.name, meta_links.get(link.name) or {})

    def add_volume_meta_link(body_xml, link):
        """Fillable/openfillable meta links: the convex volume meshes become
        non-colliding massless geoms, plus a box site fitted to their AABB so
        MuJoCo's insidesite sensor can be used for containment checks."""
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
            ET.SubElement(body_xml, "geom", geom_attrs)
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
            ET.SubElement(
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

    # Recursively build the body tree following the URDF articulation.
    children = {}  # parent link name -> list of (joint, child link)
    for joint in robot.joints:
        children.setdefault(joint.parent, []).append(joint)
    link_map = {link.name: link for link in robot.links}

    def add_body(parent_xml, link, joint):
        body_xml = ET.SubElement(parent_xml, "body", {"name": link.name})
        if joint is not None:
            body_xml.attrib.update(_pose_attrs(joint.origin))
            if joint.joint_type in ("revolute", "continuous", "prismatic"):
                joint_attrs = {
                    "name": joint.name,
                    "type": "slide" if joint.joint_type == "prismatic" else "hinge",
                    "axis": _fmt(joint.axis),
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
            add_body(body_xml, link_map[child_joint.child], child_joint)

    base_body_xml = ET.SubElement(worldbody_xml, "body", {"name": robot.base_link.name})
    ET.SubElement(base_body_xml, "freejoint")
    meta_match = META_LINK_NAME_PATTERN.fullmatch(robot.base_link.name)
    assert not meta_match, f"Base link {robot.base_link.name} should not be a meta link"
    add_regular_link(base_body_xml, robot.base_link)
    for child_joint in children.get(robot.base_link.name, []):
        add_body(base_body_xml, link_map[child_joint.child], child_joint)

    # Write the MJZ: the MJCF plus all referenced assets and per-link GLBs. The
    # mjz subdirectory mirrors the usd subdirectory of the USD dataset layout,
    # and is where the scene MJCFs (export_scenes_mujoco.py) expect to find it.
    mjcf_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    mjz_dir = os.path.join(out_obj_dir, "mjz")
    os.makedirs(mjz_dir, exist_ok=True)
    out_file = os.path.join(mjz_dir, f"{model_id}.mjz")
    try:
        with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{model_id}.xml", mjcf_str)
            for archive_name, (mode, src_path) in archive_files.items():
                if mode == "strip_obj":
                    stripped_path = os.path.join(tmp_dir, os.path.basename(archive_name))
                    _strip_obj_materials(src_path, stripped_path)
                    zf.write(stripped_path, archive_name)
                else:
                    zf.write(src_path, archive_name)
            for archive_name, image in archive_images.items():
                with zf.open(archive_name, "w") as f:
                    image.save(f, format="png")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


OBJECTS_ZIP_PATH = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "parallels" / "objects.zip"
_SOURCE_ZIP = None


def _get_source_zip():
    # One lazily opened handle per worker process.
    global _SOURCE_ZIP
    if _SOURCE_ZIP is None:
        _SOURCE_ZIP = zipfile.ZipFile(OBJECTS_ZIP_PATH)
    return _SOURCE_ZIP


def process_object(obj_prefix, members, out_obj_dir):
    """Extract one object's files straight from objects.zip into a scratch
    directory and export its MJZ."""
    b1k_pipeline.utils.TMP_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(b1k_pipeline.utils.TMP_DIR)) as scratch_dir:
        _get_source_zip().extractall(scratch_dir, members)
        urdf_to_mjz(os.path.join(scratch_dir, obj_prefix), out_obj_dir)


def main():
    with b1k_pipeline.utils.ParallelZipFS("objects_mujoco.zip", write=True) as out_fs:
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


        errors = {}
        with futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            obj_futures = {
                executor.submit(
                    process_object,
                    prefix,
                    members,
                    out_fs.makedirs(prefix).getsyspath("/"),
                ): prefix
                for prefix, members in members_by_object.items()
            }

            for future in tqdm.tqdm(
                futures.as_completed(obj_futures),
                total=len(obj_futures),
                desc="Processing objects",
            ):
                try:
                    future.result()
                except:
                    errors[obj_futures[future]] = traceback.format_exc()
                    traceback.print_exc()

        print("Finished processing")

    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    with pipeline_fs.pipeline_output().open("export_objects_mujoco.json", "w") as f:
        json.dump({"success": not errors, "errors": errors}, f)


if __name__ == "__main__":
    main()
