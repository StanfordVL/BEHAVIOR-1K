import collections
import copy
import io
import json
import logging
import os
import traceback
import xml.etree.ElementTree as ET
from xml.dom import minidom

from dask.distributed import LocalCluster, as_completed
import fs.copy
from fs.tempfs import TempFS
from fs.osfs import OSFS
import networkx as nx
import numpy as np
import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R
from PIL import Image

from b1k_pipeline import mesh_tree
import b1k_pipeline.utils
from b1k_pipeline.utils import (
    parse_name,
    get_targets,
    save_mesh,
)

from bddl.object_taxonomy import ObjectTaxonomy

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


CHANNEL_MAPPING = {
    "Diffuse map": ("diffuse", "map_Kd"),
    "Bump map": ("normal", "map_bump"),
    "Refl. gloss.": ("roughness", "map_Pr"),
    "Metalness": ("metalness", "map_Pm"),
    "Refract map": ("refraction", "map_Tf"),
    "Reflect map": ("reflection", "map_Ks"),
    "Fresnel IOR": ("ior", "map_Ns"),
}

ALLOWED_PART_TAGS = {
    "subpart",
    "extrapart",
    "connectedpart",
}

# Objects that don't require these meta types should not have them
REQUIRED_ONLY_META_TYPES = {
    "fluidsource",
    "togglebutton",
    "heatsource",
    "particleapplier",
    "particleremover",
    "fluidsink",
    "slicer",
    "fillable",
}

CLOTH_SUBDIVISION_THRESHOLD = 0.05

LOG_SURFACE_AREA_RANGE = (-6, 4)
LOG_TEXTURE_RANGE = (4, 11)


def get_required_meta_links(object_taxonomy, category):
    synset = object_taxonomy.get_synset_from_category_or_substance(category)
    if synset is None:
        raise ValueError(f"Category {category} not found in taxonomy.")
    return object_taxonomy.get_required_meta_links_for_synset(synset)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_mesh_unit_bbox(mesh, *args, **kwargs):
    mesh_copy = mesh.copy()

    # Find how much the mesh would need to be scaled to fit into a unit cube
    bounding_box = mesh_copy.bounding_box.extents
    assert np.all(
        bounding_box > 0
    ), f"Bounding box extents are not all positive: {bounding_box}"
    scale = 1 / bounding_box

    # Scale the mesh
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] = np.diag(scale)
    mesh_copy.apply_transform(scale_matrix)

    # Save the scaled mesh
    save_mesh(mesh_copy, *args, **kwargs)

    # Return the inverse scale that needs to be applied for the mesh
    return 1 / scale


def transform_mesh(orig_mesh, translation, rotation):
    mesh = orig_mesh.copy()

    # Rotate the object around the CoM of the base link frame by the transpose of its canonical orientation
    # so that the object has identity orientation in the end

    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(rotation).as_matrix()
    transform[:3, 3] = translation
    inv_transform = trimesh.transformations.inverse_matrix(transform)
    mesh.apply_transform(inv_transform)

    return mesh


def transform_points(points, translation, rotation):
    # Rotate the object around the CoM of the base link frame by the transpose of its canonical orientation
    # so that the object has identity orientation in the end
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(rotation).as_matrix()
    transform[:3, 3] = translation
    inv_transform = trimesh.transformations.inverse_matrix(transform)
    return trimesh.transformations.transform_points(points, inv_transform)


def transform_meta_links(orig_meta_links, translation, rotation):
    meta_links = copy.deepcopy(orig_meta_links)
    rotation_inv = R.from_quat(rotation).inv()
    for meta_link_id_to_subid in meta_links.values():
        for meta_link_subid_to_link in meta_link_id_to_subid.values():
            for meta_link in meta_link_subid_to_link:
                meta_link["position"] = meta_link["position"]
                meta_link["position"] -= translation
                meta_link["position"] = np.dot(
                    rotation_inv.as_matrix(), meta_link["position"]
                )
                # meta_link["position"] += translation
                meta_link["orientation"] = (
                    rotation_inv * R.from_quat(meta_link["orientation"])
                ).as_quat()

    return meta_links


def normalize_meta_links(orig_meta_links, offset):
    meta_links = copy.deepcopy(orig_meta_links)
    for meta_link_id_to_subid in meta_links.values():
        for meta_link_subid_to_link in meta_link_id_to_subid.values():
            for meta_link in meta_link_subid_to_link:
                meta_link["position"] += offset

    return meta_links


def get_mesh_center(mesh):
    if mesh.is_watertight:
        return mesh.center_mass
    else:
        return mesh.centroid


def get_part_nodes(G, root_node):
    root_node_metadata = G.nodes[root_node]["metadata"]
    part_nodes = []
    for part_name in root_node_metadata["parts"]:
        # Find the part node
        part_name_parsed = parse_name(part_name)
        part_cat = part_name_parsed.group("category")
        part_model = part_name_parsed.group("model_id")
        part_inst_id = part_name_parsed.group("instance_id")
        part_link_name = part_name_parsed.group("link_name")
        part_link_name = "base_link" if part_link_name is None else part_link_name
        if part_link_name != "base_link":
            continue
        part_node_key = (part_cat, part_model, part_inst_id, part_link_name)
        if part_node_key not in G.nodes:
            print(list(G.nodes))
        assert part_node_key in G.nodes, f"Could not find part node {part_node_key}"
        part_nodes.append(part_node_key)

    return part_nodes


def get_bbox_data_for_mesh(mesh):
    axis_aligned_bbox = mesh.bounding_box
    axis_aligned_bbox_dict = {
        "extent": np.array(axis_aligned_bbox.primitive.extents).tolist(),
        "transform": np.array(axis_aligned_bbox.primitive.transform).tolist(),
    }

    oriented_bbox = mesh.bounding_box_oriented
    oriented_bbox_dict = {
        "extent": np.array(oriented_bbox.primitive.extents).tolist(),
        "transform": np.array(oriented_bbox.primitive.transform).tolist(),
    }

    return {"axis_aligned": axis_aligned_bbox_dict, "oriented": oriented_bbox_dict}


def compute_link_aligned_bounding_boxes(G, root_node):
    link_bounding_boxes = collections.defaultdict(dict)
    for link_node in nx.dfs_preorder_nodes(G, root_node):
        obj_cat, obj_model, obj_inst_id, link_name = link_node

        # Get the pose and transform it
        for key in ["collision", "visual"]:
            try:
                mesh = (
                    G.nodes[link_node]["visual_mesh"]
                    if key == "visual"
                    else G.nodes[link_node]["canonical_collision_mesh"]
                )
                link_bounding_boxes[link_name][key] = get_bbox_data_for_mesh(
                    transform_mesh(
                        mesh, -G.nodes[link_node]["mesh_in_link_frame"], [0, 0, 0, 1]
                    )
                )
            except Exception as e:
                print(f"Problem with {obj_cat}-{obj_model} link {link_name}: {str(e)}")

    return link_bounding_boxes


def process_link(
    G,
    link_node,
    base_link_center,
    canonical_orientation,
    required_meta_types,
    output_fs,
    tree_root,
    out_metadata,
):
    category_name, model_id, _, link_name = link_node
    raw_meta_links = G.nodes[link_node]["meta_links"]

    # Create a canonicalized copy of the lower and upper meshes.
    mesh_center = get_mesh_center(G.nodes[link_node]["lower_mesh"])
    canonical_mesh = transform_mesh(
        G.nodes[link_node]["lower_mesh"], mesh_center, canonical_orientation
    )
    meta_links = transform_meta_links(
        raw_meta_links, mesh_center, canonical_orientation
    )

    # Somehow we need to manually write the vertex normals to cache
    canonical_mesh._cache.cache["vertex_normals"] = canonical_mesh.vertex_normals

    in_edges = list(G.in_edges(link_node))
    assert (
        len(in_edges) <= 1
    ), f"Something's wrong: there's more than 1 in-edge to node {link_node}"

    # Compute the texture resolution needed
    mesh_surface_area = canonical_mesh.area
    log_area = np.clip(np.log(mesh_surface_area) / np.log(10), *LOG_SURFACE_AREA_RANGE)
    log_area_range_fraction = (log_area - LOG_SURFACE_AREA_RANGE[0]) / (
        LOG_SURFACE_AREA_RANGE[1] - LOG_SURFACE_AREA_RANGE[0]
    )
    log2_area = LOG_TEXTURE_RANGE[0] + log_area_range_fraction * (
        LOG_TEXTURE_RANGE[1] - LOG_TEXTURE_RANGE[0]
    )
    log2_area = int(np.clip(np.round(log2_area), *LOG_TEXTURE_RANGE))
    texture_res = int(2**log2_area)

    # Save the mesh
    with TempFS() as tfs:
        obj_relative_path = f"{model_id}__{link_name}.obj"
        save_mesh(canonical_mesh, tfs, obj_relative_path)

        # Move the mesh to the correct path
        obj_link_mesh_folder_fs = output_fs.makedir("shape", recreate=True)
        obj_link_visual_mesh_folder_fs = obj_link_mesh_folder_fs.makedir(
            "visual", recreate=True
        )
        obj_link_material_folder_fs = output_fs.makedir("material", recreate=True)

        # Check if a material got exported.
        material_files = [x for x in tfs.listdir("/") if x.endswith(".mtl")]
        assert material_files, "No materials found after OBJ export! Not good."
        if material_files:
            assert (
                len(material_files) == 1
            ), f"Something's wrong: there's more than 1 material file in {tfs.listdir('/')}"
            original_material_filename = material_files[0]

            # Fix texture file paths if necessary.
            material_maps = G.nodes[link_node]["material_maps"]
            moved_material_maps = {}
            for map_channel, map_path in material_maps.items():
                assert os.path.exists(
                    map_path
                ), f"Texture file {map_path} does not exist!"

                # Convert the path to a dirname + filename so that we can use an OSFS
                # to copy the file.
                src_map_dir = os.path.dirname(map_path)
                src_map_filename = os.path.basename(map_path)
                src_map_ext = os.path.splitext(src_map_filename)[1]
                src_map_fs = OSFS(src_map_dir)

                assert map_channel in CHANNEL_MAPPING, f"Unknown channel {map_channel}"
                dst_fname, _ = CHANNEL_MAPPING[map_channel]
                dst_texture_filename = (
                    f"{model_id}__{link_name}__{dst_fname}{src_map_ext}"
                )

                # Load the image
                # TODO: Re-enable this after tuning it.
                # texture = Image.open(original_material_fs.open(src_texture_file, "rb"), formats=("png",))
                # existing_texture_res = texture.size[0]
                # if existing_texture_res > texture_res:
                #     texture = texture.resize((texture_res, texture_res), Image.BILINEAR)
                # texture.save(obj_link_material_folder_fs.open(dst_texture_file, "wb"), format="png")

                fs.copy.copy_file(
                    src_map_fs,
                    src_map_filename,
                    obj_link_material_folder_fs,
                    dst_texture_filename,
                )

                moved_material_maps[map_channel] = dst_texture_filename

        # Copy the OBJ into the right spot
        fs.copy.copy_file(
            tfs, obj_relative_path, obj_link_visual_mesh_folder_fs, obj_relative_path
        )

        # Save and merge precomputed convex meshes e.g. collision, fillable
        canonical_convex_meshes = {}
        convex_mesh_filenames_and_scales = {}
        for convex_mesh_type in mesh_tree.CONVEX_MESH_TYPES:
            convex_mesh_key = f"{convex_mesh_type}_mesh"
            if convex_mesh_key not in G.nodes[link_node]:
                continue
            canonical_convex_meshes[convex_mesh_type] = []
            convex_mesh_filenames_and_scales[convex_mesh_type] = []
            for i, convex_mesh in enumerate(G.nodes[link_node][convex_mesh_key]):
                canonical_convex_mesh = transform_mesh(
                    convex_mesh, mesh_center, canonical_orientation
                )
                canonical_convex_mesh._cache.cache["vertex_normals"] = (
                    canonical_convex_mesh.vertex_normals
                )
                convex_filename = obj_relative_path.replace(
                    ".obj", f"__{convex_mesh_type}__{i}.obj"
                )
                obj_link_convex_mesh_folder_fs = obj_link_mesh_folder_fs.makedir(
                    convex_mesh_type, recreate=True
                )
                convex_scale = save_mesh_unit_bbox(
                    canonical_convex_mesh,
                    obj_link_convex_mesh_folder_fs,
                    convex_filename,
                )
                convex_mesh_filenames_and_scales[convex_mesh_type].append(
                    (convex_filename, convex_scale)
                )
                canonical_convex_meshes[convex_mesh_type].append(canonical_convex_mesh)

        # Store the final meshes
        G.nodes[link_node]["visual_mesh"] = canonical_mesh.copy()
        G.nodes[link_node]["canonical_collision_mesh"] = trimesh.util.concatenate(
            canonical_convex_meshes["collision"]
        )

        if material_files:
            # Modify MTL reference in OBJ file
            mtl_name = f"{model_id}__{link_name}.mtl"
            with obj_link_visual_mesh_folder_fs.open(obj_relative_path, "r") as f:
                new_lines = []
                for line in f.readlines():
                    if f"mtllib {original_material_filename}" in line:
                        line = f"mtllib {mtl_name}\n"
                    new_lines.append(line)

            with obj_link_visual_mesh_folder_fs.open(obj_relative_path, "w") as f:
                for line in new_lines:
                    f.write(line)

            # Modify texture reference in MTL file
            with tfs.open(original_material_filename, "r") as f:
                new_lines = []
                for line in f.readlines():
                    if "map_Kd material_0.png" in line:
                        line = ""
                        for (
                            channel_name,
                            channel_filename,
                        ) in moved_material_maps.items():
                            _, mtl_key = CHANNEL_MAPPING[channel_name]
                            line += f"{mtl_key} ../../material/{channel_filename}\n"
                    new_lines.append(line)

            with obj_link_visual_mesh_folder_fs.open(mtl_name, "w") as f:
                for line in new_lines:
                    f.write(line)

    # Create the link in URDF
    link_xml = ET.SubElement(tree_root, "link")
    link_xml.attrib = {"name": link_name}
    visual_xml = ET.SubElement(link_xml, "visual")
    visual_origin_xml = ET.SubElement(visual_xml, "origin")
    visual_origin_xml.attrib = {"xyz": " ".join([str(item) for item in [0.0] * 3])}
    visual_geometry_xml = ET.SubElement(visual_xml, "geometry")
    visual_mesh_xml = ET.SubElement(visual_geometry_xml, "mesh")
    visual_mesh_xml.attrib = {
        "filename": os.path.join("..", "shape", "visual", obj_relative_path).replace(
            "\\", "/"
        ),
        "scale": " ".join([str(item) for item in np.ones(3)]),
    }

    collision_origin_xmls = []
    for collision_filename, collision_scale in convex_mesh_filenames_and_scales[
        "collision"
    ]:
        collision_xml = ET.SubElement(link_xml, "collision")
        collision_xml.attrib = {"name": collision_filename.replace(".obj", "")}
        collision_origin_xml = ET.SubElement(collision_xml, "origin")
        collision_origin_xml.attrib = {
            "xyz": " ".join([str(item) for item in [0.0] * 3])
        }
        collision_geometry_xml = ET.SubElement(collision_xml, "geometry")
        collision_mesh_xml = ET.SubElement(collision_geometry_xml, "mesh")
        collision_mesh_xml.attrib = {
            "filename": os.path.join(
                "..", "shape", "collision", collision_filename
            ).replace("\\", "/"),
            "scale": " ".join([str(item) for item in collision_scale]),
        }
        collision_origin_xmls.append(collision_origin_xml)

    # This object might be a base link and thus without an in-edge. Nothing to do then.
    if len(in_edges) == 0:
        G.nodes[link_node]["link_frame_in_base"] = np.zeros(3)
        G.nodes[link_node]["mesh_in_link_frame"] = np.zeros(3)
    else:
        # Grab the lone edge to the parent.
        (edge,) = in_edges
        parent_node, child_node = edge
        assert (
            child_node == link_node
        ), f"Something's wrong: the child node of the edge is not the link node {link_node}"
        joint_type = G.edges[edge]["joint_type"]
        parent_frame = G.nodes[parent_node]["link_frame_in_base"]

        rotated_parent_frame = R.from_quat(canonical_orientation).apply(parent_frame)

        # Load the meshes.
        lower_canonical_points = transform_points(
            G.nodes[child_node]["lower_points"],
            base_link_center + rotated_parent_frame,
            canonical_orientation,
        )

        # Get the center of mass of the child link in the parent frame.
        child_center = transform_points(
            np.array([mesh_center]),
            base_link_center + rotated_parent_frame,
            canonical_orientation,
        )[0]

        # Create the joint in the URDF
        joint_xml = ET.SubElement(tree_root, "joint")
        joint_xml.attrib = {
            "name": f"j_{child_node[3]}",
            "type": {"P": "prismatic", "R": "revolute", "F": "fixed"}[joint_type],
        }

        joint_parent_xml = ET.SubElement(joint_xml, "parent")
        joint_parent_xml.attrib = {"link": parent_node[3]}
        joint_child_xml = ET.SubElement(joint_xml, "child")
        joint_child_xml.attrib = {"link": child_node[3]}

        mesh_offset = np.zeros(3)
        if joint_type in ("P", "R"):
            upper_canonical_points = transform_points(
                G.nodes[child_node]["upper_points"],
                base_link_center + rotated_parent_frame,
                canonical_orientation,
            )

            if joint_type == "R":
                # Revolute joint
                num_v_lower = lower_canonical_points.shape[0]
                num_v_upper = upper_canonical_points.shape[0]
                assert (
                    num_v_lower == num_v_upper
                ), f"{child_node} lower mesh has {num_v_lower} vertices while upper has {num_v_upper}. These should match."
                num_v = num_v_lower
                random_index = np.random.choice(num_v, min(num_v, 20), replace=False)
                from_vertices = lower_canonical_points[random_index]
                to_vertices = upper_canonical_points[random_index]

                # Find joint axis and joint limit
                r = R.align_vectors(
                    to_vertices - np.mean(to_vertices, axis=0),
                    from_vertices - np.mean(from_vertices, axis=0),
                )[0]
                upper_limit = r.magnitude()
                assert upper_limit < np.deg2rad(
                    180
                ), "upper limit of revolute joint should be <180 degrees"
                joint_axis = r.as_rotvec() / r.magnitude()

                # Let X = from_vertices_mean, Y = to_vertices_mean, R is rotation, T is translation
                # R * (X - T) + T = Y
                # => (I - R) T = Y - R * X
                # Find the translation part of the joint origin
                r_mat = r.as_matrix()
                from_vertices_mean = from_vertices.mean(axis=0)
                to_vertices_mean = to_vertices.mean(axis=0)
                left_mat = np.eye(3) - r_mat
                arbitrary_point_on_joint_axis = np.linalg.lstsq(
                    left_mat,
                    (to_vertices_mean - np.dot(r_mat, from_vertices_mean)),
                    rcond=None,
                )[0]

                # The joint origin has infinite number of solutions along the joint axis
                # Find the translation part of the joint origin that is closest to the CoM of the link
                # by projecting the CoM onto the joint axis
                arbitrary_point_to_center = child_center - arbitrary_point_on_joint_axis
                joint_origin = arbitrary_point_on_joint_axis + (
                    np.dot(arbitrary_point_to_center, joint_axis) * joint_axis
                )

                # Assign visual and collision mesh origin so that the offset from the joint origin is removed.
                mesh_offset = child_center - joint_origin
                visual_origin_xml.attrib = {
                    "xyz": " ".join([str(item) for item in mesh_offset])
                }

                for collision_origin_xml in collision_origin_xmls:
                    collision_origin_xml.attrib = {
                        "xyz": " ".join([str(item) for item in mesh_offset])
                    }

                meta_links = normalize_meta_links(meta_links, mesh_offset)
            elif joint_type == "P":
                # Prismatic joint
                diff = np.mean(upper_canonical_points, axis=0) - np.mean(
                    lower_canonical_points, axis=0
                )

                # Find joint axis and joint limit
                if not np.allclose(diff, 0):
                    upper_limit = np.linalg.norm(diff)
                    joint_axis = diff / upper_limit
                else:
                    upper_limit = 0
                    joint_axis = np.array([0, 0, 1])

                # Assign the joint origin relative to the parent CoM
                joint_origin = child_center

            # Save these joints' data
            joint_origin_xml = ET.SubElement(joint_xml, "origin")
            joint_origin_xml.attrib = {
                "xyz": " ".join([str(item) for item in joint_origin])
            }
            joint_axis_xml = ET.SubElement(joint_xml, "axis")
            joint_axis_xml.attrib = {
                "xyz": " ".join([str(item) for item in joint_axis])
            }
            joint_limit_xml = ET.SubElement(joint_xml, "limit")
            joint_limit_xml.attrib = {"lower": str(0.0), "upper": str(upper_limit)}
        else:
            # Fixed joints are quite simple.
            joint_origin = child_center

            if joint_type == "F":
                joint_origin_xml = ET.SubElement(joint_xml, "origin")
                joint_origin_xml.attrib = {
                    "xyz": " ".join([str(item) for item in joint_origin])
                }
            else:
                raise ValueError("Unexpected joint type: " + str(joint_type))

        G.nodes[link_node]["link_frame_in_base"] = parent_frame + joint_origin
        G.nodes[link_node]["mesh_in_link_frame"] = mesh_offset

    # # Compute the moment of inertia
    # moment_of_inertia = G.nodes[link_node][
    #     "canonical_collision_mesh"
    # ].moment_inertia_frame(trimesh.transformations.translation_matrix(com))

    # Annotate mass, center of mass and moment of inertia directly on the URDF
    inertial_xml = ET.SubElement(link_xml, "inertial")
    inertial_origin_xml = ET.SubElement(inertial_xml, "origin")
    inertial_origin_xml.attrib = {
        "xyz": "0 0 0",
        "rpy": "0 0 0",
    }
    inertial_mass_xml = ET.SubElement(inertial_xml, "mass")
    inertial_mass_xml.attrib = {"value": "1.0"}
    # inertial_inertia_xml = ET.SubElement(inertial_xml, "inertia")
    # inertial_inertia_xml.attrib = {
    #     "ixx": str(moment_of_inertia[0, 0]),
    #     "ixy": str(moment_of_inertia[0, 1]),
    #     "ixz": str(moment_of_inertia[0, 2]),
    #     "iyy": str(moment_of_inertia[1, 1]),
    #     "iyz": str(moment_of_inertia[1, 2]),
    #     "izz": str(moment_of_inertia[2, 2]),
    # }

    # Walk over the meta links to add visual meshes for non-collision convex meshes, since this
    # is the only way to get them into the URDF.
    for cm_type in mesh_tree.NON_COLLISION_CONVEX_MESH_TYPES:
        # Skip the convex mesh type if the object does not have one
        if f"{cm_type}_mesh" not in G.nodes[link_node]:
            continue

        # Unpack some info
        cm_link_name = f"meta__{link_name}_{cm_type}_0_0_link"
        cm_joint_name = f"meta__{link_name}_{cm_type}_0_0_joint"

        # Create the link in URDF
        cm_link_xml = ET.SubElement(tree_root, "link")
        cm_link_xml.attrib = {"name": cm_link_name}
        cm_inertial_xml = ET.SubElement(cm_link_xml, "inertial")
        cm_inertial_origin_xml = ET.SubElement(cm_inertial_xml, "origin")
        cm_inertial_origin_xml.attrib = {
            "xyz": "0 0 0",
            "rpy": "0 0 0",
        }
        cm_inertial_mass_xml = ET.SubElement(cm_inertial_xml, "mass")
        cm_inertial_mass_xml.attrib = {"value": "0.001"}
        # cm_inertial_inertia_xml = ET.SubElement(cm_inertial_xml, "inertia")
        # cm_moment_of_inertia = np.eye(3) * 1e-7
        # cm_inertial_inertia_xml.attrib = {
        #     "ixx": str(cm_moment_of_inertia[0, 0]),
        #     "ixy": str(cm_moment_of_inertia[0, 1]),
        #     "ixz": str(cm_moment_of_inertia[0, 2]),
        #     "iyy": str(cm_moment_of_inertia[1, 1]),
        #     "iyz": str(cm_moment_of_inertia[1, 2]),
        #     "izz": str(cm_moment_of_inertia[2, 2]),
        # }

        # Create the joint in URDF
        cm_joint_xml = ET.SubElement(tree_root, "joint")
        cm_joint_xml.attrib = {
            "name": cm_joint_name,
            "type": "fixed",
        }
        cm_joint_parent_xml = ET.SubElement(cm_joint_xml, "parent")
        cm_joint_parent_xml.attrib = {"link": link_name}
        cm_joint_child_xml = ET.SubElement(cm_joint_xml, "child")
        cm_joint_child_xml.attrib = {"link": cm_link_name}
        cm_joint_origin_xml = ET.SubElement(cm_joint_xml, "origin")
        cm_joint_origin_xml.attrib = {"xyz": "0 0 0"}

        # Save the meshes into the visual mesh FS and record them in the URDF
        for cm_filename, cm_scale in convex_mesh_filenames_and_scales[cm_type]:
            cm_visual_xml = ET.SubElement(cm_link_xml, "visual")
            cm_visual_origin_xml = ET.SubElement(cm_visual_xml, "origin")
            cm_visual_origin_xml.attrib = {"xyz": "0 0 0"}
            cm_visual_geometry_xml = ET.SubElement(cm_visual_xml, "geometry")
            cm_visual_mesh_xml = ET.SubElement(cm_visual_geometry_xml, "mesh")
            cm_visual_mesh_xml.attrib = {
                "filename": os.path.join("..", "shape", cm_type, cm_filename).replace(
                    "\\", "/"
                ),
                "scale": " ".join([str(item) for item in cm_scale]),
            }

    # Filter non-required meta links
    found_nonoptional_meta_types = set(meta_links.keys()) & REQUIRED_ONLY_META_TYPES
    found_extra_nonoptional_meta_types = (
        found_nonoptional_meta_types - required_meta_types
    )
    if found_extra_nonoptional_meta_types:
        for extra_meta_type in found_extra_nonoptional_meta_types:
            del meta_links[extra_meta_type]

    # Assert no CoM meta link anywhere except the base link
    assert not (
        "com" in meta_links and link_name != "base_link"
    ), f"CoM meta link found in {link_name} of {link_node}. Only base link should have CoM."

    out_metadata["meta_links"][link_name] = meta_links
    out_metadata["link_tags"][link_name] = G.nodes[link_node]["tags"]


def process_object(root_node, target, relevant_nodes, requried_meta_types, output_dir):
    obj_cat, obj_model, obj_inst_id, _ = root_node

    G = mesh_tree.build_mesh_tree(
        target,
        filter_nodes=relevant_nodes,
    )

    with OSFS(output_dir) as output_fs:
        # Prepare the URDF tree
        tree_root = ET.Element("robot")
        tree_root.attrib = {"name": obj_model}

        # Extract base link orientation and position
        canonical_orientation = np.array(G.nodes[root_node]["canonical_orientation"])
        base_link_mesh = G.nodes[root_node]["lower_mesh"]
        base_link_center = get_mesh_center(base_link_mesh)

        out_metadata = {
            "meta_links": {},
            "link_tags": {},
            "object_parts": [],
        }

        # Iterate over each link.
        for link_node in nx.dfs_preorder_nodes(G, root_node):
            process_link(
                G,
                link_node,
                base_link_center,
                canonical_orientation,
                requried_meta_types,
                output_fs,
                tree_root,
                out_metadata,
            )

        # Save the URDF file.
        xmlstr = minidom.parseString(ET.tostring(tree_root)).toprettyxml(indent="   ")
        xmlio = io.StringIO(xmlstr)
        tree = ET.parse(xmlio)

        urdf_fs = output_fs.makedir("urdf", recreate=True)
        with urdf_fs.open(f"{obj_model}.urdf", "wb") as f:
            tree.write(f, xml_declaration=True)

        bbox_size = np.array(G.nodes[root_node]["object_bounding_box"]["extent"])
        bbox_world_pos = np.array(G.nodes[root_node]["object_bounding_box"]["position"])
        base_link_offset_in_world = bbox_world_pos - base_link_center
        base_link_offset = (
            R.from_quat(canonical_orientation).inv().apply(base_link_offset_in_world)
        )

        # Compute part information
        for part_node_key in get_part_nodes(G, root_node):
            # Get the part node bounding box
            part_bb_size = G.nodes[part_node_key]["object_bounding_box"]["extent"]
            part_bb_in_world_pos = G.nodes[part_node_key]["object_bounding_box"][
                "position"
            ]
            part_bb_in_world_rot = R.from_quat(
                G.nodes[part_node_key]["object_bounding_box"]["rotation"]
            )

            # Convert into our base link frame
            our_transform = np.eye(4)
            our_transform[:3, 3] = base_link_center
            our_transform[:3, :3] = R.from_quat(canonical_orientation).as_matrix()
            bb_transform = np.eye(4)
            bb_transform[:3, 3] = part_bb_in_world_pos
            bb_transform[:3, :3] = part_bb_in_world_rot.as_matrix()
            bb_transform_in_our = np.linalg.inv(our_transform) @ bb_transform
            bb_pos_in_our = bb_transform_in_our[:3, 3]
            bb_quat_in_our = R.from_matrix(bb_transform_in_our[:3, :3]).as_quat()

            # Get the part type
            part_tags = set(G.nodes[part_node_key]["tags"]) & ALLOWED_PART_TAGS
            assert (
                len(part_tags) == 1
            ), f"Part node {part_node_key} has multiple part tags: {part_tags}"
            (part_type,) = part_tags

            # Add the metadata
            out_metadata["object_parts"].append(
                {
                    "category": part_node_key[0],
                    "model": part_node_key[1],
                    "type": part_type,
                    "bb_pos": bb_pos_in_our,
                    "bb_orn": bb_quat_in_our,
                    "bb_size": part_bb_size,
                }
            )

            # If it's a connectedpart, we also need to generate the corresponding female attachment point.
            if part_type == "connectedpart":
                base_link_meta_links = out_metadata["meta_links"]["base_link"]
                if "attachment" not in base_link_meta_links:
                    base_link_meta_links["attachment"] = {}
                attachment_type = f"{part_node_key[1]}parent".lower() + "F"
                if attachment_type not in base_link_meta_links["attachment"]:
                    base_link_meta_links["attachment"][attachment_type] = {}
                next_id = len(base_link_meta_links["attachment"][attachment_type])

                # pretend that the attachment point is at the center of the part bbox with its transform
                # actually add the point
                base_link_meta_links["attachment"][attachment_type][str(next_id)] = {
                    "position": bb_pos_in_our,
                    "orientation": bb_quat_in_our,
                }

        # Similarly, if we are a connectedpart, we need to generate the corresponding male attachment point.
        if "connectedpart" in G.nodes[root_node]["tags"]:
            base_link_meta_links = out_metadata["meta_links"]["base_link"]
            if "attachment" not in base_link_meta_links:
                base_link_meta_links["attachment"] = {}
            attachment_type = f"{obj_model}parent".lower() + "M"
            if attachment_type not in base_link_meta_links["attachment"]:
                base_link_meta_links["attachment"][attachment_type] = {}
            next_id = len(base_link_meta_links["attachment"][attachment_type])

            # Pretend that the attachment point is at the center of the part bbox with its transform
            next_id = len(base_link_meta_links["attachment"][attachment_type])
            base_link_meta_links["attachment"][attachment_type][str(next_id)] = {
                "position": base_link_offset.tolist(),
                "orientation": [0.0, 0.0, 0.0, 1.0],
            }

        openable_joint_ids = [
            (i, joint.attrib["name"])
            for i, joint in enumerate(tree.findall("joint"))
            if "openable"
            in out_metadata["link_tags"].get(joint.find("child").attrib["link"], [])
        ]

        # If there is a center of mass annotation for the base link, we need to pop it to store it
        # separately in the metadata.
        if "com" in out_metadata["meta_links"]["base_link"]:
            assert (
                len(out_metadata["meta_links"]["base_link"]["com"]) == 1
            ), f"Something's wrong: there's more than 1 CoM in {link_node}"
            com_links = list(out_metadata["meta_links"]["base_link"]["com"].values())[0]
            assert (
                len(com_links) == 1
            ), f"Something's wrong: there's more than 1 CoM in {link_node}"
            com = np.array(com_links[0]["position"])
            del out_metadata["meta_links"]["base_link"]["com"]
            out_metadata["center_of_mass"] = com.tolist()

        # Save metadata json
        out_metadata.update(
            {
                "base_link_offset": base_link_offset.tolist(),
                "bbox_size": bbox_size.tolist(),
                "orientations": [],
                "link_bounding_boxes": compute_link_aligned_bounding_boxes(
                    G, root_node
                ),
            }
        )
        if openable_joint_ids:
            out_metadata["openable_joint_ids"] = openable_joint_ids
        with output_fs.makedir("misc").open("metadata.json", "w") as f:
            json.dump(out_metadata, f, cls=NumpyEncoder)


def process_target(target, objects_path, object_taxonomy, model_whitelist, dask_client):
    object_futures = {}

    # Build the mesh tree using our mesh tree library. The scene code also uses this system.
    G = mesh_tree.build_mesh_tree(target, load_meshes=False)

    # Go through each object.
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

    # Only save the 0th instance.
    saveable_roots = [
        root_node
        for root_node in roots
        if int(root_node[2]) == 0
        and not G.nodes[root_node]["is_broken"]
        and (not model_whitelist or root_node[1] in model_whitelist)
    ]
    for root_node in saveable_roots:
        # Start processing the object. We start by creating an object-specific
        # copy of the mesh tree (also including info about any parts)
        relevant_nodes = set(nx.dfs_tree(G, root_node).nodes())
        relevant_nodes |= {
            node
            for part_root_node in get_part_nodes(
                G, root_node
            )  # Get every part root node
            for node in nx.dfs_tree(G, part_root_node).nodes()
        }  # Get the subtree of each part

        obj_cat, obj_model, obj_inst_id, _ = root_node
        output_dirname = f"{obj_cat}/{obj_model}"
        output_dirname_abs = os.path.join(objects_path, output_dirname)
        os.makedirs(output_dirname_abs, exist_ok=True)
        object_futures[
            dask_client.submit(
                process_object,
                root_node,
                target,
                relevant_nodes,
                get_required_meta_links(object_taxonomy, obj_cat),
                output_dirname_abs,
            )
        ] = str(root_node)

    return object_futures


def main():
    object_taxonomy = ObjectTaxonomy()

    # If this variable is set, we will only export the models in this list. This is useful for quickly
    # iterating on scenes (usdify otherwise takes 8+ hours). If this list is empty, we will export all models.
    SCENES_TO_INCLUDE_MODELS_FOR = []  # ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper"]

    # Start with a whitelist of some hardcoded objects
    model_whitelist = set()
    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    for scene in SCENES_TO_INCLUDE_MODELS_FOR:
        target_output_fs = pipeline_fs.target_output(f"scenes/{scene}")

        # Load the object list for the file
        with target_output_fs.open("object_list.json", "r") as f:
            object_list = json.load(f)

        # Get the needed object models
        needed_objects = {x.split("-")[-1] for x in object_list["needed_objects"]}

        # Remove the ones that only show up as clutter
        found_as_non_clutter = set()
        for name in object_list["meshes"]:
            pn = b1k_pipeline.utils.parse_name(name)
            if not pn:
                continue
            model_id = pn.group("model_id")
            is_clutter = pn.group("loose") == "C-"
            if not is_clutter:
                found_as_non_clutter.add(model_id)
        needed_objects = needed_objects & found_as_non_clutter

        # Add the needed models to the whitelist
        model_whitelist.update(needed_objects)

    with b1k_pipeline.utils.ParallelZipFS("objects.zip", write=True) as archive_fs:
        objects_dir = archive_fs.makedir("objects").getsyspath("/")
        # Load the mesh list from the object list json.
        errors = {}

        cluster = LocalCluster()
        dask_client = cluster.get_client()

        targets = get_targets("combined")

        obj_futures = {}

        for target in tqdm.tqdm(targets, desc="Processing targets to queue objects"):
            obj_futures.update(
                process_target(
                    target, objects_dir, object_taxonomy, model_whitelist, dask_client
                )
            )

        for future in tqdm.tqdm(
            as_completed(obj_futures.keys()),
            total=len(obj_futures),
            desc="Processing objects",
        ):
            try:
                future.result()
            except:
                name = obj_futures[future]
                errors[name] = traceback.format_exc()

        print("Finished processing")

    with pipeline_fs.pipeline_output().open("export_objs.json", "w") as f:
        json.dump({"success": not errors, "errors": errors}, f)


if __name__ == "__main__":
    main()
