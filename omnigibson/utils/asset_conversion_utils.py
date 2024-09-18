import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from os.path import exists
from pathlib import Path

import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
import trimesh
from omnigibson.objects import DatasetObject
from omnigibson.scenes import Scene
from omnigibson.utils.usd_utils import create_primitive_mesh
import omnigibson.utils.transform_utils as T

_LIGHT_MAPPING = {
    0: "Rect",
    2: "Sphere",
    4: "Disk",
}

_OBJECT_STATE_TEXTURES = {
    "burnt",
    "cooked",
    "frozen",
    "soaked",
    "toggledon",
}

_MTL_MAP_TYPE_MAPPINGS = {
    "map_kd": "albedo",
    "map_bump": "normal",
    "map_pr": "roughness",
    "map_pm": "metalness",
    "map_tf": "opacity",
    "map_ke": "emission",
    "map_ks": "ao",
    "map_": "metalness",
}

_SPLIT_COLLISION_MESHES = False

_META_LINK_RENAME_MAPPING = {
    "fillable": "container",
    "fluidsink": "particlesink",
    "fluidsource": "particlesource",
}

_ALLOWED_META_TYPES = {
    "particlesource": "dimensionless",
    "togglebutton": "primitive",
    "attachment": "dimensionless",
    "heatsource": "dimensionless",
    "particleapplier": "primitive",
    "particleremover": "primitive",
    "particlesink": "primitive",
    "slicer": "primitive",
    "container": "primitive",
    "collision": "convexmesh",
    "lights": "light",
}


class _TorchEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, th.Tensor):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def _space_string_to_tensor(string):
    """
    Converts a array string in mujoco xml to th.Tensor.
    Examples:
        "0 1 2" => [0, 1, 2]
    Args:
        string (str): String to convert to an array
    Returns:
        th.Tensor: Numerical array equivalent of @string
    """
    return th.tensor([float(x) for x in string.split(" ")])


def _tensor_to_space_script(array):
    """
    Converts a numeric array into the string format in mujoco.
    Examples:
        [0, 1, 2] => "0 1 2"
    Args:
        array (th.Tensor): Array to convert to a string
    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array.tolist()])


def _split_obj_file_into_connected_components(obj_fpath):
    """
    Splits obj file at @obj_fpath into individual obj files that each contain a single connected mesh.
    """
    # Open file in trimesh
    obj = trimesh.load(obj_fpath, file_type="obj", force="mesh")

    # Split to grab all individual bodies
    obj_bodies = obj.split(only_watertight=False)

    # Procedurally create new files in the same folder as obj_fpath
    out_fpath = os.path.dirname(obj_fpath)
    out_fname_root = os.path.splitext(os.path.basename(obj_fpath))[0]

    for i, obj_body in enumerate(obj_bodies):
        # Write to a new file
        obj_body.export(f"{out_fpath}/{out_fname_root}_{i}.obj", "obj")

    # We return the number of splits we had
    return len(obj_bodies)


def _split_all_objs_in_urdf(urdf_fpath, name_suffix="split", mesh_fpath_offset="."):
    """
    Splits the obj references in the given URDF
    """
    tree = ET.parse(urdf_fpath)
    root = tree.getroot()
    urdf_dir = os.path.dirname(urdf_fpath)
    out_fname_root = os.path.splitext(os.path.basename(urdf_fpath))[0]

    def recursively_find_collision_meshes(ele):
        # Finds all collision meshes starting at @ele
        cols = []
        for child in ele:
            if child.tag == "collision":
                # If the nested geom type is a mesh, add this to our running list along with its parent node
                if child.find("./geometry/mesh") is not None:
                    cols.append((child, ele))
            elif child.tag == "visual":
                # There will be no collision mesh internally here so we simply pass
                continue
            else:
                # Recurisvely look through all children of the child
                cols += recursively_find_collision_meshes(ele=child)

        return cols

    # Iterate over the tree and find all collision entries
    col_elements = recursively_find_collision_meshes(ele=root)

    # For each collision element and its parent, we remove the original one and create a set of new ones with their
    # filename references changed
    for col, parent in col_elements:
        # Don't change the original
        col_copy = deepcopy(col)
        # Delete the original
        parent.remove(col)
        # Create new objs first so we know how many we need to create in the URDF
        obj_fpath = col_copy.find("./geometry/mesh").attrib["filename"]
        n_new_objs = _split_obj_file_into_connected_components(obj_fpath=f"{urdf_dir}/{mesh_fpath_offset}/{obj_fpath}")
        # Create the new objs in the URDF
        for i in range(n_new_objs):
            # Copy collision again
            col_copy_copy = deepcopy(col_copy)
            # Modify the filename
            fname = col_copy_copy.find("./geometry/mesh").attrib["filename"]
            fname = fname.split(".obj")[0] + f"_{i}.obj"
            col_copy_copy.find("./geometry/mesh").attrib["filename"] = fname
            # Add to parent
            parent.append(col_copy_copy)

    # Finally, write this to a new file
    urdf_out_path = f"{urdf_dir}/{out_fname_root}_{name_suffix}.urdf"
    tree.write(urdf_out_path)

    # Return the urdf it wrote to
    return urdf_out_path


def _set_mtl_albedo(mtl_prim, texture):
    mtl = "diffuse_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _set_mtl_normal(mtl_prim, texture):
    mtl = "normalmap_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _set_mtl_ao(mtl_prim, texture):
    mtl = "ao_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _set_mtl_roughness(mtl_prim, texture):
    mtl = "reflectionroughness_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    lazy.omni.usd.create_material_input(
        mtl_prim,
        "reflection_roughness_texture_influence",
        1.0,
        lazy.pxr.Sdf.ValueTypeNames.Float,
    )
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _set_mtl_metalness(mtl_prim, texture):
    mtl = "metallic_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    lazy.omni.usd.create_material_input(mtl_prim, "metallic_texture_influence", 1.0, lazy.pxr.Sdf.ValueTypeNames.Float)
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _set_mtl_opacity(mtl_prim, texture):
    mtl = "opacity_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    lazy.omni.usd.create_material_input(mtl_prim, "enable_opacity", True, lazy.pxr.Sdf.ValueTypeNames.Bool)
    lazy.omni.usd.create_material_input(mtl_prim, "enable_opacity_texture", True, lazy.pxr.Sdf.ValueTypeNames.Bool)
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _set_mtl_emission(mtl_prim, texture):
    mtl = "emissive_color_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    lazy.omni.usd.create_material_input(mtl_prim, "enable_emission", True, lazy.pxr.Sdf.ValueTypeNames.Bool)
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _rename_prim(prim, name):
    path_from = prim.GetPrimPath().pathString
    path_to = f"{'/'.join(path_from.split('/')[:-1])}/{name}"
    lazy.omni.kit.commands.execute("MovePrim", path_from=path_from, path_to=path_to)
    return lazy.omni.isaac.core.utils.prims.get_prim_at_path(path_to)


def _get_visual_objs_from_urdf(urdf_path):
    # Will return a dictionary mapping link name (e.g.: base_link) to dictionary of owned visual meshes mapping mesh
    # name to visual obj file for that mesh
    visual_objs = OrderedDict()
    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for ele in root:
        if ele.tag == "link":
            name = ele.get("name").replace("-", "_")
            visual_objs[name] = OrderedDict()
            for sub_ele in ele:
                if sub_ele.tag == "visual":
                    visual_mesh_name = sub_ele.get("name", "visuals").replace("-", "_")
                    obj_file = None if sub_ele.find(".//mesh") is None else sub_ele.find(".//mesh").get("filename")
                    if obj_file is None:
                        print(f"Warning: No obj file found associated with {name}/{visual_mesh_name}!")
                    visual_objs[name][visual_mesh_name] = obj_file

    return visual_objs


def _copy_object_state_textures(obj_category, obj_model, dataset_root):
    obj_root_dir = f"{dataset_root}/objects/{obj_category}/{obj_model}"
    old_mat_fpath = f"{obj_root_dir}/material"
    new_mat_fpath = f"{obj_root_dir}/usd/materials"
    for mat_file in os.listdir(old_mat_fpath):
        should_copy = False
        for object_state in _OBJECT_STATE_TEXTURES:
            if object_state in mat_file.lower():
                should_copy = True
                break
        if should_copy:
            shutil.copy(f"{old_mat_fpath}/{mat_file}", new_mat_fpath)


def _import_rendering_channels(obj_prim, obj_category, obj_model, model_root_path, usd_path, dataset_root):
    usd_dir = os.path.dirname(usd_path)
    # # mat_dir = f"{model_root_path}/material/{obj_category}" if \
    # #     obj_category in {"ceilings", "walls", "floors"} else f"{model_root_path}/material"
    # mat_dir = f"{model_root_path}/material"
    # # Compile all material files we have
    # mat_files = set(os.listdir(mat_dir))

    # Remove the material prims as we will create them explictly later.
    # TODO: Be a bit more smart about this. a material procedurally generated will lose its material without it having
    # be regenerated
    stage = lazy.omni.usd.get_context().get_stage()
    for prim in obj_prim.GetChildren():
        looks_prim = None
        if prim.GetName() == "Looks":
            looks_prim = prim
        elif prim.GetPrimTypeInfo().GetTypeName() == "Xform":
            looks_prim_path = f"{str(prim.GetPrimPath())}/Looks"
            looks_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(looks_prim_path)
        if not looks_prim:
            continue
        for subprim in looks_prim.GetChildren():
            if subprim.GetPrimTypeInfo().GetTypeName() != "Material":
                continue
            print(
                f"Removed material prim {subprim.GetPath()}:",
                stage.RemovePrim(subprim.GetPath()),
            )

    # # Create new default material for this object.
    # mtl_created_list = []
    # lazy.omni.kit.commands.execute(
    #     "CreateAndBindMdlMaterialFromLibrary",
    #     mdl_name="OmniPBR.mdl",
    #     mtl_name="OmniPBR",
    #     mtl_created_list=mtl_created_list,
    # )
    # default_mat = lazy.omni.isaac.core.utils.prims.get_prim_at_path(mtl_created_list[0])
    # default_mat = rename_prim(prim=default_mat, name=f"default_material")
    # print("Created default material:", default_mat.GetPath())
    #
    # # We may delete this default material if it's never used
    # default_mat_is_used = False

    # Grab all visual objs for this object
    urdf_path = f"{dataset_root}/objects/{obj_category}/{obj_model}/{obj_model}_with_metalinks.urdf"
    visual_objs = _get_visual_objs_from_urdf(urdf_path)

    # Extract absolute paths to mtl files for each link
    link_mtl_files = OrderedDict()  # maps link name to dictionary mapping mesh name to mtl file
    mtl_infos = OrderedDict()  # maps mtl name to dictionary mapping material channel name to png file
    mat_files = OrderedDict()  # maps mtl name to corresponding list of material filenames
    mtl_old_dirs = OrderedDict()  # maps mtl name to corresponding directory where the mtl file exists
    mat_old_paths = OrderedDict()  # maps mtl name to corresponding list of relative mat paths from mtl directory
    for link_name, link_meshes in visual_objs.items():
        link_mtl_files[link_name] = OrderedDict()
        for mesh_name, obj_file in link_meshes.items():
            # Get absolute path and open the obj file if it exists:
            if obj_file is not None:
                obj_path = f"{dataset_root}/objects/{obj_category}/{obj_model}/{obj_file}"
                with open(obj_path, "r") as f:
                    mtls = []
                    for line in f.readlines():
                        if "mtllib" in line and line[0] != "#":
                            mtls.append(line.split("mtllib ")[-1].split("\n")[0])

                if mtls:
                    assert len(mtls) == 1, f"Only one mtl is supported per obj file in omniverse -- found {len(mtls)}!"
                    mtl = mtls[0]
                    # TODO: Make name unique
                    mtl_name = ".".join(os.path.basename(mtl).split(".")[:-1]).replace("-", "_").replace(".", "_")
                    mtl_old_dir = f"{'/'.join(obj_path.split('/')[:-1])}"
                    link_mtl_files[link_name][mesh_name] = mtl_name
                    mtl_infos[mtl_name] = OrderedDict()
                    mtl_old_dirs[mtl_name] = mtl_old_dir
                    mat_files[mtl_name] = []
                    mat_old_paths[mtl_name] = []
                    # Open the mtl file
                    mtl_path = f"{mtl_old_dir}/{mtl}"
                    with open(mtl_path, "r") as f:
                        # Read any lines beginning with map that aren't commented out
                        for line in f.readlines():
                            if line[:4] == "map_":
                                map_type, map_file = line.split(" ")
                                map_file = map_file.split("\n")[0]
                                map_filename = os.path.basename(map_file)
                                mat_files[mtl_name].append(map_filename)
                                mat_old_paths[mtl_name].append(map_file)
                                mtl_infos[mtl_name][_MTL_MAP_TYPE_MAPPINGS[map_type.lower()]] = map_filename

    # Next, for each material information, we create a new material and port the material files to the USD directory
    mat_new_fpath = os.path.join(usd_dir, "materials")
    Path(mat_new_fpath).mkdir(parents=True, exist_ok=True)
    shaders = OrderedDict()  # maps mtl name to shader prim
    rendering_channel_mappings = {
        "diffuse": _set_mtl_albedo,
        "albedo": _set_mtl_albedo,
        "normal": _set_mtl_normal,
        "ao": _set_mtl_ao,
        "roughness": _set_mtl_roughness,
        "metalness": _set_mtl_metalness,
        "opacity": _set_mtl_opacity,
        "emission": _set_mtl_emission,
    }
    for mtl_name, mtl_info in mtl_infos.items():
        for mat_old_path in mat_old_paths[mtl_name]:
            shutil.copy(os.path.join(mtl_old_dirs[mtl_name], mat_old_path), mat_new_fpath)

        # Create the new material
        mtl_created_list = []
        lazy.omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=mtl_created_list,
        )
        mat = lazy.omni.isaac.core.utils.prims.get_prim_at_path(mtl_created_list[0])

        # Apply all rendering channels for this material
        for mat_type, mat_file in mtl_info.items():
            render_channel_fcn = rendering_channel_mappings.get(mat_type, None)
            if render_channel_fcn is not None:
                render_channel_fcn(mat, os.path.join("materials", mat_file))
            else:
                # Warn user that we didn't find the correct rendering channel
                print(f"Warning: could not find rendering channel function for material: {mat_type}, skipping")

        # Rename material
        mat = _rename_prim(prim=mat, name=mtl_name)
        shade = lazy.pxr.UsdShade.Material(mat)
        shaders[mtl_name] = shade
        print(f"Created material {mtl_name}:", mtl_created_list[0])

    # Bind each (visual) mesh to its appropriate material in the object
    # We'll loop over each link, create a list of 2-tuples each consisting of (mesh_prim_path, mtl_name) to be bound
    root_prim_path = obj_prim.GetPrimPath().pathString
    for link_name, mesh_mtl_names in link_mtl_files.items():
        # Special case -- omni always calls the visuals "visuals" by default if there's only a single visual mesh for the
        # given
        if len(mesh_mtl_names) == 1:
            mesh_mtl_infos = [
                (
                    f"{root_prim_path}/{link_name}/visuals",
                    list(mesh_mtl_names.values())[0],
                )
            ]
        else:
            mesh_mtl_infos = []
            for mesh_name, mtl_name in mesh_mtl_names.items():
                # Omni only accepts a-z, A-Z as valid start characters for prim names
                # So we check if there is an invalid character, and modify it as we know Omni does
                if not ord("a") <= ord(mesh_name[0]) <= ord("z") and not ord("A") <= ord(mesh_name[0]) <= ord("Z"):
                    mesh_name = "a_" + mesh_name[1:]
                mesh_mtl_infos.append((f"{root_prim_path}/{link_name}/visuals/{mesh_name}", mtl_name))
        for mesh_prim_path, mtl_name in mesh_mtl_infos:
            visual_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(mesh_prim_path)
            assert visual_prim, f"Error: Did not find valid visual prim at {mesh_prim_path}!"
            # Bind the created link material to the visual prim
            print(f"Binding material {mtl_name}, shader {shaders[mtl_name]}, to prim {mesh_prim_path}...")
            lazy.pxr.UsdShade.MaterialBindingAPI(visual_prim).Bind(
                shaders[mtl_name], lazy.pxr.UsdShade.Tokens.strongerThanDescendants
            )

    # Lastly, we copy object_state texture maps that are state-conditioned; e.g.: cooked, soaked, etc.
    _copy_object_state_textures(obj_category=obj_category, obj_model=obj_model, dataset_root=dataset_root)

    # ###################################
    #
    # # Iterate over all children of the object prim, if /<obj_name>/<link_name>/visual exists, then we
    # # know <link_name> is a valid link, and we check explicitly for these material files in our set
    # # Note: we assume that the link name is included as a string within the mat_file!
    # for prim in obj_prim.GetChildren():
    #     if prim.GetPrimTypeInfo().GetTypeName() == "Xform":
    #         # This could be a link, check if it owns a visual subprim
    #         link_name = prim.GetName()
    #         visual_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(f"{prim.GetPrimPath().pathString}/visuals")
    #         print(f"path: {prim.GetPrimPath().pathString}/visuals")
    #         print(f"visual prim: {visual_prim}")
    #
    #         if visual_prim:
    #             # Aggregate all material files for this prim
    #             link_mat_files = []
    #             for mat_file in deepcopy(mat_files):
    #                 if link_name in mat_file:
    #                     # Add this mat file and remove it from the set
    #                     link_mat_files.append(mat_file)
    #                     mat_files.remove(mat_file)
    #             # Potentially write material files for this prim if we have any valid materials
    #             print("link_mat_files:", link_mat_files)
    #             if not link_mat_files:
    #                 # Bind default material to the visual prim
    #                 shade = lazy.pxr.UsdShade.Material(default_mat)
    #                 lazy.pxr.UsdShade.MaterialBindingAPI(visual_prim).Bind(shade, lazy.pxr.UsdShade.Tokens.strongerThanDescendants)
    #                 default_mat_is_used = True
    #             else:
    #                 # Create new material for this link
    #                 mtl_created_list = []
    #                 lazy.omni.kit.commands.execute(
    #                     "CreateAndBindMdlMaterialFromLibrary",
    #                     mdl_name="OmniPBR.mdl",
    #                     mtl_name="OmniPBR",
    #                     mtl_created_list=mtl_created_list,
    #                 )
    #                 print(f"Created material for link {link_name}:", mtl_created_list[0])
    #                 mat = lazy.omni.isaac.core.utils.prims.get_prim_at_path(mtl_created_list[0])
    #
    #                 shade = lazy.pxr.UsdShade.Material(mat)
    #                 # Bind the created link material to the visual prim
    #                 lazy.pxr.UsdShade.MaterialBindingAPI(visual_prim).Bind(shade, lazy.pxr.UsdShade.Tokens.strongerThanDescendants)
    #
    #                 # Iterate over all material channels and write them to the material
    #                 for link_mat_file in link_mat_files:
    #                     # Copy this file into the materials folder
    #                     mat_fpath = os.path.join(usd_dir, "materials")
    #                     shutil.copy(os.path.join(mat_dir, link_mat_file), mat_fpath)
    #                     # Check if any valid rendering channel
    #                     mat_type = link_mat_file.split("_")[-1].split(".")[0].lower()
    #                     # Apply the material if it exists
    #                     render_channel_fcn = rendering_channel_mappings.get(mat_type, None)
    #                     if render_channel_fcn is not None:
    #                         render_channel_fcn(mat, os.path.join("materials", link_mat_file))
    #                     else:
    #                         # Warn user that we didn't find the correct rendering channel
    #                         print(f"Warning: could not find rendering channel function for material: {mat_type}, skipping")
    #
    #                 # Rename material
    #                 mat = rename_prim(prim=mat, name=f"material_{link_name}")
    #
    # # For any remaining materials, we write them to the default material
    # # default_mat = lazy.omni.isaac.core.utils.prims.get_prim_at_path(f"{obj_prim.GetPrimPath().pathString}/Looks/material_material_0")
    # # default_mat = lazy.omni.isaac.core.utils.prims.get_prim_at_path(f"{obj_prim.GetPrimPath().pathString}/Looks/material_default")
    # print(f"default mat: {default_mat}, obj: {obj_category}, {prim.GetPrimPath().pathString}")
    # for mat_file in mat_files:
    #     # Copy this file into the materials folder
    #     mat_fpath = os.path.join(usd_dir, "materials")
    #     shutil.copy(os.path.join(mat_dir, mat_file), mat_fpath)
    #     # Check if any valid rendering channel
    #     mat_type = mat_file.split("_")[-1].split(".")[0].lower()
    #     # Apply the material if it exists
    #     render_channel_fcn = rendering_channel_mappings.get(mat_type, None)
    #     if render_channel_fcn is not None:
    #         render_channel_fcn(default_mat, os.path.join("materials", mat_file))
    #         default_mat_is_used = True
    #     else:
    #         # Warn user that we didn't find the correct rendering channel
    #         print(f"Warning: could not find rendering channel function for material: {mat_type}")
    #
    # # Possibly delete the default material prim if it was never used
    # if not default_mat_is_used:
    #     stage.RemovePrim(default_mat.GetPrimPath())


def _add_xform_properties(prim):
    properties_to_remove = [
        "xformOp:rotateX",
        "xformOp:rotateXZY",
        "xformOp:rotateY",
        "xformOp:rotateYXZ",
        "xformOp:rotateYZX",
        "xformOp:rotateZ",
        "xformOp:rotateZYX",
        "xformOp:rotateZXY",
        "xformOp:rotateXYZ",
        "xformOp:transform",
    ]
    prop_names = prim.GetPropertyNames()
    xformable = lazy.pxr.UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    # TODO: wont be able to delete props for non root links on articulated objects
    for prop_name in prop_names:
        if prop_name in properties_to_remove:
            prim.RemoveProperty(prop_name)
    if "xformOp:scale" not in prop_names:
        xform_op_scale = xformable.AddXformOp(
            lazy.pxr.UsdGeom.XformOp.TypeScale,
            lazy.pxr.UsdGeom.XformOp.PrecisionDouble,
            "",
        )
        xform_op_scale.Set(lazy.pxr.Gf.Vec3d([1.0, 1.0, 1.0]))
    else:
        xform_op_scale = lazy.pxr.UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))

    if "xformOp:translate" not in prop_names:
        xform_op_translate = xformable.AddXformOp(
            lazy.pxr.UsdGeom.XformOp.TypeTranslate,
            lazy.pxr.UsdGeom.XformOp.PrecisionDouble,
            "",
        )
    else:
        xform_op_translate = lazy.pxr.UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))

    if "xformOp:orient" not in prop_names:
        xform_op_rot = xformable.AddXformOp(
            lazy.pxr.UsdGeom.XformOp.TypeOrient,
            lazy.pxr.UsdGeom.XformOp.PrecisionDouble,
            "",
        )
    else:
        xform_op_rot = lazy.pxr.UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
    xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])


def _process_meta_link(stage, obj_model, meta_link_type, meta_link_infos):
    """
    Process a meta link by creating visual meshes or lights below it
    """
    # TODO: Reenable after fillable meshes are backported into 3ds Max.
    # Temporarily disable importing of fillable meshes.
    if meta_link_type in ["container"]:
        return

    assert meta_link_type in _ALLOWED_META_TYPES
    if _ALLOWED_META_TYPES[meta_link_type] not in ["primitive", "light"] and meta_link_type != "particlesource":
        return

    is_light = _ALLOWED_META_TYPES[meta_link_type] == "light"

    for link_id, mesh_info_list in meta_link_infos.items():
        if len(mesh_info_list) == 0:
            continue

        # TODO: Remove this after this is fixed.
        if type(mesh_info_list) == dict:
            keys = [str(x) for x in range(len(mesh_info_list))]
            assert set(mesh_info_list.keys()) == set(keys), "Unexpected keys"
            mesh_info_list = [mesh_info_list[k] for k in keys]

        if meta_link_type in [
            "togglebutton",
            "particleapplier",
            "particleremover",
            "particlesink",
            "particlesource",
        ]:
            assert len(mesh_info_list) == 1, f"Invalid number of meshes for {meta_link_type}"

        meta_link_in_parent_link_pos, meta_link_in_parent_link_orn = (
            mesh_info_list[0]["position"],
            mesh_info_list[0]["orientation"],
        )

        # For particle applier only, the orientation of the meta link matters (particle should shoot towards the negative z-axis)
        # If the meta link is created based on the orientation of the first mesh that is a cone, we need to rotate it by 180 degrees
        # because the cone is pointing in the wrong direction. This is already done in update_obj_urdf_with_metalinks;
        # we just need to make sure meta_link_in_parent_link_orn is updated correctly.
        if meta_link_type == "particleapplier" and mesh_info_list[0]["type"] == "cone":
            meta_link_in_parent_link_orn = T.quat_multiply(
                meta_link_in_parent_link_orn, T.axisangle2quat(th.tensor([math.pi, 0.0, 0.0]))
            )

        for i, mesh_info in enumerate(mesh_info_list):
            is_mesh = False
            if is_light:
                # Create a light
                light_type = _LIGHT_MAPPING[mesh_info["type"]]
                prim_path = f"/{obj_model}/lights_{link_id}_0_link/light_{i}"
                prim = getattr(lazy.pxr.UsdLux, f"{light_type}Light").Define(stage, prim_path).GetPrim()
                lazy.pxr.UsdLux.ShapingAPI.Apply(prim).GetShapingConeAngleAttr().Set(180.0)
            else:
                if meta_link_type == "particlesource":
                    mesh_type = "Cylinder"
                else:
                    # Create a primitive shape
                    mesh_type = mesh_info["type"].capitalize() if mesh_info["type"] != "box" else "Cube"
                prim_path = f"/{obj_model}/{meta_link_type}_{link_id}_0_link/mesh_{i}"
                assert hasattr(lazy.pxr.UsdGeom, mesh_type)
                # togglebutton has to be a sphere
                if meta_link_type in ["togglebutton"]:
                    is_mesh = True
                # particle applier has to be a cone or cylinder because of the visualization of the particle flow
                elif meta_link_type in ["particleapplier"]:
                    assert mesh_type in [
                        "Cone",
                        "Cylinder",
                    ], f"Invalid mesh type for particleapplier: {mesh_type}"
                prim = (
                    create_primitive_mesh(prim_path, mesh_type, stage=stage).GetPrim()
                    if is_mesh
                    else getattr(lazy.pxr.UsdGeom, mesh_type).Define(stage, prim_path).GetPrim()
                )

            _add_xform_properties(prim=prim)
            # Make sure mesh_prim has XForm properties
            xform_prim = lazy.omni.isaac.core.prims.xform_prim.XFormPrim(prim_path=prim_path)

            # Get the mesh/light pose in the parent link frame
            mesh_in_parent_link_pos, mesh_in_parent_link_orn = th.tensor(mesh_info["position"]), th.tensor(
                mesh_info["orientation"]
            )

            # Get the mesh/light pose in the meta link frame
            mesh_in_parent_link_tf = th.eye(4)
            mesh_in_parent_link_tf[:3, :3] = T.quat2mat(mesh_in_parent_link_orn)
            mesh_in_parent_link_tf[:3, 3] = mesh_in_parent_link_pos
            meta_link_in_parent_link_tf = th.eye(4)
            meta_link_in_parent_link_tf[:3, :3] = T.quat2mat(meta_link_in_parent_link_orn)
            meta_link_in_parent_link_tf[:3, 3] = meta_link_in_parent_link_pos
            mesh_in_meta_link_tf = th.linalg.inv(meta_link_in_parent_link_tf) @ mesh_in_parent_link_tf
            mesh_in_meta_link_pos, mesh_in_meta_link_orn = (
                mesh_in_meta_link_tf[:3, 3],
                T.mat2quat(mesh_in_meta_link_tf[:3, :3]),
            )

            if is_light:
                xform_prim.prim.GetAttribute("inputs:color").Set(
                    lazy.pxr.Gf.Vec3f(*(th.tensor(mesh_info["color"]) / 255.0).tolist())
                )
                xform_prim.prim.GetAttribute("inputs:intensity").Set(mesh_info["intensity"])
                if light_type == "Rect":
                    xform_prim.prim.GetAttribute("inputs:width").Set(mesh_info["length"])
                    xform_prim.prim.GetAttribute("inputs:height").Set(mesh_info["width"])
                elif light_type == "Disk":
                    xform_prim.prim.GetAttribute("inputs:radius").Set(mesh_info["length"])
                elif light_type == "Sphere":
                    xform_prim.prim.GetAttribute("inputs:radius").Set(mesh_info["length"])
                else:
                    raise ValueError(f"Invalid light type: {light_type}")
            else:
                if mesh_type == "Cylinder":
                    if not is_mesh:
                        xform_prim.prim.GetAttribute("radius").Set(0.5)
                        xform_prim.prim.GetAttribute("height").Set(1.0)
                    if meta_link_type == "particlesource":
                        desired_radius = 0.0125
                        desired_height = 0.05
                        height_offset = -desired_height / 2.0
                    else:
                        desired_radius = mesh_info["size"][0]
                        desired_height = mesh_info["size"][2]
                        height_offset = desired_height / 2.0
                    xform_prim.prim.GetAttribute("xformOp:scale").Set(
                        lazy.pxr.Gf.Vec3f(desired_radius * 2, desired_radius * 2, desired_height)
                    )
                    # Offset the position by half the height because in 3dsmax the origin of the cylinder is at the center of the base
                    mesh_in_meta_link_pos += T.quat_apply(mesh_in_meta_link_orn, th.tensor([0.0, 0.0, height_offset]))
                elif mesh_type == "Cone":
                    if not is_mesh:
                        xform_prim.prim.GetAttribute("radius").Set(0.5)
                        xform_prim.prim.GetAttribute("height").Set(1.0)
                    desired_radius = mesh_info["size"][0]
                    desired_height = mesh_info["size"][2]
                    height_offset = -desired_height / 2.0
                    xform_prim.prim.GetAttribute("xformOp:scale").Set(
                        lazy.pxr.Gf.Vec3f(desired_radius * 2, desired_radius * 2, desired_height)
                    )
                    # Flip the orientation of the z-axis because in 3dsmax the cone is pointing in the opposite direction
                    mesh_in_meta_link_orn = T.quat_multiply(
                        mesh_in_meta_link_orn, T.axisangle2quat(th.tensor([math.pi, 0.0, 0.0]))
                    )
                    # Offset the position by half the height because in 3dsmax the origin of the cone is at the center of the base
                    mesh_in_meta_link_pos += T.quat_apply(mesh_in_meta_link_orn, th.tensor([0.0, 0.0, height_offset]))
                elif mesh_type == "Cube":
                    if not is_mesh:
                        xform_prim.prim.GetAttribute("size").Set(1.0)
                    xform_prim.prim.GetAttribute("xformOp:scale").Set(lazy.pxr.Gf.Vec3f(*mesh_info["size"]))
                    height_offset = mesh_info["size"][2] / 2.0
                    mesh_in_meta_link_pos += T.quat_apply(mesh_in_meta_link_orn, th.tensor([0.0, 0.0, height_offset]))
                elif mesh_type == "Sphere":
                    if not is_mesh:
                        xform_prim.prim.GetAttribute("radius").Set(0.5)
                    desired_radius = mesh_info["size"][0]
                    xform_prim.prim.GetAttribute("xformOp:scale").Set(
                        lazy.pxr.Gf.Vec3f(desired_radius * 2, desired_radius * 2, desired_radius * 2)
                    )
                else:
                    raise ValueError(f"Invalid mesh type: {mesh_type}")

                # Make invisible
                lazy.pxr.UsdGeom.Imageable(xform_prim.prim).MakeInvisible()

            xform_prim.set_local_pose(
                translation=mesh_in_meta_link_pos,
                orientation=mesh_in_meta_link_orn[[3, 0, 1, 2]],
            )


def _process_glass_link(prim):
    # Update any glass parts to use the glass material instead
    glass_prim_paths = []
    for gchild in prim.GetChildren():
        if gchild.GetTypeName() == "Mesh":
            # check if has col api, if not, this is visual
            if not gchild.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI):
                glass_prim_paths.append(gchild.GetPath().pathString)
        elif gchild.GetTypeName() == "Scope":
            # contains multiple additional prims, check those
            for ggchild in gchild.GetChildren():
                if ggchild.GetTypeName() == "Mesh":
                    # check if has col api, if not, this is visual
                    if not ggchild.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI):
                        glass_prim_paths.append(ggchild.GetPath().pathString)

    assert glass_prim_paths

    stage = lazy.omni.isaac.core.utils.stage.get_current_stage()
    root_path = stage.GetDefaultPrim().GetPath().pathString
    glass_mtl_prim_path = f"{root_path}/Looks/OmniGlass"
    if not lazy.omni.isaac.core.utils.prims.get_prim_at_path(glass_mtl_prim_path):
        mtl_created = []
        lazy.omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniGlass.mdl",
            mtl_name="OmniGlass",
            mtl_created_list=mtl_created,
        )

    for glass_prim_path in glass_prim_paths:
        lazy.omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=glass_prim_path,
            material_path=glass_mtl_prim_path,
            strength=None,
        )


def import_obj_metadata(obj_category, obj_model, dataset_root, import_render_channels=False):
    # Check if filepath exists
    model_root_path = f"{dataset_root}/objects/{obj_category}/{obj_model}"
    usd_path = f"{model_root_path}/usd/{obj_model}.usd"
    print("Loading", usd_path, "for metadata import.")

    print("Start metadata import")

    # Load model
    lazy.omni.isaac.core.utils.stage.open_stage(usd_path)
    stage = lazy.omni.isaac.core.utils.stage.get_current_stage()
    prim = stage.GetDefaultPrim()

    data = dict()
    for data_group in {"metadata", "mvbb_meta", "material_groups", "heights_per_link"}:
        data_path = f"{model_root_path}/misc/{data_group}.json"
        if exists(data_path):
            # Load data
            with open(data_path, "r") as f:
                data[data_group] = json.load(f)

    # If certain metadata doesn't exist, populate with some core info
    if "base_link_offset" not in data["metadata"]:
        data["metadata"]["base_link_offset"] = [0, 0, 0]
    if "bbox_size" not in data["metadata"]:
        raise ValueError("We cannot work without a bbox size.")

    # Pop bb and base link offset and meta links info
    base_link_offset = data["metadata"].pop("base_link_offset")
    default_bb = data["metadata"].pop("bbox_size")

    # Manually modify material groups info
    if "material_groups" in data:
        data["material_groups"] = {
            "groups": data["material_groups"][0],
            "links": data["material_groups"][1],
        }

    # Manually modify metadata
    if "openable_joint_ids" in data["metadata"]:
        data["metadata"]["openable_joint_ids"] = {
            str(pair[0]): pair[1] for pair in data["metadata"]["openable_joint_ids"]
        }

    # Grab light info if any
    meta_links = data["metadata"].get("meta_links", dict())

    print("Process meta links")

    # TODO: Use parent link name
    for link_name, link_metadata in meta_links.items():
        for meta_link_type, meta_link_infos in link_metadata.items():
            _process_meta_link(stage, obj_model, meta_link_type, meta_link_infos)

    # Apply temporary fillable meshes.
    # TODo: Disable after fillable meshes are backported into 3ds Max.
    fillable_path = f"{model_root_path}/fillable.obj"
    if exists(fillable_path):
        mesh = trimesh.load(fillable_path, force="mesh")
        import_fillable_mesh(stage, obj_model, mesh)

    print("Done processing meta links")

    # Iterate over dict and replace any lists of dicts as dicts of dicts (with each dict being indexed by an integer)
    data = _recursively_replace_list_of_dict(data)

    print("Done recursively replacing")

    # Create attributes for bb, offset, category, model and store values
    prim.CreateAttribute("ig:nativeBB", lazy.pxr.Sdf.ValueTypeNames.Vector3f)
    prim.CreateAttribute("ig:offsetBaseLink", lazy.pxr.Sdf.ValueTypeNames.Vector3f)
    prim.CreateAttribute("ig:category", lazy.pxr.Sdf.ValueTypeNames.String)
    prim.CreateAttribute("ig:model", lazy.pxr.Sdf.ValueTypeNames.String)
    prim.GetAttribute("ig:nativeBB").Set(lazy.pxr.Gf.Vec3f(*default_bb))
    prim.GetAttribute("ig:offsetBaseLink").Set(lazy.pxr.Gf.Vec3f(*base_link_offset))
    prim.GetAttribute("ig:category").Set(obj_category)
    prim.GetAttribute("ig:model").Set(obj_model)

    print(f"data: {data}")

    # Store remaining data as metadata
    prim.SetCustomData(data)

    # Add material channels
    # print(f"prim children: {prim.GetChildren()}")
    # looks_prim_path = f"{str(prim.GetPrimPath())}/Looks"
    # looks_prim = prim.GetChildren()[0] #lazy.omni.isaac.core.utils.prims.get_prim_at_path(looks_prim_path)
    # mat_prim_path = f"{str(prim.GetPrimPath())}/Looks/material_material_0"
    # mat_prim = looks_prim.GetChildren()[0] #lazy.omni.isaac.core.utils.prims.get_prim_at_path(mat_prim_path)
    # print(f"looks children: {looks_prim.GetChildren()}")
    # print(f"mat prim: {mat_prim}")
    if import_render_channels:
        _import_rendering_channels(
            obj_prim=prim,
            obj_category=obj_category,
            obj_model=obj_model,
            model_root_path=model_root_path,
            usd_path=usd_path,
            dataset_root=dataset_root,
        )
    for link, link_tags in data["metadata"]["link_tags"].items():
        if "glass" in link_tags:
            _process_glass_link(prim.GetChild(link))

    # Save stage
    stage.Save()

    # Delete stage reference and clear the sim stage variable, opening the dummy stage along the way
    del stage


def _recursively_replace_list_of_dict(dic):
    for k, v in dic.items():
        print(f"k: {k}")
        if v is None:
            # Replace None
            dic[k] = lazy.pxr.lazy.pxr.UsdGeom.Tokens.none
        elif isinstance(v, list) or isinstance(v, tuple):
            if len(v) == 0:
                dic[k] = lazy.pxr.Vt.Vec3fArray()
            elif isinstance(v[0], dict):
                # Replace with dict in place
                v = {str(i): vv for i, vv in enumerate(v)}
                dic[k] = v
            elif isinstance(v[0], list) or isinstance(v[0], tuple):
                # # Flatten the lists
                # dic[k] = []
                # for vv in v:
                #     dic[k] += vv
                print("v0: ", v[0])
                if len(v[0]) == 1:
                    # Do nothing
                    pass
                if len(v[0]) == 2:
                    dic[k] = lazy.pxr.Vt.Vec2fArray(v)
                elif len(v[0]) == 3:
                    dic[k] = lazy.pxr.Vt.Vec3fArray(v)
                elif len(v[0]) == 4:
                    dic[k] = lazy.pxr.Vt.Vec4fArray(v)
                else:
                    raise ValueError(f"No support for storing matrices of length {len(v[0])}!")
            elif isinstance(v[0], int):
                # if len(v) == 1:
                #     # Do nothing
                #     pass
                # elif len(v) == 2:
                #     dic[k] = lazy.pxr.Gf.Vec2i(v)
                # elif len(v) == 3:
                #     dic[k] = lazy.pxr.Gf.Vec3i(v)
                # elif len(v) == 4:
                #     dic[k] = lazy.pxr.Gf.Vec4i(v)
                # else:
                dic[k] = lazy.pxr.Vt.IntArray(v)
                # raise ValueError(f"No support for storing numeric arrays of length {len(v)}! Array: {v}")
            elif isinstance(v[0], float):
                # if len(v) == 1:
                #     # Do nothing
                #     pass
                # elif len(v) == 2:
                #     dic[k] = lazy.pxr.Gf.Vec2f(v)
                # elif len(v) == 3:
                #     dic[k] = lazy.pxr.Gf.Vec3f(v)
                # elif len(v) == 4:
                #     dic[k] = lazy.pxr.Gf.Vec4f(v)
                # else:
                dic[k] = lazy.pxr.Vt.FloatArray(v)
                # raise ValueError(f"No support for storing numeric arrays of length {len(v)}! Array: {v}")
            else:
                # Replace any Nones
                for i, ele in enumerate(v):
                    if ele is None:
                        v[i] = lazy.pxr.lazy.pxr.UsdGeom.Tokens.none
        if isinstance(v, dict):
            # Iterate through nested dictionaries
            dic[k] = _recursively_replace_list_of_dict(v)

    return dic


def _create_urdf_import_config():
    # Set up import configuration
    _, import_config = lazy.omni.kit.commands.execute("URDFCreateImportConfig")
    drive_mode = (
        import_config.default_drive_type.__class__
    )  # Hacky way to get class for default drive type, options are JOINT_DRIVE_{NONE / POSITION / VELOCITY}

    import_config.set_merge_fixed_joints(False)
    import_config.set_convex_decomp(True)
    import_config.set_fix_base(False)
    import_config.set_import_inertia_tensor(False)
    import_config.set_distance_scale(1.0)
    import_config.set_density(0.0)
    import_config.set_default_drive_type(drive_mode.JOINT_DRIVE_NONE)
    import_config.set_default_drive_strength(0.0)
    import_config.set_default_position_drive_damping(0.0)
    import_config.set_self_collision(False)
    import_config.set_up_vector(0, 0, 1)
    import_config.set_make_default_prim(True)
    import_config.set_create_physics_scene(True)
    return import_config


def import_obj_urdf(obj_category, obj_model, dataset_root, skip_if_exist=False):
    # Preprocess input URDF to account for metalinks
    urdf_path = _add_metalinks_to_urdf(
        obj_category=obj_category, obj_model=obj_model, dataset_root=dataset_root
    )
    # Import URDF
    cfg = _create_urdf_import_config()
    # Check if filepath exists
    usd_path = f"{dataset_root}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd"
    if not (skip_if_exist and exists(usd_path)):
        if _SPLIT_COLLISION_MESHES:
            print(f"Converting collision meshes from {obj_category}, {obj_model}...")
            urdf_path = _split_all_objs_in_urdf(urdf_fpath=urdf_path, name_suffix="split")
        print(f"Importing {obj_category}, {obj_model} into path {usd_path}...")
        # Only import if it doesn't exist
        lazy.omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=cfg,
            dest_path=usd_path,
        )
        print(f"Imported {obj_category}, {obj_model}")


def _pretty_print_xml(current, parent=None, index=-1, depth=0, use_tabs=False):
    space = "\t" if use_tabs else " " * 4
    for i, node in enumerate(current):
        _pretty_print_xml(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = "\n" + (space * depth)
        else:
            parent[index - 1].tail = "\n" + (space * depth)
        if index == len(parent) - 1:
            current.tail = "\n" + (space * (depth - 1))


def _convert_to_xml_string(inp):
    """
    Converts any type of {bool, int, float, list, tuple, array, string, th.Tensor} into an mujoco-xml compatible string.
        Note that an input string / th.Tensor results in a no-op action.
    Args:
        inp: Input to convert to string
    Returns:
        str: String equivalent of @inp
    """
    if type(inp) in {list, tuple, th.Tensor}:
        return _tensor_to_space_script(inp)
    elif type(inp) in {int, float, bool, th.float32, th.float64, th.int32, th.int64}:
        return str(inp).lower()
    elif type(inp) in {str}:
        return inp
    else:
        raise ValueError("Unsupported type received: got {}".format(type(inp)))


def _create_urdf_joint(
    name,
    parent,
    child,
    pos=(0, 0, 0),
    rpy=(0, 0, 0),
    joint_type="fixed",
    axis=None,
    damping=None,
    friction=None,
    limits=None,
):
    """
    Generates XML joint
    Args:
        name (str): Name of this joint
        parent (str or ET.Element): Name of parent link or parent link element itself for this joint
        child (str or ET.Element): Name of child link or child link itself for this joint
        pos (list or tuple or th.Tensor): (x,y,z) offset pos values when creating the collision body
        rpy (list or tuple or th.Tensor): (r,p,y) offset rot values when creating the joint
        joint_type (str): What type of joint to create. Must be one of {fixed, revolute, prismatic}
        axis (None or 3-tuple): If specified, should be (x,y,z) axis corresponding to DOF
        damping (None or float): If specified, should be damping value to apply to joint
        friction (None or float): If specified, should be friction value to apply to joint
        limits (None or 2-tuple): If specified, should be min / max limits to the applied joint
    Returns:
        ET.Element: Generated joint element
    """
    # Create the initial joint
    jnt = ET.Element("joint", name=name, type=joint_type)
    # Create origin subtag
    origin = ET.SubElement(
        jnt,
        "origin",
        attrib={"rpy": _convert_to_xml_string(rpy), "xyz": _convert_to_xml_string(pos)},
    )
    # Make sure parent and child are both names (str) -- if they're not str already, we assume it's the element ref
    if not isinstance(parent, str):
        parent = parent.get("name")
    if not isinstance(child, str):
        child = child.get("name")
    # Create parent and child subtags
    parent = ET.SubElement(jnt, "parent", link=parent)
    child = ET.SubElement(jnt, "child", link=child)
    # Add additional parameters if specified
    if axis is not None:
        ax = ET.SubElement(jnt, "axis", xyz=_convert_to_xml_string(axis))
    dynamic_params = {}
    if damping is not None:
        dynamic_params["damping"] = _convert_to_xml_string(damping)
    if friction is not None:
        dynamic_params["friction"] = _convert_to_xml_string(friction)
    if dynamic_params:
        dp = ET.SubElement(jnt, "dynamics", **dynamic_params)
    if limits is not None:
        lim = ET.SubElement(jnt, "limit", lower=limits[0], upper=limits[1])

    # Return this element
    return jnt


def _create_urdf_link(name, subelements=None, mass=None, inertia=None):
    """
    Generates XML link element
    Args:
        name (str): Name of this link
        subelements (None or list): If specified, specifies all nested elements that should belong to this link
            (e.g.: visual, collision body elements)
        mass (None or float): If specified, will add an inertial tag with specified mass value
        inertia (None or 6-array): If specified, will add an inertial tag with specified inertia value
            Value should be (ixx, iyy, izz, ixy, ixz, iyz)
    Returns:
        ET.Element: Generated link
    """
    # Create the initial link
    link = ET.Element("link", name=name)
    # Add all subelements if specified
    if subelements is not None:
        for ele in subelements:
            link.append(ele)
    # Add mass subelement if requested
    if mass is not None or inertia is not None:
        inertial = ET.SubElement(link, "inertial")
    if mass is not None:
        ET.SubElement(inertial, "mass", value=_convert_to_xml_string(mass))
    if inertia is not None:
        axes = ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]
        inertia_vals = {ax: str(i) for ax, i in zip(axes, inertia)}
        ET.SubElement(inertial, "inertia", **inertia_vals)

    # Return this element
    return link


def _create_urdf_metalink(
    root_element,
    metalink_name,
    parent_link_name="base_link",
    pos=(0, 0, 0),
    rpy=(0, 0, 0),
):
    # Create joint
    jnt = _create_urdf_joint(
        name=f"{metalink_name}_joint",
        parent=parent_link_name,
        child=f"{metalink_name}_link",
        pos=pos,
        rpy=rpy,
        joint_type="fixed",
    )
    # Create child link
    link = _create_urdf_link(
        name=f"{metalink_name}_link",
        mass=0.0001,
        inertia=[0.00001, 0.00001, 0.00001, 0, 0, 0],
    )

    # Add to root element
    root_element.append(jnt)
    root_element.append(link)


def _save_xmltree_as_urdf(root_element, name, dirpath, unique_urdf=False):
    """
    Generates a URDF file corresponding to @xmltree at @dirpath with name @name.urdf.
    Args:
        root_element (ET.Element): Element tree that compose the URDF
        name (str): Name of this file (name assigned to robot tag)
        dirpath (str): Absolute path to the location / filename for the generated URDF
        unique_urdf (bool): Whether to use a unique identifier when naming urdf (uses current datetime)
    Returns:
        str: Path to newly created urdf (fpath/<name>.urdf)
    """
    # Write to fpath, making sure the directory exists (if not, create it)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    # Get file
    date = datetime.now().isoformat(timespec="microseconds").replace(".", "_").replace(":", "_").replace("-", "_")
    fname = f"{name}_{date}.urdf" if unique_urdf else f"{name}.urdf"
    fpath = os.path.join(dirpath, fname)
    with open(fpath, "w") as f:
        # Write top level header line first
        f.write('<?xml version="1.0" ?>\n')
        # Convert xml to string form and write to file
        _pretty_print_xml(current=root_element)
        xml_str = ET.tostring(root_element, encoding="unicode")
        f.write(xml_str)

    # Return path to file
    return fpath


def _add_metalinks_to_urdf(obj_category, obj_model, dataset_root):
    # Check if filepath exists
    model_root_path = f"{dataset_root}/objects/{obj_category}/{obj_model}"
    urdf_path = f"{model_root_path}/{obj_model}.urdf"

    # Load urdf
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Load metadata
    metadata_fpath = f"{model_root_path}/misc/metadata.json"
    with open(metadata_fpath, "r") as f:
        metadata = json.load(f)

    # Pop meta links
    assert not (
        "links" in metadata and "meta_links" in metadata
    ), "Only expected one of links and meta_links to be found in metadata, but found both!"

    if "meta_links" in metadata:
        # Rename meta links, e.g. from "fillable" to "container"
        for link, meta_link in metadata["meta_links"].items():
            for meta_link_name in list(meta_link.keys()):
                meta_link_attrs = meta_link[meta_link_name]
                if meta_link_name in _META_LINK_RENAME_MAPPING:
                    metadata["meta_links"][link][_META_LINK_RENAME_MAPPING[meta_link_name]] = meta_link_attrs
                    del metadata["meta_links"][link][meta_link_name]

        with open(metadata_fpath, "w") as f:
            json.dump(metadata, f)

        meta_links = metadata.pop("meta_links")
        print("meta_links:", meta_links)
        for parent_link_name, child_link_attrs in meta_links.items():
            for meta_link_name, ml_attrs in child_link_attrs.items():
                assert (
                    meta_link_name in _ALLOWED_META_TYPES
                ), f"meta_link_name {meta_link_name} not in {_ALLOWED_META_TYPES}"

                # TODO: Reenable after fillable meshes are backported into 3ds Max.
                # Temporarily disable importing of fillable meshes.
                if meta_link_name in ["container"]:
                    continue

                for ml_id, attrs_list in ml_attrs.items():
                    if len(attrs_list) > 0:
                        if _ALLOWED_META_TYPES[meta_link_name] != "dimensionless":
                            # If not dimensionless, we create one meta link for a list of meshes below it
                            attrs_list = [attrs_list[0]]
                        else:
                            # Otherwise, we create one meta link for each frame
                            # For non-attachment meta links, we expect only one instance per type
                            # E.g. heatsource_leftstove_0, heatsource_rightstove_0, but not heatsource_leftstove_1
                            if meta_link_name != "attachment":
                                assert (
                                    len(attrs_list) == 1
                                ), f"Expected only one instance for meta_link {meta_link_name}_{ml_id}, but found {len(attrs_list)}"

                        # TODO: Remove this after this is fixed.
                        if type(attrs_list) == dict:
                            keys = [str(x) for x in range(len(attrs_list))]
                            assert set(attrs_list.keys()) == set(keys), "Unexpected keys"
                            attrs_list = [attrs_list[k] for k in keys]

                        for i, attrs in enumerate(attrs_list):
                            pos = attrs["position"]
                            quat = attrs["orientation"]

                            # For particle applier only, the orientation of the meta link matters (particle should shoot towards the negative z-axis)
                            # If the meta link is created based on the orientation of the first mesh that is a cone, we need to rotate it by 180 degrees
                            # because the cone is pointing in the wrong direction.
                            if meta_link_name == "particleapplier" and attrs["type"] == "cone":
                                assert (
                                    len(attrs_list) == 1
                                ), f"Expected only one instance for meta_link {meta_link_name}_{ml_id}, but found {len(attrs_list)}"
                                quat = T.quat_multiply(quat, T.axisangle2quat(th.tensor([math.pi, 0.0, 0.0])))

                            # Create metalink
                            _create_urdf_metalink(
                                root_element=root,
                                metalink_name=f"{meta_link_name}_{ml_id}_{i}",
                                parent_link_name=parent_link_name,
                                pos=pos,
                                rpy=T.quat2euler(quat),
                            )

    # Export this URDF
    return _save_xmltree_as_urdf(
        root_element=root,
        name=f"{obj_model}_with_metalinks",
        dirpath=model_root_path,
        unique_urdf=False,
    )


def convert_scene_urdf_to_json(urdf, json_path):
    # First, load the requested objects from the URDF into OG
    _load_scene_from_urdf(urdf=urdf)

    # Play the simulator, then save
    og.sim.play()
    Path(os.path.dirname(json_path)).mkdir(parents=True, exist_ok=True)
    og.sim.save(json_path=json_path)

    # Load the json, remove the init_info because we don't need it, then save it again
    with open(json_path, "r") as f:
        scene_info = json.load(f)

    scene_info.pop("init_info")

    with open(json_path, "w+") as f:
        json.dump(scene_info, f, cls=_TorchEncoder, indent=4)


def _load_scene_from_urdf(urdf):
    # First, grab object info from the urdf
    objs_info = _get_objects_config_from_scene_urdf(urdf=urdf)

    # Load all the objects manually into a scene
    scene = Scene(use_floor_plane=False)
    og.sim.import_scene(scene)

    for obj_name, obj_info in objs_info.items():
        try:
            if not os.path.exists(
                DatasetObject.get_usd_path(obj_info["cfg"]["category"], obj_info["cfg"]["model"]).replace(
                    ".usd", ".encrypted.usd"
                )
            ):
                print("Missing object", obj_name)
                continue
            obj = DatasetObject(
                name=obj_name,
                **obj_info["cfg"],
            )
            og.sim.import_object(obj)
            obj.set_bbox_center_position_orientation(position=obj_info["bbox_pos"], orientation=obj_info["bbox_quat"])
        except Exception as e:
            raise ValueError(f"Failed to load object {obj_name}") from e

    # Take a sim step
    og.sim.step()


def _get_objects_config_from_scene_urdf(urdf):
    tree = ET.parse(urdf)
    root = tree.getroot()
    objects_cfg = dict()
    _get_objects_config_from_element(root, model_pose_info=objects_cfg)
    return objects_cfg


def _get_objects_config_from_element(element, model_pose_info):
    # First pass through, populate the joint pose info
    for ele in element:
        if ele.tag == "joint":
            name, pos, quat, fixed_jnt = _get_joint_info(ele)
            name = name.replace("-", "_")
            model_pose_info[name] = {
                "bbox_pos": pos,
                "bbox_quat": quat,
                "cfg": {
                    "fixed_base": fixed_jnt,
                },
            }

    # Second pass through, import object models
    for ele in element:
        if ele.tag == "link":
            # This is a valid object, import the model
            name = ele.get("name").replace("-", "_")
            if name == "world":
                # Skip this
                pass
            else:
                print(name)
                assert name in model_pose_info, f"Did not find {name} in current model pose info!"
                model_pose_info[name]["cfg"]["category"] = ele.get("category")
                model_pose_info[name]["cfg"]["model"] = ele.get("model")
                model_pose_info[name]["cfg"]["bounding_box"] = (
                    _space_string_to_tensor(ele.get("bounding_box")) if "bounding_box" in ele.keys() else None
                )
                in_rooms = ele.get("rooms", "")
                if in_rooms:
                    in_rooms = in_rooms.split(",")
                model_pose_info[name]["cfg"]["in_rooms"] = in_rooms
                model_pose_info[name]["cfg"]["scale"] = (
                    _space_string_to_tensor(ele.get("scale")) if "scale" in ele.keys() else None
                )
                model_pose_info[name]["cfg"]["bddl_object_scope"] = ele.get("object_scope", None)

        # If there's children nodes, we iterate over those
        for child in ele:
            _get_objects_config_from_element(child, model_pose_info=model_pose_info)


def _get_joint_info(joint_element):
    child, pos, quat, fixed_jnt = None, None, None, None
    fixed_jnt = joint_element.get("type") == "fixed"
    for ele in joint_element:
        if ele.tag == "origin":
            quat = T.euler2quat(_space_string_to_tensor(ele.get("rpy")))
            pos = _space_string_to_tensor(ele.get("xyz"))
        elif ele.tag == "child":
            child = ele.get("link")
    return child, pos, quat, fixed_jnt
