from igibson import app, ig_dataset_path
import igibson.utils.transform_utils as T
import pxr.Vt
from pxr import Usd
from pxr import Gf, UsdShade, UsdLux, UsdPhysics, PhysxSchema, UsdGeom
from pxr.Sdf import ValueTypeNames as VT
import numpy as np
import xml.etree.ElementTree as ET
import json
from os.path import exists
from pxr.UsdGeom import Tokens
from omni.usd import create_material_input, get_shader_from_material
from omni.isaac.core.utils.prims import get_prim_at_path
import os
import shutil
import omni
from copy import deepcopy
from omni.isaac.core.utils.stage import open_stage, get_current_stage, close_stage
from omni.isaac.core.prims.xform_prim import XFormPrim
import igibson.utils.transform_utils as T
import trimesh
from igibson.utils.usd_utils import BoundingBoxAPI
from igibson.utils.asset_utils import encrypt_file
from igibson.macros import gm
from collections import OrderedDict
from pathlib import Path


##### SET THIS ######
URDF = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best.urdf"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


LIGHT_MAPPING = {
    0: "Rect",
    2: "Sphere",
    4: "Disk",
}


def set_mtl_albedo(mtl_prim, texture):
    mtl = "diffuse_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")

def set_mtl_normal(mtl_prim, texture):
    mtl = "normalmap_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")

def set_mtl_ao(mtl_prim, texture):
    mtl = "ao_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")

def set_mtl_roughness(mtl_prim, texture):
    mtl = "reflectionroughness_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    create_material_input(mtl_prim, "reflection_roughness_texture_influence", 1.0, VT.Float)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")

def set_mtl_metalness(mtl_prim, texture):
    mtl = "metallic_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    create_material_input(mtl_prim, "metallic_texture_influence", 1.0, VT.Float)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")

def set_mtl_opacity(mtl_prim, texture):
    mtl = "opacity_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    create_material_input(mtl_prim, "enable_opacity", True, VT.Bool)
    create_material_input(mtl_prim, "enable_opacity_texture", True, VT.Bool)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")

def set_mtl_emission(mtl_prim, texture):
    mtl = "emissive_color_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    create_material_input(mtl_prim, "enable_emission", True, VT.Bool)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")



RENDERING_CHANNEL_MAPPINGS = {
    "diffuse": set_mtl_albedo,
    "albedo": set_mtl_albedo,
    "normal": set_mtl_normal,
    "ao": set_mtl_ao,
    "roughness": set_mtl_roughness,
    "metalness": set_mtl_metalness,
    "opacity": set_mtl_opacity,
    "emission": set_mtl_emission,
}

MTL_MAP_TYPE_MAPPINGS = {
    "map_kd": "albedo",
    "map_bump": "normal",
    "map_pr": "roughness",
    "map_pm": "metalness",
    "map_tf": "opacity",
    "map_ke": "emission",
    "map_ks": "ao",
}


def string_to_array(string):
    """
    Converts a array string in mujoco xml to np.array.
    Examples:
        "0 1 2" => [0, 1, 2]
    Args:
        string (str): String to convert to an array
    Returns:
        np.array: Numerical array equivalent of @string
    """
    return np.array([float(x) for x in string.split(" ")])


def import_models_metadata_from_scene(urdf, import_render_channels=False):
    tree = ET.parse(urdf)
    root = tree.getroot()
    import_nested_models_metadata_from_element(root, model_pose_info={}, import_render_channels=import_render_channels)


def import_nested_models_metadata_from_element(element, model_pose_info, import_render_channels=False):
    obj_infos = set()
    # First pass through, populate the joint pose info
    for ele in element:
        if ele.tag == "joint":
            name, pos, quat = get_joint_info(ele)
            name = name.replace("-", "_")
            model_pose_info[name] = {
                "pos": pos,
                "quat": quat,
            }

    # Second pass through, import object models
    for ele in element:
        if ele.tag == "link":
            # This is a valid object, import the model
            name = ele.get("name").replace("-", "_")
            category = ele.get("category")
            model = ele.get("model")
            obj_info = (category, model)
            if name == "world":
                # Skip this
                pass
            # # Process ceiling, walls, floor separately
            # elif category in {"ceilings", "walls", "floors"} and obj_info not in obj_infos:
            #     import_building_metadata(obj_category=category, obj_model=model, name=name, import_render_channels=import_render_channels)
            #     obj_infos.add(obj_info)
            elif obj_info not in obj_infos:
                print(name)
                # bb = string_to_array(ele.get("bounding_box"))
                # pos = model_pose_info[name]["pos"]
                # quat = model_pose_info[name]["quat"]
                import_obj_metadata(obj_category=category, obj_model=model, import_render_channels=import_render_channels)
                obj_infos.add(obj_info)

        # If there's children nodes, we iterate over those
        for child in ele:
            import_nested_models_metadata_from_element(child, model_pose_info=model_pose_info)


def get_joint_info(joint_element):
    child, pos, quat = None, None, None
    for ele in joint_element:
        if ele.tag == "origin":
            quat = T.convert_quat(T.mat2quat(T.euler2mat(string_to_array(ele.get("rpy")))), "wxyz")
            pos = string_to_array(ele.get("xyz"))
        elif ele.tag == "child":
            child = ele.get("link")
    return child, pos, quat


def rename_prim(prim, name):
    path_from = prim.GetPrimPath().pathString
    path_to = f"{'/'.join(path_from.split('/')[:-1])}/{name}"
    omni.kit.commands.execute("MovePrim", path_from=path_from, path_to=path_to)
    return get_prim_at_path(path_to)


def get_visual_objs_from_urdf(urdf_path):
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
                    visual_mesh_name = sub_ele.get('name', "visuals").replace("-", "_")
                    obj_file = None if sub_ele.find(".//mesh") is None else sub_ele.find(".//mesh").get("filename")
                    if obj_file is None:
                        print(f"Warning: No obj file found associated with {name}/{visual_mesh_name}!")
                    visual_objs[name][visual_mesh_name] = obj_file

    return visual_objs


def import_rendering_channels(obj_prim, obj_category, obj_model, model_root_path, usd_path):
    usd_dir = "/".join(usd_path.split("/")[:-1])
    # # mat_dir = f"{model_root_path}/material/{obj_category}" if \
    # #     obj_category in {"ceilings", "walls", "floors"} else f"{model_root_path}/material"
    # mat_dir = f"{model_root_path}/material"
    # # Compile all material files we have
    # mat_files = set(os.listdir(mat_dir))

    # Remove the material prims as we will create them explictly later.
    # TODO: Be a bit more smart about this. a material procedurally generated will lose its material without it having
    # be regenerated
    stage = omni.usd.get_context().get_stage()
    for prim in obj_prim.GetChildren():
        looks_prim = None
        if prim.GetName() == "Looks":
            looks_prim = prim
        elif prim.GetPrimTypeInfo().GetTypeName() == "Xform":
            looks_prim_path = f"{str(prim.GetPrimPath())}/Looks"
            looks_prim = get_prim_at_path(looks_prim_path)
        if not looks_prim:
            continue
        for subprim in looks_prim.GetChildren():
            if subprim.GetPrimTypeInfo().GetTypeName() != "Material":
                continue
            print(f"Removed material prim {subprim.GetPath()}:", stage.RemovePrim(subprim.GetPath()))

    # # Create new default material for this object.
    # mtl_created_list = []
    # omni.kit.commands.execute(
    #     "CreateAndBindMdlMaterialFromLibrary",
    #     mdl_name="OmniPBR.mdl",
    #     mtl_name="OmniPBR",
    #     mtl_created_list=mtl_created_list,
    # )
    # default_mat = get_prim_at_path(mtl_created_list[0])
    # default_mat = rename_prim(prim=default_mat, name=f"default_material")
    # print("Created default material:", default_mat.GetPath())
    #
    # # We may delete this default material if it's never used
    # default_mat_is_used = False

    # Grab all visual objs for this object
    urdf_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}/{obj_model}_split.urdf"
    visual_objs = get_visual_objs_from_urdf(urdf_path)

    # Extract absolute paths to mtl files for each link
    link_mtl_files = OrderedDict()          # maps link name to dictionary mapping mesh name to mtl file
    mtl_infos = OrderedDict()               # maps mtl name to dictionary mapping material channel name to png file
    mat_files = OrderedDict()               # maps mtl name to corresponding list of material filenames
    mtl_old_dirs = OrderedDict()            # maps mtl name to corresponding directory where the mtl file exists
    mat_old_paths = OrderedDict()           # maps mtl name to corresponding list of relative mat paths from mtl directory
    for link_name, link_meshes in visual_objs.items():
        link_mtl_files[link_name] = OrderedDict()
        for mesh_name, obj_file in link_meshes.items():
            # Get absolute path and open the obj file if it exists:
            if obj_file is not None:
                obj_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}/{obj_file}"
                with open(obj_path, "r") as f:
                    mtls = []
                    for line in f.readlines():
                        if "mtllib" in line and line[0] != "#":
                            mtls.append(line.split("mtllib ")[-1].split("\n")[0])
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
                            map_filename = map_file.split("/")[-1]
                            mat_files[mtl_name].append(map_filename)
                            mat_old_paths[mtl_name].append(map_file)
                            mtl_infos[mtl_name][MTL_MAP_TYPE_MAPPINGS[map_type.lower()]] = map_filename

    # Next, for each material information, we create a new material and port the material files to the USD directory
    mat_new_fpath = os.path.join(usd_dir, "materials")
    Path(mat_new_fpath).mkdir(parents=True, exist_ok=True)
    shaders = OrderedDict()                 # maps mtl name to shader prim
    for mtl_name, mtl_info in mtl_infos.items():
        for mat_old_path in mat_old_paths[mtl_name]:
            shutil.copy(os.path.join(mtl_old_dirs[mtl_name], mat_old_path), mat_new_fpath)

        # Create the new material
        mtl_created_list = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=mtl_created_list,
        )
        mat = get_prim_at_path(mtl_created_list[0])

        # Apply all rendering channels for this material
        for mat_type, mat_file in mtl_info.items():
            render_channel_fcn = RENDERING_CHANNEL_MAPPINGS.get(mat_type, None)
            if render_channel_fcn is not None:
                render_channel_fcn(mat, os.path.join("materials", mat_file))
            else:
                # Warn user that we didn't find the correct rendering channel
                print(f"Warning: could not find rendering channel function for material: {mat_type}, skipping")

        # Rename material
        mat = rename_prim(prim=mat, name=mtl_name)
        shade = UsdShade.Material(mat)
        shaders[mtl_name] = shade
        print(f"Created material {mtl_name}:", mtl_created_list[0])

    # Bind each (visual) mesh to its appropriate material in the object
    # We'll loop over each link, create a list of 2-tuples each consisting of (mesh_prim_path, mtl_name) to be bound
    root_prim_path = obj_prim.GetPrimPath().pathString
    for link_name, mesh_mtl_names in link_mtl_files.items():
        # Special case -- omni always calls the visuals "visuals" by default if there's only a single visual mesh for the
        # given
        if len(mesh_mtl_names) == 1:
            mesh_mtl_infos = [(f"{root_prim_path}/{link_name}/visuals", list(mesh_mtl_names.values())[0])]
        else:
            mesh_mtl_infos = []
            for mesh_name, mtl_name in mesh_mtl_names.items():
                # Omni only accepts a-z, A-Z as valid start characters for prim names
                # So we check if there is an invalid character, and modify it as we know Omni does
                if not ord("a") <= ord(mesh_name[0]) <= ord("z") and not ord("A") <= ord(mesh_name[0]) <= ord("Z"):
                    mesh_name = "a_" + mesh_name[1:]
                mesh_mtl_infos.append((f"{root_prim_path}/{link_name}/visuals/{mesh_name}", mtl_name))
        for mesh_prim_path, mtl_name in mesh_mtl_infos:
            visual_prim = get_prim_at_path(mesh_prim_path)
            assert visual_prim, f"Error: Did not find valid visual prim at {mesh_prim_path}!"
            # Bind the created link material to the visual prim
            print(f"Binding material {mtl_name}, shader {shaders[mtl_name]}, to prim {mesh_prim_path}...")
            UsdShade.MaterialBindingAPI(visual_prim).Bind(shaders[mtl_name], UsdShade.Tokens.strongerThanDescendants)

    # ###################################
    #
    # # Iterate over all children of the object prim, if /<obj_name>/<link_name>/visual exists, then we
    # # know <link_name> is a valid link, and we check explicitly for these material files in our set
    # # Note: we assume that the link name is included as a string within the mat_file!
    # for prim in obj_prim.GetChildren():
    #     if prim.GetPrimTypeInfo().GetTypeName() == "Xform":
    #         # This could be a link, check if it owns a visual subprim
    #         link_name = prim.GetName()
    #         visual_prim = get_prim_at_path(f"{prim.GetPrimPath().pathString}/visuals")
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
    #                 shade = UsdShade.Material(default_mat)
    #                 UsdShade.MaterialBindingAPI(visual_prim).Bind(shade, UsdShade.Tokens.strongerThanDescendants)
    #                 default_mat_is_used = True
    #             else:
    #                 # Create new material for this link
    #                 mtl_created_list = []
    #                 omni.kit.commands.execute(
    #                     "CreateAndBindMdlMaterialFromLibrary",
    #                     mdl_name="OmniPBR.mdl",
    #                     mtl_name="OmniPBR",
    #                     mtl_created_list=mtl_created_list,
    #                 )
    #                 print(f"Created material for link {link_name}:", mtl_created_list[0])
    #                 mat = get_prim_at_path(mtl_created_list[0])
    #
    #                 shade = UsdShade.Material(mat)
    #                 # Bind the created link material to the visual prim
    #                 UsdShade.MaterialBindingAPI(visual_prim).Bind(shade, UsdShade.Tokens.strongerThanDescendants)
    #
    #                 # Iterate over all material channels and write them to the material
    #                 for link_mat_file in link_mat_files:
    #                     # Copy this file into the materials folder
    #                     mat_fpath = os.path.join(usd_dir, "materials")
    #                     shutil.copy(os.path.join(mat_dir, link_mat_file), mat_fpath)
    #                     # Check if any valid rendering channel
    #                     mat_type = link_mat_file.split("_")[-1].split(".")[0].lower()
    #                     # Apply the material if it exists
    #                     render_channel_fcn = RENDERING_CHANNEL_MAPPINGS.get(mat_type, None)
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
    # # default_mat = get_prim_at_path(f"{obj_prim.GetPrimPath().pathString}/Looks/material_material_0")
    # # default_mat = get_prim_at_path(f"{obj_prim.GetPrimPath().pathString}/Looks/material_default")
    # print(f"default mat: {default_mat}, obj: {obj_category}, {prim.GetPrimPath().pathString}")
    # for mat_file in mat_files:
    #     # Copy this file into the materials folder
    #     mat_fpath = os.path.join(usd_dir, "materials")
    #     shutil.copy(os.path.join(mat_dir, mat_file), mat_fpath)
    #     # Check if any valid rendering channel
    #     mat_type = mat_file.split("_")[-1].split(".")[0].lower()
    #     # Apply the material if it exists
    #     render_channel_fcn = RENDERING_CHANNEL_MAPPINGS.get(mat_type, None)
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

def add_xform_properties(prim):
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
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    # TODO: wont be able to delete props for non root links on articulated objects
    for prop_name in prop_names:
        if prop_name in properties_to_remove:
            prim.RemoveProperty(prop_name)
    if "xformOp:scale" not in prop_names:
        xform_op_scale = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
        xform_op_scale.Set(Gf.Vec3d([1.0, 1.0, 1.0]))
    else:
        xform_op_scale = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))

    if "xformOp:translate" not in prop_names:
        xform_op_translate = xformable.AddXformOp(
            UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
        )
    else:
        xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))

    if "xformOp:orient" not in prop_names:
        xform_op_rot = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
    else:
        xform_op_rot = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
    xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])


# TODO: Handle metalinks
# TODO: Import heights per link folder into USD folder
def import_obj_metadata(obj_category, obj_model, import_render_channels=False):
    # Check if filepath exists
    model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
    usd_path = f"{model_root_path}/usd/{obj_model}.usd"

    # Load model
    open_stage(usd_path)
    stage = get_current_stage()
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
        low_bb, high_bb = BoundingBoxAPI.compute_aabb(prim.GetPrimPath().pathString)
        data["metadata"]["bbox_size"] = (high_bb - low_bb).tolist()

    # Pop bb and base link offset and meta links info
    base_link_offset = data["metadata"].pop("base_link_offset")
    default_bb = data["metadata"].pop("bbox_size")

    # Pop meta inks
    if "links" in data["metadata"]:
        meta_links = data["metadata"].pop("links")
        print("meta_links:", meta_links)
        for link_name,atrr in meta_links.items():
            # # Create new Xform prim that will contain info
            link_prim = get_prim_at_path(f"{prim.GetPath()}/{link_name}_link")

            link_prim.CreateAttribute("ig:is_metalink", VT.Bool)
            link_prim.GetAttribute("ig:is_metalink").Set(True)

            # # TODO! Validate that this works
            # # test on water sink 02: water sink location is 0.1, 0.048, 0.32
            # # water source location is -0.03724, 0.008, 0.43223
            # add_xform_properties(prim=link_prim)
            # link_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*atrr["xyz"]))
            # if atrr["rpy"] is not None:
            #     link_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(*(T.euler2quat(atrr["rpy"])[[3, 0, 1, 2]])))

            # link_prim.CreateAttribute("ig:orientation", VT.Quatf)
            # link_prim.GetAttribute("ig:orientation").Set(Gf.Quatf(*atrr["rpy"]))

    # Manually modify material groups info
    if "material_groups" in data:
        data["material_groups"] = {
            "groups": data["material_groups"][0],
            "links": data["material_groups"][1],
        }

    # Manually modify metadata
    if "openable_joint_ids" in data["metadata"]:
        data["metadata"]["openable_joint_ids"] = {str(pair[0]): pair[1] for pair in data["metadata"]["openable_joint_ids"]}

    # Grab light info if any
    lights = data["metadata"].get("meta_links", dict()).get("lights", None)
    if lights is not None:
        for link_name, light_infos in lights.items():
            for light_name, light_info in light_infos.items():
                # Create the light in the scene
                light_type = LIGHT_MAPPING[light_info["type"]]
                light_prim_path = f"/{obj_model}/{link_name}/light{light_name}"
                light_prim = UsdLux.__dict__[f"{light_type}Light"].Define(stage, light_prim_path).GetPrim()
                UsdLux.ShapingAPI.Apply(light_prim).GetShapingConeAngleAttr().Set(180.0)
                add_xform_properties(prim=light_prim)
                # Make sure light_prim has XForm properties
                light = XFormPrim(prim_path=light_prim_path)
                # Set the values accordingly
                light.set_local_pose(
                    translation=np.array(light_info["position"]),
                    orientation=T.convert_quat(np.array(light_info["orientation"]), to="wxyz")
                )
                light.prim.GetAttribute("color").Set(Gf.Vec3f(*np.array(light_info["color"]) / 255.0))
                light.prim.GetAttribute("intensity").Set(light_info["intensity"])
                if light_type == "Rect":
                    light.prim.GetAttribute("height").Set(light_info["length"])
                    light.prim.GetAttribute("width").Set(light_info["width"])
                elif light_type == "Disk":
                    light.prim.GetAttribute("radius").Set(light_info["length"])
                elif light_type == "Sphere":
                    light.prim.GetAttribute("radius").Set(light_info["length"])
                else:
                    raise ValueError(f"Invalid light type: {light_type}")

    # Iterate over dict and replace any lists of dicts as dicts of dicts (with each dict being indexed by an integer)
    data = recursively_replace_list_of_dict(data)

    # Create attributes for bb, offset, category, model and store values
    prim.CreateAttribute("ig:nativeBB", VT.Vector3f)
    prim.CreateAttribute("ig:offsetBaseLink", VT.Vector3f)
    prim.CreateAttribute("ig:category", VT.String)
    prim.CreateAttribute("ig:model", VT.String)
    prim.GetAttribute("ig:nativeBB").Set(Gf.Vec3f(*default_bb))
    prim.GetAttribute("ig:offsetBaseLink").Set(Gf.Vec3f(*base_link_offset))
    prim.GetAttribute("ig:category").Set(obj_category)
    prim.GetAttribute("ig:model").Set(obj_model)


    print(f"data: {data}")

    # Store remaining data as metadata
    prim.SetCustomData(data)
    # for k, v in data.items():
    #     print(f"setting custom data {k}")
    #     print(v)
    #     print()
    #     prim.SetCustomDataByKey(k, v)
    #     # if k == "metadata":
    #     #     print(v)
    #     #     print()
    #     #     prim.SetCustomDataByKey(k, v["link_bounding_boxes"]["base_link"]["collision"]["axis_aligned"]["transform"])
    #     #     input("succeeded!")
    #     # else:
    #     #     prim.SetCustomDataByKey(k, v)

    # Add material channels
    # print(f"prim children: {prim.GetChildren()}")
    # looks_prim_path = f"{str(prim.GetPrimPath())}/Looks"
    # looks_prim = prim.GetChildren()[0] #get_prim_at_path(looks_prim_path)
    # mat_prim_path = f"{str(prim.GetPrimPath())}/Looks/material_material_0"
    # mat_prim = looks_prim.GetChildren()[0] #get_prim_at_path(mat_prim_path)
    # print(f"looks children: {looks_prim.GetChildren()}")
    # print(f"mat prim: {mat_prim}")
    if import_render_channels:
        import_rendering_channels(obj_prim=prim, obj_category=obj_category, obj_model=obj_model, model_root_path=model_root_path, usd_path=usd_path)

    # Save stage
    stage.Save()

    # Delete stage reference
    del stage

    if gm.USE_ENCRYPTED_ASSETS:
        encrypted_usd_path = usd_path.replace(".usd", ".encrypted.usd")
        encrypt_file(usd_path, encrypted_filename=encrypted_usd_path)
        os.remove(usd_path)


def recursively_replace_list_of_dict(dic):
    for k, v in dic.items():
        print(f"k: {k}")
        if v is None:
            # Replace None
            dic[k] = Tokens.none
        elif isinstance(v, list) or isinstance(v, tuple):
            if len(v) == 0:
                dic[k] = pxr.Vt.Vec3fArray()
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
                    dic[k] = pxr.Vt.Vec2fArray(v)
                elif len(v[0]) == 3:
                    dic[k] = pxr.Vt.Vec3fArray(v)
                elif len(v[0]) == 4:
                    dic[k] = pxr.Vt.Vec4fArray(v)
                else:
                    raise ValueError(f"No support for storing matrices of length {len(v[0])}!")
            elif isinstance(v[0], int):
                # if len(v) == 1:
                #     # Do nothing
                #     pass
                # elif len(v) == 2:
                #     dic[k] = Gf.Vec2i(v)
                # elif len(v) == 3:
                #     dic[k] = Gf.Vec3i(v)
                # elif len(v) == 4:
                #     dic[k] = Gf.Vec4i(v)
                # else:
                dic[k] = pxr.Vt.IntArray(v)
                # raise ValueError(f"No support for storing numeric arrays of length {len(v)}! Array: {v}")
            elif isinstance(v[0], float):
                # if len(v) == 1:
                #     # Do nothing
                #     pass
                # elif len(v) == 2:
                #     dic[k] = Gf.Vec2f(v)
                # elif len(v) == 3:
                #     dic[k] = Gf.Vec3f(v)
                # elif len(v) == 4:
                #     dic[k] = Gf.Vec4f(v)
                # else:
                dic[k] = pxr.Vt.FloatArray(v)
                # raise ValueError(f"No support for storing numeric arrays of length {len(v)}! Array: {v}")
            else:
                # Replace any Nones
                for i, ele in enumerate(v):
                    if ele is None:
                        v[i] = Tokens.none
        if isinstance(v, dict):
            # Iterate through nested dictionaries
            dic[k] = recursively_replace_list_of_dict(v)

    return dic


def import_building_metadata(obj_category, obj_model, name, import_render_channels=False):
    # Check if filepath exists
    model_root_path = f"{ig_dataset_path}/scenes/{obj_model}"
    usd_path = f"{model_root_path}/usd/{obj_category}/{obj_model}_{obj_category}.usd"

    # Load model
    open_stage(usd_path)
    stage = get_current_stage()
    prim = stage.GetDefaultPrim()

    data = dict()
    for data_group in {"material_groups"}:
        data_path = f"{model_root_path}/misc/{obj_category}_{data_group}.json"
        if exists(data_path):
            # Load data
            with open(data_path, "r") as f:
                data[data_group] = json.load(f)

    # Manually modify material groups info
    if "material_groups" in data:
        data["material_groups"] = {
            "groups": data["material_groups"][0],
            "links": data["material_groups"][1],
        }

    # Iterate over dict and replace any lists of dicts as dicts of dicts (with each dict being indexed by an integer)
    data = recursively_replace_list_of_dict(data)

    # Store attributes
    prim.CreateAttribute("ig:category", VT.String)
    prim.CreateAttribute("ig:model", VT.String)
    prim.GetAttribute("ig:category").Set(obj_category)
    prim.GetAttribute("ig:model").Set(obj_model)

    # Store remaining data as metadata
    prim.SetCustomData(data)

    # Add material channels
    # mat_prim_path = f"{str(prim.GetPrimPath())}/Looks/material_material_0"
    # mat_prim = get_prim_at_path(mat_prim_path)
    # print(f"mat prim: {mat_prim}")
    # TODO : Pass for now -- need to update handling of paths better, since a single link can have multiple visual meshes + materials
    # if import_render_channels:
    #     import_rendering_channels(obj_prim=prim, obj_category=obj_category, model_root_path=model_root_path, usd_path=usd_path)

    # Save stage
    stage.Save()

    # Delete stage reference
    del stage

if __name__ == "__main__":
    import_models_metadata_from_scene(urdf=URDF, import_render_channels=False)
    app.close()
