from igibson import app, ig_dataset_path
import igibson.utils.transform_utils as T
import pxr.Vt
from pxr import Usd
from pxr import Gf, UsdShade, UsdLux
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
    "albedo": set_mtl_albedo,
    "normal": set_mtl_normal,
    "ao": set_mtl_ao,
    "roughness": set_mtl_roughness,
    "metalness": set_mtl_metalness,
    "opacity": set_mtl_opacity,
    "emission": set_mtl_emission,
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
            if name == "world":
                # Skip this
                pass
            # Process ceiling, walls, floor separately
            elif category in {"ceilings", "walls", "floors"}:
                import_building_metadata(obj_category=category, obj_model=model, name=name, import_render_channels=import_render_channels)
            else:
                print(name)
                bb = string_to_array(ele.get("bounding_box"))
                pos = model_pose_info[name]["pos"]
                quat = model_pose_info[name]["quat"]
                import_obj_metadata(obj_category=category, obj_model=model, name=name, import_render_channels=import_render_channels)

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


def import_rendering_channels(obj_prim, model_root_path, usd_path):
    usd_dir = "/".join(usd_path.split("/")[:-1])
    mat_dir = f"{model_root_path}/material"

    # Compile all material files we have
    mat_files = set(os.listdir(mat_dir))

    # Iterate over all children of the object prim, if /<obj_name>/<link_name>/visual exists, then we
    # know <link_name> is a valid link, and we check explicitly for these material files in our set
    # Note: we assume that the link name is included as a string within the mat_file!
    for prim in obj_prim.GetChildren():
        if prim.GetPrimTypeInfo().GetTypeName() == "Xform":
            # This could be a link, check if it owns a visual subprim
            link_name = prim.GetName()
            visual_prim = get_prim_at_path(f"{prim.GetPrimPath().pathString}/visuals")
            print(f"path: {prim.GetPrimPath().pathString}/visuals")
            print(f"visual prim: {visual_prim}")

            if visual_prim:
                # Aggregate all material files for this prim
                link_mat_files = []
                for mat_file in deepcopy(mat_files):
                    if link_name in mat_file:
                        # Add this mat file and remove it from the set
                        link_mat_files.append(mat_file)
                        mat_files.remove(mat_file)
                # Potentially write material files for this prim if we have any valid materials
                if len(link_mat_files) > 0:
                    # Create new material for this link
                    mtl_created_list = []

                    omni.kit.commands.execute(
                        "CreateAndBindMdlMaterialFromLibrary",
                        mdl_name="OmniPBR.mdl",
                        mtl_name="OmniPBR",
                        mtl_created_list=mtl_created_list,
                    )
                    mat = get_prim_at_path(mtl_created_list[0])

                    shade = UsdShade.Material(mat)
                    # Bind this material to the visual prim
                    UsdShade.MaterialBindingAPI(visual_prim).Bind(shade, UsdShade.Tokens.strongerThanDescendants)

                    # Iterate over all material channels and write them to the material
                    for link_mat_file in link_mat_files:
                        # Copy this file into the materials folder
                        mat_fpath = os.path.join(usd_dir, "materials")
                        shutil.copy(os.path.join(mat_dir, link_mat_file), mat_fpath)
                        # Check if any valid rendering channel
                        mat_type = link_mat_file.split("_")[-1].split(".")[0]
                        # Apply the material
                        RENDERING_CHANNEL_MAPPINGS[mat_type](mat, os.path.join(mat_fpath, link_mat_file))

                    # Rename material
                    mat = rename_prim(prim=mat, name=f"material_{link_name}")

    # For any remaining materials, we write them to the default material
    default_mat = get_prim_at_path(f"{obj_prim.GetPrimPath().pathString}/Looks/material_material_0")
    for mat_file in mat_files:
        # Copy this file into the materials folder
        mat_fpath = os.path.join(usd_dir, "materials")
        shutil.copy(os.path.join(mat_dir, mat_file), mat_fpath)
        # Check if any valid rendering channel
        mat_type = mat_file.split("_")[-1].split(".")[0]
        # Apply the material
        RENDERING_CHANNEL_MAPPINGS[mat_type](default_mat, os.path.join(mat_fpath, mat_file))


# TODO: Handle metalinks
# TODO: Import heights per link folder into USD folder
def import_obj_metadata(obj_category, obj_model, name, import_render_channels=False):
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

    # Pop bb and base link offset and meta links info
    base_link_offset = data["metadata"].pop("base_link_offset")
    default_bb = data["metadata"].pop("bbox_size")

    # Pop meta links
    meta_links = data["metadata"].pop("links")
    print("meta_links:", meta_links)
    for link_name,atrr in meta_links.items():
        # Create new Xform prim that will contain info
        link_prim = stage.DefinePrim(f"{prim.GetPath()}/{link_name}", "Xform")
        
        link_prim.CreateAttribute("ig:position", VT.Vector3f)
        # link_prim.CreateAttribute("ig:orientation", VT.Quatf)

        link_prim.GetAttribute("ig:position").Set(Gf.Vec3f(*atrr["xyz"]))
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
            for i, light_info in enumerate(light_infos):
                # Create the light in the scene
                light_type = LIGHT_MAPPING[light_info["type"]]
                light_prim_path = f"/{obj_model}/{link_name}/light{i}"
                light_prim = UsdLux.__dict__[f"{light_type}Light"].Define(stage, light_prim_path).GetPrim()
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
        import_rendering_channels(obj_prim=prim, model_root_path=model_root_path, usd_path=usd_path)

    # Save stage
    stage.Save()

    # Delete stage reference
    del stage


def recursively_replace_list_of_dict(dic):
    for k, v in dic.items():
        print(f"k: {k}")
        if v is None:
            # Replace None
            dic[k] = Tokens.none
        elif isinstance(v, list) or isinstance(v, tuple):
            if len(v) == 0:
                # Empty array
                dic[k] = pxr.Vt.FloatArray()
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
    if import_render_channels:
        import_rendering_channels(obj_prim=prim, model_root_path=model_root_path, usd_path=usd_path)

    # Save stage
    stage.Save()

    # Delete stage reference
    del stage

if __name__ == "__main__":
    import_models_metadata_from_scene(urdf=URDF, import_render_channels=False)
    app.close()

## For test_states.py
#import_obj_metadata("stove", "101908", "stove")
#import_obj_metadata("microwave", "7128", "microwave")