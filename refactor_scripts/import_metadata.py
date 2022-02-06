from igibson import app, ig_dataset_path
import igibson.utils.transform_utils as T
import pxr.Vt
from pxr import Usd
from pxr import Gf
from pxr.Sdf import ValueTypeNames as VT
import numpy as np
import xml.etree.ElementTree as ET
import json
from os.path import exists
from pxr.UsdGeom import Tokens

##### SET THIS ######
URDF = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best.urdf"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####

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


def import_models_metadata_from_scene(urdf):
    tree = ET.parse(urdf)
    root = tree.getroot()
    import_nested_models_metadata_from_element(root, model_pose_info={})


def import_nested_models_metadata_from_element(element, model_pose_info):
    # First pass through, populate the joint pose info
    for ele in element:
        if ele.tag == "joint":
            name, pos, quat = get_joint_info(ele)
            model_pose_info[name] = {
                "pos": pos,
                "quat": quat,
            }

    # Second pass through, import object models
    for ele in element:
        if ele.tag == "link":
            # This is a valid object, import the model
            name = ele.get("name")
            category = ele.get("category")
            model = ele.get("model")
            if name == "world":
                # Skip this
                pass
            # Process ceiling, walls, floor separately
            elif category in {"ceilings", "walls", "floors"}:
                import_building_metadata(obj_category=category, obj_model=model, name=name)
            else:
                print(name)
                bb = string_to_array(ele.get("bounding_box"))
                pos = model_pose_info[name]["pos"]
                quat = model_pose_info[name]["quat"]
                import_obj_metadata(obj_category=category, obj_model=model, name=name)

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


# TODO: Handle metalinks
# TODO: Import heights per link folder into USD folder
def import_obj_metadata(obj_category, obj_model, name):
    # Check if filepath exists
    model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
    usd_path = f"{model_root_path}/usd/{obj_model}.usd"

    # Load model
    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetDefaultPrim()

    data = dict()
    for data_group in {"metadata", "mvbb_meta", "material_groups", "heights_per_link"}:
        data_path = f"{model_root_path}/misc/{data_group}.json"
        if exists(data_path):
            # Load data
            with open(data_path, "r") as f:
                data[data_group] = json.load(f)

    # Pop bb and base link offset info
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
        data["metadata"]["openable_joint_ids"] = {str(pair[0]): pair[1] for pair in data["metadata"]["openable_joint_ids"]}

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

    # Save stage
    stage.Save()


def recursively_replace_list_of_dict(dic):
    for k, v in dic.items():
        print(f"k: {k}")
        if v is None:
            # Replace None
            dic[k] = Tokens.none
        elif isinstance(v, list) or isinstance(v, tuple):
            if isinstance(v[0], dict):
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


def import_building_metadata(obj_category, obj_model, name):
    global sim

    # Check if filepath exists
    model_root_path = f"{ig_dataset_path}/scenes/{obj_model}"
    usd_path = f"{model_root_path}/usd/{obj_category}/{obj_model}_{obj_category}.usd"

    # Load model
    stage = Usd.Stage.Open(usd_path)
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

    # Store remaining data as metadata
    prim.SetCustomData(data)

    # Save stage
    stage.Save()

import_models_metadata_from_scene(urdf=URDF)

app.close()
