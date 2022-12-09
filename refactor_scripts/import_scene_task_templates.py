import os
import json

from omnigibson import app, og_dataset_path
from omnigibson.simulator import Simulator
import pxr.Vt
from pxr import Usd
from pxr import Gf
from pxr.Sdf import ValueTypeNames as VT
import numpy as np
import xml.etree.ElementTree as ET
import omnigibson.utils.transform_utils as T
import json
from os.path import exists
from pxr.UsdGeom import Tokens
from omni.isaac.core.utils.stage import add_reference_to_stage, save_stage
from omni.isaac.core.articulations import Articulation
from omnigibson.utils.usd_utils import create_joint
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children
from omnigibson.utils.constants import JointType
from omnigibson.utils.config_utils import NumpyEncoder

##### SET THIS ######
SCENE_ID = "Rs_int"
# URDF = f"{og_dataset_path}/scenes/Rs_int/urdf/Rs_int_task_cleaning_kitchen_cupboard_0_0.urdf"
# USD_TEMPLATE_FILE = f"{og_dataset_path}/scenes/Rs_int/urdf/Rs_int_task_cleaning_kitchen_cupboard_0_0_template.usd"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####

sim = None

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


def import_models_template_from_task_scene(urdf, usd_out):
    global sim
    sim = Simulator()
    sim.clear()

    print(f"parsing urdf: {urdf}")
    tree = ET.parse(urdf)
    root = tree.getroot()
    import_nested_models_template_from_element(root, model_pose_info={})

    # Add template attribute to world
    world = get_prim_at_path("/World")
    world.CreateAttribute("ig:isTemplate", VT.Bool)
    world.GetAttribute("ig:isTemplate").Set(True)

    # Save scene
    sim.stop()
    save_stage(usd_out)


def import_nested_models_template_from_element(element, model_pose_info):
    # First pass through, populate the joint pose info
    for ele in element:
        if ele.tag == "joint":
            name, pos, quat, fixed_jnt = get_joint_info(ele)
            name = name.replace("-", "_")
            model_pose_info[name] = {
                "pos": pos,
                "quat": quat,
                "fixed_jnt": fixed_jnt,
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
            # # Process ceiling, walls, floor separately
            # elif category in {"ceilings", "walls", "floors"}:
            #     import_building_usd_fixed(obj_category=category, obj_model=model, name=name)
            else:
                print(name)
                bb = string_to_array(ele.get("bounding_box")) if "bounding_box" in ele.keys() else None
                pos = model_pose_info[name]["pos"]
                quat = model_pose_info[name]["quat"]
                fixed_jnt = model_pose_info[name]["fixed_jnt"]
                room = ele.get("rooms", "")
                random_group = ele.get("random_group", None)
                scale = string_to_array(ele.get("scale")) if "scale" in ele.keys() else None
                obj_scope = ele.get("object_scope", None)

                if category == "agent_pose":
                    # Write metadata for this
                    pose_info = [
                        {
                            "position": np.array(pos),
                            "orientation": np.array(quat)[[1, 2, 3, 0]],
                        },
                    ]
                    world = get_prim_at_path("/World")
                    world.SetCustomDataByKey("agent_poses", json.dumps(pose_info, cls=NumpyEncoder))
                else:
                    # Import as new XForm
                    import_obj_template(obj_category=category, obj_model=model, name=name, bb=bb, pos=pos, quat=quat, fixed_jnt=fixed_jnt, room=room, random_group=random_group, scale=scale, obj_scope=obj_scope)

        # If there's children nodes, we iterate over those
        for child in ele:
            import_nested_models_template_from_element(child, model_pose_info=model_pose_info)


def get_joint_info(joint_element):
    child, pos, quat, fixed_jnt = None, None, None, None
    fixed_jnt = joint_element.get("type") == "fixed"
    for ele in joint_element:
        if ele.tag == "origin":
            quat = T.convert_quat(T.mat2quat(T.euler2mat(string_to_array(ele.get("rpy")))), "wxyz")
            pos = string_to_array(ele.get("xyz"))
        elif ele.tag == "child":
            child = ele.get("link")
    return child, pos, quat, fixed_jnt


def import_building_usd_fixed(obj_category, obj_model, name):
    global sim

    # Check if filepath exists
    usd_path = f"{og_dataset_path}/scenes/{obj_model}/usd/{obj_category}/{obj_model}_{obj_category}.usd"

    print(f"usd path: {usd_path}")

    # Import model
    add_reference_to_stage(
        usd_path=usd_path,
        prim_path=f"/World/{name}",
    )

    # obj = sim.scene.add(Articulation(
    #     prim_path=f"/World/{name}",
    #     name=f"{name}",
    # ))

    # Create fixed joint
    create_joint(
        prim_path=f"/World/{name}/rootJoint",
        joint_type=JointType.JOINT_FIXED,
        body0=None,
        body1=f"/World/{name}/base_link",
        stage=sim.stage,
    )

# TODO: Handle metalinks
# TODO: Import heights per link folder into USD folder
def import_obj_template(obj_category, obj_model, name, bb, pos, quat, fixed_jnt, room, random_group, scale, obj_scope):
    global sim

    print(f"obj: {name}, fixed jnt: {fixed_jnt}")

    # Create new Xform prim that will contain info
    obj = sim.stage.DefinePrim(f"/World/{name.replace('-', '_')}", "Xform")
    obj.CreateAttribute("ig:category", VT.String)
    obj.CreateAttribute("ig:fixedJoint", VT.Bool)
    obj.CreateAttribute("ig:position", VT.Vector3f)
    obj.CreateAttribute("ig:orientation", VT.Quatf)
    obj.CreateAttribute("ig:rooms", VT.String)

    # Set these values
    obj.GetAttribute("ig:category").Set(obj_category)
    obj.GetAttribute("ig:fixedJoint").Set(fixed_jnt)
    obj.GetAttribute("ig:position").Set(Gf.Vec3f(*pos))
    obj.GetAttribute("ig:orientation").Set(Gf.Quatf(*quat.tolist()))
    obj.GetAttribute("ig:rooms").Set(room)

    # Potentially create additonal attributes
    if obj_model is not None:
        obj.CreateAttribute("ig:model", VT.String)
        obj.GetAttribute("ig:model").Set(obj_model)

    if bb is not None:
        obj.CreateAttribute("ig:boundingBox", VT.Vector3f)
        obj.GetAttribute("ig:boundingBox").Set(Gf.Vec3f(*bb))

    if scale is not None:
        obj.CreateAttribute("ig:scale", VT.Vector3f)
        obj.GetAttribute("ig:scale").Set(Gf.Vec3f(*scale))

    if random_group is not None:
        obj.CreateAttribute("ig:randomGroup", VT.String)
        obj.GetAttribute("ig:randomGroup").Set(random_group)

    if obj_scope is not None:
        obj.CreateAttribute("ig:objectScope", VT.String)
        obj.GetAttribute("ig:objectScope").Set(obj_scope)


def import_models_template_from_task_scenes(scene_id):
    urdf_dir = f"{og_dataset_path}/scenes/{scene_id}/urdf"
    usd_dir = f"{og_dataset_path}/scenes/{scene_id}/usd"
    for scene_urdf in os.listdir(urdf_dir):
        # Only parse the scenes with task
        if f"{scene_id}_task_" in scene_urdf:
            usd_template_filename = f"{scene_urdf.split('.urdf')[0]}_template.usd"
            urdf = f"{urdf_dir}/{scene_urdf}"
            usd = f"{usd_dir}/{usd_template_filename}"
            import_models_template_from_task_scene(urdf=urdf, usd_out=usd)
            # break


if __name__ == "__main__":
    import_models_template_from_task_scenes(scene_id=SCENE_ID)

    # Close app
    app.close()


