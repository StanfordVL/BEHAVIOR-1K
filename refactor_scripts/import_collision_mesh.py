import igibson
from igibson import ig_dataset_path
from igibson.simulator_omni import Simulator
from pxr import Usd, UsdGeom, Gf
import pxr.Vt
from pxr.Sdf import ValueTypeNames as VT
import numpy as np
import xml.etree.ElementTree as ET
import igibson.utils.transform_utils as T
import json
from os.path import exists
from pxr.UsdGeom import Tokens
from omni.physx.scripts.utils import setCollider

##### SET THIS ######
URDF = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best.urdf"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


sim = Simulator()

def import_models_collision_mesh_from_scene(urdf):
    tree = ET.parse(urdf)
    root = tree.getroot()
    import_nested_models_collision_mesh_from_element(root, model_pose_info={})


def import_nested_models_collision_mesh_from_element(element, model_pose_info):
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
                # skip
                pass
            else:
                import_obj_collision_mesh(obj_category=category, obj_model=model, name=name)

        # If there's children nodes, we iterate over those
        for child in ele:
            import_nested_models_collision_mesh_from_element(child, model_pose_info=model_pose_info)


def replace_all_nested_collision_meshes(root_prim):
    for prim in root_prim.GetChildren():
        if prim.GetPrimTypeInfo().GetTypeName() == "Xform" and prim.GetName() != "visuals":
            # Iterate again recursively
            replace_all_nested_collision_meshes(prim)
        elif prim.IsA(UsdGeom.Mesh) and prim.GetName() != "visuals":
            # Replace this mesh
            setCollider(prim, approximationShape="meshSimplification")
            # setCollider(prim, approximationShape="convexDecomposition")
            # prim.GetProperty("physxConvexDecompositionCollision:shrinkWrap").Set(True)


def import_obj_collision_mesh(obj_category, obj_model, name):
    # Check if filepath exists
    global sim
    model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
    usd_path = f"{model_root_path}/usd/{obj_model}.usd"

    # Load model
    sim.load_stage(usd_path=usd_path)

    base_prim = sim.stage.GetPrimAtPath("/")
    replace_all_nested_collision_meshes(base_prim)

    # Save stage
    sim.save_stage(usd_path=usd_path)

import_models_collision_mesh_from_scene(urdf=URDF)


igibson.app.close()
