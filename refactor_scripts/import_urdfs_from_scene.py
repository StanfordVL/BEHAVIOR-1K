import igibson
from igibson import app, ig_dataset_path
from igibson.simulator_omni import Simulator
import omni
import omni.kit.commands
from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools, Usd
from pxr.Sdf import ValueTypeNames as VT
from omni.isaac.urdf import URDFCreateImportConfig, URDFParseAndImportFile
import xml.etree.ElementTree as ET
from copy import deepcopy
from os.path import exists
from omni.client._omniclient import Result
import omni.client
from omni.physx.scripts import utils

##### SET THIS ######
URDF = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best.urdf"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


# Create simulator
sim = Simulator()


def create_import_config():
    # Set up import configuration
    import_config = URDFCreateImportConfig().do()
    drive_mode = import_config.default_drive_type.__class__  # Hacky way to get class for default drive type, options are JOINT_DRIVE_{NONE / POSITION / VELOCITY}
    # import_config.merge_fixed_joints = False
    # import_config.convex_decomp = False
    # import_config.import_inertia_tensor = True
    # import_config.fix_base = False
    # import_config.default_drive_type = drive_mode.JOINT_DRIVE_NONE
    # import_config.self_collision = True
    # import_config.distance_scale = 1.0
    # # import_config.density = 0
    #
    # import_config.set_merge_fixed_joints(False)
    # import_config.set_convex_decomp(False)
    # import_config.set_fix_base(True)
    # import_config.set_import_inertia_tensor(False)
    # import_config.set_distance_scale(100.0)
    # import_config.set_density(0.0)
    # import_config.set_default_drive_type(1)
    # import_config.set_default_drive_strength(0.0)
    # import_config.set_default_position_drive_damping(0.0)
    # import_config.set_self_collision(False)
    # import_config.set_up_vector(0, 0, 1)
    # import_config.set_make_default_prim(True)
    # import_config.set_create_physics_scene(True)

    import_config.set_merge_fixed_joints(False)
    import_config.set_convex_decomp(True)
    import_config.set_fix_base(False)
    import_config.set_import_inertia_tensor(True)
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


def import_objects_from_scene_urdf(urdf):
    tree = ET.parse(urdf)
    root = tree.getroot()
    import_nested_objs_from_element(root)


def import_nested_objs_from_element(element):
    # Check if this element is a link
    for ele in element:
        if ele.tag == "link":
            # This is a valid object, import the model
            category = ele.get("category")
            model = ele.get("model")
            name = ele.get("name")
            print(f"Link: name: {name}, category: {category}, model: {model}")
            # Skip world link
            if name == "world":
                pass
            elif name == "straight_chair_10":
                import_obj_urdf(obj_category=category, obj_model=model, skip_if_exist=False)
            # # Import building components in different way from default objects
            # elif category in {"ceilings", "walls", "floors"}:
            #     import_building_urdf(obj_category=category, obj_model=model, skip_if_exist=False)
            # else:
            #     import_obj_urdf(obj_category=category, obj_model=model, skip_if_exist=False)
        # If there's children nodes, we iterate over those
        for child in ele:
            import_nested_objs_from_element(child)

def import_obj_urdf(obj_category, obj_model, skip_if_exist=True):
    # Import URDF
    cfg = create_import_config()
    # Check if filepath exists
    usd_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd"
    if not (skip_if_exist and exists(usd_path)):
        print(f"Importing {obj_category}, {obj_model}...")
        # Only import if it doesn't exist
        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=f"{ig_dataset_path}/objects/{obj_category}/{obj_model}/{obj_model}.urdf",
            import_config=cfg,
            dest_path=usd_path,
        )
        add_reference_to_stage(usd_path=usd_path, import_config=cfg)


def import_building_urdf(obj_category, obj_model, skip_if_exist=True):
    # For floors, ceilings, walls
    # Import URDF
    cfg = create_import_config()
    # Check if filepath exists
    usd_path = f"{ig_dataset_path}/scenes/{obj_model}/usd/{obj_category}/{obj_model}_{obj_category}.usd"
    if not (skip_if_exist and exists(usd_path)):
        print(f"Importing {obj_category}, {obj_model}...")
        # Only import if it doesn't exist
        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=f"{ig_dataset_path}/scenes/{obj_model}/urdf/{obj_model}_{obj_category}.urdf",
            import_config=cfg,
            dest_path=usd_path,
        )
        add_reference_to_stage(usd_path=usd_path, import_config=cfg)


def add_reference_to_stage(usd_path, import_config):
    stage = Usd.Stage.Open(usd_path)
    prim_name = str(stage.GetDefaultPrim().GetName())
    current_stage = omni.usd.get_context().get_stage()
    if current_stage:
        prim_path = omni.usd.get_stage_next_free_path(
            current_stage, str(current_stage.GetDefaultPrim().GetPath()) + "/" + prim_name, False
        )
        robot_prim = current_stage.OverridePrim(prim_path)
        if "anon:" in current_stage.GetRootLayer().identifier:
            robot_prim.GetReferences().AddReference(usd_path)
        else:
            robot_prim.GetReferences().AddReference(
                omni.client.make_relative_url(current_stage.GetRootLayer().identifier, usd_path)
            )
        if import_config.create_physics_scene:
            UsdPhysics.Scene.Define(current_stage, Sdf.Path("/physicsScene"))

import_objects_from_scene_urdf(urdf=URDF)

app.close()
