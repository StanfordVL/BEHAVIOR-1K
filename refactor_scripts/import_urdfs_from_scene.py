import igibson
from igibson import app, ig_dataset_path
from igibson.simulator import Simulator
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

from refactor_scripts.expand_collision_obj_and_urdf import split_objs_in_urdf
from refactor_scripts.preprocess_urdf_for_metalinks import update_obj_urdf_with_metalinks

##### SET THIS ######
URDF = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best.urdf"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


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


def import_objects_from_scene_urdf(urdf):
    tree = ET.parse(urdf)
    root = tree.getroot()
    import_nested_objs_from_element(root)


def import_nested_objs_from_element(element):
    obj_infos = set()
    # Check if this element is a link
    for ele in element:
        if ele.tag == "link":
            # This is a valid object, import the model
            category = ele.get("category")
            model = ele.get("model")
            name = ele.get("name").replace("-", "_")
            print(f"Link: name: {name}, category: {category}, model: {model}")
            obj_info = (category, model)
            # Skip world link
            if name == "world":
                pass
            # # Import IG2 building components in different way from default objects
            # elif from_ig2 and category in {"ceilings", "walls", "floors"} and obj_info not in obj_infos:
            #     import_ig2_building_urdf(scene_id=model, category=category, skip_if_exist=False)
            #     obj_infos.add(obj_info)
            elif obj_info not in obj_infos:
                import_obj_urdf(obj_category=category, obj_model=model, skip_if_exist=False)
                obj_infos.add(obj_info)
        # If there's children nodes, we iterate over those
        for child in ele:
            import_nested_objs_from_element(child)


def import_obj_urdf(obj_category, obj_model, skip_if_exist=False):
    # Preprocess input URDF to account for metalinks
    update_obj_urdf_with_metalinks(obj_category=obj_category, obj_model=obj_model)
    # Import URDF
    cfg = create_import_config()
    # Check if filepath exists
    usd_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd"
    if not (skip_if_exist and exists(usd_path)):
        urdf_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}/{obj_model}.urdf"
        print(f"Converting collision meshes from {obj_category}, {obj_model}...")
        urdf_path = split_objs_in_urdf(urdf_fpath=urdf_path, name_suffix="split")
        print(f"Importing {obj_category}, {obj_model}...")
        # Only import if it doesn't exist
        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=cfg,
            dest_path=usd_path,
        )
        add_reference_to_stage(usd_path=usd_path, import_config=cfg)


def import_ig2_building_urdf(scene_id, category, skip_if_exist=False):
    # For floors, ceilings, walls from ig2
    # Import URDF
    cfg = create_import_config()
    # Check if filepath exists
    obj_category = f"{category}"
    obj_model = f"{scene_id}"
    usd_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd"
    if not (skip_if_exist and exists(usd_path)):
        urdf_path = f"{ig_dataset_path}/scenes/{scene_id}/urdf/{obj_model}.urdf"
        print(f"Converting IG2 building collision meshes from {scene_id}, {category}...")
        urdf_path = split_objs_in_urdf(urdf_fpath=urdf_path, name_suffix="split", mesh_fpath_offset="..")
        print(f"Importing IG2 building {scene_id}, {category}...")
        # Only import if it doesn't exist
        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
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

if __name__ == "__main__":
    import_objects_from_scene_urdf(urdf=URDF)
    app.close()
