import io
import json
import math
import os
import pathlib
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from os.path import exists
from pathlib import Path
from xml.dom import minidom

import click
import pymeshlab
import torch as th
import trimesh

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.prims.material_prim import MaterialPrim
from omnigibson.scenes import Scene
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.urdfpy_utils import URDF
from omnigibson.utils.usd_utils import create_primitive_mesh
# from omnigibson.utils.python_utils import assert_valid_key

# Create module logger
log = create_module_logger(module_name=__name__)

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

USE_VRAY_MATERIAL = True

_MTL_MAP_TYPE_MAPPINGS = {
    "map_Kd": "diffuse",
    "map_bump": "normal",
    "map_Pm": "metalness",
    "map_Pr": "glossiness",
    "map_Tf": "refraction",
    "map_Ks": "reflection",
    "map_Ns": "reflection_ior",
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

_OPACITY_CATEGORIES = {"tree", "low_resolution_tree", "bush"}

_VISUAL_ONLY_CATEGORIES = {
    "carpet",
}


class _TorchEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for PyTorch tensors.

    This encoder converts PyTorch tensors to lists, making them JSON serializable.

    Methods:
        default(o): Overrides the default method to handle PyTorch tensors.
    """

    def default(self, o):
        if isinstance(o, th.Tensor):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def _space_string_to_tensor(string):
    """
    Converts a space-separated string of numbers into a PyTorch tensor.

    Examples:
        "0 1 2" => tensor([0., 1., 2.])

    Args:
        string (str): Space-separated string of numbers to convert.

    Returns:
        torch.Tensor: Tensor containing the numerical values from the input string.
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
    Splits an OBJ file into individual OBJ files, each containing a single connected mesh.

    Args:
        obj_fpath (str): The file path to the input OBJ file.

    Returns:
        int: The number of individual connected mesh files created.

    The function performs the following steps:
    1. Loads the OBJ file using trimesh.
    2. Splits the loaded mesh into individual connected components.
    3. Saves each connected component as a separate OBJ file in the same directory as the input file.
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
    Splits the OBJ references in the given URDF file into separate files for each connected component.

    This function parses a URDF file, finds all collision mesh references, splits the referenced OBJ files into
    connected components, and updates the URDF file to reference these new OBJ files. The updated URDF file is
    saved with a new name.

    Args:
        urdf_fpath (str): The file path to the URDF file to be processed.
        name_suffix (str, optional): Suffix to append to the output URDF file name. Defaults to "split".
        mesh_fpath_offset (str, optional): Offset path to the directory containing the mesh files. Defaults to ".".

    Returns:
        str: The file path to the newly created URDF file with split OBJ references.
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


def _set_omnipbr_mtl_diffuse(mtl_prim, texture):
    mtl = "diffuse_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    log.debug(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _set_omnipbr_mtl_normal(mtl_prim, texture):
    mtl = "normalmap_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    log.debug(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _set_omnipbr_mtl_metalness(mtl_prim, texture):
    mtl = "metallic_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    lazy.omni.usd.create_material_input(mtl_prim, "metallic_texture_influence", 1.0, lazy.pxr.Sdf.ValueTypeNames.Float)
    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    log.debug(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _set_omnipbr_mtl_opacity(mtl_prim, texture):
    mtl = "opacity_texture"
    lazy.omni.usd.create_material_input(mtl_prim, mtl, texture, lazy.pxr.Sdf.ValueTypeNames.Asset)
    lazy.omni.usd.create_material_input(mtl_prim, "enable_opacity", True, lazy.pxr.Sdf.ValueTypeNames.Bool)
    lazy.omni.usd.create_material_input(mtl_prim, "enable_opacity_texture", True, lazy.pxr.Sdf.ValueTypeNames.Bool)

    # Set the opacity to use the alpha channel for its mono-channel value.
    # This defaults to some other value, which takes opaque black channels in the
    # image to be fully transparent. This is not what we want.
    lazy.omni.usd.create_material_input(mtl_prim, "opacity_mode", 0, lazy.pxr.Sdf.ValueTypeNames.Int)

    # We also need to set an opacity threshold. Our objects can include continuous alpha values for opacity
    # but the ray tracing renderer can only handle binary opacity values. The default threshold
    # leaves most objects entirely transparent, so we try to avoid that here.
    lazy.omni.usd.create_material_input(mtl_prim, "opacity_threshold", 0.1, lazy.pxr.Sdf.ValueTypeNames.Float)

    # Verify it was set
    shade = lazy.omni.usd.get_shader_from_material(mtl_prim)
    log.debug(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def _rename_prim(prim, name):
    """
    Renames a given prim to a new name.

    Args:
        prim (Usd.Prim): The prim to be renamed.
        name (str): The new name for the prim.

    Returns:
        Usd.Prim: The renamed prim at the new path.
    """
    path_from = prim.GetPrimPath().pathString
    path_to = f"{'/'.join(path_from.split('/')[:-1])}/{name}"
    lazy.omni.kit.commands.execute("MovePrim", path_from=path_from, path_to=path_to)
    return lazy.isaacsim.core.utils.prims.get_prim_at_path(path_to)


def _get_visual_objs_from_urdf(urdf_path):
    """
    Extracts visual objects from a URDF file.

    Args:
        urdf_path (str): Path to the URDF file.

    Returns:
        OrderedDict: A dictionary mapping link names to dictionaries of visual meshes. Each link name (e.g., 'base_link')
                     maps to another dictionary where the keys are visual mesh names and the values are the corresponding
                     visual object file paths. If no visual object file is found for a mesh, the value will be None.
    """
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
                        log.debug(f"Warning: No obj file found associated with {name}/{visual_mesh_name}!")
                    visual_objs[name][visual_mesh_name] = obj_file

    return visual_objs


def _import_rendering_channels(obj_prim, obj_category, obj_model, model_root_path, usd_path, dataset_root):
    """
    Imports and binds rendering channels for a given object in an Omniverse USD stage.

    This function performs the following steps:
    1. Removes existing material prims from the object.
    2. Extracts visual objects and their associated material files from the object's URDF file.
    3. Copies material files to the USD directory and creates new materials.
    4. Applies rendering channels to the new materials.
    5. Binds the new materials to the visual meshes of the object.
    6. Copies state-conditioned texture maps (e.g., cooked, soaked) for the object.

    Args:
        obj_prim (Usd.Prim): The USD prim representing the object.
        obj_category (str): The category of the object (e.g., "ceilings", "walls").
        obj_model (str): The model name of the object.
        model_root_path (str): The root path of the model files.
        usd_path (str): The path to the USD file.
        dataset_root (str): The root path of the dataset containing the object files.

    Raises:
        AssertionError: If more than one material file is found in an OBJ file.
        AssertionError: If a valid visual prim is not found for a mesh.
    """
    usd_dir = os.path.dirname(usd_path)

    # Remove the material prims as we will create them explictly later.
    stage = lazy.omni.usd.get_context().get_stage()
    for prim in obj_prim.GetChildren():
        looks_prim = None
        if prim.GetName() == "Looks":
            looks_prim = prim
        elif prim.GetPrimTypeInfo().GetTypeName() == "Xform":
            looks_prim_path = f"{str(prim.GetPrimPath())}/Looks"
            looks_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(looks_prim_path)
        if not looks_prim:
            continue
        for subprim in looks_prim.GetChildren():
            if subprim.GetPrimTypeInfo().GetTypeName() != "Material":
                continue
            log.debug(
                f"Removed material prim {subprim.GetPath()}:",
                stage.RemovePrim(subprim.GetPath()),
            )

    # Remove the materials copied over by the URDF importer
    urdf_importer_mtl_dir = os.path.join(usd_dir, "materials")
    if os.path.exists(urdf_importer_mtl_dir):
        shutil.rmtree(urdf_importer_mtl_dir)

    # Grab all visual objs for this object
    urdf_path = f"{dataset_root}/objects/{obj_category}/{obj_model}/urdf/{obj_model}_with_meta_links.urdf"
    visual_objs = _get_visual_objs_from_urdf(urdf_path)

    # Extract absolute paths to mtl files for each link
    link_mtl_files = OrderedDict()  # maps link name to dictionary mapping mesh name to mtl file
    mtl_infos = OrderedDict()  # maps mtl name to dictionary mapping material channel name to png file
    for link_name, link_meshes in visual_objs.items():
        link_mtl_files[link_name] = OrderedDict()
        for mesh_name, obj_file in link_meshes.items():
            # Get absolute path and open the obj file if it exists:
            if obj_file is not None:
                obj_path = os.path.abspath(f"{dataset_root}/objects/{obj_category}/{obj_model}/urdf/{obj_file}")
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
                    mtl_dir = os.path.dirname(obj_path)
                    link_mtl_files[link_name][mesh_name] = mtl_name
                    mtl_infos[mtl_name] = OrderedDict()
                    # Open the mtl file
                    mtl_path = os.path.join(mtl_dir, mtl)
                    with open(mtl_path, "r") as f:
                        # Read any lines beginning with map that aren't commented out
                        for line in f.readlines():
                            if line[:4] == "map_":
                                map_type, map_path_relative_to_mtl_dir = line.split(" ")
                                map_path_relative_to_mtl_dir = map_path_relative_to_mtl_dir.split("\n")[0]
                                print("Found map path in file as ", map_path_relative_to_mtl_dir)
                                map_path_absolute = os.path.abspath(os.path.join(mtl_dir, map_path_relative_to_mtl_dir))
                                print("Absolute map path is ", map_path_absolute)
                                map_path_relative_to_usd_dir = os.path.relpath(map_path_absolute, usd_dir)
                                print("USD path is ", usd_dir)
                                print("Material path relative to USD is", map_path_relative_to_usd_dir)
                                mtl_infos[mtl_name][_MTL_MAP_TYPE_MAPPINGS[map_type]] = map_path_relative_to_usd_dir

                    print("Found material file:", mtl_name, mtl_infos[mtl_name])

    # Next, for each material information, we create a new OmniPBR material
    shaders = OrderedDict()  # maps mtl name to shader prim
    for mtl_name, mtl_info in mtl_infos.items():
        # Create the Vray material
        mtl_created_list = []
        lazy.omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="omnigibson_vray_mtl.mdl",
            mtl_name="OmniGibsonVRayMtl",
            mtl_created_list=mtl_created_list,
        )
        vray_mat = lazy.omni.isaac.core.utils.prims.get_prim_at_path(mtl_created_list[0])

        # Create the OmniPBR material
        pbr_material_name = mtl_name + "_pbr"
        mtl_created_list = []
        lazy.omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=mtl_created_list,
        )
        pbr_mat = lazy.isaacsim.core.utils.prims.get_prim_at_path(mtl_created_list[0])
        rendering_channel_mappings = {
            "diffuse": _set_omnipbr_mtl_diffuse,
            "normal": _set_omnipbr_mtl_normal,
            "metalness": _set_omnipbr_mtl_metalness,
        }
        # Apply all rendering channels for this material
        for mat_type, mat_file in mtl_info.items():
            # First assign the Vray material channels. These are simple - all the channels
            # are just named x_texture for channel x.
            lazy.omni.usd.create_material_input(
                vray_mat, f"{mat_type}_texture", mat_file, lazy.pxr.Sdf.ValueTypeNames.Asset
            )

            # Do the OmniPBR material next
            # Use the alpha of the diffuse texture for opacity for trees etc.
            if mat_type == "diffuse" and obj_category in _OPACITY_CATEGORIES:
                _set_omnipbr_mtl_opacity(pbr_mat, mat_file)
            render_channel_fcn = rendering_channel_mappings.get(mat_type, None)
            if render_channel_fcn is not None:
                render_channel_fcn(pbr_mat, mat_file)
            else:
                # Warn user that we didn't find the correct rendering channel
                log.debug(f"Warning: could not find rendering channel function for material: {mat_type}, skipping")

        # Rename material
        pbr_mat = _rename_prim(prim=pbr_mat, name=pbr_material_name)
        selected_mat = vray_mat if USE_VRAY_MATERIAL else pbr_mat
        shade = lazy.pxr.UsdShade.Material(selected_mat)
        shaders[mtl_name] = shade
        log.debug(f"Created material {pbr_material_name}:", pbr_mat)

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
            visual_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(mesh_prim_path)
            assert visual_prim, f"Error: Did not find valid visual prim at {mesh_prim_path}!"
            # Bind the created link material to the visual prim
            log.debug(f"Binding material {mtl_name}, shader {shaders[mtl_name]}, to prim {mesh_prim_path}...")
            lazy.pxr.UsdShade.MaterialBindingAPI(visual_prim).Bind(
                shaders[mtl_name], lazy.pxr.UsdShade.Tokens.strongerThanDescendants
            )


def _add_xform_properties(prim):
    """
    Adds and configures transformation properties for a given USD prim.

    This function ensures that the specified USD prim has the necessary transformation
    properties (scale, translate, and orient) and removes any unwanted transformation
    properties. It also sets the order of the transformation operations.

    Args:
        prim (pxr.Usd.Prim): The USD prim to which the transformation properties will be added.

    Notes:
        - The function removes the following properties if they exist:
            "xformOp:rotateX", "xformOp:rotateXZY", "xformOp:rotateY", "xformOp:rotateYXZ",
            "xformOp:rotateYZX", "xformOp:rotateZ", "xformOp:rotateZYX", "xformOp:rotateZXY",
            "xformOp:rotateXYZ", "xformOp:transform".
        - If the prim does not have "xformOp:scale", "xformOp:translate", or "xformOp:orient",
          these properties are added with default values.
        - The order of the transformation operations is set to translate, orient, and scale.
    """
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


def _generate_meshes_for_primitive_meta_links(stage, obj_model, link_name, meta_link_type, meta_link_infos):
    """
    Process a meta link by creating visual meshes or lights below it.

    Args:
        stage (pxr.Usd.Stage): The USD stage where the meta link will be processed.
        obj_model (str): The object model name.
        link_name (str): Name of the meta link's parent link (e.g. what part of the object the meta link is attached to).
        meta_link_type (str): The type of the meta link. Must be one of the allowed meta types.
        meta_link_infos (dict): A dictionary containing meta link information. The keys are link IDs and the values are lists of mesh information dictionaries.

    Returns:
        None

    Raises:
        AssertionError: If the meta_link_type is not in the allowed meta types or if the mesh_info_list has unexpected keys or invalid number of meshes.
        ValueError: If an invalid light type or mesh type is encountered.

    Notes:
        - Handles specific meta link types such as "togglebutton", "particleapplier", "particleremover", "particlesink", and "particlesource".
        - For "particleapplier" meta link type, adjusts the orientation if the mesh type is "cone".
        - Creates lights or primitive shapes based on the meta link type and mesh information.
        - Sets various attributes for lights and meshes, including color, intensity, size, and scale.
        - Makes meshes invisible and sets their local pose.
    """
    assert meta_link_type in _ALLOWED_META_TYPES
    if _ALLOWED_META_TYPES[meta_link_type] not in ["primitive", "light"] and meta_link_type != "particlesource":
        return

    is_light = _ALLOWED_META_TYPES[meta_link_type] == "light"

    for link_id, mesh_info_list in meta_link_infos.items():
        if len(mesh_info_list) == 0:
            continue

        # TODO: Remove this after this is fixed.
        if type(mesh_info_list) is dict:
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
            th.tensor(mesh_info_list[0]["position"]),
            th.tensor(mesh_info_list[0]["orientation"]),
        )

        # For particle applier only, the orientation of the meta link matters (particle should shoot towards the negative z-axis)
        # If the meta link is created based on the orientation of the first mesh that is a cone, we need to rotate it by 180 degrees
        # because the cone is pointing in the wrong direction. This is already done in update_obj_urdf_with_meta_links;
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
                stage.DefinePrim(f"/{obj_model}/meta__{link_name}_lights_{link_id}_0_link/lights", "Scope")
                prim_path = f"/{obj_model}/meta__{link_name}_lights_{link_id}_0_link/lights/light_{i}"
                prim = getattr(lazy.pxr.UsdLux, f"{light_type}Light").Define(stage, prim_path).GetPrim()
                lazy.pxr.UsdLux.ShapingAPI.Apply(prim).GetShapingConeAngleAttr().Set(180.0)
            else:
                if meta_link_type == "particlesource":
                    mesh_type = "Cylinder"
                else:
                    # Create a primitive shape
                    mesh_type = mesh_info["type"].capitalize() if mesh_info["type"] != "box" else "Cube"
                # Create the visuals prim
                stage.DefinePrim(f"/{obj_model}/meta__{link_name}_{meta_link_type}_{link_id}_0_link/visuals", "Scope")
                prim_path = f"/{obj_model}/meta__{link_name}_{meta_link_type}_{link_id}_0_link/visuals/mesh_{i}"
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
            xform_prim = lazy.isaacsim.core.prims.xform_prim.XFormPrim(prim_path=prim_path)

            # Get the mesh/light pose in the parent link frame
            mesh_in_parent_link_pos, mesh_in_parent_link_orn = (
                th.tensor(mesh_info["position"]),
                th.tensor(mesh_info["orientation"]),
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

            xform_prim.set_local_pose(
                translation=mesh_in_meta_link_pos,
                orientation=mesh_in_meta_link_orn[[3, 0, 1, 2]],
            )


def _process_glass_link(prim):
    """
    Processes the given USD prim to update any glass parts to use the glass material.

    This function traverses the children of the given prim to find any Mesh-type prims
    that do not have a CollisionAPI, indicating they are visual elements. It collects
    the paths of these prims and ensures they are bound to a glass material.

    Args:
        prim (pxr.Usd.Prim): The USD prim to process.

    Raises:
        AssertionError: If no glass prim paths are found.
    """
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

    stage = lazy.isaacsim.core.utils.stage.get_current_stage()
    root_path = stage.GetDefaultPrim().GetPath().pathString
    glass_mtl_prim_path = f"{root_path}/Looks/OmniGlass"
    if not lazy.isaacsim.core.utils.prims.get_prim_at_path(glass_mtl_prim_path):
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


def import_obj_metadata(usd_path, obj_category, obj_model, dataset_root, import_render_channels=False):
    """
    Imports metadata for a given object model from the dataset. This metadata consist of information
    that is NOT included in the URDF file and instead included in the various JSON files shipped in
    iGibson and OmniGibson datasets.

    Args:
        usd_path (str): Path to USD file
        obj_category (str): The category of the object.
        obj_model (str): The model name of the object.
        dataset_root (str): The root directory of the dataset.
        import_render_channels (bool, optional): Flag to import rendering channels. Defaults to False.

    Raises:
        ValueError: If the bounding box size is not found in the metadata.

    Returns:
        bool: Success status of the conversion
    """
    # Check if filepath exists
    model_root_path = f"{dataset_root}/objects/{obj_category}/{obj_model}"
    log.debug("Loading", usd_path, "for metadata import.")

    # Load model
    lazy.isaacsim.core.utils.stage.open_stage(usd_path)
    stage = lazy.isaacsim.core.utils.stage.get_current_stage()
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

    log.debug("Process meta links")

    # Convert primitive meta links
    for link_name, link_metadata in meta_links.items():
        for meta_link_type, meta_link_infos in link_metadata.items():
            _generate_meshes_for_primitive_meta_links(stage, obj_model, link_name, meta_link_type, meta_link_infos)

    # Get all meta links, set them to guide purpose, and add some metadata
    # Here we want to include every link that has the meta__ prefix.
    # This includes meta links that get added into the URDF in earlier
    # stages.
    meta_link_prims = [
        p for p in prim.GetChildren() if p.GetName().startswith("meta__") and p.GetName().endswith("_link")
    ]
    for meta_prim in meta_link_prims:
        # Get meta link information
        unparsed_meta = meta_prim.GetName()[6:-5]  # remove meta__ and _link
        meta_parts = unparsed_meta.rsplit("_", 3)
        assert len(meta_parts) == 4, f"Invalid meta link name: {unparsed_meta}"
        link_name, meta_link_type, link_id, link_sub_id = meta_parts

        # Add the is_meta_link, meta_link_type, and meta_link_id attributes
        meta_prim.CreateAttribute("ig:isMetaLink", lazy.pxr.Sdf.ValueTypeNames.Bool)
        meta_prim.GetAttribute("ig:isMetaLink").Set(True)
        meta_prim.CreateAttribute("ig:metaLinkType", lazy.pxr.Sdf.ValueTypeNames.String)
        meta_prim.GetAttribute("ig:metaLinkType").Set(meta_link_type)
        meta_prim.CreateAttribute("ig:metaLinkId", lazy.pxr.Sdf.ValueTypeNames.String)
        meta_prim.GetAttribute("ig:metaLinkId").Set(link_id)
        meta_prim.CreateAttribute("ig:metaLinkSubId", lazy.pxr.Sdf.ValueTypeNames.Int)
        meta_prim.GetAttribute("ig:metaLinkSubId").Set(int(link_sub_id))

        # Set the purpose of the visual meshes to be guide
        visual_prim = meta_prim.GetChild("visuals")
        if visual_prim.IsValid():
            # If it's an imageable, set the purpose to guide
            if visual_prim.GetTypeName() == "Mesh":
                purpose_attr = lazy.pxr.UsdGeom.Imageable(visual_prim).CreatePurposeAttr()
                purpose_attr.Set(lazy.pxr.UsdGeom.Tokens.guide)
            for visual_mesh in visual_prim.GetChildren():
                if visual_mesh.GetTypeName() == "Mesh":
                    purpose_attr = lazy.pxr.UsdGeom.Imageable(visual_mesh).CreatePurposeAttr()
                    purpose_attr.Set(lazy.pxr.UsdGeom.Tokens.guide)

    log.debug("Done processing meta links")

    # Iterate over dict and replace any lists of dicts as dicts of dicts (with each dict being indexed by an integer)
    data = _recursively_replace_list_of_dict(data)

    log.debug("Done recursively replacing")

    # Create attributes for bb, offset, category, model and store values
    prim.CreateAttribute("ig:nativeBB", lazy.pxr.Sdf.ValueTypeNames.Vector3f)
    prim.CreateAttribute("ig:offsetBaseLink", lazy.pxr.Sdf.ValueTypeNames.Vector3f)
    prim.CreateAttribute("ig:category", lazy.pxr.Sdf.ValueTypeNames.String)
    prim.CreateAttribute("ig:model", lazy.pxr.Sdf.ValueTypeNames.String)
    prim.GetAttribute("ig:nativeBB").Set(lazy.pxr.Gf.Vec3f(*default_bb))
    prim.GetAttribute("ig:offsetBaseLink").Set(lazy.pxr.Gf.Vec3f(*base_link_offset))
    prim.GetAttribute("ig:category").Set(obj_category)
    prim.GetAttribute("ig:model").Set(obj_model)

    log.debug(f"data: {data}")

    # Store remaining data as metadata
    prim.SetCustomData(data)

    # Add material channels
    # log.debug(f"prim children: {prim.GetChildren()}")
    # looks_prim_path = f"{str(prim.GetPrimPath())}/Looks"
    # looks_prim = prim.GetChildren()[0] #lazy.isaacsim.core.utils.prims.get_prim_at_path(looks_prim_path)
    # mat_prim_path = f"{str(prim.GetPrimPath())}/Looks/material_material_0"
    # mat_prim = looks_prim.GetChildren()[0] #lazy.isaacsim.core.utils.prims.get_prim_at_path(mat_prim_path)
    # log.debug(f"looks children: {looks_prim.GetChildren()}")
    # log.debug(f"mat prim: {mat_prim}")
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

    # Rename model to be named <model> if not already named that
    old_prim_path = prim.GetPrimPath().pathString
    if old_prim_path.split("/")[-1] != obj_model:
        new_prim_path = "/".join(old_prim_path.split("/")[:-1]) + f"/{obj_model}"
        lazy.omni.kit.commands.execute("MovePrim", path_from=old_prim_path, path_to=new_prim_path)
        prim = stage.GetDefaultPrim()

    # Hacky way to avoid new prim being created at /World
    class DummyScene:
        prim_path = ""

    og.sim.render()

    mat_prims = find_all_prim_children_with_type(prim_type="Material", root_prim=prim)
    for i, mat_prim in enumerate(mat_prims):
        mat = MaterialPrim(mat_prim.GetPrimPath().pathString, f"mat{i}")
        mat.load(DummyScene)
        mat.shader_update_asset_paths_with_root_path(root_path=os.path.dirname(usd_path), relative=True)

    # Save stage
    stage.Save()

    # Return the root prim
    return prim


def _recursively_replace_list_of_dict(dic):
    """
    Recursively processes a dictionary to replace specific values and structures that can be stored
    in USD.

    This function performs the following transformations:
    - Replaces `None` values with `lazy.pxr.lazy.pxr.UsdGeom.Tokens.none`.
    - Converts empty lists or tuples to `lazy.pxr.Vt.Vec3fArray()`.
    - Converts lists of dictionaries to a dictionary with string keys.
    - Converts nested lists or tuples to specific `lazy.pxr.Vt` array types based on their length:
        - Length 2: `lazy.pxr.Vt.Vec2fArray`
        - Length 3: `lazy.pxr.Vt.Vec3fArray`
        - Length 4: `lazy.pxr.Vt.Vec4fArray`
    - Converts lists of integers to `lazy.pxr.Vt.IntArray`.
    - Converts lists of floats to `lazy.pxr.Vt.FloatArray`.
    - Replaces `None` values within lists with `lazy.pxr.lazy.pxr.UsdGeom.Tokens.none`.
    - Recursively processes nested dictionaries.

    Args:
        dic (dict): The dictionary to process.

    Returns:
        dict: The processed dictionary with the specified transformations applied.
    """
    for k, v in dic.items():
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
                dic[k] = lazy.pxr.Vt.IntArray(v)
            elif isinstance(v[0], float):
                dic[k] = lazy.pxr.Vt.FloatArray(v)
            else:
                # Replace any Nones
                for i, ele in enumerate(v):
                    if ele is None:
                        v[i] = lazy.pxr.lazy.pxr.UsdGeom.Tokens.none
        if isinstance(v, dict):
            # Iterate through nested dictionaries
            dic[k] = _recursively_replace_list_of_dict(v)

    return dic


def _create_urdf_import_config(
    use_convex_decomposition=False,
    merge_fixed_joints=False,
):
    """
    Creates and configures a URDF import configuration.

    This function sets up the import configuration for URDF files by executing the
    "URDFCreateImportConfig" command and adjusting various settings such as drive type,
    joint merging, convex decomposition, base fixing, inertia tensor import, distance scale,
    density, drive strength, position drive damping, self-collision, up vector, default prim
    creation, and physics scene creation.

    Args:
        use_convex_decomposition (bool): Whether to have omniverse use internal convex decomposition
            on any collision meshes
        merge_fixed_joints (bool): Whether to merge fixed joints or not

    Returns:
        import_config: The configured URDF import configuration object.
    """
    # Set up import configuration
    _, import_config = lazy.omni.kit.commands.execute("URDFCreateImportConfig")
    drive_mode = (
        import_config.default_drive_type.__class__
    )  # Hacky way to get class for default drive type, options are JOINT_DRIVE_{NONE / POSITION / VELOCITY}

    import_config.set_merge_fixed_joints(merge_fixed_joints)
    import_config.set_convex_decomp(use_convex_decomposition)
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


def import_obj_urdf(
    urdf_path,
    obj_category,
    obj_model,
    dataset_root=gm.CUSTOM_DATASET_PATH,
    use_omni_convex_decomp=False,
    use_usda=False,
    merge_fixed_joints=False,
):
    """
    Imports an object from a URDF file into the current stage.

    Args:
        urdf_path (str): Path to URDF file to import
        obj_category (str): The category of the object.
        obj_model (str): The model name of the object.
        dataset_root (str): The root directory of the dataset.
        use_omni_convex_decomp (bool): Whether to use omniverse's built-in convex decomposer for collision meshes
        use_usda (bool): If set, will write files to .usda files instead of .usd
            (bigger memory footprint, but human-readable)
        merge_fixed_joints (bool): whether to merge fixed joints or not

    Returns:
        2-tuple:
            - str: Absolute path to post-processed URDF file used to generate USD
            - str: Absolute path to the imported USD file
    """
    # Preprocess input URDF to account for meta links
    urdf_path = _add_meta_links_to_urdf(
        urdf_path=urdf_path, obj_category=obj_category, obj_model=obj_model, dataset_root=dataset_root
    )
    # Import URDF
    cfg = _create_urdf_import_config(
        use_convex_decomposition=use_omni_convex_decomp,
        merge_fixed_joints=merge_fixed_joints,
    )
    # Check if filepath exists
    usd_path = f"{dataset_root}/objects/{obj_category}/{obj_model}/usd/{obj_model}.{'usda' if use_usda else 'usd'}"
    if _SPLIT_COLLISION_MESHES:
        log.debug(f"Converting collision meshes from {obj_category}, {obj_model}...")
        urdf_path = _split_all_objs_in_urdf(urdf_fpath=urdf_path, name_suffix="split")
    log.debug(f"Importing {obj_category}, {obj_model} into path {usd_path}...")
    # Only import if it doesn't exist
    lazy.omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=cfg,
        dest_path=usd_path,
    )
    log.debug(f"Imported {obj_category}, {obj_model}")

    return urdf_path, usd_path


def _pretty_print_xml(current, parent=None, index=-1, depth=0, use_tabs=False):
    """
    Recursively formats an XML element tree to be pretty-printed with indentation.

    Args:
        current (xml.etree.ElementTree.Element): The current XML element to format.
        parent (xml.etree.ElementTree.Element, optional): The parent XML element. Defaults to None.
        index (int, optional): The index of the current element in the parent's children. Defaults to -1.
        depth (int, optional): The current depth in the XML tree, used for indentation. Defaults to 0.
        use_tabs (bool, optional): If True, use tabs for indentation; otherwise, use spaces. Defaults to False.

    Returns:
        None
    """
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
    Converts any type of {bool, int, float, list, tuple, array, string, th.Tensor} into a URDF-compatible string.
    Note that an input string / th.Tensor results in a no-op action.

    Args:
        inp: Input to convert to string

    Returns:
        str: String equivalent of @inp

    Raises:
        ValueError: If the input type is unsupported.
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
    Generates URDF joint
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
    ET.SubElement(
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
        ET.SubElement(jnt, "axis", xyz=_convert_to_xml_string(axis))
    dynamic_params = {}
    if damping is not None:
        dynamic_params["damping"] = _convert_to_xml_string(damping)
    if friction is not None:
        dynamic_params["friction"] = _convert_to_xml_string(friction)
    if dynamic_params:
        ET.SubElement(jnt, "dynamics", **dynamic_params)
    if limits is not None:
        ET.SubElement(jnt, "limit", lower=limits[0], upper=limits[1])

    # Return this element
    return jnt


def _create_urdf_link(name, subelements=None, mass=None, inertia=None):
    """
    Generates URDF link element
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


def _create_urdf_meta_link(
    root_element,
    meta_link_name,
    parent_link_name="base_link",
    pos=(0, 0, 0),
    rpy=(0, 0, 0),
):
    """
    Creates the appropriate URDF joint and link for a meta link and appends it to the root element.

    Args:
        root_element (Element): The root XML element to which the meta link will be appended.
        meta_link_name (str): The name of the meta link to be created.
        parent_link_name (str, optional): The name of the parent link. Defaults to "base_link".
        pos (tuple, optional): The position of the joint in the form (x, y, z). Defaults to (0, 0, 0).
        rpy (tuple, optional): The roll, pitch, and yaw of the joint in the form (r, p, y). Defaults to (0, 0, 0).

    Returns:
        None
    """
    # Create joint
    jnt = _create_urdf_joint(
        name=f"{meta_link_name}_joint",
        parent=parent_link_name,
        child=f"{meta_link_name}_link",
        pos=pos,
        rpy=rpy,
        joint_type="fixed",
    )
    # Create child link
    link = _create_urdf_link(
        name=f"{meta_link_name}_link",
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


def _add_meta_links_to_urdf(urdf_path, obj_category, obj_model, dataset_root):
    """
    Adds meta links to a URDF file based on metadata.

    This function reads a URDF file and corresponding metadata, processes the metadata to add meta links, and then
    saves an updated version of the URDF file with these meta links.

    Args:
        urdf_path (str): Path to URDF
        obj_category (str): The category of the object.
        obj_model (str): The model name of the object.
        dataset_root (str): The root directory of the dataset.

    Returns:
        str: The path to the updated URDF file.
    """
    # Check if filepath exists
    model_root_path = f"{dataset_root}/objects/{obj_category}/{obj_model}"

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
        log.debug("meta_links:", meta_links)
        for parent_link_name, child_link_attrs in meta_links.items():
            for meta_link_name, ml_attrs in child_link_attrs.items():
                assert (
                    meta_link_name in _ALLOWED_META_TYPES
                ), f"meta_link_name {meta_link_name} not in {_ALLOWED_META_TYPES}"

                for ml_id, attrs_list in ml_attrs.items():
                    # If the attrs list is a dictionary (legacy format), convert it to a list
                    if isinstance(attrs_list, dict):
                        keys = [int(k) for k in attrs_list.keys()]
                        assert set(keys) == set(
                            range(len(keys))
                        ), f"Expected keys to be 0-indexed integers, but got {keys}"
                        int_key_dict = {int(k): v for k, v in attrs_list.items()}
                        attrs_list = [int_key_dict[i] for i in range(len(keys))]

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

                        for i, attrs in enumerate(attrs_list):
                            pos = th.as_tensor(attrs["position"])
                            quat = th.as_tensor(attrs["orientation"])

                            # For particle applier only, the orientation of the meta link matters (particle should shoot towards the negative z-axis)
                            # If the meta link is created based on the orientation of the first mesh that is a cone, we need to rotate it by 180 degrees
                            # because the cone is pointing in the wrong direction.
                            if meta_link_name == "particleapplier" and attrs["type"] == "cone":
                                assert (
                                    len(attrs_list) == 1
                                ), f"Expected only one instance for meta_link {meta_link_name}_{ml_id}, but found {len(attrs_list)}"
                                quat = T.quat_multiply(quat, T.axisangle2quat(th.tensor([math.pi, 0.0, 0.0])))

                            # Create meta link
                            _create_urdf_meta_link(
                                root_element=root,
                                meta_link_name=f"meta__{parent_link_name}_{meta_link_name}_{ml_id}_{i}",
                                parent_link_name=parent_link_name,
                                pos=pos,
                                rpy=T.quat2euler(quat),
                            )

    # Export this URDF
    return _save_xmltree_as_urdf(
        root_element=root,
        name=f"{obj_model}_with_meta_links",
        dirpath=f"{model_root_path}/urdf",
        unique_urdf=False,
    )


def convert_scene_urdf_to_json(urdf, json_path):
    """
    Converts a scene from a URDF file to a JSON file.

    This function loads the scene described by the URDF file into the OmniGibson simulator,
    plays the simulation, and saves the scene to a JSON file. After saving, it removes the
    "init_info" from the JSON file and saves it again.

    Args:
        urdf (str): The file path to the URDF file describing the scene.
        json_path (str): The file path where the JSON file will be saved.
    """
    # First, load the requested objects from the URDF into OG
    _load_scene_from_urdf(urdf=urdf)

    # Play the simulator, then save
    og.sim.play()
    Path(os.path.dirname(json_path)).mkdir(parents=True, exist_ok=True)
    og.sim.save(json_paths=[json_path])

    # Load the json, remove the init_info because we don't need it, then save it again
    with open(json_path, "r") as f:
        scene_info = json.load(f)

    scene_info.pop("init_info")

    with open(json_path, "w+") as f:
        json.dump(scene_info, f, cls=_TorchEncoder, indent=4)


def _load_scene_from_urdf(urdf):
    """
    Loads a scene from a URDF file.

    Args:
        urdf (str): Path to the URDF file.

    Raises:
        ValueError: If an object fails to load.

    This function performs the following steps:
    1. Extracts object configuration information from the URDF file.
    2. Creates a new scene without a floor plane and imports it into the simulator.
    3. Iterates over the objects' information and attempts to load each object into the scene.
       - If the USD file for an object does not exist, it prints a message and skips the object.
       - If an object fails to load, it raises a ValueError with the object's name.
    4. Sets the bounding box center position and orientation for each loaded object.
    5. Takes a simulation step to finalize the scene setup.
    """
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
                log.warning("Missing object", obj_name)
                continue
            obj = DatasetObject(
                name=obj_name,
                **obj_info["cfg"],
            )
            scene.add_object(obj)
            obj.set_bbox_center_position_orientation(position=obj_info["bbox_pos"], orientation=obj_info["bbox_quat"])
        except Exception as e:
            raise ValueError(f"Failed to load object {obj_name}") from e

    # Take a sim step
    og.sim.step()


def _get_objects_config_from_scene_urdf(urdf):
    """
    Parses a URDF file to extract object configuration information.

    Args:
        urdf (str): Path to the URDF file.

    Returns:
        dict: A dictionary containing the configuration of objects extracted from the URDF file.
    """
    tree = ET.parse(urdf)
    root = tree.getroot()
    objects_cfg = dict()
    _get_objects_config_from_element(root, model_pose_info=objects_cfg)
    return objects_cfg


def _get_objects_config_from_element(element, model_pose_info):
    """
    Extracts and populates object configuration information from an URDF element.

    This function processes an URDF element to extract joint and link information,
    populating the provided `model_pose_info` dictionary with the relevant data.

    Args:
        element (xml.etree.ElementTree.Element): The URDF element containing object configuration data.
        model_pose_info (dict): A dictionary to be populated with the extracted configuration information.

    The function performs two passes through the URDF element:
    1. In the first pass, it extracts joint information and populates `model_pose_info` with joint pose data.
    2. In the second pass, it extracts link information, imports object models, and updates `model_pose_info` with
       additional configuration details such as category, model, bounding box, rooms, scale, and object scope.

    The function also handles nested elements by recursively calling itself for child elements.

    Note:
        - Joint names with hyphens are replaced with underscores.
        - The function asserts that each link name (except "world") is present in `model_pose_info` after the first pass.
    """
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
                log.debug(name)
                assert name in model_pose_info, f"Did not find {name} in current model pose info!"
                model_pose_info[name]["cfg"]["category"] = ele.get("category")
                model_pose_info[name]["cfg"]["visual_only"] = ele.get("category") in _VISUAL_ONLY_CATEGORIES
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
    """
    Extracts joint information from an URDF element.

    Args:
        joint_element (xml.etree.ElementTree.Element): The URDF element containing joint information.

    Returns:
        tuple: A tuple containing:
            - child (str or None): The name of the child link, or None if not specified.
            - pos (numpy.ndarray or None): The position as a tensor, or None if not specified.
            - quat (numpy.ndarray or None): The orientation as a quaternion, or None if not specified.
            - fixed_jnt (bool): True if the joint is fixed, False otherwise.
    """
    child, pos, quat, fixed_jnt = None, None, None, None
    fixed_jnt = joint_element.get("type") == "fixed"
    for ele in joint_element:
        if ele.tag == "origin":
            quat = T.euler2quat(_space_string_to_tensor(ele.get("rpy")))
            pos = _space_string_to_tensor(ele.get("xyz"))
        elif ele.tag == "child":
            child = ele.get("link")
    return child, pos, quat, fixed_jnt


def make_mesh_positive(mesh_fpath, scale, output_suffix="mirror"):
    assert "." not in mesh_fpath
    for sc, letter in zip(scale, "xyz"):
        if sc < 0:
            output_suffix += f"_{letter}"
    for filetype in [".obj", ".stl", ".dae"]:
        fpath = f"{mesh_fpath}{filetype}"
        out_fpath = f"{mesh_fpath}_{output_suffix}{filetype}"
        kwargs = dict()
        if filetype == ".dae":
            kwargs["force"] = "mesh"
        if os.path.exists(fpath):
            try:
                tm = trimesh.load(fpath, **kwargs)
                tm.apply_scale(scale)
                tm.export(out_fpath)
                if filetype == ".obj":
                    # Update header lines
                    lines = []
                    with open(fpath, "r") as f:
                        for line in f.readlines():
                            if line.startswith("v "):
                                break
                            lines.append(line)
                    start = False
                    with open(out_fpath, "r") as f:
                        for line in f.readlines():
                            if line.startswith("v "):
                                start = True
                            if start:
                                lines.append(line)
                    with open(out_fpath, "w+") as f:
                        f.writelines(lines)
            except KeyError:
                # Degenerate mesh, so immediately return
                return None
    return output_suffix


def make_asset_positive(urdf_fpath, output_suffix="mirror"):
    assert urdf_fpath.endswith(".urdf")
    out_lines = []

    with open(urdf_fpath, "r") as f:
        for line in f.readlines():
            # print(line)
            out_line = line
            if "<mesh " in line and "scale=" in line:
                # Grab the scale, and possibly convert negative values
                scale_str = line.split("scale=")[1].split('"')[1]
                scale = _space_string_to_tensor(scale_str)
                if th.any(scale < 0).item():
                    mesh_rel_fpath = line.split("filename=")[1].split('"')[1]
                    base_fpath = f"{os.path.dirname(urdf_fpath)}/"
                    mesh_abs_fpath = f"{base_fpath}{mesh_rel_fpath}"
                    filetype = mesh_abs_fpath.split(".")[-1]
                    mesh_output_suffix = make_mesh_positive(
                        mesh_abs_fpath.split(".")[0], scale.cpu().numpy(), output_suffix
                    )
                    new_mesh_abs_fpath = mesh_abs_fpath.replace(f".{filetype}", f"_{mesh_output_suffix}.{filetype}")
                    new_mesh_rel_fpath = new_mesh_abs_fpath.split(base_fpath)[1]
                    out_line = line.replace(mesh_rel_fpath, new_mesh_rel_fpath).replace(scale_str, "1 1 1")
            out_lines.append(out_line)

    # Write to output file
    out_file = urdf_fpath.replace(".urdf", f"_{output_suffix}.urdf")
    with open(out_file, "w+") as f:
        f.writelines(out_lines)

    return out_file


def find_all_prim_children_with_type(prim_type, root_prim):
    """
    Recursively searches children of @root_prim to find all instances of prim that satisfy type @prim_type

    Args:
        prim_type (str): Type of the prim to search
        root_prim (Usd.Prim): Root prim to search

    Returns:
        list of Usd.Prim: All found prims whose prim type includes @prim_type
    """
    found_prims = []
    for child in root_prim.GetChildren():
        if prim_type in child.GetTypeName():
            found_prims.append(child)
        found_prims += find_all_prim_children_with_type(prim_type=prim_type, root_prim=child)

    return found_prims


def simplify_convex_hull(tm, max_vertices=60, max_faces=128):
    """
    Simplifies a convex hull mesh by using quadric edge collapse to reduce the number of faces

    Args:
        tm (Trimesh): Trimesh mesh to simply. Should be convex hull
        max_vertices (int): Maximum number of vertices to generate
    """
    # If number of faces is less than or equal to @max_faces, simply return directly
    if len(tm.vertices) <= max_vertices:
        return tm

    # Use pymeshlab to reduce
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=tm.vertices, face_matrix=tm.faces, v_normals_matrix=tm.vertex_normals))
    while len(ms.current_mesh().vertex_matrix()) > max_vertices:
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=max_faces)
        max_faces -= 2
    vertices_reduced = ms.current_mesh().vertex_matrix()
    faces_reduced = ms.current_mesh().face_matrix()
    vertex_normals_reduced = ms.current_mesh().vertex_normal_matrix()
    return trimesh.Trimesh(
        vertices=vertices_reduced,
        faces=faces_reduced,
        vertex_normals=vertex_normals_reduced,
    ).convex_hull


def generate_collision_meshes(
    trimesh_mesh, method="coacd", hull_count=32, discard_not_volume=True, error_handling=False
):
    """
    Generates a set of collision meshes from a trimesh mesh using CoACD.

    Args:
        trimesh_mesh (trimesh.Trimesh): The trimesh mesh to generate the collision mesh from.
        method (str): Method to generate collision meshes. Valid options are {"coacd", "convex"}
        hull_count (int): If @method="coacd", this sets the max number of hulls to generate
        discard_not_volume (bool): If @method="coacd" and set to True, this discards any generated hulls
            that are not proper volumes
        error_handling (bool): If true, will run coacd_runner.py and handle the coacd assertion fault by using convex hull instead

    Returns:
        List[trimesh.Trimesh]: The collision meshes.
    """
    # If the mesh is convex or the mesh is a proper volume and similar to its convex hull, simply return that directly
    if trimesh_mesh.is_convex or (
        trimesh_mesh.is_volume and (trimesh_mesh.volume / trimesh_mesh.convex_hull.volume) > 0.90
    ):
        hulls = [trimesh_mesh.convex_hull]

    elif method == "coacd":
        if error_handling:
            # Run CoACD with error handling
            import subprocess
            import sys
            import tempfile
            import pickle
            import os

            # Create separate temp files with proper extensions
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                data_path = f.name
                pickle.dump((trimesh_mesh.vertices, trimesh_mesh.faces, hull_count), f)

            script_path = tempfile.mktemp(suffix=".py")
            result_path = tempfile.mktemp(suffix=".pkl")

            # Run subprocess with clean file paths
            success = (
                subprocess.call(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "coacd_runner.py"), data_path, result_path]
                )
                == 0
            )

            # Process results or fallback
            if success and os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    result = pickle.load(f)

                # Process results as before
                hulls = []
                coacd_vol = 0.0
                for vs, fs in result:
                    hull = trimesh.Trimesh(vertices=vs, faces=fs, process=False)
                    if discard_not_volume and not hull.is_volume:
                        continue
                    hulls.append(hull)
                    coacd_vol += hull.convex_hull.volume

                # Check if we found any valid hulls
                if len(hulls) == 0:
                    print("No valid collision meshes generated, falling back to convex hull")
                    hulls = [trimesh_mesh.convex_hull]
                else:
                    # Compare volume ratios as in original code
                    vol_ratio = coacd_vol / trimesh_mesh.convex_hull.volume
                    if 0.95 < vol_ratio < 1.05:
                        print("MINIMAL CHANGE -- USING CONVEX HULL INSTEAD")
                        hulls = [trimesh_mesh.convex_hull]
            else:
                print("CoACD processing failed, falling back to convex hull")
                hulls = [trimesh_mesh.convex_hull]

            # Clean up temp files
            for path in [data_path, script_path, result_path]:
                if os.path.exists(path):
                    os.remove(path)
        else:
            try:
                import coacd
            except ImportError:
                raise ImportError("Please install the `coacd` package to use this function.")

            # Get the vertices and faces
            coacd_mesh = coacd.Mesh(trimesh_mesh.vertices, trimesh_mesh.faces)

            # Run CoACD with the hull count
            result = coacd.run_coacd(
                coacd_mesh,
                max_convex_hull=hull_count,
                max_ch_vertex=60,
            )

            # Convert the returned vertices and faces to trimesh meshes
            # and assert that they are volumes (and if not, discard them if required)
            hulls = []
            coacd_vol = 0.0
            for vs, fs in result:
                hull = trimesh.Trimesh(vertices=vs, faces=fs, process=False)
                if discard_not_volume and not hull.is_volume:
                    continue
                hulls.append(hull)
                coacd_vol += hull.convex_hull.volume

            # Assert that we got _some_ collision meshes
            assert len(hulls) > 0, "No collision meshes generated!"

            # Compare coacd's generation compared to the original mesh's convex hull
            # If the difference is small (<10% volume difference), simply keep the convex hull
            vol_ratio = coacd_vol / trimesh_mesh.convex_hull.volume
            if 0.95 < vol_ratio < 1.05:
                print("MINIMAL CHANGE -- USING CONVEX HULL INSTEAD")
                # from IPython import embed; embed()
                hulls = [trimesh_mesh.convex_hull]

    elif method == "convex":
        hulls = [trimesh_mesh.convex_hull]

    else:
        raise ValueError(f"Invalid collision mesh generation method specified: {method}")

    # Sanity check all convex hulls
    # For whatever reason, some convex hulls are not true volumes, so we take the convex hull again
    # See https://github.com/mikedh/trimesh/issues/535
    hulls = [hull.convex_hull if not hull.is_volume else hull for hull in hulls]

    # For each hull, simplify so that the complexity is guaranteed to be Omniverse-GPU compatible
    # See https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#collision-settings
    simplified_hulls = [simplify_convex_hull(hull) for hull in hulls]

    return simplified_hulls


def get_collision_approximation_for_urdf(
    urdf_path,
    collision_method="coacd",
    hull_count=32,
    coacd_links=None,
    convex_links=None,
    no_decompose_links=None,
    visual_only_links=None,
    ignore_links=None,
):
    """
    Computes collision approximation for all collision meshes (which are assumed to be non-convex) in
    the given URDF.

    NOTE: This is an in-place operation! It will overwrite @urdf_path

    Args:
        urdf_path (str): Absolute path to the URDF to decompose
        collision_method (str): Default collision method to use. Valid options are: {"coacd", "convex"}
        hull_count (int): Maximum number of convex hulls to decompose individual visual meshes into.
            Only relevant if @collision_method is "coacd"
        coacd_links (None or list of str): If specified, links that should use CoACD to decompose collision meshes
        convex_links (None or list of str): If specified, links that should use convex hull to decompose collision meshes
        no_decompose_links (None or list of str): If specified, links that should not have any special collision
            decomposition applied. This will only use the convex hull
        visual_only_links (None or list of str): If specified, link names corresponding to links that should have
            no collision associated with them (so any pre-existing collisions will be removed!)
        ignore_links (None or list of str): If specified, link names corresponding to links that should be skipped
            during collision generation process
    """
    # Load URDF
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Next, iterate over each visual mesh and define collision meshes for them
    coacd_links = set() if coacd_links is None else set(coacd_links)
    convex_links = set() if convex_links is None else set(convex_links)
    no_decompose_links = set() if no_decompose_links is None else set(no_decompose_links)
    visual_only_links = set() if visual_only_links is None else set(visual_only_links)
    ignore_links = set() if ignore_links is None else set(ignore_links)
    col_mesh_rel_folder = "meshes/collision"
    col_mesh_folder = pathlib.Path(urdf_dir) / col_mesh_rel_folder
    col_mesh_folder.mkdir(exist_ok=True, parents=True)
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        old_cols = link.findall("collision")
        # Completely skip this link if this a link to explicitly skip or we have no collision tags
        if link_name in ignore_links or len(old_cols) == 0:
            continue

        print(f"Generating collision approximation for link {link_name}...")
        generated_new_col = False
        idx = 0
        if link_name not in visual_only_links:
            for vis in link.findall("visual"):
                # Get origin
                origin = vis.find("origin")
                # Check all geometries
                geoms = vis.findall("geometry/*")
                # We should only have a single geom, so assert here
                assert len(geoms) == 1
                # Check whether we actually need to generate a collision approximation
                # No need if the geom type is not a mesh (i.e.: it's a primitive -- so we assume if a collision is already
                # specified, it's that same primitive)
                geom = geoms[0]
                if geom.tag != "mesh":
                    continue
                mesh_path = os.path.join(os.path.dirname(urdf_path), geom.attrib["filename"])
                tm = trimesh.load(mesh_path, force="mesh", process=False)

                if link_name in coacd_links:
                    method = "coacd"
                elif link_name in convex_links:
                    method = "convex"
                elif link_name in no_decompose_links:
                    # Output will just be ignored, so skip
                    continue
                else:
                    method = collision_method
                collision_meshes = generate_collision_meshes(
                    trimesh_mesh=tm,
                    method=method,
                    hull_count=hull_count,
                )
                # Save and merge precomputed collision mesh
                collision_filenames_and_scales = []
                for i, collision_mesh in enumerate(collision_meshes):
                    processed_collision_mesh = collision_mesh.copy()
                    processed_collision_mesh._cache.cache["vertex_normals"] = processed_collision_mesh.vertex_normals
                    collision_filename = f"{link_name}_col_{idx}.obj"

                    # OmniGibson requires unit-bbox collision meshes, so here we do that scaling
                    bounding_box = processed_collision_mesh.bounding_box.extents
                    assert all(
                        x > 0 for x in bounding_box
                    ), f"Bounding box extents are not all positive: {bounding_box}"
                    collision_scale = 1.0 / bounding_box
                    collision_scale_matrix = th.eye(4)
                    collision_scale_matrix[:3, :3] = th.diag(th.as_tensor(collision_scale))
                    processed_collision_mesh.apply_transform(collision_scale_matrix.numpy())
                    processed_collision_mesh.export(col_mesh_folder / collision_filename, file_type="obj")
                    collision_filenames_and_scales.append((collision_filename, 1 / collision_scale))

                    idx += 1

                for collision_filename, collision_scale in collision_filenames_and_scales:
                    collision_xml = ET.SubElement(link, "collision")
                    collision_xml.attrib = {"name": collision_filename.replace(".obj", "")}
                    # Add origin info if defined
                    if origin is not None:
                        collision_xml.append(deepcopy(origin))
                    collision_geometry_xml = ET.SubElement(collision_xml, "geometry")
                    collision_mesh_xml = ET.SubElement(collision_geometry_xml, "mesh")
                    collision_mesh_xml.attrib = {
                        "filename": os.path.join(col_mesh_rel_folder, collision_filename),
                        "scale": " ".join([str(item) for item in collision_scale]),
                    }

                if link_name not in no_decompose_links:
                    generated_new_col = True

        # If we generated a new set of collision meshes, remove the old ones
        if generated_new_col or link_name in visual_only_links:
            for col in old_cols:
                link.remove(col)

    # Save the URDF file
    _save_xmltree_as_urdf(
        root_element=root,
        name=os.path.splitext(os.path.basename(urdf_path))[0],
        dirpath=os.path.dirname(urdf_path),
        unique_urdf=False,
    )


def copy_urdf_to_dataset(
    urdf_path,
    category,
    mdl,
    urdf_dep_paths=None,
    dataset_root=gm.CUSTOM_DATASET_PATH,
    suffix="original",
    overwrite=False,
):
    """
    Copies a URDF file and its dependencies to a structured dataset directory.

    Parameters:
        urdf_path (str): Path to the source URDF file.
        category (str): Category name for organizing the model in the dataset.
        mdl (str): Model identifier/name.
        urdf_dep_paths (list, optional): List of relative paths to URDF dependencies.
            If None, dependencies will be automatically detected. Defaults to None.
        dataset_root (str, optional): Root directory of the dataset.
            Defaults to gm.CUSTOM_DATASET_PATH.
        suffix (str, optional): Suffix to append to the model name in the new URDF.
            Defaults to "original".
        overwrite (bool, optional): Whether to overwrite existing directories.
            If False, raises an assertion error if target directory exists.
            Defaults to False.

    Returns:
        str: Path to the newly created URDF file in the dataset.

    Raises:
        AssertionError: If the target directory already exists and overwrite is False.
    """
    # Create a directory for the object
    obj_dir = pathlib.Path(dataset_root) / "objects" / category / mdl / "urdf"
    if not overwrite:
        assert not obj_dir.exists(), f"Object directory {obj_dir} already exists!"
    obj_dir.mkdir(parents=True, exist_ok=True)

    # Copy over all relevant meshes to new obj directory
    old_urdf_dir = pathlib.Path(os.path.dirname(urdf_path))

    # Load urdf
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Find all mesh paths, and replace them with new obj directory
    # urdf_dep_paths should be relative paths wrt the original URDF path
    new_dirs = set() if urdf_dep_paths is None else set(urdf_dep_paths)
    for mesh_type in ["visual", "collision"]:
        for mesh_element in root.findall(f"link/{mesh_type}/geometry/mesh"):
            mesh_root_dir = mesh_element.attrib["filename"].split("/")[0]
            new_dirs.add(mesh_root_dir)
    for new_dir in new_dirs:
        shutil.copytree(old_urdf_dir / new_dir, obj_dir / new_dir, dirs_exist_ok=overwrite)

    # Export this URDF
    return _save_xmltree_as_urdf(
        root_element=root,
        name=f"{mdl}_{suffix}",
        dirpath=obj_dir,
        unique_urdf=False,
    )


def generate_urdf_for_mesh(
    asset_path,
    obj_dir,
    category,
    mdl,
    collision_method=None,
    hull_count=32,
    up_axis="z",
    scale=1.0,
    check_scale=False,
    rescale=False,
    dataset_root=None,
    overwrite=False,
    n_submesh=10,
):
    """
    Generate URDF file for either single mesh or articulated files.
    Each submesh in articulated files (glb, gltf) will be extracted as a separate link.

    Args:
        asset_path (str): Path to the input mesh file (.obj, .glb, .gltf)
        obj_dir (str): Output directory
        category (str): Category name for the object
        mdl (str): Model name
        collision_method (str or None): Method for generating collision meshes ("convex", "coacd", or None)
        hull_count (int): Maximum number of convex hulls for COACD method
        up_axis (str): Up axis for the model ("y" or "z")
        scale (float): User choice scale, will be overwritten if check_scale and rescale
        check_scale (bool): Whether to check mesh size based on heuristic
        rescale (bool): Whether to rescale mesh if check_scale
        dataset_root (str or None): Root directory for the dataset
        overwrite (bool): Whether to overwrite existing files
        n_submesh (int): If submesh number is more than n_submesh, will not convert and skip
    """

    # Validate file format
    valid_formats = trimesh.available_formats()
    mesh_format = pathlib.Path(asset_path).suffix[1:]  # Remove the dot
    assert mesh_format in valid_formats, f"Invalid mesh format: {mesh_format}. Valid formats: {valid_formats}"
    assert mesh_format in [
        "obj",
        "glb",
        "gltf",
    ], "Not obj, glb, gltf file, can only deal with these file types"

    # Convert obj_dir to Path object
    if isinstance(obj_dir, str):
        obj_dir = pathlib.Path(obj_dir)

    # Create directory structure
    if not overwrite:
        assert not obj_dir.exists(), f"Object directory {obj_dir} already exists!"
    obj_dir.mkdir(parents=True, exist_ok=True)

    obj_name = "_".join([category, mdl])

    # Dictionary to store links with their visual and collision meshes
    links = {}

    # Load and process based on file type
    if mesh_format == "obj":
        # Handle single mesh files with original loading method
        visual_mesh = trimesh.load(asset_path, force="mesh", process=False)
        if isinstance(visual_mesh, list):
            visual_mesh = visual_mesh[0]  # Take first mesh if multiple

        # Generate collision meshes if requested
        collision_meshes = []
        if collision_method is not None:
            collision_meshes = generate_collision_meshes(
                visual_mesh, method=collision_method, hull_count=hull_count, error_handling=True
            )

        # Add to links dictionary as a single link named "base_link"
        links["base_link"] = {"visual_mesh": visual_mesh, "collision_meshes": collision_meshes, "transform": th.eye(4)}

    elif mesh_format in ["glb", "gltf"]:
        # Handle articulated files
        scene = trimesh.load(asset_path)
        # Count geometries (submeshes)
        submesh_count = len(scene.geometry)
        if submesh_count > n_submesh:
            print(f"❌ Submesh count: {submesh_count} > {n_submesh}, skipping")
            return None

        # Get transforms from graph and extract each geometry as a separate link
        link_index = 0
        for node_name in scene.graph.nodes_geometry:
            geometry_name = scene.graph[node_name][1]
            if not isinstance(geometry_name, str):
                print(f"Warning: Skipping node {node_name} with non-string geometry name: {geometry_name}")
                continue

            # Get the geometry and transform
            geometry = scene.geometry[geometry_name]

            transform, _ = scene.graph.get(frame_to=node_name, frame_from=scene.graph.base_frame)
            transform_tensor = th.from_numpy(transform.copy()).float()

            # Process the geometry based on its type
            if isinstance(geometry, trimesh.Trimesh):
                # Create a link name based on the node name or index
                link_name = f"link_{link_index}"
                if node_name and isinstance(node_name, str):
                    # Clean up node name to make it a valid link name
                    link_name = "link_" + "".join(c if c.isalnum() or c == "_" else "_" for c in node_name)

                # Create a copy of the geometry
                visual_mesh = geometry.copy()

                # Generate collision meshes if requested
                collision_meshes = []
                if collision_method is not None:
                    # Create collision meshes based on the original geometry
                    # (not transformed yet - we'll handle transforms at the URDF level)
                    collision_meshes = generate_collision_meshes(
                        geometry,
                        method=collision_method,
                        hull_count=hull_count,
                        discard_not_volume=True,
                        error_handling=True,
                    )

                # Add to links dictionary with original transform
                links[link_name] = {
                    "visual_mesh": visual_mesh,
                    "collision_meshes": collision_meshes,
                    "transform": transform_tensor,
                    "node_name": node_name,
                }
                link_index += 1

            elif isinstance(geometry, (list, tuple)):
                # Handle cases where geometry is a list of meshes
                for i, submesh in enumerate(geometry):
                    if isinstance(submesh, trimesh.Trimesh):
                        # Create a link name
                        link_name = f"link_{link_index}"
                        if node_name and isinstance(node_name, str):
                            link_name = f"link_{node_name}_{i}"

                        # Create a copy of the submesh
                        visual_mesh = submesh.copy()

                        # Generate collision meshes if requested
                        collision_meshes = []

                        if collision_method is not None:
                            # Create collision meshes based on the original geometry
                            collision_meshes = generate_collision_meshes(
                                submesh,
                                method=collision_method,
                                hull_count=hull_count,
                                discard_not_volume=True,
                                error_handling=True,
                            )

                        # Add to links dictionary with original transform
                        links[link_name] = {
                            "visual_mesh": visual_mesh,
                            "collision_meshes": collision_meshes,
                            "transform": transform_tensor,
                            "node_name": f"{node_name}_{i}",
                        }
                        link_index += 1

        if not links:
            print("Warning: No valid meshes found in the scene!")
            print("Scene contents:")
            print(f"Geometries: {scene.geometry}")
            print(f"Graph: {scene.graph}")
            raise ValueError("No valid meshes found in the input file")
    else:
        raise ValueError(f"Unsupported file format: {mesh_format}")

    # Handle rotation for up_axis if needed
    if up_axis == "y":
        rotation_matrix = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        rotation_tensor = th.from_numpy(rotation_matrix).float()

        for link_name, link_data in links.items():
            # Update the transform - we'll apply the actual transforms later
            link_data["transform"] = th.matmul(rotation_tensor, link_data["transform"])

    # Compute new scale if check_scale = True
    new_scale = 1.0

    if check_scale:
        if links:
            # Find the link with the biggest bounding box
            max_bbox_size = [0, 0, 0]
            max_bbox_link = None

            for link_name, link_data in links.items():
                # Apply the transform to get the correct size
                temp_mesh = link_data["visual_mesh"].copy()
                temp_mesh.apply_transform(link_data["transform"].numpy())
                bbox_size = temp_mesh.bounding_box.extents

                # Check if this link has a bigger dimension than the current max
                if any(s > max_s for s, max_s in zip(bbox_size, max_bbox_size)):
                    max_bbox_size = bbox_size
                    max_bbox_link = link_name

            click.echo(f"Largest visual mesh bounding box size: {max_bbox_size} (link: {max_bbox_link})")

            # Check if any dimension is too large (> 100)
            if any(size > 5.0 for size in max_bbox_size):
                if any(size > 50.0 for size in max_bbox_size):
                    if any(size > 500.0 for size in max_bbox_size):
                        new_scale = 0.001
                    else:
                        new_scale = 0.01
                else:
                    new_scale = 0.1

                click.echo(
                    "Warning: The bounding box sounds a bit large. "
                    "We just wanted to confirm this is intentional. You can skip this check by passing check_scale = False."
                )

            # Check if any dimension is too small (< 0.01)
            elif all(size < 0.005 for size in max_bbox_size):
                new_scale = 1000.0
                click.echo(
                    "Warning: The bounding box sounds a bit small. "
                    "We just wanted to confirm this is intentional. You can skip this check by passing check_scale = False."
                )

            else:
                click.echo("Size is reasonable, no scaling")

        else:
            click.echo("Warning: No links found in the file!")
            return None

    # Rescale mesh if rescale= True, else scale based on function input scale
    if rescale:
        click.echo(f"Original scale {scale} be overwrtten to {new_scale}")
        scale = new_scale

    if scale != 1.0:
        click.echo(f"Adjusting scale to {scale}")
        scale_transform = trimesh.transformations.scale_matrix(scale)
        scale_tensor = th.from_numpy(scale_transform).float()

        for link_name, link_data in links.items():
            # Update the transform - we'll apply the actual transforms later
            link_data["transform"] = th.matmul(scale_tensor, link_data["transform"])

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        # Create directory structure for the output
        obj_link_mesh_folder = obj_dir / "shape"
        obj_link_mesh_folder.mkdir(exist_ok=True)
        obj_link_visual_mesh_folder = obj_link_mesh_folder / "visual"
        obj_link_visual_mesh_folder.mkdir(exist_ok=True)
        obj_link_collision_mesh_folder = obj_link_mesh_folder / "collision"
        obj_link_collision_mesh_folder.mkdir(exist_ok=True)
        obj_link_material_folder = obj_dir / "material"
        obj_link_material_folder.mkdir(exist_ok=True)

        # Dictionary to store information for URDF generation
        urdf_links = {}

        # Process each link
        for link_name, link_data in links.items():
            visual_mesh = link_data["visual_mesh"].copy()  # Create a copy to avoid modifying original
            collision_meshes = [mesh.copy() for mesh in link_data["collision_meshes"]]  # Copy all collision meshes
            transform = link_data["transform"]

            # Apply transform to visual mesh before exporting
            visual_mesh.apply_transform(transform.numpy())

            # Export the transformed mesh
            visual_filename = f"{obj_name}_{link_name}.obj"
            visual_temp_path = temp_dir_path / visual_filename
            visual_mesh.export(visual_temp_path, file_type="obj")

            # Check for material files
            material_files = [x for x in temp_dir_path.iterdir() if x.suffix == ".mtl"]
            material_filename = None

            if material_files:
                # Process material file if exists
                material_file = material_files[0]
                material_filename = f"{obj_name}_{link_name}.mtl"

                # Process MTL file (similar to original code)
                with open(visual_temp_path, "r") as f:
                    new_lines = []
                    for line in f.readlines():
                        if f"mtllib {material_file.name}" in line:
                            line = f"mtllib {material_filename}\n"
                        new_lines.append(line)

                with open(visual_temp_path, "w") as f:
                    for line in new_lines:
                        f.write(line)

                # Process texture references in MTL file
                with open(material_file, "r") as f:
                    new_lines = []
                    for line in f.readlines():
                        if "map_" in line:
                            parts = line.split(" ", 1)
                            if len(parts) > 1:
                                map_kind, texture_filename = parts
                                texture_filename = texture_filename.strip()
                                map_kind = map_kind.strip().replace("map_", "")
                                new_filename = f"../../material/{obj_name}_{link_name}_{map_kind}.png"

                                # Copy texture file
                                texture_from_path = temp_dir_path / texture_filename
                                if texture_from_path.exists():
                                    texture_to_path = (
                                        obj_link_material_folder / f"{obj_name}_{link_name}_{map_kind}.png"
                                    )
                                    if not overwrite and texture_to_path.exists():
                                        print(f"Warning: Texture file {texture_to_path} already exists!")
                                    else:
                                        shutil.copy2(texture_from_path, texture_to_path)

                                # Update line
                                line = f"{parts[0]} {new_filename}\n"
                        new_lines.append(line)

                # Write updated MTL file
                with open(obj_link_visual_mesh_folder / material_filename, "w") as f:
                    for line in new_lines:
                        f.write(line)

            # Copy visual mesh to final location
            visual_final_path = obj_link_visual_mesh_folder / visual_filename
            shutil.copy2(visual_temp_path, visual_final_path)

            # Process collision meshes
            collision_info = []
            for i, collision_mesh in enumerate(collision_meshes):
                # Apply transform to collision mesh before exporting
                collision_mesh.apply_transform(transform.numpy())

                # Export collision mesh filename
                collision_filename = visual_filename.replace(".obj", f"_collision_{i}.obj")

                # Scale collision mesh to unit bbox if needed
                bounding_box = collision_mesh.bounding_box.extents
                if all(x > 0 for x in bounding_box):
                    collision_scale = 1.0 / bounding_box
                    collision_scale_matrix = th.eye(4)
                    collision_scale_matrix[:3, :3] = th.diag(th.as_tensor(collision_scale))

                    # Create a copy to avoid modifying the original
                    scaled_collision_mesh = collision_mesh.copy()
                    scaled_collision_mesh.apply_transform(collision_scale_matrix.numpy())

                    # Export collision mesh
                    collision_path = obj_link_collision_mesh_folder / collision_filename
                    scaled_collision_mesh.export(collision_path, file_type="obj")

                    # Since we've already applied the transform, scale includes only the sizing adjustment
                    collision_info.append({"filename": collision_filename, "scale": 1.0 / collision_scale})
                else:
                    print(f"Warning: Skipping collision mesh with invalid bounding box: {bounding_box}")

            # Store information for URDF generation - now without transform since it's been applied
            urdf_links[link_name] = {
                "visual_filename": visual_filename,
                "collision_info": collision_info,
                "transform": th.eye(4),  # Identity transform since we've already applied it to the meshes
            }

    if mesh_format == "obj":
        # Change the link name from "base_link" to "obj_link"
        if "base_link" in urdf_links:
            urdf_links["obj_link"] = urdf_links.pop("base_link")

    # Generate URDF XML
    tree_root = ET.Element("robot")
    tree_root.attrib = {"name": mdl}

    # Create a base_link as the root
    base_link = ET.SubElement(tree_root, "link")
    base_link.attrib = {"name": "base_link"}

    # Add all other links and joints to connect them to the base_link
    for link_name, link_info in urdf_links.items():
        # Create link element
        link_xml = ET.SubElement(tree_root, "link")
        link_xml.attrib = {"name": link_name}

        # Add visual geometry
        visual_xml = ET.SubElement(link_xml, "visual")
        visual_origin_xml = ET.SubElement(visual_xml, "origin")
        visual_origin_xml.attrib = {"xyz": "0 0 0", "rpy": "0 0 0"}  # Zero transform since already applied
        visual_geometry_xml = ET.SubElement(visual_xml, "geometry")
        visual_mesh_xml = ET.SubElement(visual_geometry_xml, "mesh")
        visual_mesh_xml.attrib = {
            "filename": os.path.join("shape", "visual", link_info["visual_filename"]).replace("\\", "/"),
            "scale": "1 1 1",  # Using 1.0 scale since transform already applied
        }

        # Add collision geometries
        for i, collision in enumerate(link_info["collision_info"]):
            collision_xml = ET.SubElement(link_xml, "collision")
            collision_xml.attrib = {"name": f"{link_name}_collision_{i}"}
            collision_origin_xml = ET.SubElement(collision_xml, "origin")
            collision_origin_xml.attrib = {"xyz": "0 0 0", "rpy": "0 0 0"}  # Zero transform since already applied
            collision_geometry_xml = ET.SubElement(collision_xml, "geometry")
            collision_mesh_xml = ET.SubElement(collision_geometry_xml, "mesh")
            collision_mesh_xml.attrib = {
                "filename": os.path.join("shape", "collision", collision["filename"]).replace("\\", "/"),
                "scale": " ".join(str(item) for item in collision["scale"]),
            }

        # Create a joint to connect this link to the base_link
        joint_xml = ET.SubElement(tree_root, "joint")
        joint_xml.attrib = {"name": f"{link_name}_joint", "type": "fixed"}

        # Set parent and child links
        parent_xml = ET.SubElement(joint_xml, "parent")
        parent_xml.attrib = {"link": "base_link"}
        child_xml = ET.SubElement(joint_xml, "child")
        child_xml.attrib = {"link": link_name}

        # Set origin for the joint with zeros since transform was applied to meshes
        joint_origin_xml = ET.SubElement(joint_xml, "origin")
        joint_origin_xml.attrib = {"xyz": "0 0 0", "rpy": "0 0 0"}

    # Save URDF file
    xmlstr = minidom.parseString(ET.tostring(tree_root)).toprettyxml(indent="   ")
    xmlio = io.StringIO(xmlstr)
    tree = ET.parse(xmlio)

    urdf_path = obj_dir / f"{mdl}.urdf"
    with open(urdf_path, "wb") as f:
        tree.write(f, xml_declaration=True)

    return str(urdf_path)


def record_obj_metadata_from_urdf(urdf_path, obj_dir, joint_setting="zero", overwrite=False):
    """
    Records object metadata and writes it to misc/metadata.json within the object directory.

    Args:
        urdf_path (str): Path to object URDF
        obj_dir (str): Absolute path to the object's root directory
        joint_setting (str): Setting for joints when calculating canonical metadata. Valid options
            are {"low", "zero", "high"} (i.e.: lower joint limit, all 0 values, or upper joint limit)
        overwrite (bool): Whether to overwrite any pre-existing data
    """
    # Load the URDF file into urdfpy
    robot = URDF.load(urdf_path)

    # Do FK with everything at desired configuration
    if joint_setting == "zero":
        val = lambda jnt: 0.0
    elif joint_setting == "low":
        val = lambda jnt: jnt.limit.lower
    elif joint_setting == "high":
        val = lambda jnt: jnt.limit.upper
    else:
        raise ValueError(f"Got invalid joint_setting: {joint_setting}! Valid options are ['low', 'zero', 'high']")
    joint_cfg = {joint.name: val(joint) for joint in robot.joints if joint.joint_type in ("prismatic", "revolute")}
    vfk = robot.visual_trimesh_fk(cfg=joint_cfg)

    scene = trimesh.Scene()
    for mesh, transform in vfk.items():
        scene.add_geometry(geometry=mesh, transform=transform)

    # Calculate relevant metadata

    # Base link offset is pos offset from robot root link -> overall AABB center
    # Since robot is placed at origin, this is simply the AABB centroid
    base_link_offset = scene.bounding_box.centroid

    # BBox size is simply the extent of the overall AABB
    bbox_size = scene.bounding_box.extents

    # Save metadata json
    out_metadata = {
        "meta_links": {},
        "link_tags": {},
        "object_parts": [],
        "base_link_offset": base_link_offset.tolist(),
        "bbox_size": bbox_size.tolist(),
        "orientations": [],
    }
    misc_dir = pathlib.Path(obj_dir) / "misc"
    misc_dir.mkdir(exist_ok=overwrite)
    with open(misc_dir / "metadata.json", "w") as f:
        json.dump(out_metadata, f)


def import_og_asset_from_urdf(
    category,
    model,
    urdf_path=None,
    urdf_dep_paths=None,
    collision_method="coacd",
    coacd_links=None,
    convex_links=None,
    no_decompose_links=None,
    visual_only_links=None,
    merge_fixed_joints=False,
    dataset_root=gm.CUSTOM_DATASET_PATH,
    hull_count=32,
    overwrite=False,
    use_usda=False,
):
    """
    Imports an asset from URDF format into OmniGibson-compatible USD format. This will write the new USD
    (and copy the URDF if it does not already exist within @dataset_root) to @dataset_root

    Args:
        category (str): Category to assign to imported asset
        model (str): Model name to assign to imported asset
        urdf_path (None or str): If specified, external URDF that should be copied into the dataset first before
            converting into USD format. Otherwise, assumes that the urdf file already exists within @dataset_root dir
        urdf_dep_paths (None or list of str): If specified, relative paths to the @urdf_path directory that should be copied
            over to the custom dataset, e.g., relevant material directories
        collision_method (None or str): If specified, collision decomposition method to use to generate
            OmniGibson-compatible collision meshes. Valid options are {"coacd", "convex"}
        coacd_links (None or list of str): If specified, links that should use CoACD to decompose collision meshes
        convex_links (None or list of str): If specified, links that should use convex hull to decompose collision meshes
        no_decompose_links (None or list of str): If specified, links that should not have any special collision
            decomposition applied. This will only use the convex hull
        visual_only_links (None or list of str): If specified, links that should have no colliders associated with it
        merge_fixed_joints (bool): Whether to merge fixed joints or not
        dataset_root (str): Dataset root directory to use for writing imported USD file. Default is custom dataset
            path set from the global macros
        hull_count (int): Maximum number of convex hulls to decompose individual visual meshes into.
            Only relevant if @collision_method is "coacd"
        overwrite (bool): If set, will overwrite any pre-existing files
        use_usda (bool): If set, will write files to .usda files instead of .usd
            (bigger memory footprint, but human-readable)

    Returns:
        3-tuple:
            - str: Absolute path to post-processed URDF file
            - str: Absolute path to generated USD file
            - Usd.Prim: Generated root USD prim (currently on active stage)
    """
    # If URDF already exists, write it to the dataset
    if urdf_path is not None:
        print(f"Copying URDF to dataset root {dataset_root}...")
        urdf_path = copy_urdf_to_dataset(
            urdf_path=urdf_path,
            category=category,
            mdl=model,
            urdf_dep_paths=urdf_dep_paths,
            dataset_root=dataset_root,
            suffix="original",
            overwrite=overwrite,
        )
    else:
        # Verify that the object exists at the expected location
        # This is <dataset_root>/objects/<category>/<model>/urdf/<model>_original.urdf
        urdf_path = os.path.join(dataset_root, "objects", category, model, "urdf", f"{model}_original.urdf")
        assert os.path.exists(urdf_path), f"Expected urdf at dataset location {urdf_path}, but none was found!"

    # Make sure all scaling is positive
    model_dir = os.path.join(dataset_root, "objects", category, model)
    urdf_path = make_asset_positive(urdf_fpath=urdf_path)

    # Update collisions if requested
    if collision_method is not None:
        print("Generating collision approximation for URDF...")
        get_collision_approximation_for_urdf(
            urdf_path=urdf_path,
            collision_method=collision_method,
            hull_count=hull_count,
            coacd_links=coacd_links,
            convex_links=convex_links,
            no_decompose_links=no_decompose_links,
            visual_only_links=visual_only_links,
        )

    # Generate metadata
    print("Recording object metadata from URDF...")
    record_obj_metadata_from_urdf(
        urdf_path=urdf_path,
        obj_dir=model_dir,
        joint_setting="zero",
        overwrite=overwrite,
    )

    # Convert to USD
    print("Converting obj URDF to USD...")
    og.launch()
    assert len(og.sim.scenes) == 0
    urdf_path, usd_path = import_obj_urdf(
        urdf_path=urdf_path,
        obj_category=category,
        obj_model=model,
        dataset_root=dataset_root,
        use_omni_convex_decomp=False,  # We already pre-decomposed the values, so don' use omni convex decomp
        use_usda=use_usda,
        merge_fixed_joints=merge_fixed_joints,
    )

    # Copy meta links URDF to original name of object model
    shutil.copy2(urdf_path, os.path.join(dataset_root, "objects", category, model, "urdf", f"{model}.urdf"))

    prim = import_obj_metadata(
        usd_path=usd_path,
        obj_category=category,
        obj_model=model,
        dataset_root=dataset_root,
        import_render_channels=False,  # TODO: Make this True once we find a systematic / robust way to import materials of different source formats
    )
    print(
        f"\nConversion complete! Object has been successfully imported into OmniGibson-compatible USD, located at:\n\n{usd_path}\n"
    )

    return urdf_path, usd_path, prim
