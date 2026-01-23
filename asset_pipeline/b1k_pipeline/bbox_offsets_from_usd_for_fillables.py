from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import pathlib
import tempfile
from cryptography.fernet import Fernet
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import fs.path
from fs.zipfs import ZipFS

FILLABLE_DIR = pathlib.Path(r"D:\fillable-10-21")
KEY_PATH = pathlib.Path(
    r"C:\Users\cgokmen\research\OmniGibson\omnigibson\data\omnigibson.key"
)
PIPELINE_ROOT = pathlib.Path(r"D:\BEHAVIOR-1K\asset_pipeline")
MODE = "JSON"  # use one of "ATTRIBUTE", "USD" or "JSON"


def _set_xform_properties(prim, pos, quat):
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
        xform_op_scale = xformable.AddXformOp(
            UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, ""
        )
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
        xform_op_rot = xformable.AddXformOp(
            UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, ""
        )
    else:
        xform_op_rot = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
    xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])

    position = Gf.Vec3d(*pos.tolist())
    xform_op_translate.Set(position)

    orientation = quat[[3, 0, 1, 2]].tolist()
    if xform_op_rot.GetTypeName() == "quatf":
        rotq = Gf.Quatf(*orientation)
    else:
        rotq = Gf.Quatd(*orientation)
    xform_op_rot.Set(rotq)


def decrypt_file(encrypted_filename, decrypted_filename):
    with open(KEY_PATH, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
    """
    Compute Bounding Box using ComputeWorldBound at UsdGeom.Imageable
    See https://openusd.org/release/api/class_usd_geom_imageable.html

    Args:
        prim: A prim to compute the bounding box.
    Returns:
        A range (i.e. bounding box), see more at: https://openusd.org/release/api/class_gf_range3d.html
    """
    imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default()  # The time at which we compute the bounding box
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    return np.array(bound_range.GetMin()), np.array(bound_range.GetMax())


def keep_paths_to_all_visuals(stage):
    """
    Keeps only the paths to all prims named 'visuals' and their ancestors.

    Args:
        stage (Usd.Stage): The USD stage to modify.
    """

    # Find all prims named 'visuals'
    def find_visuals_prims(prim, visuals_paths):
        for child in prim.GetChildren():
            if child.GetName() == "visuals":
                visuals_paths.add(child.GetPath())
            find_visuals_prims(child, visuals_paths)

    visuals_paths = set()
    root_prim = stage.GetPseudoRoot()
    find_visuals_prims(root_prim, visuals_paths)

    if not visuals_paths:
        print("No prims named 'visuals' found.")
        return

    # Collect paths to keep (visuals and their ancestors)
    paths_to_keep = set()
    for path in visuals_paths:
        current_path = path
        while current_path != Sdf.Path.emptyPath:
            paths_to_keep.add(current_path)
            current_path = current_path.GetParentPath()

    # Traverse and prune
    def traverse_and_prune(prim):
        for child in prim.GetChildren():
            traverse_and_prune(child)

        # Remove the prim if its path is not in paths_to_keep
        if prim.GetPath() not in paths_to_keep:
            stage.RemovePrim(prim.GetPath())

    traverse_and_prune(root_prim)


def get_bounding_box_from_usd(input_usd, rotmat, tempdir):
    encrypted_filename = input_usd
    fd, decrypted_filename = tempfile.mkstemp(suffix=".usd", dir=tempdir)
    os.close(fd)
    decrypt_file(encrypted_filename, decrypted_filename)
    stage = Usd.Stage.Open(str(decrypted_filename))
    prim = stage.GetDefaultPrim()
    keep_paths_to_all_visuals(stage)
    stage.Save()

    if MODE == "ATTRIBUTE":
        base_link_size = np.array(prim.GetAttribute("ig:nativeBB").Get()) * 1000
        base_link_offset = np.array(prim.GetAttribute("ig:offsetBaseLink").Get()) * 1000

        # THIS IS INACCURATE FOR NON-AA ROTATIONS!
        return (rotmat @ base_link_offset).tolist(), (rotmat @ base_link_size).tolist()

    elif MODE == "USD":
        # Rotate the object by the rotmat and get the bounding box
        _set_xform_properties(prim, np.zeros(3), R.from_matrix(rotmat).as_quat())
        bb_min, bb_max = compute_bbox(prim)
        center = ((bb_min + bb_max) / 2) * 1000
        size = (bb_max - bb_min) * 1000
        return center.tolist(), size.tolist()

    elif MODE == "JSON":
        json_path = input_usd.parent.parent / "bbox.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        size = np.array(data["bbox_extents"]) * 1000
        center = (np.array(data["bbox_center"]) - np.array(data["base_pos"])) * 1000
        return center.tolist(), size.tolist()

    raise ValueError("Invalid mode.")


def main():
    input_usds = list(FILLABLE_DIR.glob("objects/*/*/usd/*.usd"))
    input_usds.sort(key=lambda x: x.parts[-3])
    print(len(input_usds))

    # Scale up
    futures = {}
    with tempfile.TemporaryDirectory() as tempdir:
        with ProcessPoolExecutor() as executor:
            for input_usd in tqdm(input_usds, desc="Queueing up jobs"):
                mdl = input_usd.parts[-3]
                rot = R.identity()
                euler = np.rad2deg(rot.as_euler("xyz"))
                if not np.allclose(np.remainder(np.abs(euler), 90), 0, atol=5):
                    print("Non-axis aligned rotation detected for", mdl, euler.tolist())
                future = executor.submit(
                    get_bounding_box_from_usd, input_usd, rot.as_matrix(), tempdir
                )
                futures[future] = input_usd

            # Gather the results (with a tqdm progress bar)
            results = {}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing results"
            ):
                input_usd = futures[future]
                results[input_usd.parts[-3]] = future.result()

    with open(FILLABLE_DIR / "fillable_bboxes.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
