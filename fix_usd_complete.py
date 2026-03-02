"""
Complete USD fix: flatten references + add correct Y-up → Z-up rotation.

Step 1: Flatten the USD (uses OmniGibson/Isaac Sim to access pxr)
Step 2: Add +90° X-axis rotation to each link's visuals Xform

The +90° X rotation quaternion (w,x,y,z) = (0.7071068, 0.7071068, 0, 0)
converts mesh vertices from Y-up (URDF) to Z-up (USD).

Usage:
    conda run -n behavior python fix_usd_complete.py
"""
import os
import sys
import re
import shutil

USD_PATH = "/home/robot/Desktop/BEHAVIOR-1K/datasets/objects/robot/franka_mounted_sharpa_right/usd/franka_mounted_sharpa_right.usda"
BACKUP_PATH = USD_PATH.replace(".usda", "_original.usda")

XFORM_BLOCK = """\
            quatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)
            double3 xformOp:scale = (1, 1, 1)
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
"""


def step1_flatten():
    """Flatten the USD to inline all mesh data (removes references)."""
    print("=== STEP 1: Flatten USD ===")
    print("Starting OmniGibson to access pxr...")

    import omnigibson as og
    from omnigibson.macros import gm
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = False

    og.launch()

    from pxr import Usd, UsdGeom, Sdf, Gf
    print("pxr imported successfully!")

    stage = Usd.Stage.Open(USD_PATH)
    if not stage:
        print("ERROR: Could not open stage!")
        sys.exit(1)

    print(f"Default prim: {stage.GetDefaultPrim().GetName()}")

    print("Flattening stage...")
    flattened_layer = stage.Flatten()
    print("Flatten complete.")

    # Add xformOps to root prim if missing
    flat_stage = Usd.Stage.Open(flattened_layer)
    flat_default = flat_stage.GetDefaultPrim()
    if flat_default and flat_default.IsValid():
        xformable = UsdGeom.Xformable(flat_default)
        if not xformable.GetOrderedXformOps():
            print("Adding xformOps to root prim")
            t = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
            o = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
            s = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
            t.Set(Gf.Vec3d(0, 0, 0))
            o.Set(Gf.Quatd(1, 0, 0, 0))
            s.Set(Gf.Vec3d(1, 1, 1))

    print(f"Exporting flattened USD to: {USD_PATH}")
    flattened_layer.Export(USD_PATH)

    orig_size = os.path.getsize(BACKUP_PATH)
    new_size = os.path.getsize(USD_PATH)
    print(f"Original: {orig_size / 1024 / 1024:.1f} MB -> Flattened: {new_size / 1024 / 1024:.1f} MB")

    og.shutdown()
    print("Step 1 done.\n")


def step2_add_rotation():
    """Add +90° X rotation to each link's visuals Xform.

    Handles two cases produced by different URDF converters:
      A) visuals Xform has NO existing xformOps  → insert the rotation block
      B) visuals Xform ALREADY has identity xformOps (orient = (1,0,0,0))
         → replace the identity orient with the rotation quaternion

    Without handling case B, the identity orient overwrites the inserted
    rotation (USD uses last-wins for duplicate attributes), leaving
    SharpaWave hand meshes in Y-up and rendering as invisible slivers.
    """
    print("=== STEP 2: Add Y-up → Z-up rotation ===")

    ROTATION_QUAT = "(0.7071068, 0.7071068, 0, 0)"
    IDENTITY_QUAT = "(1, 0, 0, 0)"

    with open(USD_PATH, "r") as f:
        content = f.read()

    orig_size = len(content)
    total_fixed = 0

    # --- Case B: visuals Xforms that already have identity xformOps ---
    # Replace identity orient with rotation orient inside visuals blocks.
    # Pattern: 'def Xform "visuals"\n        {\n            quatd xformOp:orient = (1, 0, 0, 0)'
    old_identity = (
        '        def Xform "visuals"\n'
        '        {\n'
        f'            quatd xformOp:orient = {IDENTITY_QUAT}'
    )
    new_rotated = (
        '        def Xform "visuals"\n'
        '        {\n'
        f'            quatd xformOp:orient = {ROTATION_QUAT}'
    )
    count_replaced = content.count(old_identity)
    if count_replaced > 0:
        content = content.replace(old_identity, new_rotated)
        print(f"  Replaced identity orient → rotation on {count_replaced} visuals Xforms (had existing xformOps)")
        total_fixed += count_replaced

    # --- Case A: visuals Xforms with NO existing xformOps ---
    # Insert a full xformOps block after the opening brace.
    # Only matches visuals that don't already start with 'quatd xformOp:orient' after '{'.
    pattern = r'(        def Xform "visuals"\n        \{\n)(?!            quatd xformOp:orient)'
    replacement = r'\g<1>' + XFORM_BLOCK + '\n'
    content, count_inserted = re.subn(pattern, replacement, content)
    if count_inserted > 0:
        print(f"  Inserted rotation block into {count_inserted} visuals Xforms (had no xformOps)")
        total_fixed += count_inserted

    # --- Safety: check for any visuals still missing rotation ---
    count_still_identity = content.count(
        'def Xform "visuals"\n        {\n            quatd xformOp:orient = ' + IDENTITY_QUAT
    )
    if count_still_identity > 0:
        print(f"  WARNING: {count_still_identity} visuals Xforms still have identity orient!")

    print(f"  Total visuals Xforms fixed: {total_fixed}")

    if total_fixed == 0:
        # Check if they're already all rotated (idempotent re-run)
        already_done = content.count(
            'def Xform "visuals"\n        {\n            quatd xformOp:orient = ' + ROTATION_QUAT
        )
        if already_done > 0:
            print(f"  All {already_done} visuals Xforms already have rotation (nothing to do)")
            return True
        print("WARNING: No visuals Xforms found to fix!")
        return False

    with open(USD_PATH, "w") as f:
        f.write(content)

    new_size = len(content)
    print(f"Size: {orig_size / 1024 / 1024:.1f} MB -> {new_size / 1024 / 1024:.1f} MB")
    print("Step 2 done.\n")
    return True


def main():
    print(f"Target USD: {USD_PATH}")
    print(f"Backup:     {BACKUP_PATH}\n")

    # Step 1: Flatten
    step1_flatten()

    # Step 2: Rotation
    if step2_add_rotation():
        # Verify
        with open(USD_PATH, "r") as f:
            content = f.read()
        n_rot = content.count("0.7071068, 0.7071068, 0, 0")
        n_vis = content.count('def Xform "visuals"')
        n_ref = content.count("prepend references")
        print(f"=== Verification ===")
        print(f"  Rotation transforms: {n_rot}")
        print(f"  Visual Xforms:       {n_vis}")
        print(f"  References remaining: {n_ref}")
        print(f"\nAll done! Re-run: conda run -n behavior python test_sharpa_robot.py")
    else:
        print("Fix failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
