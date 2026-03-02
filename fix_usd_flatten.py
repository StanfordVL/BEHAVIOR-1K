"""
Fix the converted USD by flattening references so visual mesh data is inlined.
Uses OmniGibson's simulator initialization to access the pxr module.

Usage:
    conda run -n behavior python fix_usd_flatten.py
"""
import os
import sys
import shutil

USD_PATH = "/home/robot/Desktop/BEHAVIOR-1K/datasets/objects/robot/franka_mounted_sharpa_right/usd/franka_mounted_sharpa_right.usda"
BACKUP_PATH = USD_PATH.replace(".usda", "_original.usda")


def main():
    print(f"Target USD: {USD_PATH}")

    # Backup the original
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(USD_PATH, BACKUP_PATH)
        print(f"Backed up to: {BACKUP_PATH}")

    # Initialize OmniGibson (which starts Isaac Sim and makes pxr available)
    print("Starting OmniGibson to access pxr...")
    import omnigibson as og
    from omnigibson.macros import gm
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = False

    # Create a minimal environment just to initialize the simulator
    og.launch()

    # Now pxr should be available
    from pxr import Usd, UsdGeom, Sdf, Gf
    print("pxr imported successfully!")

    # Open the stage
    stage = Usd.Stage.Open(USD_PATH)
    if not stage:
        print("ERROR: Could not open stage!")
        sys.exit(1)

    default_prim = stage.GetDefaultPrim()
    print(f"Default prim: {default_prim.GetName()}")

    root_prims = [p for p in stage.GetPseudoRoot().GetChildren()]
    print(f"Top-level prims: {[p.GetName() for p in root_prims]}")

    # Flatten the stage
    print("Flattening stage (this may take a moment for large files)...")
    flattened_layer = stage.Flatten()
    print("Flatten complete.")

    # Open flattened layer as a stage to clean up
    flat_stage = Usd.Stage.Open(flattened_layer)

    # Remove the visuals and colliders scopes (now redundant)
    for prim in list(flat_stage.GetPseudoRoot().GetChildren()):
        name = prim.GetName()
        if name in ("visuals", "colliders"):
            print(f"Removing redundant scope: /{name}")
            flat_stage.RemovePrim(prim.GetPath())

    # Add xformOps to root prim if missing
    flat_default = flat_stage.GetDefaultPrim()
    if flat_default and flat_default.IsValid():
        xformable = UsdGeom.Xformable(flat_default)
        if not xformable.GetOrderedXformOps():
            print("Adding xformOps to root prim")
            translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
            orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
            scale_op = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
            translate_op.Set(Gf.Vec3d(0, 0, 0))
            orient_op.Set(Gf.Quatd(1, 0, 0, 0))
            scale_op.Set(Gf.Vec3d(1, 1, 1))

    # Export
    print(f"Exporting to: {USD_PATH}")
    flattened_layer.Export(USD_PATH)

    orig_size = os.path.getsize(BACKUP_PATH)
    new_size = os.path.getsize(USD_PATH)
    print(f"Original: {orig_size / 1024 / 1024:.1f} MB -> Flattened: {new_size / 1024 / 1024:.1f} MB")
    print("\nDone! Now re-run: python test_sharpa_robot.py")

    # Shut down
    og.shutdown()


if __name__ == "__main__":
    main()
