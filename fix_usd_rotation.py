"""
Fix the converted USD by setting +90° X-axis rotation on each link's "visuals" Xform.

Root cause: Isaac Sim's URDF importer stores mesh vertices in Y-up (URDF/ROS convention)
but OmniGibson's renderer expects Z-up (USD convention). The working FrankaMounted USD
has pre-rotated vertex data; ours doesn't. Rather than modifying millions of vertices,
we set a rotation transform on each visuals Xform.

The rotation quaternion for +90° around X (Y-up → Z-up):
  w = cos(45°) = 0.7071068
  x = sin(45°) = 0.7071068
  y = 0, z = 0
  => (0.7071068, 0.7071068, 0, 0)

Handles two cases:
  A) visuals Xform has NO existing xformOps → insert rotation block
  B) visuals Xform has identity xformOps (orient = (1,0,0,0)) → replace with rotation

Usage:
    python fix_usd_rotation.py
"""
import re
import os

USD_PATH = "/home/robot/Desktop/BEHAVIOR-1K/datasets/objects/robot/franka_mounted_sharpa_right/usd/franka_mounted_sharpa_right.usda"

ROTATION_QUAT = "(0.7071068, 0.7071068, 0, 0)"
IDENTITY_QUAT = "(1, 0, 0, 0)"

XFORM_BLOCK = """\
            quatd xformOp:orient = {rotation}
            double3 xformOp:scale = (1, 1, 1)
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
""".format(rotation=ROTATION_QUAT)


def main():
    print(f"Reading: {USD_PATH}")
    with open(USD_PATH, "r") as f:
        content = f.read()

    orig_size = len(content)
    print(f"File size: {orig_size / 1024 / 1024:.1f} MB")

    total_fixed = 0

    # --- Case B: visuals Xforms with existing identity xformOps ---
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
        print(f"  Replaced identity → rotation on {count_replaced} visuals Xforms")
        total_fixed += count_replaced

    # --- Case A: visuals Xforms with NO existing xformOps ---
    pattern = r'(        def Xform "visuals"\n        \{\n)(?!            quatd xformOp:orient)'
    replacement = r'\g<1>' + XFORM_BLOCK + '\n'
    content, count_inserted = re.subn(pattern, replacement, content)
    if count_inserted > 0:
        print(f"  Inserted rotation block into {count_inserted} visuals Xforms")
        total_fixed += count_inserted

    print(f"Total visuals Xforms fixed: {total_fixed}")

    if total_fixed == 0:
        already_done = content.count(
            'def Xform "visuals"\n        {\n            quatd xformOp:orient = ' + ROTATION_QUAT
        )
        if already_done > 0:
            print(f"All {already_done} visuals already have rotation (nothing to do)")
        else:
            print("WARNING: No visuals Xforms found! Check indentation.")
        return

    with open(USD_PATH, "w") as f:
        content_new = content
        f.write(content_new)

    new_size = len(content)
    print(f"Written: {new_size / 1024 / 1024:.1f} MB (delta: {new_size - orig_size:+d} bytes)")
    print(f"\nDone! Now re-run: conda run -n behavior python test_sharpa_robot.py")


if __name__ == "__main__":
    main()
