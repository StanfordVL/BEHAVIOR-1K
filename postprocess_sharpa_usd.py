#!/usr/bin/env python3
"""
Post-process a converted Franka+SharpaWave USD for OmniGibson rendering.

This script is run AFTER the standard BEHAVIOR/OmniGibson URDF-to-USD
conversion pipeline (import_custom_robot.py / asset_conversion_utils.py).
It applies fixes that are specific to the converted output, without
modifying any BEHAVIOR codebase files.

*** REQUIRES Isaac Sim (conda run -n behavior) for the flatten step ***

What it fixes:

  1. FLATTEN REFERENCES (requires Isaac Sim / pxr)
     The URDF-to-USD converter produces a USDA where link visuals use
     `prepend references = </visuals/...>` pointing to a top-level scope
     with `visibility = "invisible"`.  OmniGibson can't render these.
     This step uses pxr's stage.Flatten() to inline all mesh data and
     remove the hidden reference scopes.

  2. SHARED-MESH DEDUPLICATION FIX (plain Python)
     After flattening, shared STL meshes (e.g. right_PP_visual.STL used
     by index, middle, ring, and pinky) result in empty visuals Xforms
     for all but the first link.  This step copies mesh data from the
     populated source link into the empty targets.

  3. VISUALS XFORMOPS FIX (plain Python)
     After flattening, xformOps on the visuals Xform blocks are lost.
     ARM links (panda_*) need a Y-up to Z-up rotation quaternion
     (0.707, 0.707, 0, 0) and HAND links need identity (1, 0, 0, 0).

Usage:
    # Fix right hand
    conda run -n behavior python postprocess_sharpa_usd.py --hand right

    # Fix left hand
    conda run -n behavior python postprocess_sharpa_usd.py --hand left

    # Fix both hands
    conda run -n behavior python postprocess_sharpa_usd.py --hand both

    # Skip flatten if already done (only run the mesh dedup fix)
    python postprocess_sharpa_usd.py --hand left --skip-flatten

This script is idempotent -- re-running it on an already-fixed file is safe.
"""

import argparse
import os
import re
import sys
from collections import OrderedDict
from xml.etree import ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default paths produced by merge_franka_sharpa.py and import_custom_robot.py
HAND_CONFIGS = {
    "right": {
        "usd": os.path.join(
            SCRIPT_DIR,
            "datasets/objects/robot/franka_mounted_sharpa_right/usd/franka_mounted_sharpa_right.usda",
        ),
        "urdf": os.path.join(
            SCRIPT_DIR,
            "datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_right/urdf/franka_mounted_sharpa_right.urdf",
        ),
    },
    "left": {
        "usd": os.path.join(
            SCRIPT_DIR,
            "datasets/objects/robot/franka_mounted_sharpa_left/usd/franka_mounted_sharpa_left.usda",
        ),
        "urdf": os.path.join(
            SCRIPT_DIR,
            "datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_left/urdf/franka_mounted_sharpa_left.urdf",
        ),
    },
}


# ============================================================================
# Step 1: Flatten USD references (requires Isaac Sim / pxr)
# ============================================================================

def needs_flatten(usd_path):
    """Check if the USDA still has unresolved visual references."""
    with open(usd_path, "r") as f:
        # Read just the first portion to check for references
        # (don't need to read the whole 150MB+ file)
        chunk = f.read(500_000)
    return "prepend references = </visuals/" in chunk


def flatten_usd(usd_path):
    """
    Flatten a USDA file: inline all referenced mesh data and remove hidden scopes.

    This replaces `prepend references = </visuals/...>` with actual inline
    mesh data, and removes the now-redundant top-level /visuals and /colliders
    scopes.

    Requires Isaac Sim (pxr module) -- must be run in conda environment.

    Returns:
        bool: True on success, False on error
    """
    if not os.path.exists(usd_path):
        print(f"  ERROR: USD not found: {usd_path}")
        return False

    # Check if flattening is needed
    if not needs_flatten(usd_path):
        print("  Already flattened (no visual references found). Skipping.")
        return True

    print("  Importing OmniGibson/pxr for USD flattening...")
    try:
        import omnigibson as og
        from omnigibson.macros import gm
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_FLATCACHE = False

        # Launch OmniGibson to make pxr available
        if og.sim is None or len(og.sim.scenes) == 0:
            og.launch()

        from pxr import Usd, UsdGeom, Gf
    except ImportError as e:
        print(f"  ERROR: Cannot import pxr. Run with: conda run -n behavior python postprocess_sharpa_usd.py")
        print(f"  Details: {e}")
        return False

    orig_size = os.path.getsize(usd_path)

    # Open the stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"  ERROR: Could not open USD stage: {usd_path}")
        return False

    default_prim = stage.GetDefaultPrim()
    print(f"  Default prim: {default_prim.GetName()}")
    print(f"  Flattening stage (inlining all references)...")

    flattened_layer = stage.Flatten()
    print(f"  Flatten complete.")

    # Open flattened layer to clean up
    flat_stage = Usd.Stage.Open(flattened_layer)

    # Remove the now-redundant visuals and colliders scopes
    for prim in list(flat_stage.GetPseudoRoot().GetChildren()):
        name = prim.GetName()
        if name in ("visuals", "colliders"):
            print(f"  Removing redundant scope: /{name}")
            flat_stage.RemovePrim(prim.GetPath())

    # Add xformOps to root prim if missing
    flat_default = flat_stage.GetDefaultPrim()
    if flat_default and flat_default.IsValid():
        xformable = UsdGeom.Xformable(flat_default)
        if not xformable.GetOrderedXformOps():
            print("  Adding xformOps to root prim")
            t = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
            o = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
            s = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
            t.Set(Gf.Vec3d(0, 0, 0))
            o.Set(Gf.Quatd(1, 0, 0, 0))
            s.Set(Gf.Vec3d(1, 1, 1))

    # Export
    print(f"  Exporting flattened USD to: {usd_path}")
    flattened_layer.Export(usd_path)

    new_size = os.path.getsize(usd_path)
    print(f"  Size: {orig_size / 1024 / 1024:.1f} MB -> {new_size / 1024 / 1024:.1f} MB")
    return True


# ============================================================================
# Step 2: Shared-mesh deduplication fix (plain Python, no Isaac Sim needed)
# ============================================================================

def get_shared_visual_groups(urdf_path):
    """
    Parse a merged URDF to find groups of links sharing the same visual mesh.

    Returns:
        dict: {source_link_name: [target_link_name, ...]}
              Source is the first link in URDF order that uses each shared mesh.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_visual_mesh = OrderedDict()
    for link in root.findall("link"):
        name = link.attrib["name"]
        visual = link.find("visual")
        if visual is not None:
            geom = visual.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None:
                    fn = os.path.basename(mesh.get("filename", ""))
                    link_visual_mesh[name] = fn

    mesh_to_links = OrderedDict()
    for link_name, mesh_fn in link_visual_mesh.items():
        if mesh_fn not in mesh_to_links:
            mesh_to_links[mesh_fn] = []
        mesh_to_links[mesh_fn].append(link_name)

    source_to_targets = {}
    for mesh_fn, links in mesh_to_links.items():
        if len(links) > 1:
            source_to_targets[links[0]] = links[1:]

    return source_to_targets


def find_matching_brace(content, open_pos):
    """Find the closing brace matching the one at open_pos."""
    depth = 1
    i = open_pos + 1
    while i < len(content) and depth > 0:
        ch = content[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        i += 1
    if depth != 0:
        raise ValueError(f"Unmatched brace at position {open_pos}")
    return i - 1


def find_link_visuals_blocks(content):
    """
    Find all 'def Xform "visuals"' blocks in the USDA and map them to parent links.

    Returns:
        dict: {link_name: {
            'open_brace': position of opening '{',
            'close_brace': position of closing '}',
            'has_mesh': bool
        }}
    """
    blocks = {}
    pos = 0
    while True:
        idx = content.find('def Xform "visuals"', pos)
        if idx == -1:
            break

        search_start = max(0, idx - 3000)
        segment = content[search_start:idx]
        parent_name = None
        for m in re.finditer(r'def Xform "(\w+)"', segment):
            name = m.group(1)
            if name not in ("visuals", "collisions"):
                parent_name = name

        if parent_name is None:
            pos = idx + 1
            continue

        open_brace = content.index('{', idx)
        try:
            close_brace = find_matching_brace(content, open_brace)
        except ValueError:
            print(f"  WARNING: Unmatched brace for visuals of {parent_name}")
            pos = idx + 1
            continue

        inner = content[open_brace + 1:close_brace]
        has_mesh = 'def Mesh' in inner

        blocks[parent_name] = {
            'open_brace': open_brace,
            'close_brace': close_brace,
            'has_mesh': has_mesh,
        }

        pos = close_brace + 1

    return blocks


def fix_empty_visuals(usd_path, urdf_path):
    """
    Fix empty visuals Xform blocks caused by shared-mesh deduplication.

    Returns:
        int: Number of visuals blocks fixed (0 if already done, -1 on error)
    """
    if not os.path.exists(urdf_path):
        print(f"  ERROR: URDF not found: {urdf_path}")
        return -1
    if not os.path.exists(usd_path):
        print(f"  ERROR: USD not found: {usd_path}")
        return -1

    # Check if the USDA still has references (not yet flattened)
    if needs_flatten(usd_path):
        print("  WARNING: USDA still has unresolved references. Run flatten first!")
        print("  (use: conda run -n behavior python postprocess_sharpa_usd.py --hand <hand>)")
        return -1

    source_to_targets = get_shared_visual_groups(urdf_path)
    if not source_to_targets:
        print("  No shared visual mesh groups found in URDF -- nothing to fix.")
        return 0

    print(f"  Found {len(source_to_targets)} shared mesh groups in URDF:")
    for src, tgts in source_to_targets.items():
        print(f"    {src} -> {tgts}")

    with open(usd_path, "r") as f:
        content = f.read()

    orig_size = len(content)
    blocks = find_link_visuals_blocks(content)
    print(f"  Found {len(blocks)} link visuals blocks in USDA")

    replacements = []
    already_ok = 0

    for source_link, target_links in source_to_targets.items():
        if source_link not in blocks:
            print(f"  WARNING: Source link '{source_link}' not found in USDA")
            continue

        src = blocks[source_link]
        if not src['has_mesh']:
            print(f"  WARNING: Source link '{source_link}' has no mesh data!")
            continue

        source_inner = content[src['open_brace'] + 1 : src['close_brace']]

        for target_link in target_links:
            if target_link not in blocks:
                print(f"  WARNING: Target link '{target_link}' not found in USDA")
                continue

            tgt = blocks[target_link]
            if tgt['has_mesh']:
                already_ok += 1
                continue

            replacements.append((
                tgt['open_brace'] + 1,
                tgt['close_brace'],
                source_inner,
                target_link,
            ))

    if not replacements:
        total_expected = sum(len(t) for t in source_to_targets.values())
        print(f"  All {total_expected} shared-mesh visuals blocks already have data.")
        return 0

    replacements.sort(key=lambda x: x[0], reverse=True)
    for start, end, new_content, link_name in replacements:
        content = content[:start] + new_content + content[end:]
        print(f"    Fixed empty visuals: {link_name}")

    with open(usd_path, "w") as f:
        f.write(content)

    new_size = len(content)
    print(f"  Fixed {len(replacements)} empty visuals blocks ({already_ok} were already OK)")
    print(f"  File size: {orig_size / 1024 / 1024:.1f} MB -> {new_size / 1024 / 1024:.1f} MB")
    return len(replacements)


# ============================================================================
# Step 3: Fix visuals xformOps (Y-up to Z-up rotation for arm links)
# ============================================================================

# The URDF-to-USD converter's output, after pxr Flatten(), loses the xformOps
# on the visuals Xform blocks.  In the working right-hand USDA, the arm links
# (panda_*) have a Y-up to Z-up rotation and hand links have identity.  This
# step restores those xformOps.

ARM_ROTATION_QUAT = "(0.7071068, 0.7071068, 0, 0)"
IDENTITY_QUAT = "(1, 0, 0, 0)"

XFORMOPS_TEMPLATE = """\
            quatd xformOp:orient = {quat}
            double3 xformOp:scale = (1, 1, 1)
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
"""


def fix_visuals_xformops(usd_path):
    """
    Add missing xformOps to visuals Xform blocks.

    ARM links (panda_*) get the Y-up to Z-up rotation.
    HAND links get identity (no rotation).

    Returns:
        int: Number of blocks fixed (0 if already done, -1 on error)
    """
    if not os.path.exists(usd_path):
        print(f"  ERROR: USD not found: {usd_path}")
        return -1

    with open(usd_path, "r") as f:
        content = f.read()

    # Find all visuals blocks and their parents
    blocks = find_link_visuals_blocks(content)
    if not blocks:
        print("  WARNING: No link visuals blocks found!")
        return -1

    # Check which blocks need xformOps
    fixes = []  # (insert_pos, xformops_text, link_name)
    already_ok = 0

    for link_name, info in blocks.items():
        # Check if this visuals block already has xformOps
        inner_start = info['open_brace'] + 1
        # Look at the first ~300 chars after the opening brace for existing xformOps
        preview = content[inner_start:inner_start + 300]
        if 'xformOp:orient' in preview:
            already_ok += 1
            continue

        # Determine which quaternion to use
        if link_name.startswith('panda_'):
            quat = ARM_ROTATION_QUAT
        else:
            quat = IDENTITY_QUAT

        xformops = XFORMOPS_TEMPLATE.format(quat=quat)
        fixes.append((inner_start, xformops, link_name, quat))

    if not fixes:
        print(f"  All {already_ok} visuals blocks already have xformOps.")
        return 0

    # Apply fixes in reverse order (so positions stay valid)
    fixes.sort(key=lambda x: x[0], reverse=True)
    for insert_pos, xformops_text, link_name, quat in fixes:
        content = content[:insert_pos] + "\n" + xformops_text + content[insert_pos:]
        label = "ROTATION" if quat == ARM_ROTATION_QUAT else "identity"
        print(f"    Added xformOps ({label}): {link_name}")

    with open(usd_path, "w") as f:
        f.write(content)

    print(f"  Fixed {len(fixes)} visuals blocks ({already_ok} were already OK)")
    return len(fixes)


# ============================================================================
# Main
# ============================================================================

def postprocess_one_hand(hand_side, usd_path=None, urdf_path=None, skip_flatten=False):
    """Run all post-processing fixes for one hand variant."""
    if usd_path is None:
        usd_path = HAND_CONFIGS[hand_side]["usd"]
    if urdf_path is None:
        urdf_path = HAND_CONFIGS[hand_side]["urdf"]

    print(f"\n{'=' * 60}")
    print(f"Post-processing: {hand_side} hand")
    print(f"  USD:  {usd_path}")
    print(f"  URDF: {urdf_path}")
    print(f"{'=' * 60}")

    # Step 1: Flatten references
    if skip_flatten:
        print(f"\n--- Step 1: Flatten (SKIPPED by --skip-flatten) ---")
    else:
        print(f"\n--- Step 1: Flatten USD references ---")
        ok = flatten_usd(usd_path)
        if not ok:
            return False

    # Step 2: Fix shared-mesh deduplication
    print(f"\n--- Step 2: Shared-mesh deduplication fix ---")
    result = fix_empty_visuals(usd_path, urdf_path)
    if result < 0:
        return False

    # Step 3: Fix visuals xformOps (Y-up to Z-up rotation for arm links)
    print(f"\n--- Step 3: Fix visuals xformOps ---")
    result = fix_visuals_xformops(usd_path)

    return result >= 0


def main():
    parser = argparse.ArgumentParser(
        description="Post-process converted Franka+SharpaWave USD for OmniGibson",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script is run AFTER the standard BEHAVIOR URDF-to-USD conversion.

Step 1 (flatten) requires Isaac Sim:
    conda run -n behavior python postprocess_sharpa_usd.py --hand left

Steps 2-3 (mesh fix + xformOps) can run without Isaac Sim if already flattened:
    python postprocess_sharpa_usd.py --hand left --skip-flatten

Workflow:
  1. python merge_franka_sharpa.py --hand left           # Merge URDFs
  2. conda run -n behavior python import_custom_robot ... # BEHAVIOR conversion
  3. conda run -n behavior python postprocess_sharpa_usd.py --hand left  # This script
  4. conda run -n behavior python test_sharpa_robot.py --hand left  # Test
        """,
    )
    parser.add_argument(
        "--hand",
        choices=["right", "left", "both"],
        default=None,
        help="Which hand variant to fix (default: right)",
    )
    parser.add_argument(
        "--usd-path", type=str, default=None,
        help="Explicit path to a .usda file (overrides --hand for USD)",
    )
    parser.add_argument(
        "--urdf-path", type=str, default=None,
        help="Explicit path to the merged .urdf (overrides --hand for URDF)",
    )
    parser.add_argument(
        "--skip-flatten", action="store_true",
        help="Skip the flatten step (only run mesh dedup fix). "
             "Use if the USDA is already flattened.",
    )

    args = parser.parse_args()

    if args.usd_path:
        if not args.urdf_path:
            print("ERROR: --urdf-path is required when using --usd-path")
            sys.exit(1)
        hands = [("custom", args.usd_path, args.urdf_path)]
    elif args.hand == "both":
        hands = [("right", None, None), ("left", None, None)]
    else:
        hand = args.hand or "right"
        hands = [(hand, None, None)]

    print("=" * 60)
    print("Post-processing Franka+SharpaWave USD for OmniGibson")
    print("=" * 60)

    all_ok = True
    for hand_side, usd_override, urdf_override in hands:
        ok = postprocess_one_hand(
            hand_side, usd_override, urdf_override,
            skip_flatten=args.skip_flatten,
        )
        if not ok:
            all_ok = False

    print(f"\n{'=' * 60}")
    if all_ok:
        print("Done! Run test_sharpa_robot.py to verify rendering.")
    else:
        print("Some files could not be processed (see errors above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
