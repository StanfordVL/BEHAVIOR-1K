#!/usr/bin/env python3
"""
Merge Franka Mounted arm with SharpaWave dexterous hand.

This script:
1. Loads the Franka mounted URDF (XML)
2. Strips the default 2-finger gripper (panda_hand + fingers)
3. Loads a SharpaWave hand URDF (right or left)
4. Rewrites SharpaWave mesh paths from package:// to relative paths
5. Joins the hand onto panda_link7 with a fixed joint
6. Copies all meshes into a self-contained output directory
7. Saves the merged URDF

Usage:
    python merge_franka_sharpa.py                           # Both hands, default RPY
    python merge_franka_sharpa.py --hand right              # Right hand only
    python merge_franka_sharpa.py --hand left               # Left hand only
    python merge_franka_sharpa.py --rpy 0 0 -0.785          # Custom orientation
    python merge_franka_sharpa.py --xyz 0 0 0.107            # Custom position offset

This script works at the XML level (no urdfpy) to avoid mesh path issues.
"""

import argparse
import os
import shutil
import sys
import copy
from xml.etree import ElementTree as ET


# ============================================================================
# Paths (all relative to this script / repo root)
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FRANKA_URDF = os.path.join(
    SCRIPT_DIR,
    "datasets/omnigibson-robot-assets/models/franka/franka_mounted/urdf/franka_mounted.urdf",
)
FRANKA_MESH_DIR = os.path.join(
    SCRIPT_DIR,
    "datasets/omnigibson-robot-assets/models/franka/franka_mounted/urdf/meshes",
)

SHARPA_RIGHT_URDF = os.path.join(
    SCRIPT_DIR,
    "SharpaWave_URDF_XML_USD_SU_SVL_V3.0.3/src/right_sharpa_wave/right_sharpa_wave.urdf",
)
SHARPA_RIGHT_MESHES = os.path.join(
    SCRIPT_DIR,
    "SharpaWave_URDF_XML_USD_SU_SVL_V3.0.3/src/right_sharpa_wave/meshes",
)

SHARPA_LEFT_URDF = os.path.join(
    SCRIPT_DIR,
    "SharpaWave_URDF_XML_USD_SU_SVL_V3.0.3/src/left_sharpa_wave/left_sharpa_wave.urdf",
)
SHARPA_LEFT_MESHES = os.path.join(
    SCRIPT_DIR,
    "SharpaWave_URDF_XML_USD_SU_SVL_V3.0.3/src/left_sharpa_wave/meshes",
)

OUTPUT_BASE = os.path.join(
    SCRIPT_DIR,
    "datasets/omnigibson-robot-assets/models/franka",
)

# Links and joints to remove (default Franka gripper)
GRIPPER_LINKS = {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
GRIPPER_JOINTS = {"panda_hand_joint", "panda_finger_joint1", "panda_finger_joint2"}

# Attachment point on Franka
ATTACH_LINK = "panda_link7"

# Default offset (same distance as original gripper: 10.7cm along Z)
DEFAULT_XYZ = [0.0, 0.0, 0.107]
DEFAULT_RPY = [0.0, 0.0, 0.0]


# ============================================================================
# XML-level URDF manipulation helpers
# ============================================================================

def get_link_names(root):
    """Get all link names from a URDF XML root."""
    return [link.attrib["name"] for link in root.findall("link")]


def get_joint_names(root):
    """Get all joint names from a URDF XML root."""
    return [joint.attrib["name"] for joint in root.findall("joint")]


def find_base_link(root):
    """Find the base link (a link that is never a child in any joint)."""
    child_links = set()
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            child_links.add(child.attrib["link"])
    for link in root.findall("link"):
        if link.attrib["name"] not in child_links:
            return link.attrib["name"]
    return None


def strip_gripper(root):
    """Remove the default Franka gripper links and joints from the XML tree."""
    removed_links = 0
    removed_joints = 0

    # Remove gripper links
    for link in root.findall("link"):
        if link.attrib["name"] in GRIPPER_LINKS:
            root.remove(link)
            removed_links += 1

    # Remove gripper joints
    for joint in root.findall("joint"):
        if joint.attrib["name"] in GRIPPER_JOINTS:
            root.remove(joint)
            removed_joints += 1

    return removed_links, removed_joints


def rewrite_sharpa_mesh_paths(root, mesh_subdir):
    """Rewrite all package:// mesh paths to relative paths in a SharpaWave URDF."""
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and "package://" in filename:
            basename = os.path.basename(filename)
            mesh.set("filename", "meshes/{}/{}".format(mesh_subdir, basename))


def clean_sharpa_xml(root):
    """Remove non-standard elements from SharpaWave URDF (e.g., <mujoco>)."""
    for elem_name in ["mujoco"]:
        for elem in root.findall(elem_name):
            root.remove(elem)


def merge_urdfs(franka_root, sharpa_root, attach_link, xyz, rpy, robot_name):
    """
    Merge a SharpaWave hand URDF into a (gripper-stripped) Franka URDF.

    Works at the XML level: copies all <link> and <joint> elements from the
    SharpaWave URDF into the Franka URDF, then adds a fixed joint connecting
    them.

    Parameters
    ----------
    franka_root : ET.Element
        The Franka URDF root (already stripped of gripper).
    sharpa_root : ET.Element
        The SharpaWave hand URDF root (mesh paths already rewritten).
    attach_link : str
        The Franka link to attach the hand to (e.g., "panda_link7").
    xyz : list of 3 floats
        Position offset [x, y, z] in meters.
    rpy : list of 3 floats
        Orientation offset [roll, pitch, yaw] in radians.
    robot_name : str
        Name for the merged robot.

    Returns
    -------
    ET.Element : The merged URDF root element.
    """
    # Find the SharpaWave base link
    sharpa_base = find_base_link(sharpa_root)

    # Set the robot name
    franka_root.attrib["name"] = robot_name

    # Copy all SharpaWave links into the Franka tree
    for link in sharpa_root.findall("link"):
        franka_root.append(copy.deepcopy(link))

    # Copy all SharpaWave joints into the Franka tree
    for joint in sharpa_root.findall("joint"):
        franka_root.append(copy.deepcopy(joint))

    # Copy any materials from SharpaWave
    for material in sharpa_root.findall("material"):
        franka_root.append(copy.deepcopy(material))

    # Create the fixed joint connecting panda_link7 to the hand base
    join_joint = ET.SubElement(franka_root, "joint")
    join_joint.attrib["name"] = "panda_{}_joint".format(sharpa_base)
    join_joint.attrib["type"] = "fixed"

    parent_elem = ET.SubElement(join_joint, "parent")
    parent_elem.attrib["link"] = attach_link

    child_elem = ET.SubElement(join_joint, "child")
    child_elem.attrib["link"] = sharpa_base

    origin_elem = ET.SubElement(join_joint, "origin")
    origin_elem.attrib["xyz"] = "{} {} {}".format(*xyz)
    origin_elem.attrib["rpy"] = "{} {} {}".format(*rpy)

    return franka_root


# ============================================================================
# Tree printing for verification
# ============================================================================
def print_urdf_tree(root, filepath=""):
    """Print a human-readable joint/link tree from an XML root."""
    robot_name = root.attrib.get("name", "unknown")

    links = {link.attrib["name"] for link in root.findall("link")}
    children_of = {}
    child_set = set()

    for joint in root.findall("joint"):
        jname = joint.attrib["name"]
        jtype = joint.attrib["type"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        xyz = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy = origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"

        if parent not in children_of:
            children_of[parent] = []
        children_of[parent].append((jname, jtype, child, xyz, rpy))
        child_set.add(child)

    base_links = [l for l in links if l not in child_set]

    print("=" * 70)
    print("ROBOT: {}".format(robot_name))
    if filepath:
        print("File:  {}".format(filepath))
    print("Links: {}  |  Joints: {}".format(
        len(links), sum(len(v) for v in children_of.values())
    ))
    print("BASE:  {}".format(", ".join(sorted(base_links))))
    print("=" * 70)

    def _print(link_name, indent=0):
        prefix = "  " * indent
        print("{}LINK: {}".format(prefix, link_name))
        if link_name in children_of:
            for jname, jtype, child, xyz, rpy in children_of[link_name]:
                jprefix = "  " * (indent + 1)
                print("{}|".format(jprefix))
                print("{}+-- {} ({})".format(jprefix, jname, jtype.upper()))
                print("{}|   xyz=({})  rpy=({})".format(jprefix, xyz, rpy))
                print("{}|".format(jprefix))
                _print(child, indent + 2)

    for base in sorted(base_links):
        _print(base)
    print("=" * 70)


# ============================================================================
# Main merge logic
# ============================================================================
def merge_one_hand(hand_side, xyz, rpy, dry_run=False):
    """
    Merge the Franka arm with one SharpaWave hand.

    Returns the path to the output URDF.
    """
    if hand_side == "right":
        sharpa_urdf_path = SHARPA_RIGHT_URDF
        sharpa_mesh_src = SHARPA_RIGHT_MESHES
        mesh_subdir = "sharpa_right"
    else:
        sharpa_urdf_path = SHARPA_LEFT_URDF
        sharpa_mesh_src = SHARPA_LEFT_MESHES
        mesh_subdir = "sharpa_left"

    variant_name = "franka_mounted_sharpa_{}".format(hand_side)
    output_dir = os.path.join(OUTPUT_BASE, variant_name)
    urdf_dir = os.path.join(output_dir, "urdf")
    mesh_dir = os.path.join(urdf_dir, "meshes")
    sharpa_mesh_dst = os.path.join(mesh_dir, mesh_subdir)
    output_urdf = os.path.join(urdf_dir, "{}.urdf".format(variant_name))

    print("\n" + "=" * 70)
    print("MERGING: Franka + SharpaWave {} hand".format(hand_side))
    print("=" * 70)
    print("  Franka URDF:     {}".format(FRANKA_URDF))
    print("  SharpaWave URDF: {}".format(sharpa_urdf_path))
    print("  Output:          {}".format(output_urdf))
    print("  Attach link:     {}".format(ATTACH_LINK))
    print("  XYZ offset:      {}".format(xyz))
    print("  RPY orientation: {}".format(rpy))
    print()

    if dry_run:
        print("  [DRY RUN] Skipping file operations.")
        return output_urdf

    # -----------------------------------------------------------------
    # 1. Create output directory structure
    # -----------------------------------------------------------------
    os.makedirs(urdf_dir, exist_ok=True)
    os.makedirs(sharpa_mesh_dst, exist_ok=True)

    # -----------------------------------------------------------------
    # 2. Symlink Franka mesh subdirectories
    # -----------------------------------------------------------------
    franka_mesh_subdirs = ["visual", "collision", "base"]
    for subdir in franka_mesh_subdirs:
        src = os.path.join(FRANKA_MESH_DIR, subdir)
        dst = os.path.join(mesh_dir, subdir)
        if os.path.exists(src):
            if os.path.exists(dst) or os.path.islink(dst):
                if os.path.islink(dst):
                    os.remove(dst)
                else:
                    shutil.rmtree(dst)
            rel_src = os.path.relpath(src, mesh_dir)
            os.symlink(rel_src, dst)
            print("  Symlinked meshes/{} -> {}".format(subdir, rel_src))

    # -----------------------------------------------------------------
    # 3. Copy SharpaWave STL meshes
    # -----------------------------------------------------------------
    stl_count = 0
    for fname in os.listdir(sharpa_mesh_src):
        src = os.path.join(sharpa_mesh_src, fname)
        dst = os.path.join(sharpa_mesh_dst, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            stl_count += 1
    print("  Copied {} mesh files to meshes/{}".format(stl_count, mesh_subdir))

    # -----------------------------------------------------------------
    # 4. Parse the Franka URDF
    # -----------------------------------------------------------------
    print("  Parsing Franka URDF...")
    franka_tree = ET.parse(FRANKA_URDF)
    franka_root = franka_tree.getroot()
    franka_base = find_base_link(franka_root)
    n_links_before = len(get_link_names(franka_root))
    n_joints_before = len(get_joint_names(franka_root))
    print("    Base: {}  Links: {}  Joints: {}".format(
        franka_base, n_links_before, n_joints_before
    ))

    # -----------------------------------------------------------------
    # 5. Strip the default gripper
    # -----------------------------------------------------------------
    removed_links, removed_joints = strip_gripper(franka_root)
    print("  Stripped gripper: removed {} links, {} joints".format(
        removed_links, removed_joints
    ))

    # -----------------------------------------------------------------
    # 6. Parse the SharpaWave URDF
    # -----------------------------------------------------------------
    print("  Parsing SharpaWave {} URDF...".format(hand_side))
    sharpa_tree = ET.parse(sharpa_urdf_path)
    sharpa_root = sharpa_tree.getroot()
    sharpa_base = find_base_link(sharpa_root)
    print("    Base: {}  Links: {}  Joints: {}".format(
        sharpa_base,
        len(get_link_names(sharpa_root)),
        len(get_joint_names(sharpa_root)),
    ))

    # -----------------------------------------------------------------
    # 7. Clean and rewrite SharpaWave XML
    # -----------------------------------------------------------------
    clean_sharpa_xml(sharpa_root)
    rewrite_sharpa_mesh_paths(sharpa_root, mesh_subdir)
    print("  Rewrote SharpaWave mesh paths -> meshes/{}/".format(mesh_subdir))

    # -----------------------------------------------------------------
    # 8. Merge the two URDF XML trees
    # -----------------------------------------------------------------
    print("  Merging onto {}...".format(ATTACH_LINK))
    merged_root = merge_urdfs(
        franka_root=franka_root,
        sharpa_root=sharpa_root,
        attach_link=ATTACH_LINK,
        xyz=xyz,
        rpy=rpy,
        robot_name=variant_name,
    )
    n_links_after = len(get_link_names(merged_root))
    n_joints_after = len(get_joint_names(merged_root))
    print("    Merged: {} links, {} joints".format(n_links_after, n_joints_after))

    # -----------------------------------------------------------------
    # 9. Save the merged URDF
    # -----------------------------------------------------------------
    merged_tree = ET.ElementTree(merged_root)
    ET.indent(merged_tree, space="    ")
    merged_tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print("  Saved: {}".format(output_urdf))

    # -----------------------------------------------------------------
    # 10. Verify all mesh paths resolve
    # -----------------------------------------------------------------
    print("  Verifying mesh paths...")
    missing = []
    for mesh in merged_root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            full = os.path.join(urdf_dir, fn)
            if not os.path.exists(full):
                missing.append(fn)
    if missing:
        print("  WARNING: {} mesh files not found:".format(len(missing)))
        for m in missing[:10]:
            print("    MISSING: {}".format(m))
    else:
        print("  All {} mesh paths verified OK.".format(
            sum(1 for _ in merged_root.iter("mesh"))
        ))

    return output_urdf, merged_root


# ============================================================================
# Entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Merge Franka Mounted arm with SharpaWave dexterous hand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_franka_sharpa.py                        # Both hands, default orientation
  python merge_franka_sharpa.py --hand right            # Right hand only
  python merge_franka_sharpa.py --rpy 0 0 -0.785        # Rotate hand -45 deg yaw
  python merge_franka_sharpa.py --rpy 0 0 0 --xyz 0 0 0.12  # Adjust position too
        """,
    )
    parser.add_argument(
        "--hand",
        choices=["right", "left", "both"],
        default="both",
        help="Which hand to merge (default: both)",
    )
    parser.add_argument(
        "--rpy",
        nargs=3,
        type=float,
        default=DEFAULT_RPY,
        metavar=("R", "P", "Y"),
        help="Orientation: roll pitch yaw in radians (default: 0 0 0)",
    )
    parser.add_argument(
        "--xyz",
        nargs=3,
        type=float,
        default=DEFAULT_XYZ,
        metavar=("X", "Y", "Z"),
        help="Position offset from panda_link7 in meters (default: 0 0 0.107)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without creating files",
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Don't print the joint/link tree after merging",
    )

    args = parser.parse_args()

    # Validate inputs exist
    if not os.path.exists(FRANKA_URDF):
        print("ERROR: Franka URDF not found: {}".format(FRANKA_URDF))
        sys.exit(1)

    hands = ["right", "left"] if args.hand == "both" else [args.hand]

    for hand_side in hands:
        result = merge_one_hand(
            hand_side=hand_side,
            xyz=list(args.xyz),
            rpy=list(args.rpy),
            dry_run=args.dry_run,
        )

        if not args.dry_run and not args.no_tree:
            output_urdf, merged_root = result
            print("\n")
            print_urdf_tree(merged_root, output_urdf)

    print("\n\nDone! To adjust orientation, re-run with --rpy R P Y (radians)")
    print("  Example: python merge_franka_sharpa.py --rpy 0 0 -0.785")


if __name__ == "__main__":
    main()
