"""
Fast URDF visualizer using PyBullet for debugging transforms.
No Isaac Sim needed - starts in ~2 seconds.

Usage:
    conda run -n behavior python debug_urdf.py
    conda run -n behavior python debug_urdf.py --hand left
    conda run -n behavior python debug_urdf.py --rpy 0 0 -0.785
    conda run -n behavior python debug_urdf.py --rpy 0 0 -1.571 --xyz 0 0 0.1
"""
import argparse
import time
import xml.etree.ElementTree as ET
import tempfile
import shutil
import os
import pybullet as p
import pybullet_data


def modify_urdf_joint(urdf_path, joint_name, xyz=None, rpy=None):
    """Create a temp copy of the URDF with modified joint transform."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for joint in root.findall("joint"):
        if joint.attrib.get("name") == joint_name:
            origin = joint.find("origin")
            if origin is not None:
                if xyz is not None:
                    origin.attrib["xyz"] = f"{xyz[0]} {xyz[1]} {xyz[2]}"
                if rpy is not None:
                    origin.attrib["rpy"] = f"{rpy[0]} {rpy[1]} {rpy[2]}"
                print(f"Modified {joint_name}: xyz={origin.attrib.get('xyz')}, rpy={origin.attrib.get('rpy')}")
            break

    # Save to a temp file in the same directory (so relative mesh paths still work)
    urdf_dir = os.path.dirname(urdf_path)
    tmp_path = os.path.join(urdf_dir, "_debug_temp.urdf")
    tree.write(tmp_path, xml_declaration=True)
    return tmp_path


def main():
    parser = argparse.ArgumentParser(description="Fast URDF debug viewer (PyBullet)")
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--xyz", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                        help="Override hand joint xyz")
    parser.add_argument("--rpy", nargs=3, type=float, default=None, metavar=("R", "P", "Y"),
                        help="Override hand joint rpy (radians)")
    parser.add_argument("--no-gui", action="store_true", help="Use direct mode (no GUI)")
    args = parser.parse_args()

    base = "/home/robot/Desktop/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/franka"
    urdf_path = f"{base}/franka_mounted_sharpa_{args.hand}/urdf/franka_mounted_sharpa_{args.hand}.urdf"
    joint_name = f"panda_{args.hand}_hand_C_MC_joint"

    # Optionally modify the URDF
    tmp_path = None
    if args.xyz or args.rpy:
        tmp_path = modify_urdf_joint(urdf_path, joint_name, xyz=args.xyz, rpy=args.rpy)
        load_path = tmp_path
    else:
        load_path = urdf_path

    # Start PyBullet
    mode = p.DIRECT if args.no_gui else p.GUI
    physics_client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load ground plane
    p.loadURDF("plane.urdf")

    # Load robot
    print(f"\nLoading URDF: {load_path}")
    robot_id = p.loadURDF(
        load_path,
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION,
    )

    # Print joint info
    num_joints = p.getNumJoints(robot_id)
    print(f"\nRobot loaded with {num_joints} joints:")
    print(f"{'Index':>5} {'Name':<40} {'Type':<10} {'Parent Link':<25} {'Child Link':<25}")
    print("-" * 110)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_idx = info[0]
        joint_name_b = info[1].decode()
        joint_type = {0: "revolute", 1: "prismatic", 2: "spherical", 3: "planar", 4: "fixed"}[info[2]]
        parent_link = info[12].decode()
        child_link = info[12].decode()  # link name associated with this joint
        print(f"{joint_idx:>5} {joint_name_b:<40} {joint_type:<10}")

    # Set a nice default pose (standard Franka resting config)
    franka_default = [0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75]
    arm_joint_indices = []
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        if info[2] == 0:  # revolute
            arm_joint_indices.append(i)

    # Set Franka arm joints to default pose (first 7 revolute joints)
    for idx, pos in zip(arm_joint_indices[:7], franka_default):
        p.resetJointState(robot_id, idx, pos)

    # Set all finger (non-arm) revolute joints to a slight curl so they look natural
    for idx in arm_joint_indices[7:]:
        info = p.getJointInfo(robot_id, idx)
        lower = info[8]
        upper = info[9]
        # Set to ~30% of range for a relaxed pose
        mid = lower + 0.3 * (upper - lower)
        p.resetJointState(robot_id, idx, mid)

    # Set camera to look at the hand area (end effector)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.6,
        cameraYaw=135,
        cameraPitch=-25,
        cameraTargetPosition=[0.3, 0, 1.5],
    )

    print("\n" + "=" * 60)
    print("PyBullet URDF Viewer - CONTROLS:")
    print("  Left-click drag   = ROTATE camera")
    print("  Middle-click drag  = PAN camera")
    print("  Scroll wheel       = ZOOM in/out")
    print("  Right-click drag   = ZOOM (alternative)")
    print("  Close window or Ctrl+C to exit")
    print("=" * 60)
    print(f"\nCurrent joint transform: xyz={args.xyz}, rpy={args.rpy}")
    print("Try different values:")
    print(f"  python debug_urdf.py --rpy 0 0 0")
    print(f"  python debug_urdf.py --rpy 0 0 -0.785")
    print(f"  python debug_urdf.py --rpy 0 0 -1.571")
    print(f"  python debug_urdf.py --rpy 1.571 0 0")
    print(f"  python debug_urdf.py --xyz 0 0 0.12 --rpy 0 0 -0.785")
    print()

    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        p.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
