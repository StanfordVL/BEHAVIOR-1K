"""
Helper script to download BEHAVIOR-1K dataset and assets.
"""

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import click
import torch as th
import yaml
from addict import Dict

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.utils.asset_conversion_utils import (
    _add_xform_properties,
    _space_string_to_tensor,
    find_all_prim_children_with_type,
    import_og_asset_from_urdf,
)
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.usd_utils import create_joint, create_primitive_mesh

# Make sure flatcache is NOT used so we write directly to USD
gm.ENABLE_FLATCACHE = False


_DOCSTRING = """
Imports an custom-defined robot URDF asset into an OmniGibson-compatible USD format and saves the imported asset
files to the custom dataset directory data/custom_dataset

Note that @config is expected to follow the following format (R1 config shown as an example):

\b
urdf_path: r1_pro_source.urdf       # (str) Absolute path to robot URDF to import
name: r1                            # (str) Name to assign to robot
headless: false                     # (bool) if set, run without GUI
overwrite: true                     # (bool) if set, overwrite any existing files
merge_fixed_joints: false           # (bool) whether to merge fixed joints in the robot hierarchy or not
base_motion:
  wheel_links:                      # (list of str): links corresponding to wheels
    - wheel_link1
    - wheel_link2
    - wheel_link3
  wheel_joints:                     # (list of str): joints corresponding to wheel motion
    - servo_joint1
    - servo_joint2
    - servo_joint3
    - wheel_joint1
    - wheel_joint2
    - wheel_joint3
  use_sphere_wheels: true           # (bool) whether to use sphere approximation for wheels (better stability)
  use_holonomic_joints: true        # (bool) whether to use joints to approximate a holonomic base. In this case, all
                                    #       wheel-related joints will be made into fixed joints, and 6 additional
                                    #       "virtual" joints will be added to the robot's base capturing 6DOF movement,
                                    #       with the (x,y,rz) joints being controllable by motors
collision:
  decompose_method: coacd           # (str) [coacd, convex, or null] collision decomposition method
  hull_count: 8                     # (int) per-mesh max hull count to use during decomposition, only relevant for coacd
  coacd_links: []                   # (list of str): links that should use CoACD to decompose collision meshes
  convex_links:                     # (list of str): links that should use convex hull to decompose collision meshes
    - base_link
    - wheel_link1
    - wheel_link2
    - wheel_link3
    - torso_link1
    - torso_link2
    - torso_link3
    - torso_link4
    - left_arm_link1
    - left_arm_link4
    - left_arm_link5
    - right_arm_link1
    - right_arm_link4
    - right_arm_link5
  no_decompose_links: []            # (list of str): links that should not have any post-processing done to them
  no_collision_links:               # (list of str) links that will have any associated collision meshes removed
    - servo_link1
    - servo_link2
    - servo_link3
eef_vis_links:                      # (list of dict) information for adding cameras to robot
  - link: left_eef_link             # same format as @camera_links
    parent_link: left_arm_link6
    offset:
      position: [0, 0, 0.06]
      orientation: [0, 0, 0, 1]     # NOTE: Convention for these eef vis links should be tuned such that:
                                    #   z-axis points out from the tips of the fingers
                                    #   y-axis points in the direction from the left finger to the right finger
                                    #   x-axis is automatically inferred from the above two axes
  - link: right_eef_link            # same format as @camera_links
    parent_link: right_arm_link6
    offset:
      position: [0, 0, 0.06]
      orientation: [0, 0, 0, 1]
camera_links:                       # (list of dict) information for adding cameras to robot
  - link: eyes                      # (str) link name to add camera. Must exist if @parent_link is null, else will be
                                    #       added as a child of the parent
    parent_link: torso_link4        # (str) optional parent link to use if adding new link
    offset:                         # (dict) local pos,ori offset values. if @parent_link is specified, defines offset
                                    #       between @parent_link and @link specified in @parent_link's frame.
                                    #       Otherwise, specifies offset of generated prim relative to @link's frame
      position: [0.0732, 0, 0.4525]                     # (3-tuple) (x,y,z) offset -- this is done BEFORE the rotation
      orientation: [0.4056, -0.4056, -0.5792, 0.5792]   # (4-tuple) (x,y,z,w) offset
  - link: left_eef_link
    parent_link: null
    offset:
      position: [0.05, 0, -0.05]
      orientation: [-0.7011, -0.7011, -0.0923, -0.0923]
  - link: right_eef_link
    parent_link: null
    offset:
      position: [0.05, 0, -0.05]
      orientation: [-0.7011, -0.7011, -0.0923, -0.0923]
lidar_links: []                     # (list of dict) information for adding cameras to robot
curobo:
  eef_to_gripper_info:              # (dict) Maps EEF link name to corresponding gripper links / joints
    right_eef_link:
      links: ["right_gripper_link1", "right_gripper_link2"]
      joints: ["right_gripper_axis1", "right_gripper_axis2"]
    left_eef_link:
      links: ["left_gripper_link1", "left_gripper_link2"]
      joints: ["left_gripper_axis1", "left_gripper_axis2"]
  flip_joint_limits: []             # (list of str) any joints that have a negative axis specified in the
                                    #       source URDF
  lock_joints: {}                   # (dict) Maps joint name to "locked" joint configuration. Any joints
                                    #       specified here will not be considered active when motion planning
                                    #       NOTE: All gripper joints and non-controllable holonomic joints
                                    #       will automatically be added here. Null means that the value will be
                                    #       dynamically computed at runtime based on the robot's reset qpos
  self_collision_ignore:            # (dict) Maps link name to list of other ignore links to ignore collisions
                                    #       with. Note that bi-directional specification is not necessary,
                                    #       e.g.: "torso_link1" does not need to be specified in
                                    #       "torso_link2"'s list if "torso_link2" is already specified in
                                    #       "torso_link1"'s list
    base_link: ["torso_link1", "wheel_link1", "wheel_link2", "wheel_link3"]
    torso_link1: ["torso_link2"]
    torso_link2: ["torso_link3", "torso_link4"]
    torso_link3: ["torso_link4"]
    torso_link4: ["left_arm_link1", "right_arm_link1", "left_arm_link2", "right_arm_link2"]
    left_arm_link1: ["left_arm_link2"]
    left_arm_link2: ["left_arm_link3"]
    left_arm_link3: ["left_arm_link4"]
    left_arm_link4: ["left_arm_link5"]
    left_arm_link5: ["left_arm_link6"]
    left_arm_link6: ["left_gripper_link1", "left_gripper_link2"]
    right_arm_link1: ["right_arm_link2"]
    right_arm_link2: ["right_arm_link3"]
    right_arm_link3: ["right_arm_link4"]
    right_arm_link4: ["right_arm_link5"]
    right_arm_link5: ["right_arm_link6"]
    right_arm_link6: ["right_gripper_link1", "right_gripper_link2"]
    left_gripper_link1: ["left_gripper_link2"]
    right_gripper_link1: ["right_gripper_link2"]
  collision_spheres:                # (dict) Maps link name to list of collision sphere representations,
                                    #       where each sphere is defined by its (x,y,z) "center" and "radius"
                                    #       values. This defines the collision geometry during motion planning
    base_link:
      - "center": [-0.009, -0.094, 0.131]
        "radius": 0.09128
      - "center": [-0.021, 0.087, 0.121]
        "radius": 0.0906
      - "center": [0.019, 0.137, 0.198]
        "radius": 0.07971
      - "center": [0.019, -0.14, 0.209]
        "radius": 0.07563
      - "center": [0.007, -0.018, 0.115]
        "radius": 0.08448
      - "center": [0.119, -0.176, 0.209]
        "radius": 0.05998
      - "center": [0.137, 0.118, 0.208]
        "radius": 0.05862
      - "center": [-0.152, -0.049, 0.204]
        "radius": 0.05454
    torso_link1:
      - "center": [-0.001, -0.014, -0.057]
        "radius": 0.1
      - "center": [-0.001, -0.127, -0.064]
        "radius": 0.07
      - "center": [-0.001, -0.219, -0.064]
        "radius": 0.07
      - "center": [-0.001, -0.29, -0.064]
        "radius": 0.07
      - "center": [-0.001, -0.375, -0.064]
        "radius": 0.07
      - "center": [-0.001, -0.419, -0.064]
        "radius": 0.07
    torso_link2:
      - "center": [-0.001, -0.086, -0.064]
        "radius": 0.07
      - "center": [-0.001, -0.194, -0.064]
        "radius": 0.07
      - "center": [-0.001, -0.31, -0.064]
        "radius": 0.07
    torso_link4:
      - "center": [0.005, -0.001, 0.062]
        "radius": 0.1
      - "center": [0.005, -0.001, 0.245]
        "radius": 0.15
      - "center": [0.005, -0.001, 0.458]
        "radius": 0.1
      - "center": [0.002, 0.126, 0.305]
        "radius": 0.08
      - "center": [0.002, -0.126, 0.305]
        "radius": 0.08
    left_arm_link1:
      - "center": [0.001, 0.0, 0.069]
        "radius": 0.06
    left_arm_link2:
      - "center": [-0.062, -0.016, -0.03]
        "radius": 0.06
      - "center": [-0.135, -0.019, -0.03]
        "radius": 0.06
      - "center": [-0.224, -0.019, -0.03]
        "radius": 0.06
      - "center": [-0.31, -0.022, -0.03]
        "radius": 0.06
      - "center": [-0.34, -0.027, -0.03]
        "radius": 0.06
    left_arm_link3:
      - "center": [0.037, -0.058, -0.044]
        "radius": 0.05
      - "center": [0.095, -0.08, -0.044]
        "radius": 0.03
      - "center": [0.135, -0.08, -0.043]
        "radius": 0.03
      - "center": [0.176, -0.08, -0.043]
        "radius": 0.03
      - "center": [0.22, -0.077, -0.043]
        "radius": 0.03
    left_arm_link4:
      - "center": [-0.002, 0.0, 0.276]
        "radius": 0.04
    left_arm_link5:
      - "center": [0.059, -0.001, -0.021]
        "radius": 0.035
    left_arm_link6:
      - "center": [0.0, 0.0, 0.04]
        "radius": 0.04
    right_arm_link1:
      - "center": [0.001, 0.0, 0.069]
        "radius": 0.06
    right_arm_link2:
      - "center": [-0.062, -0.016, -0.03]
        "radius": 0.06
      - "center": [-0.135, -0.019, -0.03]
        "radius": 0.06
      - "center": [-0.224, -0.019, -0.03]
        "radius": 0.06
      - "center": [-0.31, -0.022, -0.03]
        "radius": 0.06
      - "center": [-0.34, -0.027, -0.03]
        "radius": 0.06
    right_arm_link3:
      - "center": [0.037, -0.058, -0.044]
        "radius": 0.05
      - "center": [0.095, -0.08, -0.044]
        "radius": 0.03
      - "center": [0.135, -0.08, -0.043]
        "radius": 0.03
      - "center": [0.176, -0.08, -0.043]
        "radius": 0.03
      - "center": [0.22, -0.077, -0.043]
        "radius": 0.03
    right_arm_link4:
      - "center": [-0.002, 0.0, 0.276]
        "radius": 0.04
    right_arm_link5:
      - "center": [0.059, -0.001, -0.021]
        "radius": 0.035
    right_arm_link6:
      - "center": [-0.0, 0.0, 0.04]
        "radius": 0.035
    wheel_link1:
      - "center": [-0.0, 0.0, -0.03]
        "radius": 0.06
    wheel_link2:
      - "center": [0.0, 0.0, 0.03]
        "radius": 0.06
    wheel_link3:
      - "center": [0.0, 0.0, -0.03]
        "radius": 0.06
    left_gripper_link1:
      - "center": [-0.03, 0.0, -0.002]
        "radius": 0.008
      - "center": [-0.01, 0.0, -0.003]
        "radius": 0.007
      - "center": [0.005, 0.0, -0.005]
        "radius": 0.005
      - "center": [0.02, 0.0, -0.007]
        "radius": 0.003
    left_gripper_link2:
      - "center": [-0.03, 0.0, -0.002]
        "radius": 0.008
      - "center": [-0.01, 0.0, -0.003]
        "radius": 0.007
      - "center": [0.005, 0.0, -0.005]
        "radius": 0.005
      - "center": [0.02, 0.0, -0.007]
        "radius": 0.003
    right_gripper_link1:
      - "center": [-0.03, 0.0, -0.002]
        "radius": 0.008
      - "center": [-0.01, -0.0, -0.003]
        "radius": 0.007
      - "center": [0.005, -0.0, -0.005]
        "radius": 0.005
      - "center": [0.02, -0.0, -0.007]
        "radius": 0.003
    right_gripper_link2:
      - "center": [-0.03, 0.0, -0.002]
        "radius": 0.008
      - "center": [-0.01, 0.0, -0.003]
        "radius": 0.007
      - "center": [0.005, 0.0, -0.005]
        "radius": 0.005
      - "center": [0.02, 0.0, -0.007]
        "radius": 0.003

"""


def create_rigid_prim(stage, link_prim_path):
    """
    Creates a new rigid link prim nested under @root_prim

    Args:
        stage (Usd.Stage): Current active omniverse stage
        link_prim_path (str): Prim path at which link will be created. Should not already exist on the stage

    Returns:
        Usd.Prim: Newly created rigid prim
    """
    # Make sure link prim does NOT already exist (this should be a new link)
    link_prim_exists = stage.GetPrimAtPath(link_prim_path).IsValid()
    assert (
        not link_prim_exists
    ), f"Cannot create new link because there already exists a link at prim path {link_prim_path}!"

    # Manually create a new prim (specified offset)
    link_prim = lazy.pxr.UsdGeom.Xform.Define(stage, link_prim_path).GetPrim()
    _add_xform_properties(prim=link_prim)

    # Add rigid prim API to new link prim
    lazy.pxr.UsdPhysics.RigidBodyAPI.Apply(link_prim)
    lazy.pxr.PhysxSchema.PhysxRigidBodyAPI.Apply(link_prim)

    return link_prim


def add_sensor(stage, root_prim, sensor_type, link_name, parent_link_name=None, pos_offset=None, ori_offset=None):
    """
    Adds sensor to robot. This is an in-place operation on @root_prim

    Args:
        stage (Usd.Stage): Current active omniverse stage
        root_prim (Usd.Prim): Root prim of the current robot, assumed to be on the current stage
        sensor_type (str): Sensor to create. Valid options are: {Camera, Lidar, VisualSphere}
        link_name (str): Link to attach the created sensor prim to. If this link does not already exist in the robot's
            current set of links, a new one will be created as a child of @parent_link_name's link
        parent_link_name (None or str): If specified, parent link from which to create a new child link @link_name. If
            set, @link_name should NOT be a link already found on the robot!
        pos_offset (None or 3-tuple): If specified, (x,y,z) local translation offset to apply.
            If @parent_link_name is specified, defines offset of @link_name wrt @parent_link_name
            If only @link_name is specified, defines offset of the sensor prim wrt @link_name
        ori_offset (None or 3-tuple): If specified, (x,y,z,w) quaternion rotation offset to apply.
            If @parent_link_name is specified, defines offset of @link_name wrt @parent_link_name
            If only @link_name is specified, defines offset of the sensor prim wrt @link_name
    """
    # Make sure pos and ori offsets are defined
    if pos_offset is None or pos_offset == {}:  # May be {} from empty addict key
        pos_offset = (0, 0, 0)
    if ori_offset is None or ori_offset == {}:  # May be {} from empty addict key
        ori_offset = (0, 0, 0, 1)

    pos_offset = th.tensor(pos_offset, dtype=th.float)
    ori_offset = th.tensor(ori_offset, dtype=th.float)

    # Sanity check link / parent link combos
    root_prim_path = root_prim.GetPrimPath().pathString
    if parent_link_name is None or parent_link_name == {}:  # May be {} from empty addict key
        parent_link_prim = None
    else:
        parent_path = f"{root_prim_path}/{parent_link_name}"
        assert lazy.isaacsim.core.utils.prims.is_prim_path_valid(
            parent_path
        ), f"Could not find parent link within robot with name {parent_link_name}!"
        parent_link_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(parent_path)

    # If parent link is defined, link prim should NOT exist (this should be a new link)
    link_prim_path = f"{root_prim_path}/{link_name}"
    link_prim_exists = lazy.isaacsim.core.utils.prims.is_prim_path_valid(link_prim_path)
    if parent_link_prim is not None:
        assert not link_prim_exists, f"Since parent link is defined, link_name {link_name} must be a link that is NOT pre-existing within the robot's set of links!"
        # Manually create a new prim (specified offset)
        create_rigid_prim(
            stage=stage,
            link_prim_path=link_prim_path,
        )
        link_prim_xform = lazy.isaacsim.core.prims.xform_prim.XFormPrim(prim_path=link_prim_path)

        # Create fixed joint to connect the two together
        create_joint(
            prim_path=f"{parent_path}/{parent_link_name}_{link_name}_joint",
            joint_type="FixedJoint",
            body0=parent_path,
            body1=link_prim_path,
            joint_frame_in_parent_frame_pos=pos_offset,
            joint_frame_in_parent_frame_quat=ori_offset,
        )

        # Set child prim to the appropriate local pose -- this should be the parent local pose transformed by
        # the additional offset
        parent_prim_xform = lazy.isaacsim.core.prims.xform_prim.XFormPrim(prim_path=parent_path)
        parent_pos, parent_quat = parent_prim_xform.get_local_pose()
        parent_quat = parent_quat[[1, 2, 3, 0]]
        parent_pose = T.pose2mat((th.tensor(parent_pos), th.tensor(parent_quat)))
        offset_pose = T.pose2mat((pos_offset, ori_offset))
        child_pose = parent_pose @ offset_pose
        link_pos, link_quat = T.mat2pose(child_pose)
        link_prim_xform.set_local_pose(link_pos, link_quat[[3, 0, 1, 2]])

    else:
        # Otherwise, link prim MUST exist
        assert link_prim_exists, f"Since no parent link is defined, link_name {link_name} must be a link that IS pre-existing within the robot's set of links!"

    # Define functions to generate the desired sensor prim
    if sensor_type == "Camera":
        create_sensor_prim = lambda parent_prim_path: lazy.pxr.UsdGeom.Camera.Define(
            stage, f"{parent_prim_path}/Camera"
        ).GetPrim()
    elif sensor_type == "Lidar":
        create_sensor_prim = lambda parent_prim_path: lazy.omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/Lidar",
            parent=parent_prim_path,
            min_range=0.4,
            max_range=100.0,
            draw_points=False,
            draw_lines=False,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=0.4,
            vertical_resolution=4.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False,
        )[1].GetPrim()
    elif sensor_type == "VisualSphere":
        create_sensor_prim = lambda parent_prim_path: create_primitive_mesh(
            prim_path=f"{parent_prim_path}/VisualSphere",
            primitive_type="Sphere",
            extents=0.01,
            stage=stage,
        ).GetPrim()
    else:
        raise ValueError(f"Got unknown sensor type: {sensor_type}!")

    # Create the new prim as a child of the link prim
    sensor_prim = create_sensor_prim(parent_prim_path=link_prim_path)
    _add_xform_properties(sensor_prim)

    # If sensor prim is a camera, set some default values
    if sensor_type == "Camera":
        sensor_prim.GetAttribute("focalLength").Set(17.0)
        sensor_prim.GetAttribute("clippingRange").Set(lazy.pxr.Gf.Vec2f(0.001, 1000000.0))
        # Refresh visibility
        lazy.pxr.UsdGeom.Imageable(sensor_prim).MakeInvisible()
        og.sim.render()
        lazy.pxr.UsdGeom.Imageable(sensor_prim).MakeVisible()

    # If we didn't have a parent prim defined, we need to add the offset directly to this sensor
    if parent_link_prim is None:
        sensor_prim.GetAttribute("xformOp:translate").Set(lazy.pxr.Gf.Vec3d(*pos_offset.tolist()))
        sensor_prim.GetAttribute("xformOp:orient").Set(
            lazy.pxr.Gf.Quatd(*ori_offset[[3, 0, 1, 2]].tolist())
        )  # expects (w,x,y,z)


def _find_prim_with_condition(condition, root_prim):
    """
    Recursively searches children of @root_prim to find first instance of prim satisfying @condition

    Args:
        condition (function): Condition to check. Should satisfy function signature:

            def condition(prim: Usd.Prim) -> bool

            which returns True if the condition is met, else False

        root_prim (Usd.Prim): Root prim to search

    Returns:
        None or Usd.Prim: If found, first prim whose prim name includes @name
    """
    if condition(root_prim):
        return root_prim

    for child in root_prim.GetChildren():
        found_prim = _find_prim_with_condition(condition=condition, root_prim=child)
        if found_prim is not None:
            return found_prim


def _find_prims_with_condition(condition, root_prim):
    """
    Recursively searches children of @root_prim to find all instances of prim satisfying @condition

    Args:
        condition (function): Condition to check. Should satisfy function signature:

            def condition(prim: Usd.Prim) -> bool

            which returns True if the condition is met, else False

        root_prim (Usd.Prim): Root prim to search

    Returns:
        None or Usd.Prim: If found, first prim whose prim name includes @name
    """
    found_prims = []
    if condition(root_prim):
        found_prims.append(root_prim)

    for child in root_prim.GetChildren():
        found_prims += _find_prims_with_condition(condition=condition, root_prim=child)

    return found_prims


def find_prim_with_name(name, root_prim):
    """
    Recursively searches children of @root_prim to find first instance of prim including string @name

    Args:
        name (str): Name of the prim to search
        root_prim (Usd.Prim): Root prim to search

    Returns:
        None or Usd.Prim: If found, first prim whose prim name includes @name
    """
    return _find_prim_with_condition(condition=lambda prim: name in prim.GetName(), root_prim=root_prim)


def find_articulation_root_prim(root_prim):
    """
    Recursively searches children of @root_prim to find the articulation root

    Args:
        root_prim (Usd.Prim): Root prim to search

    Returns:
        None or Usd.Prim: If found, articulation root prim
    """
    return _find_prim_with_condition(
        condition=lambda prim: prim.HasAPI(lazy.pxr.UsdPhysics.ArticulationRootAPI), root_prim=root_prim
    )


def make_joint_fixed(stage, root_prim, joint_name):
    """
    Converts a revolute / prismatic joint @joint_name into a fixed joint

    NOTE: This is an in-place operation!

    Args:
        stage (Usd.Stage): Current active omniverse stage
        root_prim (Usd.Prim): Root prim of the current robot, assumed to be on the current stage
        joint_name (str): Joint to convert to be fixed
    """
    joint_prim = find_prim_with_name(name=joint_name, root_prim=root_prim)
    assert joint_prim is not None, f"Could not find joint prim with name {joint_name}!"

    # Remove its Joint APIs and add Fixed Joint API
    joint_type = joint_prim.GetTypeName()
    if joint_type != "PhysicsFixedJoint":
        assert joint_type in {
            "PhysicsRevoluteJoint",
            "PhysicsPrismaticJoint",
        }, f"Got invalid joint type: {joint_type}. Only PhysicsRevoluteJoint and PhysicsPrismaticJoint are supported!"

        lazy.omni.kit.commands.execute("RemovePhysicsComponentCommand", usd_prim=joint_prim, component=joint_type)
        lazy.pxr.UsdPhysics.FixedJoint.Define(stage, joint_prim.GetPrimPath().pathString)


def set_link_collision_approximation(stage, root_prim, link_name, approximation_type):
    """
    Sets all collision geoms under @link_name to be @approximation type
    Args:
        approximation_type (str): approximation used for collision.
            Can be one of: {"none", "convexHull", "convexDecomposition", "meshSimplification", "sdf",
                "boundingSphere", "boundingCube"}
            If None, the approximation will use the underlying triangle mesh.
    """
    # Sanity check approximation type
    assert_valid_key(
        key=approximation_type,
        valid_keys={
            "none",
            "convexHull",
            "convexDecomposition",
            "meshSimplification",
            "sdf",
            "boundingSphere",
            "boundingCube",
        },
        name="collision approximation type",
    )

    # Find the link prim first
    link_prim = find_prim_with_name(name=link_name, root_prim=root_prim)
    assert link_prim is not None, f"Could not find link prim with name {link_name}!"

    # Iterate through all children that are mesh prims
    mesh_prims = find_all_prim_children_with_type(prim_type="Mesh", root_prim=link_prim)

    # For each mesh prim, check if it is collision -- if so, update the approximation type appropriately
    for mesh_prim in mesh_prims:
        if not mesh_prim.HasAPI(lazy.pxr.UsdPhysics.MeshCollisionAPI):
            # This is a visual mesh, so skip
            continue
        mesh_collision_api = lazy.pxr.UsdPhysics.MeshCollisionAPI(mesh_prim)

        # Make sure to add the appropriate API if we're setting certain values
        if approximation_type == "convexHull" and not mesh_prim.HasAPI(
            lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI
        ):
            lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh_prim)
        elif approximation_type == "convexDecomposition" and not mesh_prim.HasAPI(
            lazy.pxr.PhysxSchema.PhysxConvexDecompositionCollisionAPI
        ):
            lazy.pxr.PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)
        elif approximation_type == "meshSimplification" and not mesh_prim.HasAPI(
            lazy.pxr.PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI
        ):
            lazy.pxr.PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI.Apply(mesh_prim)
        elif approximation_type == "sdf" and not mesh_prim.HasAPI(lazy.pxr.PhysxSchema.PhysxSDFMeshCollisionAPI):
            lazy.pxr.PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
        elif approximation_type == "none" and not mesh_prim.HasAPI(lazy.pxr.PhysxSchema.PhysxTriangleMeshCollisionAPI):
            lazy.pxr.PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh_prim)

        if approximation_type == "convexHull":
            pch_api = lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI(mesh_prim)
            # Also make sure the maximum vertex count is 60 (max number compatible with GPU)
            # https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html#collision-settings
            if pch_api.GetHullVertexLimitAttr().Get() is None:
                pch_api.CreateHullVertexLimitAttr()
            pch_api.GetHullVertexLimitAttr().Set(60)

        mesh_collision_api.GetApproximationAttr().Set(approximation_type)


def is_joint(prim, only_articulated=True):
    prim_type = prim.GetPrimTypeInfo().GetTypeName().lower()
    if only_articulated and "fixed" in prim_type:
        return False
    return "joint" in prim_type


def find_all_joints(root_prim, only_articulated=True):
    return _find_prims_with_condition(
        condition=lambda x: is_joint(x, only_articulated=only_articulated),
        root_prim=root_prim,
    )


def is_rigid_body(prim):
    prim_type = prim.GetPrimTypeInfo().GetTypeName().lower()
    has_rigid_api = prim.HasAPI(lazy.pxr.UsdPhysics.RigidBodyAPI)
    return "xform" in prim_type and has_rigid_api


def find_all_rigid_bodies(root_prim):
    return _find_prims_with_condition(
        condition=is_rigid_body,
        root_prim=root_prim,
    )


def is_mesh(prim):
    prim_type = prim.GetPrimTypeInfo().GetTypeName().lower()
    return "mesh" in prim_type


def find_all_meshes(root_prim):
    return _find_prims_with_condition(
        condition=is_mesh,
        root_prim=root_prim,
    )


def create_curobo_cfgs(robot_prim, robot_urdf_path, curobo_cfg, root_link, save_dir, is_holonomic=False):
    """
    Creates a set of curobo configs based on @robot_prim and @curobo_cfg

    Args:
        robot_prim (Usd.Prim): Top-level prim defining the robot in the current USD stage
        robot_urdf_path (str): Path to robot URDF file
        curobo_cfg (Dict): Dictionary of relevant curobo information
        root_link (str): Name of the robot's root link, BEFORE any holonomic joints are applied
        save_dir (str): Path to the directory to save generated curobo files
        is_holonomic (bool): Whether the robot has a holonomic base applied or not
    """
    robot_name = robot_prim.GetName()

    # Left, then right by default if sorted alphabetically
    ee_links = list(sorted(curobo_cfg.eef_to_gripper_info.keys()))

    # Find all joints that have a negative axis specified so we know to flip them in curobo
    tree = ET.parse(robot_urdf_path)
    root = tree.getroot()
    flip_joints = dict()
    flip_joint_limits = []
    for joint in root.findall("joint"):
        if joint.attrib["type"] != "fixed":
            axis = th.round(_space_string_to_tensor(joint.find("axis").attrib["xyz"]))
            axis_idx = th.nonzero(axis).squeeze().item()
            flip_joints[joint.attrib["name"]] = "XYZ"[axis_idx]
            is_negative = (axis[axis_idx] < 0).item()
            if is_negative:
                flip_joint_limits.append(joint.attrib["name"])

    def get_joint_upper_limit(root_prim, joint_name):
        joint_prim = find_prim_with_name(name=joint_name, root_prim=root_prim)
        assert joint_prim is not None, f"Could not find joint prim with name {joint_name}!"
        return joint_prim.GetAttribute("physics:upperLimit").Get()

    # The original format from Lula is a list of dicts, so we need to convert to a single dict
    if isinstance(curobo_cfg.collision_spheres, list):
        collision_spheres = {k: v for c in curobo_cfg.collision_spheres for k, v in c.to_dict().items()}
    else:
        collision_spheres = curobo_cfg.collision_spheres.to_dict()

    # Generate list of collision link names -- this is simply the list of all link names from the
    # collision spheres specification
    all_collision_link_names = list(collision_spheres.keys())

    joint_prims = find_all_joints(robot_prim, only_articulated=True)
    all_joint_names = [joint_prim.GetName() for joint_prim in joint_prims]
    lock_joints = curobo_cfg.lock_joints.to_dict() if curobo_cfg.lock_joints else {}
    if is_holonomic:
        # Move the final six joints to the beginning, since the holonomic joints are added at the end
        all_joint_names = list(reversed(all_joint_names[-6:])) + all_joint_names[:-6]
        lock_joints["base_footprint_z_joint"] = None
        lock_joints["base_footprint_rx_joint"] = None
        lock_joints["base_footprint_ry_joint"] = None

    default_generated_cfg = {
        "usd_robot_root": f"/{robot_prim.GetName()}",
        "usd_flip_joints": flip_joints,
        "usd_flip_joint_limits": flip_joint_limits,
        "base_link": "base_footprint_x" if is_holonomic else root_link,
        "ee_link": ee_links[0],
        "link_names": ee_links[1:],
        "lock_joints": lock_joints,
        "extra_links": {},
        "collision_link_names": deepcopy(all_collision_link_names),
        "collision_spheres": collision_spheres,
        "collision_sphere_buffer": 0.002,
        "extra_collision_spheres": {},
        "self_collision_ignore": curobo_cfg.self_collision_ignore.to_dict(),
        "self_collision_buffer": curobo_cfg.self_collision_buffer.to_dict(),
        "use_global_cumul": True,
        "mesh_link_names": deepcopy(all_collision_link_names),
        "external_asset_path": None,
        "cspace": {
            "joint_names": all_joint_names,
            "retract_config": None,
            "null_space_weight": [1] * len(all_joint_names),
            "cspace_distance_weight": [1] * len(all_joint_names),
            "max_jerk": 500.0,
            "max_acceleration": 15.0,
        },
    }

    for eef_link_name, gripper_info in curobo_cfg.eef_to_gripper_info.items():
        attached_obj_link_name = f"attached_object_{eef_link_name}"
        for jnt_name in gripper_info["joints"]:
            default_generated_cfg["lock_joints"][jnt_name] = None
        default_generated_cfg["extra_links"][attached_obj_link_name] = {
            "parent_link_name": eef_link_name,
            "link_name": attached_obj_link_name,
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],  # (x,y,z,w,x,y,z)
            "joint_type": "FIXED",
            "joint_name": f"{attached_obj_link_name}_joint",
        }
        default_generated_cfg["collision_link_names"].append(attached_obj_link_name)
        default_generated_cfg["extra_collision_spheres"][attached_obj_link_name] = 32
        for link_name in gripper_info["links"]:
            if link_name not in default_generated_cfg["self_collision_ignore"]:
                default_generated_cfg["self_collision_ignore"][link_name] = []
            default_generated_cfg["self_collision_ignore"][link_name].append(attached_obj_link_name)
        default_generated_cfg["mesh_link_names"].append(attached_obj_link_name)

    # Save generated file
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fpath = f"{save_dir}/{robot_name}_description_curobo_default.yaml"
    with open(save_fpath, "w+") as f:
        yaml.dump({"robot_cfg": {"kinematics": default_generated_cfg}}, f)

    # Permute the default config to have additional base only / arm only configs
    # Only relevant if robot is holonomic
    if is_holonomic:
        # Create base only config
        base_only_cfg = deepcopy(default_generated_cfg)
        base_only_cfg["ee_link"] = root_link
        base_only_cfg["link_names"] = []
        for jnt_name in all_joint_names:
            if jnt_name not in base_only_cfg["lock_joints"] and "base_footprint" not in jnt_name:
                # Lock this joint
                base_only_cfg["lock_joints"][jnt_name] = None
        save_base_fpath = f"{save_dir}/{robot_name}_description_curobo_base.yaml"
        with open(save_base_fpath, "w+") as f:
            yaml.dump({"robot_cfg": {"kinematics": base_only_cfg}}, f)

        # Create arm only config
        arm_only_cfg = deepcopy(default_generated_cfg)
        for jnt_name in {"base_footprint_x_joint", "base_footprint_y_joint", "base_footprint_rz_joint"}:
            arm_only_cfg["lock_joints"][jnt_name] = None
        save_arm_fpath = f"{save_dir}/{robot_name}_description_curobo_arm.yaml"
        with open(save_arm_fpath, "w+") as f:
            yaml.dump({"robot_cfg": {"kinematics": arm_only_cfg}}, f)

        # Create arm only no torso config
        arm_only_no_torso_cfg = deepcopy(arm_only_cfg)
        for jnt_name in curobo_cfg.torso_joints:
            arm_only_no_torso_cfg["lock_joints"][jnt_name] = None
        save_arm_no_torso_fpath = f"{save_dir}/{robot_name}_description_curobo_arm_no_torso.yaml"
        with open(save_arm_no_torso_fpath, "w+") as f:
            yaml.dump({"robot_cfg": {"kinematics": arm_only_no_torso_cfg}}, f)


@click.command(help=_DOCSTRING)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Absolute path to robot config yaml file to import",
)
def import_custom_robot(config):
    # Load config
    with open(config, "r") as f:
        cfg = Dict(yaml.load(f, yaml.Loader))

    # Convert URDF -> USD
    urdf_path, usd_path, prim = import_og_asset_from_urdf(
        category="robot",
        model=cfg.name,
        urdf_path=cfg.urdf_path,
        collision_method=cfg.collision.decompose_method,
        coacd_links=cfg.collision.coacd_links,
        convex_links=cfg.collision.convex_links,
        no_decompose_links=cfg.collision.no_decompose_links,
        visual_only_links=cfg.collision.no_collision_links,
        merge_fixed_joints=cfg.merge_fixed_joints,
        hull_count=cfg.collision.hull_count,
        overwrite=cfg.overwrite,
        use_usda=True,
    )

    # Get current stage
    stage = lazy.isaacsim.core.utils.stage.get_current_stage()

    # Add visual spheres, cameras, and lidars
    if cfg.eef_vis_links:
        for eef_vis_info in cfg.eef_vis_links:
            add_sensor(
                stage=stage,
                root_prim=prim,
                sensor_type="VisualSphere",
                link_name=eef_vis_info.link,
                parent_link_name=eef_vis_info.parent_link,
                pos_offset=eef_vis_info.offset.position,
                ori_offset=eef_vis_info.offset.orientation,
            )
    if cfg.camera_links:
        for camera_info in cfg.camera_links:
            add_sensor(
                stage=stage,
                root_prim=prim,
                sensor_type="Camera",
                link_name=camera_info.link,
                parent_link_name=camera_info.parent_link,
                pos_offset=camera_info.offset.position,
                ori_offset=camera_info.offset.orientation,
            )
    if cfg.lidar_links:
        for lidar_info in cfg.lidar_links:
            add_sensor(
                stage=stage,
                root_prim=prim,
                sensor_type="Lidar",
                link_name=lidar_info.link,
                parent_link_name=lidar_info.parent_link,
                pos_offset=lidar_info.offset.position,
                ori_offset=lidar_info.offset.orientation,
            )

    # Make wheels sphere approximations if requested
    if cfg.base_motion.use_sphere_wheels:
        for wheel_link in cfg.base_motion.wheel_links:
            set_link_collision_approximation(
                stage=stage,
                root_prim=prim,
                link_name=wheel_link,
                approximation_type="boundingSphere",
            )

    # Get reference to articulation root link
    articulation_root_prim = find_articulation_root_prim(root_prim=prim)
    assert articulation_root_prim is not None, "Could not find any valid articulation root prim!"
    root_prim_name = articulation_root_prim.GetName()

    # We always want our robot to have its canonical frame corresponding to the bottom surface
    # So, we compute the AABB dynamically to calculate the current z-offset, and then apply the offset to the following:
    #   - root link CoM
    #   - root link's visual / collision meshes
    #   - all of the root link's immediate child joints
    #   - all of the root link's descendant links (all links except self)

    # Compute AABB
    bbox_cache = lazy.isaacsim.core.utils.bounds.create_bbox_cache(use_extents_hint=False)
    aabb = lazy.isaacsim.core.utils.bounds.compute_aabb(bbox_cache=bbox_cache, prim_path=prim.GetPrimPath().pathString)
    z_offset = aabb[2]

    # Update the root link's CoM
    com_attr = articulation_root_prim.GetAttribute("physics:centerOfMass")
    com = com_attr.Get()
    com[2] -= z_offset
    com_attr.Set(com)

    # Grab all of the root prim's nested children and joints, and modify their offsets based on AABB z-lower bound
    root_meshes = find_all_meshes(root_prim=articulation_root_prim)
    for mesh in root_meshes:
        translate_attr = mesh.GetAttribute("xformOp:translate")
        local_pos = translate_attr.Get()
        local_pos[2] -= z_offset
        translate_attr.Set(local_pos)
    root_joints = find_all_joints(root_prim=articulation_root_prim, only_articulated=False)
    root_prim_path = articulation_root_prim.GetPrimPath().pathString
    for joint in root_joints:
        body0_targets = joint.GetProperty("physics:body0").GetTargets()
        # Don't include any joints where the articulation root link is not the parent
        if len(body0_targets) == 0 or body0_targets[0].pathString != root_prim_path:
            continue
        pos0_attr = joint.GetAttribute("physics:localPos0")
        pos0 = pos0_attr.Get()
        pos0[2] -= z_offset
        pos0_attr.Set(pos0)

    # Update all links that are not the root link
    all_links = find_all_rigid_bodies(root_prim=prim)
    for link in all_links:
        if link == articulation_root_prim:
            continue
        translate_attr = link.GetAttribute("xformOp:translate")
        local_pos = translate_attr.Get()
        local_pos[2] -= z_offset
        translate_attr.Set(local_pos)

    # Add holonomic base if requested
    if cfg.base_motion.use_holonomic_joints:
        # Convert all wheel joints into fixed joints
        for wheel_joint in cfg.base_motion.wheel_joints:
            make_joint_fixed(
                stage=stage,
                root_prim=prim,
                joint_name=wheel_joint,
            )

        # Remove the articulation root from the original root link
        articulation_root_prim.RemoveAPI(lazy.pxr.UsdPhysics.ArticulationRootAPI)

        # Add 6DOF virtual joints ("base_footprint_<AXIS>")
        # Create in backwards order so that the child always exists
        child_prim = articulation_root_prim
        robot_prim_path = prim.GetPrimPath().pathString
        for prefix, joint_type, drive_type in zip(("r", ""), ("Revolute", "Prismatic"), ("angular", "linear")):
            for axis in ("z", "y", "x"):
                joint_suffix = f"{prefix}{axis}"
                parent_name = f"base_footprint_{joint_suffix}"
                # Create new link
                parent_prim_path = f"{robot_prim_path}/{parent_name}"
                parent_prim = create_rigid_prim(
                    stage=stage,
                    link_prim_path=parent_prim_path,
                )

                # Create new joint
                joint_prim_path = f"{parent_prim_path}/{parent_name}_joint"
                joint = create_joint(
                    prim_path=joint_prim_path,
                    joint_type=f"{joint_type}Joint",
                    body0=parent_prim_path,
                    body1=child_prim.GetPrimPath().pathString,
                )
                joint.GetAttribute("physics:axis").Set(axis.upper())

                # Add JointState API, and also Drive API only if the joint is in {x,y,rz}
                lazy.pxr.PhysxSchema.JointStateAPI.Apply(joint, drive_type)
                if joint_suffix in {"x", "y", "rz"}:
                    lazy.pxr.UsdPhysics.DriveAPI.Apply(joint, drive_type)

                # Update child
                child_prim = parent_prim

        # Re-add the articulation root API to the new virtual footprint link
        lazy.pxr.UsdPhysics.ArticulationRootAPI.Apply(parent_prim)

    # Save stage
    stage.Save()

    # Import auxiliary files necessary for CuRobo motion planning
    if bool(cfg.curobo):
        create_curobo_cfgs(
            robot_prim=prim,
            robot_urdf_path=urdf_path,
            root_link=root_prim_name,
            curobo_cfg=cfg.curobo,
            save_dir="/".join(usd_path.split("/")[:-2]) + "/curobo",
            is_holonomic=cfg.base_motion.use_holonomic_joints,
        )

    # Visualize if not headless
    if not cfg.headless:
        click.echo("The asset has been successfully imported. You can view it and make changes and save if you'd like.")
        while True:
            og.sim.render()


if __name__ == "__main__":
    import_custom_robot()
