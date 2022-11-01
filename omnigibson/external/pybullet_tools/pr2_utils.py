"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by iGibson team.
"""
import math
import os
import random
import re
from collections import namedtuple
from itertools import combinations

import numpy as np

from .pr2_never_collisions import NEVER_COLLISIONS
from .utils import multiply, get_link_pose, set_joint_position, joints_from_names, \
    set_joint_positions, get_joint_positions, get_min_limit, get_max_limit, quat_from_euler, read_pickle, set_pose, get_pose, euler_from_quat, link_from_name, point_from_pose, invert, Pose, \
    unit_pose, joints_from_names, PoseSaver, get_aabb, get_joint_limits, ConfSaver, get_bodies, create_mesh, remove_body, unit_from_theta, angle_between, violates_limit, \
    violates_limits, add_line, get_body_name, get_num_joints, approximate_as_cylinder, \
    approximate_as_prism, unit_quat, unit_point, clip, tform_point, angle_between, quat_from_pose, compute_jacobian, \
    movable_from_joints, quat_from_axis_angle, LockRenderer, Euler, get_links, get_link_name,\
    get_extend_fn, get_moving_links, link_pairs_collision, get_link_subtree, \
    clone_body, get_all_links, set_color, pairwise_collision, tform_point

# TODO: restrict number of pr2 rotations to prevent from wrapping too many times

ARM_NAMES = ('left', 'right')

def arm_from_arm(arm): # TODO: rename
    assert (arm in ARM_NAMES)
    return '{}_arm'.format(arm)

def gripper_from_arm(arm):
    assert (arm in ARM_NAMES)
    return '{}_gripper'.format(arm)


PR2_GROUPS = {
    'base': ['x', 'y', 'theta'],
    'torso': ['torso_lift_joint'],
    'head': ['head_pan_joint', 'head_tilt_joint'],
    arm_from_arm('left'): ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint',
                 'l_elbow_flex_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint'],
    arm_from_arm('right'): ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
                  'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint'],
    gripper_from_arm('left'): ['l_gripper_l_finger_joint', 'l_gripper_r_finger_joint',
                     'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_tip_joint'],
    gripper_from_arm('right'): ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint',
                      'r_gripper_l_finger_tip_joint', 'r_gripper_r_finger_tip_joint'],
    # r_gripper_joint & l_gripper_joint are not mimicked
}

HEAD_LINK_NAME = 'high_def_optical_frame' # high_def_optical_frame | high_def_frame | wide_stereo_l_stereo_camera_frame
# kinect - 'head_mount_kinect_rgb_optical_frame' | 'head_mount_kinect_rgb_link'

PR2_TOOL_FRAMES = {
    'left': 'l_gripper_tool_frame',  # l_gripper_palm_link | l_gripper_tool_frame
    'right': 'r_gripper_tool_frame',  # r_gripper_palm_link | r_gripper_tool_frame
    'head': HEAD_LINK_NAME,
}

PR2_GRIPPER_ROOTS = {
    'left': 'l_gripper_palm_link',
    'right': 'r_gripper_palm_link',
}

PR2_BASE_LINK = 'base_footprint'

# Arm tool poses
#TOOL_POSE = ([0.18, 0., 0.], [0., 0.70710678, 0., 0.70710678]) # l_gripper_palm_link
TOOL_POSE = Pose(euler=Euler(pitch=np.pi/2)) # l_gripper_tool_frame (+x out of gripper arm)
#TOOL_DIRECTION = [0., 0., 1.]

#####################################

# Special configurations

TOP_HOLDING_LEFT_ARM = [0.67717021, -0.34313199, 1.2, -1.46688405, 1.24223229, -1.95442826, 2.22254125]
SIDE_HOLDING_LEFT_ARM = [0.39277395, 0.33330058, 0., -1.52238431, 2.72170996, -1.21946936, -2.98914779]
REST_LEFT_ARM = [2.13539289, 1.29629967, 3.74999698, -0.15000005, 10000., -0.10000004, 10000.]
WIDE_LEFT_ARM = [1.5806603449288885, -0.14239066980481405, 1.4484623937179126, -1.4851759349218694, 1.3911839347271555,
                 -1.6531320011389408, -2.978586584568441]
CENTER_LEFT_ARM = [-0.07133691252641006, -0.052973836083405494, 1.5741805775919033, -1.4481146328076862,
                   1.571782540186805, -1.4891468812835686, -9.413338322697955]
# WIDE_RIGHT_ARM = [-1.3175723551150083, -0.09536552225976803, -1.396727055561703, -1.4433371993320296, -1.5334243909312468, -1.7298129320065025, 6.230244924007009]

PR2_LEFT_CARRY_CONFS = {
    'top': TOP_HOLDING_LEFT_ARM,
    'side': SIDE_HOLDING_LEFT_ARM,
}

#####################################

PR2_URDF = "models/pr2_description/pr2.urdf" # 87 joints
#PR2_URDF = "models/pr2_description/pr2_hpn.urdf"
#PR2_URDF = "models/pr2_description/pr2_kinect.urdf"
DRAKE_PR2_URDF = "models/drake/pr2_description/urdf/pr2_simplified.urdf" # 82 joints

def is_drake_pr2(robot): # 87
    return (get_body_name(robot) == 'pr2') and (get_num_joints(robot) == 82)

#####################################

# TODO: for when the PR2 is copied and loses it's joint names
# PR2_JOINT_NAMES = []
#
# def set_pr2_joint_names(pr2):
#     for joint in get_joints(pr2):
#         PR2_JOINT_NAMES.append(joint)
#
# def get_pr2_joints(joint_names):
#     joint_from_name = dict(zip(PR2_JOINT_NAMES, range(len(PR2_JOINT_NAMES))))
#     return [joint_from_name[name] for name in joint_names]

#####################################

def get_base_pose(pr2):
    return get_link_pose(pr2, link_from_name(pr2, PR2_BASE_LINK))

def rightarm_from_leftarm(config):
    right_from_left = np.array([-1, 1, -1, 1, -1, 1, -1])
    return config * right_from_left

def arm_conf(arm, left_config):
    if arm == 'left':
        return left_config
    elif arm == 'right':
        return rightarm_from_leftarm(left_config)
    raise ValueError(arm)

def get_carry_conf(arm, grasp_type):
    return arm_conf(arm, PR2_LEFT_CARRY_CONFS[grasp_type])

def get_other_arm(arm):
    for other_arm in ARM_NAMES:
        if other_arm != arm:
            return other_arm
    raise ValueError(arm)

#####################################

def get_disabled_collisions(pr2):
    #disabled_names = PR2_ADJACENT_LINKS
    #disabled_names = PR2_DISABLED_COLLISIONS
    disabled_names = NEVER_COLLISIONS
    #disabled_names = PR2_DISABLED_COLLISIONS + NEVER_COLLISIONS
    link_mapping = {get_link_name(pr2, link): link for link in get_links(pr2)}
    return {(link_mapping[name1], link_mapping[name2])
            for name1, name2 in disabled_names if (name1 in link_mapping) and (name2 in link_mapping)}


def load_dae_collisions():
    # pr2-beta-static.dae: link 0 = base_footprint
    # pybullet: link -1 = base_footprint
    dae_file = 'models/pr2_description/pr2-beta-static.dae'
    dae_string = open(dae_file).read()
    link_regex = r'<\s*link\s+sid="(\w+)"\s+name="(\w+)"\s*>'
    link_mapping = dict(re.findall(link_regex, dae_string))
    ignore_regex = r'<\s*ignore_link_pair\s+link0="kmodel1/(\w+)"\s+link1="kmodel1/(\w+)"\s*/>'
    disabled_collisions = []
    for link1, link2 in re.findall(ignore_regex, dae_string):
        disabled_collisions.append((link_mapping[link1], link_mapping[link2]))
    return disabled_collisions


def load_srdf_collisions():
    srdf_file = 'models/pr2_description/pr2.srdf'
    srdf_string = open(srdf_file).read()
    regex = r'<\s*disable_collisions\s+link1="(\w+)"\s+link2="(\w+)"\s+reason="(\w+)"\s*/>'
    disabled_collisions = []
    for link1, link2, reason in re.findall(regex, srdf_string):
        if reason == 'Never':
            disabled_collisions.append((link1, link2))
    return disabled_collisions

#####################################

def get_group_joints(robot, group):
    return joints_from_names(robot, PR2_GROUPS[group])

def get_group_conf(robot, group):
    return get_joint_positions(robot, get_group_joints(robot, group))

def set_group_conf(robot, group, positions):
    set_joint_positions(robot, get_group_joints(robot, group), positions)

#####################################

# End-effectors

def get_arm_joints(robot, arm):
    return get_group_joints(robot, arm_from_arm(arm))


def get_torso_arm_joints(robot, arm):
    return joints_from_names(robot, PR2_GROUPS['torso'] + PR2_GROUPS[arm_from_arm(arm)])


#def get_arm_conf(robot, arm):
#    return get_joint_positions(robot, get_arm_joints(robot, arm))


def set_arm_conf(robot, arm, conf):
    set_joint_positions(robot, get_arm_joints(robot, arm), conf)


def get_gripper_link(robot, arm):
    assert arm in ARM_NAMES
    return link_from_name(robot, PR2_TOOL_FRAMES[arm])


# def get_gripper_pose(robot):
#    # world_from_gripper * gripper_from_tool * tool_from_object = world_from_object
#    pose = multiply(get_link_pose(robot, link_from_name(robot, LEFT_ARM_LINK)), TOOL_POSE)
#    #pose = get_link_pose(robot, link_from_name(robot, LEFT_TOOL_NAME))
#    return pose


def get_gripper_joints(robot, arm):
    return get_group_joints(robot, gripper_from_arm(arm))


def set_gripper_position(robot, arm, position):
    gripper_joints = get_gripper_joints(robot, arm)
    set_joint_positions(robot, gripper_joints, [position] * len(gripper_joints))


def open_arm(robot, arm): # These are mirrored on the pr2
    for joint in get_gripper_joints(robot, arm):
        set_joint_position(robot, joint, get_max_limit(robot, joint))


def close_arm(robot, arm):
    for joint in get_gripper_joints(robot, arm):
        set_joint_position(robot, joint, get_min_limit(robot, joint))

# TODO: use these names
open_gripper = open_arm
close_gripper = close_arm

#####################################

# Box grasps

#GRASP_LENGTH = 0.04
GRASP_LENGTH = 0.
#GRASP_LENGTH = -0.01

#MAX_GRASP_WIDTH = 0.07
MAX_GRASP_WIDTH = np.inf

SIDE_HEIGHT_OFFSET = 0.03 # z distance from top of object

def get_top_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                   max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH):
    # TODO: rename the box grasps
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, h / 2 - grasp_length])
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    grasps = []
    if w <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, math.pi / 2 + i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    if l <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    return grasps

def get_side_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                    max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH, top_offset=SIDE_HEIGHT_OFFSET):
    # TODO: compute bounding box width wrt tool frame
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    grasps = []
    #x_offset = 0
    x_offset = h/2 - top_offset
    for j in range(1 + under):
        swap_xz = Pose(euler=[0, -math.pi / 2 + j * math.pi, 0])
        if w <= max_width:
            translate_z = Pose(point=[x_offset, 0, l / 2 - grasp_length])
            for i in range(2):
                rotate_z = Pose(euler=[math.pi / 2 + i * math.pi, 0, 0])
                grasps += [multiply(tool_pose, translate_z, rotate_z, swap_xz,
                                    translate_center, body_pose)]  # , np.array([w])
        if l <= max_width:
            translate_z = Pose(point=[x_offset, 0, w / 2 - grasp_length])
            for i in range(2):
                rotate_z = Pose(euler=[i * math.pi, 0, 0])
                grasps += [multiply(tool_pose, translate_z, rotate_z, swap_xz,
                                    translate_center, body_pose)]  # , np.array([l])
    return grasps

#####################################

# Cylinder grasps

def get_top_cylinder_grasps(body, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                            max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH):
    # Apply transformations right to left on object pose
    center, (diameter, height) = approximate_as_cylinder(body, body_pose=body_pose)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, height / 2 - grasp_length])
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    if max_width < diameter:
        return
    while True:
        theta = random.uniform(0, 2*np.pi)
        rotate_z = Pose(euler=[0, 0, theta])
        yield multiply(tool_pose, translate_z, rotate_z,
                       reflect_z, translate_center, body_pose)

def get_side_cylinder_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                             max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH,
                             top_offset=SIDE_HEIGHT_OFFSET):
    center, (diameter, height) = approximate_as_cylinder(body, body_pose=body_pose)
    translate_center = Pose(point_from_pose(body_pose)-center)
    #x_offset = 0
    x_offset = height/2 - top_offset
    if max_width < diameter:
        return
    while True:
        theta = random.uniform(0, 2*np.pi)
        translate_rotate = ([x_offset, 0, diameter / 2 - grasp_length], quat_from_euler([theta, 0, 0]))
        for j in range(1 + under):
            swap_xz = Pose(euler=[0, -math.pi / 2 + j * math.pi, 0])
            yield multiply(tool_pose, translate_rotate, swap_xz, translate_center, body_pose)

def get_edge_cylinder_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                             grasp_length=GRASP_LENGTH):
    center, (diameter, height) = approximate_as_cylinder(body, body_pose=body_pose)
    translate_yz = Pose(point=[0, diameter/2, height/2 - grasp_length])
    reflect_y = Pose(euler=[0, math.pi, 0])
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    while True:
        theta = random.uniform(0, 2*np.pi)
        rotate_z = Pose(euler=[0, 0, theta])
        for i in range(1 + under):
            rotate_under = Pose(euler=[0, 0, i * math.pi])
            yield multiply(tool_pose, rotate_under, translate_yz, rotate_z,
                           reflect_y, translate_center, body_pose)

#####################################

# Cylinder pushes

def get_cylinder_push(body, theta, under=False, body_quat=unit_quat(),
                      tilt=0., base_offset=0.02, side_offset=0.03):
    body_pose = (unit_point(), body_quat)
    center, (diameter, height) = approximate_as_cylinder(body, body_pose=body_pose)
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    tilt_gripper = Pose(euler=Euler(pitch=tilt))
    translate_x = Pose(point=[-diameter / 2 - side_offset, 0, 0]) # Compute as a function of theta
    translate_z = Pose(point=[0, 0, -height / 2 + base_offset])
    rotate_x = Pose(euler=Euler(yaw=theta))
    reflect_z = Pose(euler=Euler(pitch=math.pi))
    grasps = []
    for i in range(1 + under):
        rotate_z = Pose(euler=Euler(yaw=i * math.pi))
        grasps.append(multiply(tilt_gripper, translate_x, translate_z, rotate_x, rotate_z,
                               reflect_z, translate_center, body_pose))
    return grasps

#####################################

# Button presses

PRESS_OFFSET = 0.02

def get_x_presses(body, max_orientations=1, body_pose=unit_pose(), top_offset=PRESS_OFFSET):
    # gripper_from_object
    # TODO: update
    center, (w, _, h) = approximate_as_prism(body, body_pose=body_pose)
    translate_center = Pose(-center)
    press_poses = []
    for j in range(max_orientations):
        swap_xz = Pose(euler=[0, -math.pi / 2 + j * math.pi, 0])
        translate = Pose(point=[0, 0, w / 2 + top_offset])
        press_poses += [multiply(TOOL_POSE, translate, swap_xz, translate_center, body_pose)]
    return press_poses

def get_top_presses(body, tool_pose=TOOL_POSE, body_pose=unit_pose(), top_offset=PRESS_OFFSET, **kwargs):
    center, (_, height) = approximate_as_cylinder(body, body_pose=body_pose, **kwargs)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, height / 2 + top_offset])
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    while True:
        theta = random.uniform(0, 2*np.pi)
        rotate_z = Pose(euler=[0, 0, theta])
        yield multiply(tool_pose, translate_z, rotate_z,
                       reflect_z, translate_center, body_pose)

GET_GRASPS = {
    'top': get_top_grasps,
    'side': get_side_grasps,
    # 'press': get_x_presses,
}
# TODO: include approach/carry info

#####################################

# Inverse reachability

DATABASES_DIR = '../databases'
IR_FILENAME = '{}_{}_ir.pickle'
IR_CACHE = {}

def get_database_file(filename):
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, DATABASES_DIR, filename)


def load_inverse_reachability(arm, grasp_type):
    key =  (arm, grasp_type)
    if key not in IR_CACHE:
        filename = IR_FILENAME.format(grasp_type, arm)
        path = get_database_file(filename)
        IR_CACHE[key] = read_pickle(path)['gripper_from_base']
    return IR_CACHE[key]


def learned_forward_generator(robot, base_pose, arm, grasp_type):
    gripper_from_base_list = list(load_inverse_reachability(arm, grasp_type))
    random.shuffle(gripper_from_base_list)
    for gripper_from_base in gripper_from_base_list:
        yield multiply(base_pose, invert(gripper_from_base))


def learned_pose_generator(robot, gripper_pose, arm, grasp_type):
    # TODO: record collisions with the reachability database
    gripper_from_base_list = load_inverse_reachability(arm, grasp_type)
    random.shuffle(gripper_from_base_list)
    #handles = []
    for gripper_from_base in gripper_from_base_list:
        base_point, base_quat = multiply(gripper_pose, gripper_from_base)
        x, y, _ = base_point
        _, _, theta = euler_from_quat(base_quat)
        base_values = (x, y, theta)
        #handles.extend(draw_point(np.array([x, y, -0.1]), color=(1, 0, 0), size=0.05))
        #set_base_values(robot, base_values)
        #yield get_pose(robot)
        yield base_values

#####################################

# Camera

# TODO: this is only for high_def_optical_frame
MAX_VISUAL_DISTANCE = 5.0
MAX_KINECT_DISTANCE = 2.5

def get_camera_matrix(width, height, fx, fy):
    # cx, cy = 320.5, 240.5
    cx, cy = width / 2., height / 2.
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

def clip_pixel(pixel, width, height):
    x, y = pixel
    return clip(x, 0, width-1), clip(y, 0, height-1)

def ray_from_pixel(camera_matrix, pixel):
    return np.linalg.inv(camera_matrix).dot(np.append(pixel, 1))

def pixel_from_ray(camera_matrix, ray):
    return camera_matrix.dot(np.array(ray) / ray[2])[:2]

PR2_CAMERA_MATRIX = get_camera_matrix(
    width=640, height=480, fx=772.55, fy=772.5)

def dimensions_from_camera_matrix(camera_matrix):
    width, height = 2 * np.array(camera_matrix)[:2, 2]
    return width, height

def get_pr2_view_section(z, camera_matrix=None):
    if camera_matrix is None:
        camera_matrix = PR2_CAMERA_MATRIX
    width, height = dimensions_from_camera_matrix(camera_matrix)
    pixels = [(0, 0), (width, height)]
    return [z*ray_from_pixel(camera_matrix, p) for p in pixels]

def get_pr2_field_of_view(**kwargs):
    z = 1
    view_lower, view_upper = get_pr2_view_section(z=z, **kwargs)
    horizontal = angle_between([view_lower[0], 0, z],
                               [view_upper[0], 0, z]) # 0.7853966439794928
    vertical = angle_between([0, view_lower[1], z],
                             [0, view_upper[1], z]) # 0.6024511557247721
    return horizontal, vertical

def is_visible_point(camera_matrix, depth, point_world, camera_pose=unit_pose()):
    point_camera = tform_point(invert(camera_pose), point_world)
    if not (0 <= point_camera[2] < depth):
        return False
    px, py = pixel_from_ray(camera_matrix, point_camera)
    width, height = dimensions_from_camera_matrix(camera_matrix)
    return (0 <= px < width) and (0 <= py < height)

def is_visible_aabb(aabb, **kwargs):
    # TODO: do intersect as well for identifying new obstacles
    body_lower, body_upper = aabb
    z = body_lower[2]
    if z < 0:
        return False
    view_lower, view_upper = get_pr2_view_section(z, **kwargs)
    # TODO: bounding box methods?
    return not (np.any(body_lower[:2] < view_lower[:2]) or
                np.any(view_upper[:2] < body_upper[:2]))

def support_from_aabb(aabb):
    lower, upper = aabb
    min_x, min_y, z = lower
    max_x, max_y, _ = upper
    return [(min_x, min_y, z), (min_x, max_y, z),
            (max_x, max_y, z), (max_x, min_y, z)]

#####################################

def cone_vertices_from_base(base):
    return [np.zeros(3)] + base

def cone_wires_from_support(support):
    #vertices = cone_vertices_from_base(support)
    # TODO: could obtain from cone_mesh_from_support
    # TODO: could also just return vertices and indices
    apex = np.zeros(3)
    lines = []
    for vertex in support:
        lines.append((apex, vertex))
    #for i, v2 in enumerate(support):
    #    v1 = support[i-1]
    #    lines.append((v1, v2))
    for v1, v2 in combinations(support, 2):
        lines.append((v1, v2))
    center = np.average(support, axis=0)
    lines.append((apex, center))
    return lines

def cone_mesh_from_support(support):
    assert(len(support) == 4)
    vertices = cone_vertices_from_base(support)
    faces = [(1, 4, 3), (1, 3, 2)]
    for i in range(len(support)):
        index1 = 1+i
        index2 = 1+(i+1)%len(support)
        faces.append((0, index1, index2))
    return vertices, faces

def get_viewcone_base(depth=MAX_VISUAL_DISTANCE, camera_matrix=None):
    if camera_matrix is None:
        camera_matrix = PR2_CAMERA_MATRIX
    width, height = dimensions_from_camera_matrix(camera_matrix)
    vertices = []
    for pixel in [(0, 0), (width, 0), (width, height), (0, height)]:
        ray = depth * ray_from_pixel(camera_matrix, pixel)
        vertices.append(ray[:3])
    return vertices

def get_viewcone(depth=MAX_VISUAL_DISTANCE, camera_matrix=None, **kwargs):
    mesh = cone_mesh_from_support(get_viewcone_base(
        depth=depth, camera_matrix=camera_matrix))
    assert (mesh is not None)
    return create_mesh(mesh, **kwargs)

def attach_viewcone(robot, head_name=HEAD_LINK_NAME, depth=MAX_VISUAL_DISTANCE,
                    camera_matrix=None, color=(1, 0, 0), **kwargs):
    # TODO: head_name likely needs to have a visual geometry to attach
    head_link = link_from_name(robot, head_name)
    lines = []
    for v1, v2 in cone_wires_from_support(get_viewcone_base(
            depth=depth, camera_matrix=camera_matrix)):
        if is_optical(head_name):
            rotation = Pose()
        else:
            rotation = Pose(euler=Euler(roll=-np.pi/2, yaw=-np.pi/2)) # Apply in reverse order
        p1 = tform_point(rotation, v1)
        p2 = tform_point(rotation, v2)
        lines.append(add_line(p1, p2, color=color, parent=robot, parent_link=head_link, **kwargs))
    return lines

def draw_viewcone(pose, depth=MAX_VISUAL_DISTANCE,
                  camera_matrix=None, color=(1, 0, 0), **kwargs):
    lines = []
    for v1, v2 in cone_wires_from_support(get_viewcone_base(
            depth=depth, camera_matrix=camera_matrix)):
        p1 = tform_point(pose, v1)
        p2 = tform_point(pose, v2)
        lines.append(add_line(p1, p2, color=color, **kwargs))
    return lines

#####################################

def is_optical(link_name):
    return 'optical' in link_name

def inverse_visibility(pr2, point, head_name=HEAD_LINK_NAME,
                       max_iterations=100, step_size=0.5, tolerance=np.pi*1e-2):
    # https://github.com/PR2/pr2_controllers/blob/kinetic-devel/pr2_head_action/src/pr2_point_frame.cpp
    head_joints = joints_from_names(pr2, PR2_GROUPS['head'])
    head_link = link_from_name(pr2, head_name)
    camera_axis = np.array([0, 0, 1]) if is_optical(head_name) else np.array([1, 0, 0])
    # TODO: could also set the target orientation for inverse kinematics
    head_conf = np.zeros(len(head_joints))
    with LockRenderer():
        with ConfSaver(pr2):
            for _ in range(max_iterations):
                set_joint_positions(pr2, head_joints, head_conf)
                world_from_head = get_link_pose(pr2, head_link)
                point_head = tform_point(invert(world_from_head), point)
                error_angle = angle_between(camera_axis, point_head)
                if abs(error_angle) <= tolerance:
                    break
                normal_head = np.cross(camera_axis, point_head)
                normal_world = tform_point((unit_point(), quat_from_pose(world_from_head)), normal_head)
                correction_quat = quat_from_axis_angle(normal_world, step_size*error_angle)
                correction_euler = euler_from_quat(correction_quat)
                _, angular = compute_jacobian(pr2, head_link)
                correction_conf = np.array([np.dot(angular[mj], correction_euler)
                                            for mj in movable_from_joints(pr2, head_joints)])
                head_conf += correction_conf
                #wait_for_user()
            else:
                return None
    if violates_limits(pr2, head_joints, head_conf):
        return None
    return head_conf

def plan_scan_path(pr2, tilt=0):
    head_joints = joints_from_names(pr2, PR2_GROUPS['head'])
    start_conf = get_joint_positions(pr2, head_joints)
    lower_limit, upper_limit = get_joint_limits(pr2, head_joints[0])

    first_conf = np.array([lower_limit, tilt])
    second_conf = np.array([upper_limit, tilt])
    if start_conf[0] > 0:
        first_conf, second_conf = second_conf, first_conf
    return [first_conf, second_conf]
    #return [start_conf, first_conf, second_conf]
    #third_conf = np.array([0, tilt])
    #return [start_conf, first_conf, second_conf, third_conf]

def plan_pause_scan_path(pr2, tilt=0):
    head_joints = joints_from_names(pr2, PR2_GROUPS['head'])
    assert(not violates_limit(pr2, head_joints[1], tilt))
    theta, _ = get_pr2_field_of_view()
    lower_limit, upper_limit = get_joint_limits(pr2, head_joints[0])
    # Add one because half visible on limits
    n = int(np.math.ceil((upper_limit - lower_limit) / theta) + 1)
    epsilon = 1e-3
    return [np.array([pan, tilt]) for pan in np.linspace(lower_limit + epsilon,
                                                         upper_limit - epsilon, n, endpoint=True)]

#####################################

Detection = namedtuple('Detection', ['body', 'distance'])

def get_view_aabb(body, view_pose, **kwargs):
    with PoseSaver(body):
        body_view = multiply(invert(view_pose), get_pose(body))
        set_pose(body, body_view)
        return get_aabb(body, **kwargs)

def get_detection_cone(pr2, body, camera_link=HEAD_LINK_NAME, depth=MAX_VISUAL_DISTANCE, **kwargs):
    head_link = link_from_name(pr2, camera_link)
    body_aabb = get_view_aabb(body, get_link_pose(pr2, head_link))
    lower_z = body_aabb[0][2]
    if depth < lower_z:
        return None, lower_z
    if not is_visible_aabb(body_aabb, **kwargs):
        return None, lower_z
    return cone_mesh_from_support(support_from_aabb(body_aabb)), lower_z

def get_detections(pr2, p_false_neg=0, camera_link=HEAD_LINK_NAME,
                   exclude_links=set(), color=None, **kwargs):
    camera_pose = get_link_pose(pr2, link_from_name(pr2, camera_link))
    detections = []
    for body in get_bodies():
        if (pr2 == body) or (np.random.random() < p_false_neg):
            continue
        mesh, z = get_detection_cone(pr2, body, camera_link=camera_link, **kwargs)
        if mesh is None:
            continue
        cone = create_mesh(mesh, color=color)
        set_pose(cone, camera_pose)
        if not any(pairwise_collision(cone, obst)
                   for obst in set(get_bodies()) - {pr2, body, cone}) \
                and not any(link_pairs_collision(pr2, [link], cone)
                            for link in set(get_all_links(pr2)) - exclude_links):
            detections.append(Detection(body, z))
        #wait_for_user()
        remove_body(cone)
    return detections

def get_visual_detections(pr2, **kwargs):
    return [body for body, _ in get_detections(pr2, depth=MAX_VISUAL_DISTANCE, **kwargs)]

def get_kinect_registrations(pr2, **kwargs):
    return [body for body, _ in get_detections(pr2, depth=MAX_KINECT_DISTANCE, **kwargs)]

# TODO: Gaussian on resulting pose

#####################################

# TODO: base motion with some stochasticity
def visible_base_generator(robot, target_point, base_range):
    #base_from_table = point_from_pose(get_pose(robot))[:2]
    while True:
        base_from_table = unit_from_theta(np.random.uniform(0, 2 * np.pi))
        look_distance = np.random.uniform(*base_range)
        base_xy = target_point[:2] - look_distance * base_from_table
        base_theta = np.math.atan2(base_from_table[1], base_from_table[0]) # TODO: stochastic orientation?
        base_q = np.append(base_xy, base_theta)
        yield base_q


def get_base_extend_fn(robot):
    # TODO: rotate such that in field of view of the camera first
    # TODO: plan base movements while checking edge feasibility with camera
    raise NotImplementedError()

#####################################

def close_until_collision(robot, gripper_joints, bodies=[], num_steps=25, **kwargs):
    if not gripper_joints:
        return None
    closed_conf = [get_min_limit(robot, joint) for joint in gripper_joints]
    open_conf = [get_max_limit(robot, joint) for joint in gripper_joints]
    resolutions = np.abs(np.array(open_conf) - np.array(closed_conf)) / num_steps
    extend_fn = get_extend_fn(robot, gripper_joints, resolutions=resolutions)
    close_path = [open_conf] + list(extend_fn(open_conf, closed_conf))
    collision_links = frozenset(get_moving_links(robot, gripper_joints))

    for i, conf in enumerate(close_path):
        set_joint_positions(robot, gripper_joints, conf)
        if any(pairwise_collision((robot, collision_links), body, **kwargs) for body in bodies):
            if i == 0:
                return None
            return close_path[i-1][0]
    return close_path[-1][0]


def compute_grasp_width(robot, arm, body, grasp_pose, **kwargs):
    gripper_joints = get_gripper_joints(robot, arm)
    tool_link = link_from_name(robot, PR2_TOOL_FRAMES[arm])
    tool_pose = get_link_pose(robot, tool_link)
    body_pose = multiply(tool_pose, grasp_pose)
    set_pose(body, body_pose)
    return close_until_collision(robot, gripper_joints, bodies=[body], **kwargs)


def create_gripper(robot, arm, visual=True):
    link_name = PR2_GRIPPER_ROOTS[arm]
    # gripper = load_pybullet(os.path.join(get_data_path(), 'pr2_gripper.urdf'))
    # gripper = load_pybullet(os.path.join(get_models_path(), 'pr2_description/pr2_l_gripper.urdf'), fixed_base=False)
    # pybullet.error: Error receiving visual shape info for the DRAKE_PR2
    links = get_link_subtree(robot, link_from_name(robot, link_name))
    gripper = clone_body(robot, links=links, visual=False, collision=True)  # TODO: joint limits
    if not visual:
        for link in get_all_links(gripper):
            set_color(gripper, np.zeros(4), link)
    return gripper
