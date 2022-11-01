"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by iGibson team.
"""
import numpy as np

from pybullet_tools.pr2_utils import DRAKE_PR2_URDF, set_group_conf, REST_LEFT_ARM, rightarm_from_leftarm
from pybullet_tools.utils import HideOutput, load_model, base_values_from_pose, has_joint, set_joint_position, \
    joint_from_name, get_box_geometry, create_shape, Pose, Point, set_pose, \
    get_cylinder_geometry, get_sphere_geometry, create_shape_array, create_body


def parse_point(point_json):
    return tuple(point_json[key] for key in ['x', 'y', 'z'])


def parse_quat(quat_json):
    return tuple(quat_json[key] for key in ['x', 'y', 'z', 'w'])


def parse_pose(pose_json):
    return parse_point(pose_json['point']), parse_quat(pose_json['quat'])


def parse_color(color_json):
    return tuple(color_json[key] for key in ['r', 'g', 'b', 'a'])


def parse_robot(robot_json):
    pose = parse_pose(robot_json)
    if robot_json['name'] == 'pr2':
        with HideOutput():
            robot_id = load_model(DRAKE_PR2_URDF, fixed_base=True)
        set_group_conf(robot_id, 'base',  base_values_from_pose(pose))
    else:
        # TODO: set the z?
        #set_pose(robot_id, pose)
        raise NotImplementedError(robot_json['name'])

    for joint, values in robot_json['conf'].items():
        [value] = values
        if has_joint(robot_id, joint):
            set_joint_position(robot_id, joint_from_name(robot_id, joint), value)
        else:
            print('Robot {} lacks joint {}'.format(robot_json['name'], joint))

    if robot_json['name'] == 'pr2':
        set_group_conf(robot_id, 'torso', [0.2])
        set_group_conf(robot_id, 'left_arm', REST_LEFT_ARM)
        set_group_conf(robot_id, 'right_arm', rightarm_from_leftarm(REST_LEFT_ARM))

    return robot_id


def parse_region(region):
    lower = np.min(region['hull'], axis=0)
    upper = np.max(region['hull'], axis=0)
    x, y = (lower + upper) / 2.
    w, h = (upper - lower)  # / 2.
    geom = get_box_geometry(w, h, 1e-3)
    collision_id, visual_id = create_shape(geom, pose=Pose(Point(x, y)), color=parse_color(region['color']))
    #region_id = create_body(NULL_ID, visual_id)
    region_id = create_body(collision_id, visual_id)
    set_pose(region_id, parse_pose(region))
    return region_id

def parse_geometry(geometry):
    # TODO: can also just make fixed links
    geom = None
    if geometry['type'] == 'box':
        geom = get_box_geometry(*2 * np.array(geometry['extents']))
    elif geometry['type'] == 'cylinder':
        geom = get_cylinder_geometry(geometry['radius'], geometry['height'])
    elif geometry['type'] == 'sphere':
        # TODO: does sphere not work?
        geom = get_sphere_geometry(geometry['radius'])
    elif geometry['type'] == 'trimesh':
        pass
    else:
        raise NotImplementedError(geometry['type'])
    pose = parse_pose(geometry)
    color = parse_color(geometry['color'])  # specular=geometry['specular'])
    return geom, pose, color

def parse_body(body, important=False):
    [link] = body['links']
    # for geometry in link['geometries']:
    geoms = []
    poses = []
    colors = []
    skipped = False
    for geometry in link:
        geom, pose, color = parse_geometry(geometry)
        if geom == None:
            skipped = True
        else:
            geoms.append(geom)
            poses.append(pose)
            colors.append(color)

    if skipped:
        if important:
            center = body['aabb']['center']
            extents = 2*np.array(body['aabb']['extents'])
            geoms = [get_box_geometry(*extents)]
            poses = [Pose(center)]
            colors = [(.5, .5, .5, 1)]
        else:
            return None
    if not geoms:
        return None
    if len(geoms) == 1:
        collision_id, visual_id = create_shape(geoms[0], pose=poses[0], color=colors[0])
    else:
        collision_id, visual_id = create_shape_array(geoms, poses, colors)
    body_id = create_body(collision_id, visual_id)
    set_pose(body_id, parse_pose(body))
    return body_id
