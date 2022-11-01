"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by iGibson team.
"""
import random
from collections import namedtuple

from ..utils import matrix_from_quat, point_from_pose, quat_from_pose, quat_from_matrix, \
    get_joint_limits, get_joint_position, get_joint_positions, get_distance

# TODO: lookup robot & tool in dictionary and use if exists

IKFastInfo = namedtuple('IKFastInfo', ['module_name', 'base_link', 'ee_link', 'free_joints'])

USE_ALL = False
USE_CURRENT = None


def compute_forward_kinematics(fk_fn, conf):
    pose = fk_fn(list(conf))
    pos, rot = pose
    quat = quat_from_matrix(rot) # [X,Y,Z,W]
    return pos, quat


def compute_inverse_kinematics(ik_fn, pose, sampled=[]):
    pos = point_from_pose(pose)
    rot = matrix_from_quat(quat_from_pose(pose)).tolist()
    if len(sampled) == 0:
        solutions = ik_fn(list(rot), list(pos))
    else:
        solutions = ik_fn(list(rot), list(pos), list(sampled))
    if solutions is None:
        return []
    return solutions


def get_ik_limits(robot, joint, limits=USE_ALL):
    if limits is USE_ALL:
        return get_joint_limits(robot, joint)
    elif limits is USE_CURRENT:
        value = get_joint_position(robot, joint)
        return value, value
    return limits


def select_solution(body, joints, solutions, nearby_conf=USE_ALL, **kwargs):
    if not solutions:
        return None
    if nearby_conf is USE_ALL:
        return random.choice(solutions)
    if nearby_conf is USE_CURRENT:
        nearby_conf = get_joint_positions(body, joints)
    # TODO: sort by distance before collision checking
    # TODO: search over neighborhood of sampled joints when nearby_conf != None
    return min(solutions, key=lambda conf: get_distance(nearby_conf, conf, **kwargs))
