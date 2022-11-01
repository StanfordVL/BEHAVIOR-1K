"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by iGibson team.
"""
import numpy as np
import time

from itertools import islice

from pybullet_tools.ikfast.utils import IKFastInfo
from ..utils import compute_inverse_kinematics
from ...utils import multiply, get_link_pose, link_from_name, invert, joints_from_names, \
    randomize, violates_limits, \
    get_joint_positions, INF, get_difference_fn, get_link_ancestors, prune_fixed_joints, \
    get_length, unit_generator, get_min_limits, get_max_limits, elapsed_time

PANDA_INFO = IKFastInfo(module_name='ikfast_panda_arm', base_link='panda_link0',
                        ee_link='panda_link8', free_joints=['panda_joint7'])

def import_ikfast(ikfast_info):
    # TODO: relative import
    import pybullet_tools.ikfast.franka_panda.ikfast_panda_arm as ikfast
    return ikfast
    #return importlib.import_module(ikfast_info.module_name, package=pybullet_tools.ikfast.franka_panda)

def is_ik_compiled(ikfast_info):
    try:
        import_ikfast(ikfast_info)
        return True
    except ImportError:
        return False

def get_base_from_ee(robot, ikfast_info, tool_link, world_from_target):
    world_from_base = get_link_pose(robot, link_from_name(robot, ikfast_info.base_link))
    world_from_ee = get_link_pose(robot, link_from_name(robot, ikfast_info.ee_link))
    world_from_tool = get_link_pose(robot, tool_link)
    tool_from_ee = multiply(invert(world_from_tool), world_from_ee)
    base_from_ee = multiply(invert(world_from_base), world_from_target, tool_from_ee)
    return base_from_ee

def get_ordered_ancestors(robot, link):
    #return prune_fixed_joints(robot, get_link_ancestors(robot, link)[1:] + [link])
    return get_link_ancestors(robot, link)[1:] + [link]

def get_ik_joints(robot, ikfast_info, tool_link):
    # Get joints between base and ee
    # Ensure no joints between ee and tool
    base_link = link_from_name(robot, ikfast_info.base_link)
    ee_link = link_from_name(robot, ikfast_info.ee_link)
    ee_ancestors = get_ordered_ancestors(robot, ee_link)
    tool_ancestors = get_ordered_ancestors(robot, tool_link)
    assert prune_fixed_joints(robot, ee_ancestors) == prune_fixed_joints(robot, tool_ancestors)
    assert base_link in ee_ancestors
    ik_joints = prune_fixed_joints(robot, ee_ancestors[ee_ancestors.index(base_link):])
    free_joints = joints_from_names(robot, ikfast_info.free_joints)
    assert set(free_joints) <= set(ik_joints)
    assert len(ik_joints) == 6 + len(free_joints)
    return ik_joints

def ikfast_inverse_kinematics(robot, ikfast_info, tool_link, world_from_target,
                              max_attempts=INF, norm=INF, max_distance=INF, **kwargs):
    if max_distance is None:
        max_distance = INF
    #assert is_ik_compiled(ikfast_info)
    ikfast = import_ikfast(ikfast_info)
    ik_joints = get_ik_joints(robot, ikfast_info, tool_link)
    free_joints = joints_from_names(robot, ikfast_info.free_joints)
    base_from_ee = get_base_from_ee(robot, ikfast_info, tool_link, world_from_target)
    difference_fn = get_difference_fn(robot, ik_joints)
    current_conf = get_joint_positions(robot, ik_joints)
    # TODO: handle circular joints
    current_positions = get_joint_positions(robot, free_joints)
    free_deltas = max_distance*np.ones(len(free_joints))
    lower_limits = np.maximum(get_min_limits(robot, free_joints), current_positions - free_deltas)
    upper_limits = np.minimum(get_max_limits(robot, free_joints), current_positions + free_deltas)
    generator = (weights*lower_limits + (1-weights)*upper_limits
                 for weights in unit_generator(d=len(free_joints), **kwargs))
    if max_attempts < INF:
        generator = islice(generator, max_attempts)
    for free_positions in generator:
        for conf in randomize(compute_inverse_kinematics(ikfast.get_ik, base_from_ee, free_positions)):
            difference = difference_fn(current_conf, conf)
            if not violates_limits(robot, ik_joints, conf) and get_length(difference, norm=norm):
                #set_joint_positions(robot, ik_joints, conf)
                yield conf

def closest_inverse_kinematics(robot, ikfast_info, tool_link, world_from_target,
                               max_attempts=INF, max_time=INF, norm=INF, **kwargs):
    assert (max_attempts < INF) or (max_time < INF)
    start_time = time.time()
    ik_joints = get_ik_joints(robot, ikfast_info, tool_link)
    current_conf = get_joint_positions(robot, ik_joints)
    generator = ikfast_inverse_kinematics(robot, ikfast_info, tool_link, world_from_target,
                                          max_attempts=max_attempts, norm=norm, **kwargs)
    solutions = []
    for conf in generator:
        solutions.append(conf)
        if max_time < elapsed_time(start_time):
            break
    #print('{} confs in {:.3f} seconds'.format(len(solutions), elapsed_time(start_time)))
    difference_fn = get_difference_fn(robot, ik_joints)
    #distance_fn = get_distance_fn(robot, ik_joints)
    #set_joint_positions(robot, ik_joints, closest_conf)
    return iter(sorted(solutions, key=lambda q: get_length(difference_fn(q, current_conf), norm=norm)))
