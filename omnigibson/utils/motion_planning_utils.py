import numpy as np
from ompl import base as ob
from ompl import geometric as ompl_geo

import omnigibson as og
from omnigibson.object_states import ContactBodies
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import RigidContactAPI

def plan_base_motion(
    robot,
    end_conf,
    planning_time = 100.0,
    **kwargs,
):
    def state_valid_fn(q):
        x = q.getX()
        y = q.getY()
        yaw = q.getYaw()
        robot.set_position_orientation(
            [x, y, 0.05], T.euler2quat((0, 0, yaw))
        )
        og.sim.step(render=False)
        state_valid = not detect_robot_collision(robot)
        return state_valid

    pos = robot.get_position()
    yaw = T.quat2euler(robot.get_orientation())[2]
    start_conf = (pos[0], pos[1], yaw)

    # create an SE2 state space
    space = ob.SE2StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-7.0)
    bounds.setHigh(7.0)
    space.setBounds(bounds)

    # create a simple setup object
    ss = ompl_geo.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(state_valid_fn))

    si = ss.getSpaceInformation()
    planner = ompl_geo.LBKPIECE1(si)
    ss.setPlanner(planner)

    start = ob.State(space)
    start().setX(start_conf[0])
    start().setY(start_conf[1])
    start().setYaw(T.wrap_angle(start_conf[2]))
    print(start)

    goal = ob.State(space)
    goal().setX(end_conf[0])
    goal().setY(end_conf[1])
    goal().setYaw(T.wrap_angle(end_conf[2]))
    print(goal)

    ss.setStartAndGoalStates(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(planning_time)

    if solved:
        # try to shorten the path
        ss.simplifySolution()
        # print the simplified path
        sol_path = ss.getSolutionPath()
        return_path = []
        for i in range(sol_path.getStateCount()):
            x = sol_path.getState(i).getX()
            y = sol_path.getState(i).getY()
            yaw = sol_path.getState(i).getYaw()
            return_path.append([x, y, yaw])
        return remove_unnecessary_rotations(return_path)
    return None

def plan_arm_motion(
    robot,
    end_conf,
    planning_time = 100.0,
    **kwargs,
):
    joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    dim = len(joint_control_idx)

    def state_valid_fn(q):
        joint_pos = [q[i] for i in range(dim)]
        robot.set_joint_positions(joint_pos, joint_control_idx)
        og.sim.step(render=False)
        state_valid = not detect_robot_collision(robot)
        return state_valid
    
    # create an SE2 state space
    space = ob.RealVectorStateSpace(dim)

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(dim)
    joints = np.array([joint for joint in robot.joints.values()])
    arm_joints = joints[joint_control_idx]
    for i, joint in enumerate(arm_joints):
        bounds.setLow(i, float(joint.lower_limit))
        bounds.setHigh(i, float(joint.upper_limit))
    space.setBounds(bounds)

    # create a simple setup object
    ss = ompl_geo.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(state_valid_fn))

    si = ss.getSpaceInformation()
    planner = ompl_geo.LBKPIECE1(si)
    ss.setPlanner(planner)

    start_conf = robot.get_joint_positions()[joint_control_idx]
    start = ob.State(space)
    for i in range(dim):
        start[i] = float(start_conf[i])

    goal = ob.State(space)
    for i in range(dim):
        goal[i] = float(end_conf[i])
    ss.setStartAndGoalStates(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(planning_time)

    if solved:
        # try to shorten the path
        ss.simplifySolution()

        sol_path = ss.getSolutionPath()
        return_path = []
        for i in range(sol_path.getStateCount()):
            joint_pos = [sol_path.getState(i)[j] for j in range(dim)]
            return_path.append(joint_pos)
        return return_path
    return None

def detect_robot_collision(robot, filter_objs=[]):
    filter_categories = ["floors"]
    
    obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
    if obj_in_hand is not None:
        filter_objs.append(obj_in_hand)

    collision_prims = list(robot.states[ContactBodies].get_value(ignore_objs=tuple(filter_objs)))

    for col_prim in collision_prims:
        tokens = col_prim.prim_path.split("/")
        obj_prim_path = "/".join(tokens[:-1])
        col_obj = og.sim.scene.object_registry("prim_path", obj_prim_path)
        if col_obj.category in filter_categories:
            collision_prims.remove(col_prim)

    return len(collision_prims) > 0 or detect_self_collision(robot)

def detect_self_collision(robot):
    contacts = robot.contact_list()
    robot_links = [link.prim_path for link in robot.links.values()]
    disabled_pairs = [set(p) for p in robot.disabled_collision_pairs]
    for c in contacts:
        link0 = c.body0.split("/")[-1]
        link1 = c.body1.split("/")[-1]
        if {link0, link1} not in disabled_pairs and c.body0 in robot_links and c.body1 in robot_links:
            return True
    return False

def detect_hand_collision(robot, joint_pos, control_idx):
    robot.set_joint_positions(joint_pos, control_idx)
    return detect_robot_collision(robot)

def remove_unnecessary_rotations(path):
    for start_idx in range(len(path) - 1):
        start = np.array(path[start_idx][:2])
        end = np.array(path[start_idx + 1][:2])
        segment = end - start
        theta = np.arctan2(segment[1], segment[0])
        path[start_idx] = (start[0], start[1], theta)
    return path