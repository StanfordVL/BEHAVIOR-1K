import numpy as np
from ompl import base as ob
from ompl import geometric as ompl_geo

import omnigibson as og
from omnigibson.object_states import ContactBodies
import omnigibson.utils.transform_utils as T

def plan_base_motion(
    robot,
    obj_in_hand,
    end_conf,
    planning_time = 10.0,
    **kwargs,
):
    distance_fn = lambda q1, q2: np.linalg.norm(np.array(q2[:2]) - np.array(q1[:2]))

    def collision_fn(q):
        robot.set_position_orientation(
            [q[0], q[1], 0.05], T.euler2quat((0, 0, q[2]))
        )
        og.sim.step(render=False)
        return detect_robot_collision(robot, obj_in_hand)

    pos = robot.get_position()
    yaw = T.quat2euler(robot.get_orientation())[2]
    start_conf = (pos[0], pos[1], yaw)


       # create an SE2 state space
    space = ob.SE2StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-100)
    bounds.setHigh(100)
    space.setBounds(bounds)

    # create a simple setup object
    ss = ompl_geo.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(collision_fn))

    start = ob.State(space)
    start().setX(start_conf[0])
    start().setY(start_conf[1])
    start().setYaw(start_conf[2])

    goal = ob.State(space)
    goal().setX(end_conf[0])
    goal().setY(end_conf[1])
    goal().setYaw(end_conf[2])

    ss.setStartAndGoalStates(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(10.0)

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
        return return_path
    return None

def detect_robot_collision(robot, obj_in_hand=None):
    # filter_objects = ["floor"]
    # if obj_in_hand is not None:
    #     filter_objects.append(obj_in_hand.name)
    collision_objects = list(filter(lambda obj : "floor" not in obj.name, robot.states[ContactBodies].get_value()))
    # collision_objects = robot.states[ContactBodies].get_value()
    # for col_obj in collision_objects:
    return len(collision_objects) > 0