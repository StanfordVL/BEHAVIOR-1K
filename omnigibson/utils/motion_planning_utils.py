import numpy as np
from ompl import base as ob
from ompl import geometric as o_geo

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
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)
    # construct an instance of space information from this state space
    si = ob.SpaceInformation(space)
    # set state validity checking for this space
    si.setStateValidityChecker(ob.StateValidityCheckerFn(collision_fn))
    # create a problem instance
    pdef = ob.ProblemDefinition(si)
    # set the start and goal states
    pdef.setStartAndGoalStates(start_conf, end_conf)
    # create a planner for the defined space
    planner = o_geo.RRTConnect(si)
    # set the problem we are trying to solve for the planner
    planner.setProblemDefinition(pdef)
    # perform setup steps for the planner
    planner.setup()
    # attempt to solve the problem within one second of planning time
    solved = planner.solve(planning_time)

    if solved:
        # get the goal representation from the problem definition (not the same as the goal state)
        # and inquire about the found path
        path = pdef.getSolutionPath()
        print("Found solution:\n%s" % path)
    else:
        print("No solution found")

    return path

def detect_robot_collision(robot, obj_in_hand=None):
    # filter_objects = ["floor"]
    # if obj_in_hand is not None:
    #     filter_objects.append(obj_in_hand.name)
    collision_objects = list(filter(lambda obj : "floor" not in obj.name, robot.states[ContactBodies].get_value()))
    # collision_objects = robot.states[ContactBodies].get_value()
    # for col_obj in collision_objects:
    return len(collision_objects) > 0