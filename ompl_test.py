import numpy as np
from ompl import base as ob
from ompl import geometric as ompl_geo

def plan_base_motion(
    robot,
    obj_in_hand,
    end_conf,
    planning_time = 100.0,
    **kwargs,
):

    def state_valid_fn(q):
       return True
    
    dim = 3
    # create an SE2 state space
    space = ob.RealVectorStateSpace(3)

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-np.pi)
    bounds.setHigh(np.pi)
    space.setBounds(bounds)

    # create a simple setup object
    ss = ompl_geo.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(state_valid_fn))

    si = ss.getSpaceInformation()
    planner = ompl_geo.LBKPIECE1(si)
    ss.setPlanner(planner)

    start_conf = [0.0, 0.0, 0.0]
    start = ob.State(space)
    start[0] = start_conf[0]
    start[1] = start_conf[1]
    start[2] = start_conf[2]
    print(start)

    goal = ob.State(space)
    goal[0] = 1
    goal[1] = 1
    goal[2] = 1
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
            joint_pos = [sol_path.getState(i)[j] for j in range(dim)]
            return_path.append(joint_pos)
        return return_path
    return None

print(plan_base_motion(None, None, None))