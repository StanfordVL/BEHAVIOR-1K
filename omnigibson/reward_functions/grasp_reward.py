import math
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
import omnigibson.utils.transform_utils as T


class GraspReward(BaseRewardFunction):
    """
    Grasp reward
    """

    def __init__(self, obj_name, dist_coeff, grasp_reward):
        # Store internal vars
        self.prev_grasping = False
        self.obj_name = obj_name
        self.obj = None
        self.dist_coeff = dist_coeff
        self.grasp_reward = grasp_reward

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        self.obj = env.scene.object_registry("name", self.obj_name) if self.obj is None else self.obj

        robot = env.robots[0]
        obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        current_grasping = True if obj_in_hand == self.obj else False
        
        # Reward varying based on combination of whether the robot was previously grasping the desired and object
        # and is currently grasping the desired object

        # not grasping -> not grasping = Distance between eef and obj reward
        # not grasping -> grasping = Minimizing MOI + grasp reward
        # grapsing -> not grapsing = Distance between eef and obj reward
        # grasping -> grasping = Minimizing MOI + grasp reward

        reward = 0.

        if not self.prev_grasping and not current_grasping:
            eef_pos = robot.get_eef_position(robot.default_arm)
            obj_center = self.obj.aabb_center
            dist = T.l2_distance(eef_pos, obj_center)
            reward += math.exp(-dist) * self.dist_coeff

        elif not self.prev_grasping and current_grasping:
            robot_center = robot.aabb_center
            obj_center = self.obj.aabb_center
            dist = T.l2_distance(robot_center, obj_center)
            dist_reward =  math.exp(-dist) * self.dist_coeff
            reward += dist_reward + self.grasp_reward
        
        elif self.prev_grasping and not current_grasping:
            eef_pos = robot.get_eef_position(robot.default_arm)
            obj_center = self.obj.aabb_center
            dist = T.l2_distance(eef_pos, obj_center)
            reward +=  math.exp(-dist) * self.dist_coeff
        
        elif self.prev_grasping and current_grasping:
            robot_center = robot.aabb_center
            obj_center = self.obj.aabb_center
            dist = T.l2_distance(robot_center, obj_center)
            dist_reward =  math.exp(-dist) * self.dist_coeff
            reward += dist_reward + self.grasp_reward

        # Overwrite reward if robot is in collision
        # The one step where it grasps the object, it is in collision and this triggers
        if detect_robot_collision_in_sim(robot, ignore_obj_in_hand=True):
            reward += -1.

        self.prev_grasping = current_grasping
        return reward, {}
