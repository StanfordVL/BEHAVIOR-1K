from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
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

        # not grasping -> not grasping = Distance reward
        # not grasping -> grasping = Large reward
        # grapsing -> not grapsing = Large negative reward
        # grasping -> grasping = Minimizing moment of inertia

        reward = None

        if not self.prev_grasping and not current_grasping:
            eef_pos = robot.get_eef_position(robot.arm)
            obj_pos = self.obj.get_position()
            reward = T.l2_distance(eef_pos, obj_pos) * self.dist_coeff

        elif not self.prev_grasping and current_grasping:
            reward = self.grasp_reward
        
        elif self.prev_grasping and not current_grasping:
            reward = -self.grasp_reward
        
        elif self.prev_grasping and current_grasping:
            # Need to finish
            reward = 0

        self.prev_grasping = current_grasping
        return reward, {}
