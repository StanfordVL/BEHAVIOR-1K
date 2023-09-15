import random
import numpy as np
import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import UndoableContext
from omnigibson.reward_functions.grasp_reward import GraspReward

from omnigibson.tasks.task_base import BaseTask
from omnigibson.scenes.scene_base import Scene
from omnigibson.termination_conditions.grasp_goal import GraspGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
from omnigibson.utils.motion_planning_utils import set_arm_and_detect_collision
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.sim_utils import land_object

DIST_COEFF = 0.1
GRASP_REWARD = 1.0
MAX_JOINT_RANDOMIZATION_ATTEMPTS = 50

class GraspTask(BaseTask):
    """
    Grasp task
    """

    def __init__(
        self,
        obj_name,
        termination_config=None,
        reward_config=None,
    ):
        self.obj_name = obj_name
        super().__init__(termination_config=termination_config, reward_config=reward_config)
        
    def _load(self, env):
        pass

    def _create_termination_conditions(self):
        terminations = dict()
        terminations["graspgoal"] = GraspGoal(
            self.obj_name
        )
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])

        return terminations

    def _create_reward_functions(self):
        rewards = dict()
        rewards["grasp"] = GraspReward(
            self.obj_name,
            dist_coeff=self._reward_config["r_dist_coeff"],
            grasp_reward=self._reward_config["r_grasp"]
        )
        return rewards
    
    def _reset_agent(self, env):
        robot = env.robots[0]
        # # Randomize the robots joint positions
        # # @TODO: SHOULD WORK, BUT NEED TO TEST
        # # joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        # # joint_combined_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
        # # initial_joint_pos = np.array(robot.get_joint_positions()[joint_combined_idx])
        # # control_idx_in_joint_pos = np.where(np.in1d(joint_combined_idx, joint_control_idx))[0]

        # # with UndoableContext(self.primitive_controller.robot, self.primitive_controller.robot_copy, "original") as context:
        # #     for _ in range(MAX_JOINT_RANDOMIZATION_ATTEMPTS):
        # #         joint_pos, joint_control_idx = self._get_random_joint_position(robot)
        # #         initial_joint_pos[control_idx_in_joint_pos] = joint_pos
        # #         if not set_arm_and_detect_collision(context, initial_joint_pos):
        # #             robot.set_joint_positions(joint_pos, joint_control_idx)
        # #             og.sim.step()
        # #             break

        # Randomize the robot's 2d pose
        obj = env.scene.object_registry("name", self.obj_name)
        if obj is not None:  
            grasp_poses = get_grasp_poses_for_object_sticky(obj)
            grasp_pose, _ = random.choice(grasp_poses)
            sampled_pose_2d = self.primitive_controller._sample_pose_near_object(obj, pose_on_obj=grasp_pose)
            robot_pose = self.primitive_controller._get_robot_pose_from_2d_pose(sampled_pose_2d)
            robot.set_position_orientation(*robot_pose)

    def _get_random_joint_position(self, robot):
        joint_positions = []
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        joints = np.array([joint for joint in robot.joints.values()])
        arm_joints = joints[joint_control_idx]
        for i, joint in enumerate(arm_joints):
            val = random.uniform(joint.lower_limit, joint.upper_limit)
            joint_positions.append(val)
        return joint_positions, joint_control_idx

    def _get_obs(self, env):
        # No task-specific obs of any kind
        return dict(), dict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()
    
    def add_primitive_controller(self, controller):
        self.primitive_controller = controller

    @classproperty
    def valid_scene_types(cls):
        # Any scene works
        return {Scene}

    @classproperty
    def default_termination_config(cls):
        return {
            "max_steps": 100000
        }

    @classproperty
    def default_reward_config(cls):
        return {
            "r_dist_coeff": DIST_COEFF,
            "r_grasp": GRASP_REWARD,
        }

