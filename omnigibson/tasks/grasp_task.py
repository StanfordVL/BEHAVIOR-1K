import random
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R
import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import PlanningContext
from omnigibson.reward_functions.grasp_reward import GraspReward

import omnigibson.utils.transform_utils as T
from omnigibson.tasks.task_base import BaseTask
from omnigibson.scenes.scene_base import Scene
from omnigibson.termination_conditions.falling import Falling
from omnigibson.termination_conditions.grasp_goal import GraspGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
from omnigibson.utils.motion_planning_utils import set_arm_and_detect_collision
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.sim_utils import land_object
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

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
        self._primitive_controller = None
        path = os.path.dirname(__file__)
        f = open(path + "/../../rl/reset_poses.json")
        self.reset_poses = json.load(f)
        super().__init__(termination_config=termination_config, reward_config=reward_config)
        
    def _load(self, env):
        pass

    def _create_termination_conditions(self):
        terminations = dict()
        # terminations["graspgoal"] = GraspGoal(
        #     self.obj_name
        # )
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        # terminations["falling"] = Falling()

        return terminations

    def _create_reward_functions(self):
        rewards = dict()
        rewards["grasp"] = GraspReward(
            self.obj_name,
            **self._reward_config
        )
        return rewards
    
    def _reset_agent(self, env):
        if self._primitive_controller is None:
            self._primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

        # Reset the robot with primitive controller
        ###########################################
        # robot = env.robots[0]
        # # Randomize the robots joint positions
        # joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        # dim = len(joint_control_idx)
        # # For Tiago
        # if "combined" in robot.robot_arm_descriptor_yamls:
        #     joint_combined_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
        #     initial_joint_pos = np.array(robot.get_joint_positions()[joint_combined_idx])
        #     control_idx_in_joint_pos = np.where(np.in1d(joint_combined_idx, joint_control_idx))[0]
        # # For Fetch
        # else:
        #     initial_joint_pos = np.array(robot.get_joint_positions()[joint_control_idx])
        #     control_idx_in_joint_pos = np.arange(dim)

        # with PlanningContext(self._primitive_controller.robot, self._primitive_controller.robot_copy, "original") as context:
        #     for _ in range(MAX_JOINT_RANDOMIZATION_ATTEMPTS):
        #         joint_pos, joint_control_idx = self._get_random_joint_position(robot)
        #         initial_joint_pos[control_idx_in_joint_pos] = joint_pos
        #         if not set_arm_and_detect_collision(context, initial_joint_pos):
        #             robot.set_joint_positions(joint_pos, joint_control_idx)
        #             og.sim.step()
        #             break

        # # Randomize the robot's 2d pose
        # obj = env.scene.object_registry("name", self.obj_name)
        # grasp_poses = get_grasp_poses_for_object_sticky(obj)
        # grasp_pose, _ = random.choice(grasp_poses)
        # sampled_pose_2d = self._primitive_controller._sample_pose_near_object(obj, pose_on_obj=grasp_pose)
        # # sampled_pose_2d = [-0.433881, -0.210183, -2.96118]
        # robot_pose = self._primitive_controller._get_robot_pose_from_2d_pose(sampled_pose_2d)
        # robot.set_position_orientation(*robot_pose)
            
        # Reset the robot with cached reset poses
        ###########################################
        robot = env.robots[0]
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        robot_pose = random.choice(self.reset_poses)
        robot.set_joint_positions(robot_pose['joint_pos'], joint_control_idx)
        robot.set_position_orientation(robot_pose["base_pos"], robot_pose["base_ori"])

        # Settle robot
        for _ in range(10):
            og.sim.step()

        for _ in range(100):
            og.sim.step()
            if np.linalg.norm(robot.get_linear_velocity()) > 1e-2:
                continue 
            if np.linalg.norm(robot.get_angular_velocity()) > 1e-2:
                continue
            # otherwise we've stopped
            break
        else:
            raise ValueError("Robot could not settle")

        # Check if the robot has toppled
        rotation = R.from_quat(robot.get_orientation())
        robot_up = rotation.apply(np.array([0, 0, 1]))
        if robot_up[2] < 0.75:
            raise ValueError("Robot has toppled over")

        print("Reset robot pose to: ", robot_pose)

    # Overwrite reset by only removeing reset scene
    def reset(self, env):
        """
        Resets this task in the environment

        Args:
            env (Environment): environment instance to reset
        """
        # Reset the scene, agent, and variables
        for _ in range(20):
            try:
                self._reset_scene(env)
                self._reset_agent(env)
                break
            except Exception as e:
                print("Resetting error: ", e)
        else:
            raise ValueError("Could not reset task.")
        self._reset_variables(env)

        # Also reset all termination conditions and reward functions
        for termination_condition in self._termination_conditions.values():
            termination_condition.reset(self, env)
        for reward_function in self._reward_functions.values():
            reward_function.reset(self, env)

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
        obj = env.scene.object_registry("name", self.obj_name)
        robot = env.robots[0]
        relative_pos, _ = T.relative_pose_transform(*obj.get_position_orientation(), *robot.get_position_orientation())

        return {"obj_pos": relative_pos}, dict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

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
            "dist_coeff": DIST_COEFF,
            "grasp_reward": GRASP_REWARD,
            "eef_position_penalty_coef": 0.01,
            "eef_orientation_penalty_coef": 0.001,
        }

