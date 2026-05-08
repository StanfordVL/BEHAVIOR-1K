import json
import random

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.objects.usd_object import REGISTERED_OBJECTS
from omnigibson.reward_functions.grasp_reward import GraspReward
from omnigibson.scenes.scene_base import Scene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
from omnigibson.utils.python_utils import classproperty, create_class_from_registry_and_config

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
        include_obs=True,
        precached_reset_pose_path=None,
        objects_config=None,
    ):
        self.obj_name = obj_name
        self._primitive_controllers = None  # list per scene
        self._reset_poses = None
        self._objects_config = objects_config
        if precached_reset_pose_path is not None:
            with open(precached_reset_pose_path) as f:
                self._reset_poses = json.load(f)
        super().__init__(termination_config=termination_config, reward_config=reward_config, include_obs=include_obs)

    def _load(self, env):
        for scene in env.scenes:
            for obj_config in self._objects_config:
                obj = scene.object_registry("name", obj_config["name"])
                # Create object
                if obj is None:
                    obj = create_class_from_registry_and_config(
                        cls_name=obj_config["type"],
                        cls_registry=REGISTERED_OBJECTS,
                        cfg=obj_config,
                        cls_type_descriptor="object",
                    )
                    # Import the object into the simulator and set the pose
                    scene.add_object(obj)

                obj_pos = [0.0, 0.0, 0.0] if "position" not in obj_config else obj_config["position"]
                obj_orn = [0.0, 0.0, 0.0, 1.0] if "orientation" not in obj_config else obj_config["orientation"]
                obj.set_position_orientation(position=obj_pos, orientation=obj_orn, frame="scene")

    def _create_termination_conditions(self):
        terminations = dict()
        # terminations["graspgoal"] = GraspGoal(
        #     self.obj_name
        # )
        # This helpes to prevent resets happening at different times
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        # terminations["falling"] = Falling()

        return terminations

    def _create_reward_functions(self):
        rewards = dict()
        rewards["grasp"] = GraspReward(self.obj_name, **self._reward_config)
        return rewards

    def _reset_agent(self, env, env_indices):
        for env_idx in env_indices:
            robot = env.scenes[env_idx].robots[0]
            for arm in robot.arm_names:
                robot.release_grasp_immediately(arm=arm)

            # If available, reset the robot with cached reset poses.
            # This is significantly faster than randomizing using the primitives.
            if self._reset_poses is not None:
                joint_control_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
                robot_pose = random.choice(self._reset_poses)
                joint_pos = th.tensor(robot_pose["joint_pos"])
                robot.set_joint_positions(joint_pos, joint_control_idx)
                robot_pos = th.tensor(robot_pose["base_pos"])
                robot_orn = th.tensor(robot_pose["base_ori"])
                robot.set_position_orientation(position=robot_pos, orientation=robot_orn, frame="scene")
            # Otherwise, reset using the primitive controller (requires holonomic base robot).
            elif not robot.is_holonomic_base:
                raise ValueError(
                    f"Robot '{robot.model}' does not have a holonomic base. "
                    "CuRobo-based reset requires a holonomic base robot (e.g. Tiago, R1). "
                    "Please provide precached_reset_pose_path for non-holonomic robots."
                )
            else:
                if self._primitive_controllers is None:
                    self._primitive_controllers = [None] * env.num_envs
                if self._primitive_controllers[env_idx] is None:
                    self._primitive_controllers[env_idx] = StarterSemanticActionPrimitives(
                        env, robot, enable_head_tracking=False
                    )
                pc = self._primitive_controllers[env_idx]

                # Randomize the robots joint positions
                joint_control_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
                for _ in range(MAX_JOINT_RANDOMIZATION_ATTEMPTS):
                    joint_pos, joint_control_idx = self._get_random_joint_position(robot)
                    all_joint_pos = robot.get_joint_positions().clone()
                    all_joint_pos[joint_control_idx] = joint_pos
                    collision_detected = pc._motion_generator.check_collisions(
                        [all_joint_pos],
                    ).cpu()[0]
                    if not collision_detected:
                        robot.set_joint_positions(joint_pos, joint_control_idx)
                        break

                # Randomize the robot's 2d pose
                obj = env.scenes[env_idx].object_registry("name", self.obj_name)
                grasp_poses = get_grasp_poses_for_object_sticky(obj)
                grasp_pose = random.choice(grasp_poses)
                sampled_pose_2d = pc._sample_pose_near_object(obj, eef_pose=grasp_pose)
                if sampled_pose_2d is None:
                    raise ValueError("Could not sample a valid 2d pose near the object")
                robot_pose = pc._get_robot_pose_from_2d_pose(sampled_pose_2d)
                robot.set_position_orientation(*robot_pose)

                # Check if the robot has toppled
                robot_up = T.quat_apply(robot.get_position_orientation()[1], th.tensor([0, 0, 1], dtype=th.float32))
                if robot_up[2] < 0.75:
                    raise ValueError("Robot has toppled over")

    def _reset_scene(self, env, env_indices):
        super()._reset_scene(env, env_indices)

        # Reset objects
        for idx in env_indices:
            for obj_config in self._objects_config:
                # Get object in the scene
                obj_name = obj_config["name"]
                obj = env.scenes[idx].object_registry("name", obj_name)
                if obj is None:
                    raise ValueError("Object {} not found in scene".format(obj_name))

                # Set object pose
                obj_pos = [0.0, 0.0, 0.0] if "position" not in obj_config else obj_config["position"]
                obj_orn = [0.0, 0.0, 0.0, 1.0] if "orientation" not in obj_config else obj_config["orientation"]
                obj.set_position_orientation(position=obj_pos, orientation=obj_orn, frame="scene")

    # Overwrite reset by only removing reset scene
    def reset(self, env, env_indices=None):
        """
        Resets this task in the environment

        Args:
            env (Environment): environment instance to reset
            env_indices (th.Tensor): indices of environments to reset
        """
        if env_indices is None:
            env_indices = th.arange(self._num_envs)

        # Reset the scene, agent, and variables
        # Try up to 20 times per env
        for _ in range(20):
            try:
                self._reset_scene(env, env_indices)
                self._reset_agent(env, env_indices)
                break
            except Exception as e:
                print("Resetting error: ", e)
        else:
            raise ValueError("Could not reset task.")
        self._reset_variables(env, env_indices)

        # Also reset all termination conditions and reward functions
        for termination_condition in self._termination_conditions.values():
            termination_condition.reset(self, env, env_indices)
        for reward_function in self._reward_functions.values():
            reward_function.reset(self, env, env_indices)

    def _get_random_joint_position(self, robot):
        joint_positions = []
        joint_control_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        all_joints = list(robot.joints.values())
        arm_joints = [all_joints[idx] for idx in joint_control_idx]
        for joint in arm_joints:
            val = random.uniform(joint.lower_limit, joint.upper_limit)
            joint_positions.append(val)
        return th.tensor(joint_positions), joint_control_idx

    def _get_obs(self, env, env_idx):
        obj = env.scenes[env_idx].object_registry("name", self.obj_name)
        robot = env.scenes[env_idx].robots[0]
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
        return {"max_steps": 100000}

    @classproperty
    def default_reward_config(cls):
        return {
            "dist_coeff": 0.1,
            "grasp_reward": 1.0,
            "collision_penalty": 1.0,
            "eef_position_penalty_coef": 0.01,
            "eef_orientation_penalty_coef": 0.001,
            "regularization_coef": 0.01,
        }
