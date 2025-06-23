import omnigibson as og
from omnigibson.envs import EnvMetric
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.utils.constants import STRUCTURE_CATEGORIES, GROUND_CATEGORIES
from omnigibson.utils.backend_utils import _compute_backend as cb
import omnigibson.utils.transform_utils as T
from omnigibson.utils.sim_utils import prim_paths_to_rigid_prims
from omnigibson.robots import LocomotionRobot
from gello.robots.sim_robot.og_teleop_utils import GHOST_APPEAR_THRESHOLD
import torch as th
import numpy as np
import operator


class MotionMetric(EnvMetric):
    def __init__(self, step_dt):
        """
        Args:
            step_dt (float): Amount of time between steps, used to differentiate from pos -> vel -> acc -> jerk
        """
        self.step_dt = step_dt

        super().__init__()

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            # Record velocity (we'll derive accel -> jerk at the end of the episode)
            step_metrics[f"robot{i}::pos"] = robot.get_joint_positions()
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        episode_metrics = dict()
        for robot, (pos_key, positions) in zip(env.robots, episode_info.items()):
            arm_idxs = th.cat([arm_control_idx for arm_control_idx in robot.arm_control_idx.values()])
            arm_vel_limits = th.tensor([jnt.max_velocity for jnt in robot.joints.values()])[arm_idxs]
            positions = th.stack(positions, dim=0)[:, arm_idxs]
            vels = (positions[1:] - positions[:-1]) / self.step_dt
            n_vels = len(vels)
            accs = (vels[1:] - vels[:-1]) / self.step_dt
            jerks = (accs[1:] - accs[:-1]) / self.step_dt
            # Only keep absolute values
            vels, accs, jerks = th.abs(vels), th.abs(accs), th.abs(jerks)
            episode_metrics[f"{pos_key}::vel_avg"] = vels.mean(dim=0)
            episode_metrics[f"{pos_key}::vel_prop_over_05max"] = th.any(vels > arm_vel_limits.unsqueeze(0) * 0.5, dim=-1).sum() / n_vels
            episode_metrics[f"{pos_key}::vel_prop_over_06max"] = th.any(vels > arm_vel_limits.unsqueeze(0) * 0.6, dim=-1).sum() / n_vels
            episode_metrics[f"{pos_key}::vel_prop_over_07max"] = th.any(vels > arm_vel_limits.unsqueeze(0) * 0.7, dim=-1).sum() / n_vels
            episode_metrics[f"{pos_key}::vel_prop_over_08max"] = th.any(vels > arm_vel_limits.unsqueeze(0) * 0.8, dim=-1).sum() / n_vels
            episode_metrics[f"{pos_key}::vel_prop_over_09max"] = th.any(vels > arm_vel_limits.unsqueeze(0) * 0.9, dim=-1).sum() / n_vels
            episode_metrics[f"{pos_key}::vel_prop_over_10max"] = th.any(vels > arm_vel_limits.unsqueeze(0) * 1.0, dim=-1).sum() / n_vels
            episode_metrics[f"{pos_key}::acc_avg"] = accs.mean(dim=0)
            episode_metrics[f"{pos_key}::jerk_avg"] = jerks.mean(dim=0)
            episode_metrics[f"{pos_key}::vel_std"] = vels.std(dim=0)
            episode_metrics[f"{pos_key}::acc_std"] = accs.std(dim=0)
            episode_metrics[f"{pos_key}::jerk_std"] = jerks.std(dim=0)
            episode_metrics[f"{pos_key}::vel_max"] = vels.max(dim=0).values
            episode_metrics[f"{pos_key}::acc_max"] = accs.max(dim=0).values
            episode_metrics[f"{pos_key}::jerk_max"] = jerks.max(dim=0).values

        return episode_metrics

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            vel_avg_limit=None,
            vel_max_limit=None,
            vel_prop_over_05max=None,
            vel_prop_over_06max=None,
            vel_prop_over_07max=None,
            vel_prop_over_08max=None,
            vel_prop_over_09max=None,
            vel_prop_over_10max=None,
            acc_avg_limit=None,
            acc_max_limit=None,
            jerk_avg_limit=None,
            jerk_max_limit=None,
    ):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            vel_avg_limit (None or float): If specified, maximum average velocity that is acceptable for the episode to
                    be validated
            vel_max_limit (None or float): If specified, maximum peak velocity that is acceptable for the episode to
                    be validated
            vel_prop_over_05max (None or float): If specified, maximum acceptable proportion of frames where any
                joint's velocity is over 0.5 of its max joint velocity limit
            vel_prop_over_06max (None or float): If specified, maximum acceptable proportion of frames where any
                joint's velocity is over 0.6 of its max joint velocity limit
            vel_prop_over_07max (None or float): If specified, maximum acceptable proportion of frames where any
                joint's velocity is over 0.7 of its max joint velocity limit
            vel_prop_over_08max (None or float): If specified, maximum acceptable proportion of frames where any
                joint's velocity is over 0.8 of its max joint velocity limit
            vel_prop_over_09max (None or float): If specified, maximum acceptable proportion of frames where any
                joint's velocity is over 0.9 of its max joint velocity limit
            vel_prop_over_10max (None or float): If specified, maximum acceptable proportion of frames where any
                joint's velocity is over 1.0 of its max joint velocity limit
            acc_avg_limit (None or float): If specified, maximum average acceleration that is acceptable for the
                episode to be validated
            acc_max_limit (None or float): If specified, maximum acceleration that is acceptable for the episode to
                be validated
            jerk_avg_limit (None or float): If specified, maximum average jerk that is acceptable for the episode to
                    be validated
            jerk_max_limit (None or float): If specified, maximum jerk that is acceptable for the episode to
                be validated

        Returns:
            dict: Keyword-mapped dictionary mapping each validation test name to {"success": bool, "feedback": str} dict
                where "success" is True if the given @episode_metrics pass that specific test; if False, "feedback"
                provides information as to why the test failed
        """
        results = dict()
        for val_max_limit, val_name in zip(
                (vel_avg_limit, vel_max_limit, vel_prop_over_05max, vel_prop_over_06max, vel_prop_over_07max, vel_prop_over_08max, vel_prop_over_09max, vel_prop_over_10max, acc_avg_limit, acc_max_limit, jerk_avg_limit, jerk_max_limit),
                ("vel_avg", "vel_max", "vel_prop_over_05max", "vel_prop_over_06max", "vel_prop_over_07max", "vel_prop_over_08max", "vel_prop_over_09max", "vel_prop_over_10max", "acc_avg", "acc_max", "jerk_avg", "jerk_max"),
        ):
            if val_max_limit is not None:
                for name, metric in episode_metrics.items():
                    if f"::{val_name}" in name:
                        test_name = name
                        success = metric <= val_max_limit
                        if isinstance(success, th.Tensor):
                            success = th.all(success).item()
                        feedback = None if success else f"Robot's {val_name} is too high ({metric}), must be <= {val_max_limit}"
                        results[test_name] = {"success": success, "feedback": feedback}

        return results


class CollisionMetric(EnvMetric):
    def __init__(self, default_color=(0.8235, 0.8235, 1.0000)):
        """
        Args:
            default_color (3-tuple): Default color to assign to the robot's geoms when there's no colored collision
                active
        """
        self.checks = dict()
        self.check_colors = dict()
        self.color_is_active = dict()
        self.default_color = th.tensor(default_color)
        self.active_color = self.default_color

        super().__init__()

    def add_check(self, name, check, color_robots=None):
        """
        Adds a collision check to this metric, which can be queried by @name

        Args:
            name (str): name of the check
            check (function): Collision checker function, with the following signature:

                def check(env: Environment) -> bool

                which should return True if there is collision, else False

            color_robots (None or 3-element tuple): If set, should be the (R,G,B) color to visualize for the entire
                robots' set of visual geoms when this check is active. Else, will not do any coloring. Note that
                downstream checks added are prioritized in terms of coloring
        """
        self.checks[name] = check
        self.check_colors[name] = color_robots

    def remove_check(self, name):
        """
        Removes check with corresponding @name

        Args:
            name (str): name of the check to remove
        """
        self.checks.pop(name)
        self.check_colors.pop(name)

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        active_color = self.default_color
        for name, check in self.checks.items():
            active = check(env)
            step_metrics[f"{name}"] = active
            if active:
                color = self.check_colors[name]
                if color is not None:
                    active_color = color
        if th.any(active_color != self.active_color).item():
            # Our color has changed, update accordingly
            for robot in env.robots:
                for link in robot.links.values():
                    for vm in link.visual_meshes.values():
                        if vm.material is not None:
                            vm.material.diffuse_color_constant = active_color
            self.active_color = active_color

        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Compute any collisions from this step
        episode_metrics = dict()
        for name, collisions in episode_info.items():
            collisions = th.tensor(collisions)
            episode_metrics[f"{name}::n_collision"] = collisions.sum().item()

        return episode_metrics

    def reset(self, env):
        super().reset(env=env)

        # Reset all robot colors
        if th.any(self.active_color != self.default_color).item():
            # Our color has changed, reset to default color
            for robot in env.robots:
                for link in robot.links.values():
                    for vm in link.visual_meshes.values():
                        vm.material.diffuse_color_constant = self.default_color
            self.active_color = self.default_color

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            collision_limits=None,
    ):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            collision_limits (None or dict): If specified, should map collision check name (i.e.: the name
                passed in during self.add_check() to corresponding maximum acceptable collision count)

        Returns:
            dict: Keyword-mapped dictionary mapping each validation test name to {"success": bool, "feedback": str} dict
                where "success" is True if the given @episode_metrics pass that specific test; if False, "feedback"
                provides information as to why the test failed
        """
        results = dict()
        if collision_limits is not None:
            for name, collision_limit in collision_limits.items():
                test_name = name
                if collision_limit is not None:
                    n_collisions = episode_metrics[f"{name}::n_collision"]
                    success = n_collisions <= collision_limit
                    feedback = None if success else f"Too many collisions ({n_collisions}) when checking {name}, must be <= {collision_limit}"
                    results[test_name] = {"success": success, "feedback": feedback}

        return results


class TaskSuccessMetric(EnvMetric):

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        # Record whether task is done (terminated is true but not truncated)
        return {"done": terminated and not truncated}

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        return {"success": th.any(th.tensor(episode_info["done"])).item()}

    @classmethod
    def validate_episode(cls, episode_metrics):
        success = episode_metrics["success"]
        feedback = None if success else "Task was a not a success!"
        return {"task_success": {"success": success, "feedback": feedback}}


class GhostHandAppearanceMetric(EnvMetric):

    def __init__(self, color_arms=True):
        self.color_arms = color_arms
        self.robot_arm_colors = dict()

        super().__init__()

    @classmethod
    def is_compatible(cls, env):
        valid = super().is_compatible(env=env)
        if valid:
            # We must be using a binary / smooth gripper controller for each robot to ensure that we can
            # infer un/grasping intent
            for robot in env.robots:
                for arm in robot.arm_names:
                    gripper_controller = robot.controllers[f"gripper_{arm}"]
                    is_1d = gripper_controller.command_dim == 1
                    is_normalized = (th.all(cb.to_torch(gripper_controller.command_input_limits[0]) == -1.0).item() and
                                     th.all(cb.to_torch(gripper_controller.command_input_limits[1]) == 1.0).item())
                    valid = is_1d and is_normalized
                    if not valid:
                        break
                if not valid:
                    break
        return valid

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        # Record whether task is done (terminated is true but not truncated)
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            robot_qpos = robot.get_joint_positions(normalized=False)
            gripper_action_idxs = robot.gripper_action_idx
            for arm in robot.arm_names:
                active = th.max(th.abs(
                        robot_qpos[robot.arm_control_idx[arm]] - action[robot.arm_action_idx[arm]]
                )).item() > GHOST_APPEAR_THRESHOLD
                gripper_controller = robot.controllers[f"gripper_{arm}"]
                step_metrics[f"robot{i}::arm_{arm}::active"] = active
                if self.color_arms:
                    if robot.name not in self.robot_arm_colors:
                        self.robot_arm_colors[robot.name] = {a: False for a in robot.arm_names}
                    robot_arm_color_is_active = self.robot_arm_colors[robot.name][arm]
                    if active != robot_arm_color_is_active:
                        # Update the coloring
                        # TODO: Overfit to R1Pro at the moment
                        color = th.tensor([1.0, 0, 0]) if active else th.tensor([0.8235, 0.8235, 1.0000])
                        for link in robot.arm_links[arm]:
                            for vm in link.visual_meshes.values():
                                vm.material.diffuse_color_constant = color
                        self.robot_arm_colors[robot.name][arm] = active
                op = operator.lt if gripper_controller._inverted else operator.ge
                step_metrics[f"robot{i}::arm_{arm}::open_cmd"] = th.all(op(action[gripper_action_idxs[arm]], 0.0)).item()
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Aggregate number of steps the ghost hands have appeared per-arm, and whether the robot was releasing a grasp
        # during that time
        episode_metrics = dict()
        for i, robot in enumerate(env.robots):
            for arm in robot.arm_names:
                pf = f"robot{i}::arm_{arm}"
                active = th.tensor(episode_info[f"{pf}::active"])
                open_cmd = th.tensor(episode_info[f"{pf}::open_cmd"])
                ungrasping = open_cmd[1:] & ~open_cmd[:-1]
                episode_metrics[f"{pf}::n_steps_total"] = active.sum().item()
                episode_metrics[f"{pf}::n_steps_while_ungrasping"] = (active[1:] & ungrasping).sum().item()
        return episode_metrics

    def reset(self, env):
        """
        Resets this metric with respect to @env

        Args:
            env (EnvironmentWrapper): Environment being tracked
        """
        super().reset(env=env)

        # Reset all robot colors
        for i, robot in enumerate(env.robots):
            if self.color_arms and robot.name in self.robot_arm_colors:
                for arm in robot.arm_names:
                    if self.robot_arm_colors[robot.name][arm]:
                        # TODO: Overfit to R1Pro at the moment
                        # Reset to original color
                        color = th.tensor([0.8235, 0.8235, 1.0000])
                        for link in robot.arm_links[arm]:
                            for vm in link.visual_meshes.values():
                                vm.material.diffuse_color_constant = color

        self.robot_arm_colors = dict()

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            gh_appearance_limit=None,
            gh_appearance_limit_while_ungrasping=None,
    ):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            gh_appearance_limit (None or int): If specified, maximum acceptable number of steps where the ghost hand
                was triggered in the episode
            gh_appearance_limit_while_ungrasping (None or int): If specified, maximum number of steps where the
                ghost hand was triggered during an ungrasp

        Returns:
            dict: Keyword-mapped dictionary mapping each validation test name to {"success": bool, "feedback": str} dict
                where "success" is True if the given @episode_metrics pass that specific test; if False, "feedback"
                provides information as to why the test failed
        """
        results = dict()
        if gh_appearance_limit is not None or gh_appearance_limit_while_ungrasping is not None:
            for name, metric in episode_metrics.items():
                test_name = name
                if "::n_steps_total" in name:
                    if gh_appearance_limit is None:
                        continue
                    success = episode_metrics[name] <= gh_appearance_limit
                    limit = gh_appearance_limit
                elif "::n_steps_while_ungrasping" in name:
                    if gh_appearance_limit_while_ungrasping is None:
                        continue
                    success = episode_metrics[name] <= gh_appearance_limit_while_ungrasping
                    limit = gh_appearance_limit_while_ungrasping
                else:
                    raise ValueError(f"Got invalid metric name: {name}")
                feedback = None if success else f"Too many ghost hand appearances ({episode_metrics[name]}) when checking {name}, must be <= {limit}"
                results[test_name] = {"success": success, "feedback": feedback}

        return results


class ProlongedPauseMetric(MotionMetric):

    def __init__(self, step_dt, vel_threshold=0.001):
        """
        Args:
            step_dt (float): Amount of time between steps, used to differentiate from pos -> vel -> acc -> jerk
            vel_threshold (float): Per-joint vel threshold for determining whether there's any motion occurring
                at a given step
        """
        self.vel_threshold = vel_threshold
        super().__init__(step_dt=step_dt)

    def _compute_episode_metrics(self, env, episode_info):
        # Derive velocities, then count consecutive steps that contain values greater than our threshold
        episode_metrics = dict()
        for pos_key, positions in episode_info.items():
            positions = th.stack(positions, dim=0)
            vels = (positions[1:] - positions[:-1]) / self.step_dt
            in_motions = th.any(th.abs(vels) > self.vel_threshold, dim=-1)
            max_pause_length = 0
            current_pause_length = 0
            for in_motion in in_motions:
                if not in_motion.item():
                    current_pause_length += 1
                    if current_pause_length > max_pause_length:
                        max_pause_length = current_pause_length
                else:
                    current_pause_length = 0
            episode_metrics[f"{pos_key}::max_pause_length"] = max_pause_length

        return episode_metrics

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            pause_steps_limit=None,
    ):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            pause_steps_limit (None or int): If specified, maximum acceptable number of consecutive steps without
                robot motion

        Returns:
            dict: Keyword-mapped dictionary mapping each validation test name to {"success": bool, "feedback": str} dict
                where "success" is True if the given @episode_metrics pass that specific test; if False, "feedback"
                provides information as to why the test failed
        """
        results = dict()
        if pause_steps_limit is not None:
            for name, metric in episode_metrics.items():
                test_name = name
                success = episode_metrics[name] <= pause_steps_limit
                feedback = None if success else f"Too many consecutive steps ({episode_metrics[name]}) without robot motion, must be <= {pause_steps_limit}"
                results[test_name] = {"success": success, "feedback": feedback}

        return results


class FailedGraspMetric(EnvMetric):

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            # Record whether fingers are closed (values ~ 0) -- this implies a failed grasp
            for arm in robot.arm_names:
                step_metrics[f"robot{i}::arm_{arm}::fingers_closed"] = th.allclose(robot.get_joint_positions()[robot.gripper_control_idx[arm]], th.zeros(2), atol=1e-4)
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Compute all fingers_closed upticks to get total failed grasp count
        episode_metrics = dict()
        for i, robot in enumerate(env.robots):
            for arm in robot.arm_names:
                pf = f"robot{i}::arm_{arm}"
                fingers_closed = th.tensor(episode_info[f"{pf}::fingers_closed"])
                fingers_closed_transition = fingers_closed[1:] & ~fingers_closed[:-1]
                episode_metrics[f"{pf}::failed_grasp_count"] = fingers_closed_transition.sum().item()

        return episode_metrics

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            failed_grasp_limit=None,
    ):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            failed_grasp_limit (None or int): If specified, maximum acceptable number of failed (empty) grasps

        Returns:
            dict: Keyword-mapped dictionary mapping each validation test name to {"success": bool, "feedback": str} dict
                where "success" is True if the given @episode_metrics pass that specific test; if False, "feedback"
                provides information as to why the test failed
        """
        results = dict()
        if failed_grasp_limit is not None:
            for name, metric in episode_metrics.items():
                test_name = name
                success = episode_metrics[name] <= failed_grasp_limit
                feedback = None if success else f"Too many ({episode_metrics[name]}) failed grasps, must be <= {failed_grasp_limit}"
                results[test_name] = {"success": success, "feedback": feedback}

        return results


class TaskRelevantObjectVelocityMetric(EnvMetric):
    def __init__(self, step_dt):
        """
        Args:
            step_dt (float): Amount of time between steps, used to differentiate from pos -> vel
        """
        self.step_dt = step_dt

        super().__init__()

    @classmethod
    def is_compatible(cls, env):
        return isinstance(env.task, BehaviorTask)

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for name, bddl_inst in env.task.object_scope.items():
            if bddl_inst.is_system or not bddl_inst.exists or bddl_inst.fixed_base or "agent" in name:
                continue
            step_metrics[f"{name}::pos"] = bddl_inst.get_position_orientation()[0]
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        episode_metrics = dict()
        for pos_key, positions in episode_info.items():
            positions = th.stack(positions, dim=0)
            vels = th.norm(positions[1:] - positions[:-1], dim=-1) / self.step_dt
            episode_metrics[f"{pos_key}::vel_avg"] = vels.mean().item()
            episode_metrics[f"{pos_key}::vel_std"] = vels.std().item()
            episode_metrics[f"{pos_key}::vel_max"] = vels.max().item()

        return episode_metrics

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            vel_max_limit=None,
    ):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            vel_max_limit (None or float): If specified, maximum velocity that is acceptable for the episode to
                be validated

        Returns:
            dict: Keyword-mapped dictionary mapping each validation test name to {"success": bool, "feedback": str} dict
                where "success" is True if the given @episode_metrics pass that specific test; if False, "feedback"
                provides information as to why the test failed
        """
        results = dict()
        if vel_max_limit is not None:
            for name, metric in episode_metrics.items():
                if f"::vel_max" in name:
                    test_name = name
                    success = metric <= vel_max_limit
                    feedback = None if success else f"{name} is too high({metric}), must be <= {vel_max_limit}"
                    results[test_name] = {"success": success, "feedback": feedback}

        return results


class FieldOfViewMetric(EnvMetric):
    """
    When teleoperator grasp/release, the gripper needs to be in field of view
    """
    @classmethod
    def is_compatible(cls, env):
        valid = super().is_compatible(env=env)
        if valid:
            # We must be using a binary / smooth gripper controller for each robot to ensure that we can
            # infer un/grasping intent
            for robot in env.robots:
                for arm in robot.arm_names:
                    gripper_controller = robot.controllers[f"gripper_{arm}"]
                    is_1d = gripper_controller.command_dim == 1
                    is_normalized = (th.all(cb.to_torch(gripper_controller.command_input_limits[0]) == -1.0).item() and
                                     th.all(cb.to_torch(gripper_controller.command_input_limits[1]) == 1.0).item())
                    valid = is_1d and is_normalized
                    if not valid:
                        break
                if not valid:
                    break
        return valid

    def __init__(self, head_camera, gripper_link_paths):
        """
        Args:
            head_camera (VisionSensor): The head camera of the robot
            gripper_link_paths (dict): The paths of the gripper links
        """
        self.head_camera = head_camera
        self.gripper_link_paths = gripper_link_paths

        assert "seg_instance_id" in self.head_camera.modalities, "FieldOfViewMetric requires instance_id_segmentation modality"

        super().__init__()

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            _, info = self.head_camera.get_obs()
            links_in_fov = set(info["seg_instance_id"].values())
            gripper_action_idxs = robot.gripper_action_idx
            for arm in robot.arm_names:
                gripper_controller = robot.controllers[f"gripper_{arm}"]
                op = operator.lt if gripper_controller._inverted else operator.ge
                # check if any of the gripper link for this arm is in the field of view
                gripper_in_fov = len(links_in_fov.intersection(self.gripper_link_paths[arm])) > 0

                step_metrics[f"robot{i}::arm_{arm}::open_cmd"] = th.all(op(action[gripper_action_idxs[arm]], 0.0)).item()
                step_metrics[f"robot{i}::arm_{arm}::gripper_in_fov"] = gripper_in_fov
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        episode_metrics = dict()

        for i, robot in enumerate(env.robots):
            for arm in robot.arm_names:
                pf = f"robot{i}::arm_{arm}"
                open_cmd = th.tensor(episode_info[f"{pf}::open_cmd"])
                gripper_in_fov = th.tensor(episode_info[f"{pf}::gripper_in_fov"])

                # Detect grasping state changes (comparing current with previous)
                # For index 0, we assume no change (start of episode), so we only evaluate index 1 - end
                grasping_changes = open_cmd[1:] != open_cmd[:-1]

                # Count steps when gripper is not in field of view
                episode_metrics[f"robot{i}::arm_{arm}::gripper_outside_fov"] = (gripper_in_fov == 0).sum().item()

                # Count changes when gripper was NOT in field of view (undesired behavior)
                episode_metrics[f"robot{i}::arm_{arm}::grasp_changes_outside_fov"] = (grasping_changes & ~gripper_in_fov[1:]).sum().item()
        return episode_metrics

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            gripper_changes_outside_fov_limit=None,
    ):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            gripper_changes_outside_fov_limit (None or float): If specified, maximum acceptable number of instances
                where the gripper was outside the fov during an un/grasp change

        Returns:
            dict: Keyword-mapped dictionary mapping each validation test name to {"success": bool, "feedback": str} dict
                where "success" is True if the given @episode_metrics pass that specific test; if False, "feedback"
                provides information as to why the test failed
        """
        results = dict()
        if gripper_changes_outside_fov_limit is not None:
            for name, metric in episode_metrics.items():
                if f"::grasp_changes_outside_fov" in name:
                    test_name = name
                    success = metric <= gripper_changes_outside_fov_limit
                    feedback = None if success else f"{name} is too high ({metric}) (too many times the gripper was toggled outside of the robot's main FOV), must be <= {gripper_changes_outside_fov_limit}"
                    results[test_name] = {"success": success, "feedback": feedback}

        return results


class HeadCameraUprightMetric(EnvMetric):
    """
    When robot navigate for prolonged periods, head camera link should not be tilted up or down too much
    """
    @classmethod
    def is_compatible(cls, env):
        return super().is_compatible(env) and all(isinstance(robot, LocomotionRobot) for robot in env.robots)
    
    def __init__(self, head_camera_link_name, step_dt, navigation_window=3.0, translation_threshold=0.1, rotation_threshold=0.05, camera_tilt_threshold=0.4):
        """
        Args:
            head_camera_link_name (str): head camera link name
            step_dt (float): Amount of time between steps
            navigation_window (float): window size for detecting navigation in seconds
            translation_threshold (float): threshold for translation velocity
            rotation_threshold (float): threshold for rotation velocity
            camera_tilt_threshold (float): threshold for camera tilt
        """
        self.head_camera_link_name = head_camera_link_name
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.camera_tilt_threshold = camera_tilt_threshold
        
        self.navigation_window_in_steps = int(navigation_window / step_dt)
        self.step_dt = step_dt

        super().__init__()

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            _, ori = robot.links[self.head_camera_link_name].get_position_orientation()
            step_metrics[f"robot{i}::head_link_y_ori"] = T.quat2euler(ori)[1]
            base_pos, base_ori = robot.get_position_orientation()
            step_metrics[f"robot{i}::base_pos_x"] = base_pos[0]
            step_metrics[f"robot{i}::base_pos_y"] = base_ori[1]
            step_metrics[f"robot{i}::base_ori_yaw"] = T.quat2euler(base_ori)[2]
            
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        episode_metrics = dict()

        for i, robot in enumerate(env.robots):
            # Get the stored data from step metrics
            head_y_ori = th.tensor(episode_info[f"robot{i}::head_link_y_ori"])
            base_pos_x = th.tensor(episode_info[f"robot{i}::base_pos_x"])
            base_pos_y = th.tensor(episode_info[f"robot{i}::base_pos_y"])
            base_ori_yaw = th.tensor(episode_info[f"robot{i}::base_ori_yaw"])
            
            # Calculate base position and orientation changes to detect navigation
            base_pos_diff_x = base_pos_x[1:] - base_pos_x[:-1]
            base_pos_diff_y = base_pos_y[1:] - base_pos_y[:-1]
            # TODO: handle angle wrapping here
            base_ori_diff_yaw = base_ori_yaw[1:] - base_ori_yaw[:-1]
            
            # Detect when robot is navigating (has significant position or orientation change)
            translation_velocity = th.sqrt(base_pos_diff_x**2 + base_pos_diff_y**2) / self.step_dt
            is_translating = translation_velocity > self.translation_threshold
            is_rotating = th.abs(base_ori_diff_yaw / self.step_dt) > self.rotation_threshold
            is_navigating = is_translating | is_rotating
            
            # Add a placeholder for the first timestep (assume not navigating)
            is_navigating = th.cat([th.tensor([False]), is_navigating])
            is_tilted = th.abs(head_y_ori) > self.camera_tilt_threshold
            prolonged_navigation_mask = th.zeros_like(is_navigating, dtype=th.bool)

            consecutive_count = 0
            for j in range(len(is_navigating)):
                if is_navigating[j]:
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                    
                # Consider navigation "prolonged" after several consecutive frames
                if consecutive_count >= self.navigation_window_in_steps:
                    prolonged_navigation_mask[j] = True

            episode_metrics[f"robot{i}::head_camera_tilted_during_navigation"] = (is_tilted & prolonged_navigation_mask).sum().item()

        return episode_metrics

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            head_camera_tilt_during_navigation_limit=None,
    ):
        """
        Validates the given @episode_metrics from self.aggregate_results using any specific @kwargs

        Args:
            episode_metrics (dict): Metrics aggregated using self.aggregate_results
            head_camera_tilt_during_navigation_limit (None or float): If specified, maximum acceptable number of instances
                where the head camera was tilted during navigation

        Returns:
            dict: Keyword-mapped dictionary mapping each validation test name to {"success": bool, "feedback": str} dict
                where "success" is True if the given @episode_metrics pass that specific test; if False, "feedback"
                provides information as to why the test failed
        """
        results = dict()
        if head_camera_tilt_during_navigation_limit is not None:
            for name, metric in episode_metrics.items():
                if f"::head_camera_tilted_during_navigation" in name:
                    test_name = name
                    success = metric <= head_camera_tilt_during_navigation_limit
                    feedback = None if success else f"{name} is too high ({metric}) (too many steps where the robot head is tilted during navigation), must be <= {head_camera_tilt_during_navigation_limit}"
                    results[test_name] = {"success": success, "feedback": feedback}

        return results


def check_robot_self_collision(env, min_threshold=None):
    # TODO: What about gripper finger self collision?
    for robot in env.robots:
        link_paths = robot.link_prim_paths
        if min_threshold is None:
            if RigidContactAPI.in_contact(link_paths, link_paths):
                return True
        else:
            if th.any(th.norm(RigidContactAPI.get_impulses(link_paths, link_paths), dim=-1) > min_threshold).item():
                return True
    return False


def check_robot_base_nonarm_nonkinematic_collision(env):
    # TODO: How to check for wall collisions? They're kinematic only
    # # One solution: Make them non-kinematic only during QA checking
    # floor_link_paths = []
    # for structure_category in STRUCTURE_CATEGORIES:
    #     for structure in env.scene.object_registry("category", structure_category):
    #         floor_link_paths += structure.link_prim_paths
    # floor_link_col_idxs = {RigidContactAPI.get_body_col_idx(link_path) for link_path in floor_link_paths}

    for robot in env.robots:
        robot_link_paths = set(robot.link_prim_paths)
        for arm in robot.arm_names:
            robot_link_paths -= set(link.prim_path for link in robot.arm_links[arm])
            robot_link_paths -= set(link.prim_path for link in robot.gripper_links[arm])
            robot_link_paths -= set(link.prim_path for link in robot.finger_links[arm])
    robot_link_idxs = [RigidContactAPI.get_body_col_idx(link_path)[1] for link_path in robot_link_paths]
    robot_contacts = RigidContactAPI.get_all_impulses(env.scene.idx)[robot_link_idxs]

    return th.any(robot_contacts).item()


def check_robot_nonarm_nonground_collision(env):
    for robot in env.robots:
        robot_arm_paths = set()
        robot_prim_path = robot.prim_path
        for arm in robot.arm_names:
            robot_arm_paths = robot_arm_paths.union(set(link.prim_path for link in robot.arm_links[arm]))
            robot_arm_paths = robot_arm_paths.union(set(link.prim_path for link in robot.gripper_links[arm]))
            robot_arm_paths = robot_arm_paths.union(set(link.prim_path for link in robot.finger_links[arm]))
        for link in robot.links.values():
            # Skip if link is an arm link
            if link.prim_path in robot_arm_paths:
                continue
            for c in link.contact_list():
                # Skip if it's a self-collision
                if robot_prim_path in c.body0:
                    if robot_prim_path in c.body1:
                        continue
                    else:
                        c_prim_path = c.body1
                else:
                    c_prim_path = c.body0
                # Ignore if zero-impulse
                if np.linalg.norm(tuple(c.impulse)) == 0:
                    continue
                # Check which object this is
                rigid_prims = prim_paths_to_rigid_prims([c_prim_path], robot.scene)
                # Skip if obj is part of ground categories
                assert len(rigid_prims) == 1
                obj = next(iter(rigid_prims))[0]
                if obj.category in GROUND_CATEGORIES:
                    continue
                # Otherwise this is a valid contact, so immediately return True
                return True

    return False


def create_collision_metric(
        include_robot_self_collision=True,
        include_robot_nonarm_nonkinematic_collision=True,
        include_robot_nonarm_nonground_collision=True,
):
    col_metric = CollisionMetric()
    if include_robot_self_collision:
        col_metric.add_check(name="robot_self", check=check_robot_self_collision, color_robots=th.tensor([1.0, 0, 0]))
    if include_robot_nonarm_nonkinematic_collision:
        col_metric.add_check(name="robot_nonarm_nonstructure", check=check_robot_base_nonarm_nonkinematic_collision)
    if include_robot_nonarm_nonground_collision:
        col_metric.add_check(name="robot_nonarm_nonground", check=check_robot_nonarm_nonground_collision)
    return col_metric

