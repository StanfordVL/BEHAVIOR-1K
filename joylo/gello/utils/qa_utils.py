"""
Quality Assurance (QA) Metrics for robot trajectory validation.

This module provides a comprehensive framework for validating robot demonstration
episodes based on various quality metrics including motion smoothness, collision
detection, task success, and more. Each metric is implemented as a subclass of
OmniGibson's MetricBase and can be configured with different enforcement modes
(DISABLED, SOFT, HARD).

The metrics are designed to work with the BEHAVIOR-1K dataset collection framework
and can be used to filter out low-quality demonstrations during data collection.

Example:
    >>> from gello.utils.qa_utils import ALL_QA_METRICS, aggregate_episode_validation
    >>> success, results = aggregate_episode_validation(task_name, episode_metrics)
    >>> if not success:
    ...     print("Episode failed validation")
"""

import omnigibson.utils.transform_utils as T
import operator
import torch as th
from enum import IntEnum
from omnigibson.metrics.metric_base import MetricBase
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.utils.constants import GROUND_CATEGORIES
from omnigibson.utils.backend_utils import _compute_backend as cb

from gello.utils.og_teleop_utils import GHOST_APPEAR_THRESHOLD


COMMON_QA_METRICS = {
    "motion",
    "collision",
    "task_success",
    "ghost_hand_appearance",
    "prolonged_pause",
    "failed_grasp",
    "task_relevant_obj_vel",
    "gripper_in_fov",
    "head_camera_upright_during_navigation",
}


class MetricMode(IntEnum):
    """Defines the enforcement mode for QA metrics.

    Attributes:
        DISABLED: Metric is computed but not used for validation.
        SOFT: Metric generates warnings but does not fail validation.
        HARD: Metric must pass or validation fails.
    """

    DISABLED = 0
    SOFT = 1
    HARD = 2


def aggregate_episode_validation(task, all_episode_metrics):
    """Validates the given @all_episode_metrics for a specific task.

    This function aggregates per-metric validation results and determines whether
    an episode passes QA checks based on the configured MetricMode for each metric.

    Args:
        task (str): The name of the task whose QA metrics are being aggregated.
        all_episode_metrics (dict): Keyword-mapped aggregated episode metrics
            with keys in the format "metric_name::sub_metric_name".

    Returns:
        2-tuple:
            - bool: Whether the validation succeeded or not (requires all metric
              validation checks to pass based on their mode).
            - dict: Per-metric validation information containing success status
              and optional feedback messages.

    Example:
        >>> task = "pick_up_apple"
        >>> metrics = {"motion::vel_avg": tensor([0.1]), "collision::n_collision": 0}
        >>> success, results = aggregate_episode_validation(task, metrics)
    """
    results = dict()
    sorted_metrics = dict()
    for name, val in all_episode_metrics.items():
        metric_name = name.split("::")[0]
        metric_val_name = "::".join(name.split("::")[1:])
        if metric_name not in sorted_metrics:
            sorted_metrics[metric_name] = dict()
        sorted_metrics[metric_name][metric_val_name] = val
    for metric_name, episode_metrics in sorted_metrics.items():
        metric_info = ALL_QA_METRICS[metric_name]
        if metric_info["mode"] == MetricMode.DISABLED:
            continue
        if (
            metric_info["task_whitelist"] is not None
            and task not in metric_info["task_whitelist"]
        ) or (
            metric_info["task_blacklist"] is not None
            and task in metric_info["task_blacklist"]
        ):
            continue
        results[metric_name] = metric_info["cls"].validate_episode(
            episode_metrics=episode_metrics,
            **metric_info["validate_kwargs"],
        )
        if (
            not all(v["success"] for v in results[metric_name].values())
            and metric_info["mode"] == MetricMode.SOFT
        ):
            results[metric_name]["warning"] = metric_info["warning"]

    success = all(
        v.get("success", True) for res in results.values() for v in res.values()
    )
    return success, results


class MotionMetric(MetricBase):
    """Metric for validating robot motion quality.

    Computes velocity, acceleration, and jerk statistics for robot arm joints
    throughout an episode. Helps identify jerky or unsafe movements that may
    indicate teleoperation issues or trajectory problems.

    Attributes:
        step_dt (float): Time between simulation steps in seconds.

    Example:
        >>> metric = MotionMetric(step_dt=0.1)
        >>> # During episode, automatically tracks joint positions
        >>> # At episode end, computes motion statistics
    """

    def __init__(self, step_dt):
        """
        Args:
            step_dt (float): Amount of time between steps, used to differentiate from pos -> vel -> acc -> jerk
        """
        self.step_dt = step_dt

        super().__init__()

    def _compute_step_metrics(
        self, env, action, obs, reward, terminated, truncated, info
    ):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            step_metrics[f"robot{i}::pos"] = robot.get_joint_positions()
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        episode_metrics = dict()
        for robot, (pos_key, positions) in zip(env.robots, episode_info.items()):
            arm_idxs = th.cat(
                [arm_control_idx for arm_control_idx in robot.arm_control_idx.values()]
            )
            arm_vel_limits = th.tensor(
                [jnt.max_velocity for jnt in robot.joints.values()]
            )[arm_idxs]
            positions = th.stack(positions, dim=0)[:, arm_idxs]
            vels = (positions[1:] - positions[:-1]) / self.step_dt
            n_vels = len(vels)
            accs = (vels[1:] - vels[:-1]) / self.step_dt
            jerks = (accs[1:] - accs[:-1]) / self.step_dt
            vels, accs, jerks = th.abs(vels), th.abs(accs), th.abs(jerks)
            episode_metrics[f"{pos_key}::vel_avg"] = vels.mean(dim=0)
            episode_metrics[f"{pos_key}::vel_prop_over_05max"] = (
                th.any(vels > arm_vel_limits.unsqueeze(0) * 0.5, dim=-1).sum() / n_vels
            )
            episode_metrics[f"{pos_key}::vel_prop_over_06max"] = (
                th.any(vels > arm_vel_limits.unsqueeze(0) * 0.6, dim=-1).sum() / n_vels
            )
            episode_metrics[f"{pos_key}::vel_prop_over_07max"] = (
                th.any(vels > arm_vel_limits.unsqueeze(0) * 0.7, dim=-1).sum() / n_vels
            )
            episode_metrics[f"{pos_key}::vel_prop_over_08max"] = (
                th.any(vels > arm_vel_limits.unsqueeze(0) * 0.8, dim=-1).sum() / n_vels
            )
            episode_metrics[f"{pos_key}::vel_prop_over_09max"] = (
                th.any(vels > arm_vel_limits.unsqueeze(0) * 0.9, dim=-1).sum() / n_vels
            )
            episode_metrics[f"{pos_key}::vel_prop_over_10max"] = (
                th.any(vels > arm_vel_limits.unsqueeze(0) * 1.0, dim=-1).sum() / n_vels
            )
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
        results = dict()
        for val_max_limit, val_name in zip(
            (
                vel_avg_limit,
                vel_max_limit,
                vel_prop_over_05max,
                vel_prop_over_06max,
                vel_prop_over_07max,
                vel_prop_over_08max,
                vel_prop_over_09max,
                vel_prop_over_10max,
                acc_avg_limit,
                acc_max_limit,
                jerk_avg_limit,
                jerk_max_limit,
            ),
            (
                "vel_avg",
                "vel_max",
                "vel_prop_over_05max",
                "vel_prop_over_06max",
                "vel_prop_over_07max",
                "vel_prop_over_08max",
                "vel_prop_over_09max",
                "vel_prop_over_10max",
                "acc_avg",
                "acc_max",
                "jerk_avg",
                "jerk_max",
            ),
        ):
            if val_max_limit is not None:
                for name, metric in episode_metrics.items():
                    if f"::{val_name}" in name:
                        test_name = name
                        success = metric <= val_max_limit
                        if isinstance(success, th.Tensor):
                            success = th.all(success).item()
                        feedback = (
                            None
                            if success
                            else f"Robot's {val_name} is too high ({metric}), must be <= {val_max_limit}"
                        )
                        results[test_name] = {"success": success, "feedback": feedback}

        return results


class CollisionMetric(MetricBase):
    """Metric for detecting robot collisions during episodes.

    Tracks collisions between different robot components (self-collisions,
    collisions with environment objects) and can visually highlight colliding
    links by changing their color. Supports adding custom collision checks.

    Attributes:
        default_color (tuple): Default RGB color for robot links (default: light blue).

    Example:
        >>> metric = CollisionMetric()
        >>> metric.add_check("robot_self", check_robot_self_collision, color_robots=red_color)
    """

    def __init__(self, default_color=(0.8235, 0.8235, 1.0000)):
        self.checks = dict()
        self.check_colors = dict()
        self.color_is_active = dict()
        self.default_color = th.tensor(default_color)
        self.active_color = self.default_color

        super().__init__()

    def add_check(self, name, check, color_robots=None):
        self.checks[name] = check
        self.check_colors[name] = color_robots

    def remove_check(self, name):
        self.checks.pop(name)
        self.check_colors.pop(name)

    def _compute_step_metrics(
        self, env, action, obs, reward, terminated, truncated, info
    ):
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
            for robot in env.robots:
                for link in robot.links.values():
                    for vm in link.visual_meshes.values():
                        if vm.material is not None:
                            vm.material.diffuse_color_constant = active_color
            self.active_color = active_color

        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        episode_metrics = dict()
        for name, collisions in episode_info.items():
            collisions = th.tensor(collisions)
            episode_metrics[f"{name}::n_collision"] = collisions.sum().item()

        return episode_metrics

    def reset(self, env):
        super().reset(env=env)

        if th.any(self.active_color != self.default_color).item():
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
        results = dict()
        if collision_limits is not None:
            for name, collision_limit in collision_limits.items():
                test_name = name
                if collision_limit is not None:
                    n_collisions = episode_metrics[f"{name}::n_collision"]
                    success = n_collisions <= collision_limit
                    feedback = (
                        None
                        if success
                        else f"Too many collisions ({n_collisions}) when checking {name}, must be <= {collision_limit}"
                    )
                    results[test_name] = {"success": success, "feedback": feedback}

        return results


class TaskSuccessMetric(MetricBase):
    """Metric for tracking task completion status.

    Simple metric that records whether the episode terminated with success
    (i.e., task was completed before timeout or failure conditions).
    """

    def _compute_step_metrics(
        self, env, action, obs, reward, terminated, truncated, info
    ):
        return {"done": terminated and not truncated}

    def _compute_episode_metrics(self, env, episode_info):
        return {"success": th.any(th.tensor(episode_info["done"])).item()}

    @classmethod
    def validate_episode(cls, episode_metrics):
        success = episode_metrics["success"]
        feedback = None if success else "Task was a not a success!"
        return {"task_success": {"success": success, "feedback": feedback}}


class GhostHandAppearanceMetric(MetricBase):
    """Metric for detecting ghost hand appearances during teleoperation.

    Detects when the operator's ghost hand (visualized in VR) diverges
    significantly from the actual robot gripper position. This can indicate
    tracking issues or operator fatigue. Optionally colors robot arms to
    highlight when ghost hand is active.

    Attributes:
        color_arms (bool): Whether to color robot arms when ghost hand is active
            (default: True).
    """

    def __init__(self, color_arms=True):
        self.color_arms = color_arms
        self.robot_arm_colors = dict()

        super().__init__()

    @classmethod
    def is_compatible(cls, env):
        valid = super().is_compatible(env=env)
        if valid:
            for robot in env.robots:
                for arm in robot.arm_names:
                    gripper_controller = robot.controllers[f"gripper_{arm}"]
                    is_1d = gripper_controller.command_dim == 1
                    is_normalized = (
                        th.all(
                            cb.to_torch(gripper_controller.command_input_limits[0])
                            == -1.0
                        ).item()
                        and th.all(
                            cb.to_torch(gripper_controller.command_input_limits[1])
                            == 1.0
                        ).item()
                    )
                    valid = is_1d and is_normalized
                    if not valid:
                        break
                if not valid:
                    break
        return valid

    def _compute_step_metrics(
        self, env, action, obs, reward, terminated, truncated, info
    ):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            robot_qpos = robot.get_joint_positions(normalized=False)
            gripper_action_idxs = robot.gripper_action_idx
            for arm in robot.arm_names:
                active = (
                    th.max(
                        th.abs(
                            robot_qpos[robot.arm_control_idx[arm]]
                            - action[robot.arm_action_idx[arm]]
                        )
                    ).item()
                    > GHOST_APPEAR_THRESHOLD
                )
                gripper_controller = robot.controllers[f"gripper_{arm}"]
                step_metrics[f"robot{i}::arm_{arm}::active"] = active
                if self.color_arms:
                    if robot.name not in self.robot_arm_colors:
                        self.robot_arm_colors[robot.name] = {
                            a: False for a in robot.arm_names
                        }
                    robot_arm_color_is_active = self.robot_arm_colors[robot.name][arm]
                    if active != robot_arm_color_is_active:
                        color = (
                            th.tensor([1.0, 0, 0])
                            if active
                            else th.tensor([0.8235, 0.8235, 1.0000])
                        )
                        for link in robot.arm_links[arm]:
                            for vm in link.visual_meshes.values():
                                vm.material.diffuse_color_constant = color
                        self.robot_arm_colors[robot.name][arm] = active
                op = operator.lt if gripper_controller._inverted else operator.ge
                step_metrics[f"robot{i}::arm_{arm}::open_cmd"] = th.all(
                    op(action[gripper_action_idxs[arm]], 0.0)
                ).item()
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        episode_metrics = dict()
        for i, robot in enumerate(env.robots):
            for arm in robot.arm_names:
                pf = f"robot{i}::arm_{arm}"
                active = th.tensor(episode_info[f"{pf}::active"])
                open_cmd = th.tensor(episode_info[f"{pf}::open_cmd"])
                ungrasping = open_cmd[1:] & ~open_cmd[:-1]
                episode_metrics[f"{pf}::n_steps_total"] = active.sum().item()
                episode_metrics[f"{pf}::n_steps_while_ungrasping"] = (
                    (active[1:] & ungrasping).sum().item()
                )
        return episode_metrics

    def reset(self, env):
        super().reset(env=env)

        for i, robot in enumerate(env.robots):
            if self.color_arms and robot.name in self.robot_arm_colors:
                for arm in robot.arm_names:
                    if self.robot_arm_colors[robot.name][arm]:
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
        results = dict()
        if (
            gh_appearance_limit is not None
            or gh_appearance_limit_while_ungrasping is not None
        ):
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
                    success = (
                        episode_metrics[name] <= gh_appearance_limit_while_ungrasping
                    )
                    limit = gh_appearance_limit_while_ungrasping
                else:
                    raise ValueError(f"Got invalid metric name: {name}")
                feedback = (
                    None
                    if success
                    else f"Too many ghost hand appearances ({episode_metrics[name]}) when checking {name}, must be <= {limit}"
                )
                results[test_name] = {"success": success, "feedback": feedback}

        return results


class ProlongedPauseMetric(MotionMetric):
    """Metric for detecting extended periods of robot immobility.

    Inherits from MotionMetric but focuses on identifying long pauses during
    the episode that may indicate teleoperation issues or task difficulties.
    A pause is defined as consecutive steps where robot velocity stays below
    a specified threshold.

    Attributes:
        step_dt (float): Time between simulation steps in seconds.
        vel_threshold (float): Velocity threshold below which robot is considered
            paused (default: 0.001).
    """

    def __init__(self, step_dt, vel_threshold=0.001):
        self.vel_threshold = vel_threshold
        super().__init__(step_dt=step_dt)

    def _compute_episode_metrics(self, env, episode_info):
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
        results = dict()
        if pause_steps_limit is not None:
            for name, metric in episode_metrics.items():
                test_name = name
                success = episode_metrics[name] <= pause_steps_limit
                feedback = (
                    None
                    if success
                    else f"Too many consecutive steps ({episode_metrics[name]}) without robot motion, must be <= {pause_steps_limit}"
                )
                results[test_name] = {"success": success, "feedback": feedback}

        return results


class FailedGraspMetric(MetricBase):
    """Metric for detecting failed grasp attempts.

    Tracks when fingers close completely without successful object grasping.
    A failed grasp is identified when gripper transitions from open to closed
    state without acquiring an object.
    """

    def _compute_step_metrics(
        self, env, action, obs, reward, terminated, truncated, info
    ):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            for arm in robot.arm_names:
                step_metrics[f"robot{i}::arm_{arm}::fingers_closed"] = th.allclose(
                    robot.get_joint_positions()[robot.gripper_control_idx[arm]],
                    th.zeros(2),
                    atol=1e-4,
                )
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        episode_metrics = dict()
        for i, robot in enumerate(env.robots):
            for arm in robot.arm_names:
                pf = f"robot{i}::arm_{arm}"
                fingers_closed = th.tensor(episode_info[f"{pf}::fingers_closed"])
                fingers_closed_transition = fingers_closed[1:] & ~fingers_closed[:-1]
                episode_metrics[f"{pf}::failed_grasp_count"] = (
                    fingers_closed_transition.sum().item()
                )

        return episode_metrics

    @classmethod
    def validate_episode(
        cls,
        episode_metrics,
        failed_grasp_limit=None,
    ):
        results = dict()
        if failed_grasp_limit is not None:
            for name, metric in episode_metrics.items():
                test_name = name
                success = episode_metrics[name] <= failed_grasp_limit
                feedback = (
                    None
                    if success
                    else f"Too many ({episode_metrics[name]}) failed grasps, must be <= {failed_grasp_limit}"
                )
                results[test_name] = {"success": success, "feedback": feedback}

        return results


class TaskRelevantObjectVelocityMetric(MetricBase):
    """Metric for tracking velocity of task-relevant objects.

    Computes velocity statistics (mean, std, max) for all non-fixed, non-system
    objects in the scene. Helps identify if objects are moving too fast (which
    may indicate improper handling) or not moving at all.

    Only compatible with BehaviorTask environments.

    Attributes:
        step_dt (float): Time between simulation steps in seconds.
    """

    def __init__(self, step_dt):
        self.step_dt = step_dt

        super().__init__()

    @classmethod
    def is_compatible(cls, env):
        return isinstance(env.task, BehaviorTask)

    def _compute_step_metrics(
        self, env, action, obs, reward, terminated, truncated, info
    ):
        step_metrics = dict()
        for name, bddl_inst in env.task.object_scope.items():
            if (
                bddl_inst.is_system
                or not bddl_inst.exists
                or bddl_inst.fixed_base
                or "agent" in name
            ):
                continue
            step_metrics[f"{name}::pos"] = bddl_inst.get_position_orientation()[0]
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
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
        results = dict()
        if vel_max_limit is not None:
            for name, metric in episode_metrics.items():
                if f"::vel_max" in name:
                    test_name = name
                    success = metric <= vel_max_limit
                    feedback = (
                        None
                        if success
                        else f"{name} is too high({metric}), must be <= {vel_max_limit}"
                    )
                    results[test_name] = {"success": success, "feedback": feedback}

        return results


class FieldOfViewMetric(MetricBase):
    """Metric for tracking gripper visibility in robot's main camera.

    Tracks whether the gripper is visible in the robot's head camera's field
    of view. Generates warnings when grasp state changes occur while the
    gripper is outside the FOV (which may indicate poor camera positioning
    or unreachable grasps).

    Only compatible with environments that have instance segmentation enabled
    on the head camera.

    Attributes:
        head_camera: The head camera sensor providing observation data.
        gripper_link_paths (dict): Mapping from arm name to set of gripper
            link prim paths for FOV detection.
    """

    @classmethod
    def is_compatible(cls, env):
        valid = super().is_compatible(env=env)
        if valid:
            for robot in env.robots:
                for arm in robot.arm_names:
                    gripper_controller = robot.controllers[f"gripper_{arm}"]
                    is_1d = gripper_controller.command_dim == 1
                    is_normalized = (
                        th.all(
                            cb.to_torch(gripper_controller.command_input_limits[0])
                            == -1.0
                        ).item()
                        and th.all(
                            cb.to_torch(gripper_controller.command_input_limits[1])
                            == 1.0
                        ).item()
                    )
                    valid = is_1d and is_normalized
                    if not valid:
                        break
                if not valid:
                    break
        return valid

    def __init__(self, head_camera, gripper_link_paths):
        self.head_camera = head_camera
        self.gripper_link_paths = gripper_link_paths

        assert "seg_instance_id" in self.head_camera.modalities, (
            "FieldOfViewMetric requires instance_id_segmentation modality"
        )

        super().__init__()

    def _compute_step_metrics(
        self, env, action, obs, reward, terminated, truncated, info
    ):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            _, info = self.head_camera.get_obs()
            links_in_fov = set(info["seg_instance_id"].values())
            gripper_action_idxs = robot.gripper_action_idx
            for arm in robot.arm_names:
                gripper_controller = robot.controllers[f"gripper_{arm}"]
                op = operator.lt if gripper_controller._inverted else operator.ge
                gripper_in_fov = (
                    len(links_in_fov.intersection(self.gripper_link_paths[arm])) > 0
                )

                step_metrics[f"robot{i}::arm_{arm}::open_cmd"] = th.all(
                    op(action[gripper_action_idxs[arm]], 0.0)
                ).item()
                step_metrics[f"robot{i}::arm_{arm}::gripper_in_fov"] = gripper_in_fov
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        episode_metrics = dict()

        for i, robot in enumerate(env.robots):
            for arm in robot.arm_names:
                pf = f"robot{i}::arm_{arm}"
                open_cmd = th.tensor(episode_info[f"{pf}::open_cmd"])
                gripper_in_fov = th.tensor(episode_info[f"{pf}::gripper_in_fov"])

                grasping_changes = open_cmd[1:] != open_cmd[:-1]

                episode_metrics[f"robot{i}::arm_{arm}::gripper_outside_fov"] = (
                    (gripper_in_fov == 0).sum().item()
                )

                episode_metrics[f"robot{i}::arm_{arm}::grasp_changes_outside_fov"] = (
                    (grasping_changes & ~gripper_in_fov[1:]).sum().item()
                )
        return episode_metrics

    @classmethod
    def validate_episode(
        cls,
        episode_metrics,
        gripper_changes_outside_fov_limit=None,
    ):
        results = dict()
        if gripper_changes_outside_fov_limit is not None:
            for name, metric in episode_metrics.items():
                if f"::grasp_changes_outside_fov" in name:
                    test_name = name
                    success = metric <= gripper_changes_outside_fov_limit
                    feedback = (
                        None
                        if success
                        else f"{name} is too high ({metric}) (too many times the gripper was toggled outside of the robot's main FOV), must be <= {gripper_changes_outside_fov_limit}"
                    )
                    results[test_name] = {"success": success, "feedback": feedback}

        return results


class HeadCameraUprightMetric(MetricBase):
    """Metric for detecting head camera tilt during navigation.

    Tracks whether the head camera is tilted excessively while the robot is
    navigating (translating or rotating). Prolonged tilted navigation may
    indicate improper head positioning or comfort issues for the operator.

    Attributes:
        head_camera_link_name (str): Name of the link containing the head camera.
        step_dt (float): Time between simulation steps in seconds.
        navigation_window (float): Time window in seconds to consider as "prolonged"
            navigation (default: 3.0).
        translation_threshold (float): Velocity threshold for detecting translation
            in m/s (default: 0.1).
        rotation_threshold (float): Angular velocity threshold for detecting rotation
            in rad/s (default: 0.05).
        camera_tilt_threshold (float): Euler Y-angle threshold in radians for
            considering camera tilted (default: 0.4).
    """

    @classmethod
    def is_compatible(cls, env):
        return super().is_compatible(env)

    def __init__(
        self,
        head_camera_link_name,
        step_dt,
        navigation_window=3.0,
        translation_threshold=0.1,
        rotation_threshold=0.05,
        camera_tilt_threshold=0.4,
    ):
        self.head_camera_link_name = head_camera_link_name
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.camera_tilt_threshold = camera_tilt_threshold

        self.navigation_window_in_steps = int(navigation_window / step_dt)
        self.step_dt = step_dt

        super().__init__()

    def _compute_step_metrics(
        self, env, action, obs, reward, terminated, truncated, info
    ):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            _, ori = robot.links[self.head_camera_link_name].get_position_orientation()
            step_metrics[f"robot{i}::head_link_y_ori"] = T.quat2euler(ori)[1]
            base_pos, base_ori = robot.get_position_orientation()
            step_metrics[f"robot{i}::base_pos_x"] = base_pos[0]
            step_metrics[f"robot{i}::base_pos_y"] = base_pos[1]
            step_metrics[f"robot{i}::base_ori_yaw"] = T.quat2euler(base_ori)[2]

        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        episode_metrics = dict()

        for i, robot in enumerate(env.robots):
            head_y_ori = th.tensor(episode_info[f"robot{i}::head_link_y_ori"])
            base_pos_x = th.tensor(episode_info[f"robot{i}::base_pos_x"])
            base_pos_y = th.tensor(episode_info[f"robot{i}::base_pos_y"])
            base_ori_yaw = th.tensor(episode_info[f"robot{i}::base_ori_yaw"])

            base_pos_diff_x = base_pos_x[1:] - base_pos_x[:-1]
            base_pos_diff_y = base_pos_y[1:] - base_pos_y[:-1]
            base_ori_diff_yaw = base_ori_yaw[1:] - base_ori_yaw[:-1]

            translation_velocity = (
                th.sqrt(base_pos_diff_x**2 + base_pos_diff_y**2) / self.step_dt
            )
            is_translating = translation_velocity > self.translation_threshold
            is_rotating = (
                th.abs(base_ori_diff_yaw / self.step_dt) > self.rotation_threshold
            )
            is_navigating = is_translating | is_rotating

            is_navigating = th.cat([th.tensor([False]), is_navigating])
            is_tilted = th.abs(head_y_ori) > self.camera_tilt_threshold
            prolonged_navigation_mask = th.zeros_like(is_navigating, dtype=th.bool)

            consecutive_count = 0
            for j in range(len(is_navigating)):
                if is_navigating[j]:
                    consecutive_count += 1
                else:
                    consecutive_count = 0

                if consecutive_count >= self.navigation_window_in_steps:
                    prolonged_navigation_mask[j] = True

            episode_metrics[f"robot{i}::head_camera_tilted_during_navigation"] = (
                (is_tilted & prolonged_navigation_mask).sum().item()
            )

        return episode_metrics

    @classmethod
    def validate_episode(
        cls,
        episode_metrics,
        head_camera_tilt_during_navigation_limit=None,
    ):
        results = dict()
        if head_camera_tilt_during_navigation_limit is not None:
            for name, metric in episode_metrics.items():
                if f"::head_camera_tilted_during_navigation" in name:
                    test_name = name
                    success = metric <= head_camera_tilt_during_navigation_limit
                    feedback = (
                        None
                        if success
                        else f"{name} is too high ({metric}) (too many steps where the robot head is tilted during navigation), must be <= {head_camera_tilt_during_navigation_limit}"
                    )
                    results[test_name] = {"success": success, "feedback": feedback}

        return results


def check_robot_self_collision(env):
    """Check if any part of the robot is in collision with itself.

    Args:
        env: The OmniGibson environment instance.

    Returns:
        bool: True if robot self-collision detected, False otherwise.
    """
    for robot in env.robots:
        link_paths = list(robot.link_prim_paths)
        if RigidContactAPI.is_in_contact(
            env.scene.idx,
            link_paths,
            with_set=link_paths,
            ignore_set=None,
            current_only=False,
        ):
            return True
    return False


def check_robot_base_nonarm_nonkinematic_collision(env):
    """Check if robot base (excluding arms) collides with non-kinematic objects.

    Args:
        env: The OmniGibson environment instance.

    Returns:
        bool: True if collision with non-arm, non-kinematic objects detected.
    """
    for robot in env.robots:
        robot_link_paths = set(robot.link_prim_paths)
        for arm in robot.arm_names:
            robot_link_paths -= set(link.prim_path for link in robot.arm_links[arm])
            robot_link_paths -= set(link.prim_path for link in robot.gripper_links[arm])
            robot_link_paths -= set(link.prim_path for link in robot.finger_links[arm])
        if RigidContactAPI.is_in_contact(
            env.scene.idx,
            robot_link_paths,
            with_set=None,
            ignore_set={robot},
            current_only=False,
        ):
            return True

    return False


def check_robot_nonarm_nonground_collision(env):
    """Check if robot non-arm links collide with non-ground objects.

    This check identifies collisions between robot base/torso links and
    objects in the scene (excluding the ground plane and the robot itself).

    Args:
        env: The OmniGibson environment instance.

    Returns:
        bool: True if collision with non-ground, non-robot objects detected.
    """
    ground_objects = []
    for cat in GROUND_CATEGORIES:
        ground_objects.extend(env.scene.object_registry("category", cat, []))

    for robot in env.robots:
        robot_arm_paths = set()
        for arm in robot.arm_names:
            robot_arm_paths = robot_arm_paths.union(
                set(link.prim_path for link in robot.arm_links[arm])
            )
            robot_arm_paths = robot_arm_paths.union(
                set(link.prim_path for link in robot.gripper_links[arm])
            )
            robot_arm_paths = robot_arm_paths.union(
                set(link.prim_path for link in robot.finger_links[arm])
            )
        non_arm_links = set(
            link
            for link in robot.links.values()
            if link.prim_path not in robot_arm_paths
        )

        if RigidContactAPI.is_in_contact(
            env.scene.idx,
            non_arm_links,
            with_set=None,
            ignore_set=ground_objects + [robot],
            current_only=False,
        ):
            return True

    return False


def create_collision_metric(
    include_robot_self_collision=True,
    include_robot_nonarm_nonkinematic_collision=True,
    include_robot_nonarm_nonground_collision=True,
):
    """Create a CollisionMetric with specified collision checks enabled.

    Factory function for creating a CollisionMetric configured with specific
    collision detection checks. Each check can be independently enabled or
    disabled based on the data collection requirements.

    Args:
        include_robot_self_collision (bool): Enable robot self-collision detection
            (default: True).
        include_robot_nonarm_nonkinematic_collision (bool): Enable collision detection
            between robot non-arm parts and non-kinematic objects (default: True).
        include_robot_nonarm_nonground_collision (bool): Enable collision detection
            between robot non-arm parts and non-ground objects (default: True).

    Returns:
        CollisionMetric: Configured collision metric with the specified checks.

    Example:
        >>> metric = create_collision_metric(
        ...     include_robot_self_collision=True,
        ...     include_robot_nonarm_nonground_collision=True
        ... )
    """
    col_metric = CollisionMetric()
    if include_robot_self_collision:
        col_metric.add_check(
            name="robot_self",
            check=check_robot_self_collision,
            color_robots=th.tensor([1.0, 0, 0]),
        )
    if include_robot_nonarm_nonkinematic_collision:
        col_metric.add_check(
            name="robot_nonarm_nonstructure",
            check=check_robot_base_nonarm_nonkinematic_collision,
        )
    if include_robot_nonarm_nonground_collision:
        col_metric.add_check(
            name="robot_nonarm_nonground", check=check_robot_nonarm_nonground_collision
        )
    return col_metric


ALL_QA_METRICS = {
    """Dictionary of all available QA metrics with their configuration.

    Each key corresponds to a metric name, and the value is a dictionary with:
        - cls: The metric class.
        - init: Optional factory function for metric initialization.
        - mode: MetricMode enum (DISABLED, SOFT, or HARD) for validation enforcement.
        - warning: Optional warning message for SOFT mode failures.
        - task_whitelist: List of tasks where this metric applies (None = all).
        - task_blacklist: List of tasks where this metric is excluded (None = none).
        - validate_kwargs: Default validation thresholds for this metric.

    The available metrics are:
        - motion: Joint velocity, acceleration, and jerk limits.
        - collision: Robot self and environment collision detection.
        - task_success: Whether the episode completed successfully.
        - ghost_hand_appearance: VR ghost hand tracking issues.
        - prolonged_pause: Extended periods without robot motion.
        - failed_grasp: Gripper closes without successful grasp.
        - task_relevant_obj_vel: Object velocity limits during episodes.
        - gripper_in_fov: Gripper visibility in camera field of view.
        - head_camera_upright_during_navigation: Camera tilt during navigation.

    Example:
        >>> motion_config = ALL_QA_METRICS["motion"]
        >>> print(motion_config["mode"])  # MetricMode.HARD
    """
    "motion": {
        "cls": MotionMetric,
        "init": None,
        "mode": MetricMode.HARD,
        "warning": None,
        "task_whitelist": None,
        "task_blacklist": None,
        "validate_kwargs": dict(
            vel_avg_limit=0.15,
            vel_max_limit=None,
            vel_prop_over_05max=None,
            vel_prop_over_06max=None,
            vel_prop_over_07max=None,
            vel_prop_over_08max=None,
            vel_prop_over_09max=0.005,
            vel_prop_over_10max=None,
            acc_avg_limit=3.0,
            acc_max_limit=None,
            jerk_avg_limit=100.0,
            jerk_max_limit=None,
        ),
    },
    "collision": {
        "cls": CollisionMetric,
        "init": lambda: create_collision_metric(
            include_robot_self_collision=True,
            include_robot_nonarm_nonkinematic_collision=False,
            include_robot_nonarm_nonground_collision=True,
        ),
        "mode": MetricMode.HARD,
        "warning": None,
        "task_whitelist": None,
        "task_blacklist": None,
        "validate_kwargs": dict(
            collision_limits=dict(
                robot_self=0,
                robot_nonarm_nonstructure=None,
                robot_nonarm_nonground=0,
            ),
        ),
    },
    "task_success": {
        "cls": TaskSuccessMetric,
        "init": None,
        "mode": MetricMode.HARD,
        "warning": None,
        "task_whitelist": None,
        "task_blacklist": None,
        "validate_kwargs": dict(),
    },
    "ghost_hand_appearance": {
        "cls": GhostHandAppearanceMetric,
        "init": None,
        "mode": MetricMode.HARD,
        "warning": None,
        "task_whitelist": None,
        "task_blacklist": None,
        "validate_kwargs": dict(
            gh_appearance_limit=None,
            gh_appearance_limit_while_ungrasping=0,
        ),
    },
    "prolonged_pause": {
        "cls": ProlongedPauseMetric,
        "init": None,
        "mode": MetricMode.HARD,
        "warning": None,
        "task_whitelist": None,
        "task_blacklist": None,
        "validate_kwargs": dict(
            pause_steps_limit=50,
        ),
    },
    "failed_grasp": {
        "cls": FailedGraspMetric,
        "init": None,
        "mode": MetricMode.SOFT,
        "warning": None,
        "task_whitelist": None,
        "task_blacklist": None,
        "validate_kwargs": dict(
            failed_grasp_limit=0,
        ),
    },
    "task_relevant_obj_vel": {
        "cls": TaskRelevantObjectVelocityMetric,
        "init": None,
        "mode": MetricMode.DISABLED,
        "warning": None,
        "task_whitelist": None,
        "task_blacklist": None,
        "validate_kwargs": dict(
            vel_max_limit=4.0,
        ),
    },
    "gripper_in_fov": {
        "cls": FieldOfViewMetric,
        "init": None,
        "mode": MetricMode.SOFT,
        "warning": "Some grasps occurred out of view. Please make sure these grasps are necessary (e.g.: grasping deep inside a cabinet / fridges for an object)",
        "task_whitelist": None,
        "task_blacklist": None,
        "validate_kwargs": dict(
            gripper_changes_outside_fov_limit=0,
        ),
    },
    "head_camera_upright_during_navigation": {
        "cls": HeadCameraUprightMetric,
        "init": None,
        "mode": MetricMode.SOFT,
        "warning": "Head seems to be tilted while navigating. Please make sure the head camera is faced upright and forward",
        "task_whitelist": None,
        "task_blacklist": ["putting_away_Halloween_decorations"],
        "validate_kwargs": dict(
            head_camera_tilt_during_navigation_limit=30,
        ),
    },
}
