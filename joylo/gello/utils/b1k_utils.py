from gello.utils.qa_utils import *
from enum import IntEnum


class MetricMode(IntEnum):
    DISABLED = 0
    SOFT = 1
    HARD = 2


# Maps QA metric class to dict with metric name, initializer, and kwargs to pass into the metric's respective
# @validate_episode calls
ALL_QA_METRICS = {
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

# Set of common metric names to use
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

# Task specific qa metrics
TASK_QA_METRICS = {

}


def aggregate_episode_validation(task, all_episode_metrics):
    """
    Validates the given @all_episode_metrics

    Args:
        task (str): The name of the task whose QA metrics are being aggregated
        all_episode_metrics (dict): Keyword-mapped aggregated episode metrics

    Returns:
        2-tuple:
            - bool: Whether the validation succeeded or not (requires all metric validation checks to pass)
            - dict: Per-metric information
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
        # If disabled, skip
        if metric_info["mode"] == MetricMode.DISABLED:
            continue
        # If the task is not in the whitelist or is in the blacklist, skip
        if ((metric_info["task_whitelist"] is not None and task not in metric_info["task_whitelist"])
                or (metric_info["task_blacklist"] is not None and task in metric_info["task_blacklist"])):
            continue
        results[metric_name] = metric_info["cls"].validate_episode(
            episode_metrics=episode_metrics,
            **metric_info["validate_kwargs"],
        )
        if not all(v["success"] for v in results[metric_name].values()) and metric_info["mode"] == MetricMode.SOFT:
            # Add warning feedback for manual QA
            results[metric_name]["warning"] = metric_info["warning"]

    # Passes if all metric validations pass
    success = all(v.get("success", True) for res in results.values() for v in res.values())
    return success, results
