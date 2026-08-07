import cv2
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from signal import SIGINT, signal
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

import omnigibson as og
import omnigibson.utils.transform_utils as T
from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.utils.og_teleop_utils import (
    augment_rooms,
    get_task_relevant_room_types,
    load_available_tasks,
)
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    EVAL_TIMEOUT_MULTIPLIER,
    NUM_HIDDEN_TEST_INSTANCES,
    NUM_PUBLIC_TEST_INSTANCES,
    TASK_NAMES_TO_INDICES,
    TEST_INSTANCE_IDS,
    flatten_obs_dict,
    generate_basic_environment_config,
    get_robot_camera_names,
)
from omnigibson.eval.utils.light_utils import LightToggleSynchronizer, set_light_control_toggles
from omnigibson.eval.utils.obs_utils import create_video_writer, write_video
from omnigibson.eval.utils.score_utils import load_human_stats
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import Robot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import create_module_logger


TORCH_NUM_THREADS = None
TORCH_NUM_INTEROP_THREADS = None

if TORCH_NUM_THREADS is not None:
    th.set_num_threads(TORCH_NUM_THREADS)
if TORCH_NUM_INTEROP_THREADS is not None:
    th.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)

LIGHT_EVAL_TASKS = {"turning_out_all_lights_before_sleep"}
EVAL_BASE_LINK_MASS = 250.0
EVAL_HEAD_HORIZONTAL_APERTURE = 40.0
DEFAULT_ROBOT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "r1pro.yaml")
EVAL_MODES = ("train", "public_test", "hidden_test")

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def _to_plain_dict(cfg: Any) -> dict | None:
    if cfg is None:
        return None
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def resolve_instance_ids(task_name: str, instance_indices: list[int], mode: str = "public_test") -> list[int]:
    assert mode in EVAL_MODES, f"Mode must be one of {EVAL_MODES}, got {mode}"
    if mode == "train":
        return [int(instance_id) for instance_id in instance_indices]

    test_instances = (
        TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES]
        if mode == "public_test"
        else TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:]
    )
    num_split_instances = NUM_PUBLIC_TEST_INSTANCES if mode == "public_test" else NUM_HIDDEN_TEST_INSTANCES
    assert set(instance_indices).issubset(
        set(range(num_split_instances))
    ), f"Instance indices must be in range({num_split_instances}) for mode {mode}"
    return [int(test_instances[i]) for i in instance_indices]


def evaluate_instances_batched(
    instances: Sequence,
    num_envs: int,
    load_fn: Callable[[Dict[int, object]], None],
    step_fn: Callable[[List[int]], "tuple[Sequence[bool], Sequence[bool]]"],
    record_fn: Callable[..., object],
    max_steps: Optional[int] = None,
) -> "Dict[object, object]":
    """
    Drive one parallel evaluation batch with exactly one instance per logical environment. Pure
    orchestration: all sim work is delegated to the injected load_fn / step_fn / record_fn. When an
    instance finishes, its environment is removed from the active list and remains frozen until the
    complete batch finishes. Returns {instance id -> record_fn's return}.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    instances = list(instances)
    if len(instances) != num_envs:
        raise ValueError(
            f"Evaluation requires exactly one instance per logical environment: "
            f"got {len(instances)} instances and num_envs={num_envs}."
        )

    results: Dict[object, object] = {}
    env_idx_to_instance: Dict[int, object] = {env_idx: instance for env_idx, instance in enumerate(instances)}
    load_fn(dict(env_idx_to_instance))

    active = {env_idx: True for env_idx in env_idx_to_instance}
    step = 0
    while any(active.values()):
        active_env_indices = sorted(env_idx for env_idx, is_active in active.items() if is_active)
        terminated, truncated = step_fn(active_env_indices)
        step += 1

        hit_cap = max_steps is not None and step >= max_steps
        for env_idx in active_env_indices:
            term = bool(terminated[env_idx])
            trunc = bool(truncated[env_idx]) or hit_cap
            if term or trunc:
                results[env_idx_to_instance[env_idx]] = record_fn(
                    env_idx=env_idx,
                    instance=env_idx_to_instance[env_idx],
                    step=step,
                    terminated=term,
                    truncated=trunc,
                )
                active[env_idx] = False

    return results


@dataclass(frozen=True)
class InstanceEnvAccessor:
    """Read-only access to one logical environment inside a shared vectorized environment."""

    shared_env: EnvironmentWrapper
    env_idx: int

    @property
    def scene(self):
        return self.shared_env.scenes[self.env_idx]

    @property
    def robot(self) -> Robot:
        robots = self.scene.robots
        assert len(robots) == 1, f"Evaluation assumes one robot per environment, got {len(robots)}."
        return robots[0]

    @property
    def object_scope(self):
        return self.shared_env.task.object_scope[self.env_idx]

    @property
    def success(self) -> bool:
        return bool(self.shared_env.task.success[self.env_idx])

    def get_goal_option_satisfaction(self):
        return self.shared_env.task.get_goal_option_satisfaction(self.env_idx)


@dataclass
class InstanceEvaluationState:
    """Mutable rollout state for one task instance assigned to one logical environment."""

    env_accessor: InstanceEnvAccessor
    metrics: List[MetricBase] = field(default_factory=list)
    instance_id: Optional[int] = None
    obs: Optional[dict] = None
    video_writer: Any = None
    light_synchronizer: Optional[LightToggleSynchronizer] = None
    active: bool = False

    @property
    def env_idx(self) -> int:
        return self.env_accessor.env_idx


class BatchedEvaluator:
    """
    Batched evaluator engine for BEHAVIOR tasks. Owns one shared vectorized OmniGibson environment and
    one batched policy. Each batch entry is represented by an :class:`InstanceEvaluationState`,
    which binds one task instance to one logical environment through an
    :class:`InstanceEnvAccessor`. ``num_envs=1`` reproduces single-environment evaluation.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_name = None
        self.robot_eval_config = {}
        self.robot_camera_names = {}

        # Number of logical environments evaluated concurrently. Defaults to 1.
        self.num_envs = int(cfg.get("num_envs", 1))
        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        assert (
            self.env.num_envs == self.num_envs
        ), f"Env created with num_envs={self.env.num_envs} but BatchedEvaluator expected {self.num_envs}."

        self.instance_eval_states = self._create_instance_eval_states()
        # All environments share the same robot config, so resolve eval camera names once and
        # validate that the configured camera sensors exist on the robot.
        self.robot_camera_names = get_robot_camera_names(
            self.instance_eval_states[0].env_accessor.robot.name, self.robot_eval_config
        )
        self._validate_robot_eval_config()
        self._apply_robot_eval_settings()
        self.policy = self.load_policy()
        for instance_eval_state in self.instance_eval_states:
            instance_eval_state.metrics = self.load_metrics(instance_eval_state.env_accessor)

        # Initialize the physics views so the first reset()/load can read/restore robot joint state.
        og.sim.update_handles()
        self.env._current_episode = 0

    @property
    def should_sync_lights(self) -> bool:
        return self.env.task.activity_name in LIGHT_EVAL_TASKS

    def _reset_light_synchronizer(self, instance_eval_state: InstanceEvaluationState) -> None:
        if self.should_sync_lights:
            synchronizer = LightToggleSynchronizer(instance_eval_state.env_accessor.scene)
            synchronizer.reset_from_current_state()
            instance_eval_state.light_synchronizer = synchronizer
        else:
            instance_eval_state.light_synchronizer = None

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"

        self.human_stats = load_human_stats(task_name)

        task_cfg = available_tasks[task_name][0]
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        robot_cfg = self._build_robot_config(task_name=task_name, task_cfg=task_cfg)
        self.robot_name = robot_cfg["name"]
        cfg["robots"] = [robot_cfg]

        # Camera resolution + modalities are decided by the chosen eval wrapper and baked into the
        # robot config HERE, before env creation. This is required (not just cleaner) because in
        # multi-env mode robot cameras are batched into a single TiledVisionSensor whose resolution is
        # fixed at creation -- a post-hoc per-sensor resize in the wrapper would be silently ignored.
        # Resolve the wrapper class (without instantiating it -- that needs the env) to read its spec.
        camera_spec = get_class(env_wrapper["_target_"]).camera_spec()
        cfg["robots"][0]["obs_modalities"] = ["proprio", *camera_spec["modalities"]]
        # Per-camera resolution via per-link sensor_config overrides. The link key is "<link>:Camera:0"
        # (see robots/robot.py), which takes precedence over the class-level VisionSensor default. Map
        # each eval camera role to its robot sensor via the (config-driven) eval.camera_sensor_names.
        sensor_config = cfg["robots"][0].setdefault("sensor_config", {})
        for camera_id, sensor_name in self.robot_eval_config.get("camera_sensor_names", {}).items():
            if camera_id not in camera_spec["resolution"]:
                continue
            link_key = sensor_name.split(":", 1)[1]
            height, width = camera_spec["resolution"][camera_id]
            sensor_config[link_key] = {"sensor_kwargs": {"image_height": height, "image_width": width}}

        if self.cfg.max_steps is None:
            max_steps = int(self.human_stats["length"] * EVAL_TIMEOUT_MULTIPLIER)
            logger.info(
                f"Setting timeout to be {EVAL_TIMEOUT_MULTIPLIER}x the average length of human demos: {max_steps}"
            )
            cfg["task"]["termination_config"]["max_steps"] = max_steps
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False

        # Run num_envs instances of the same scene model + task in parallel environments.
        cfg.setdefault("env", {})
        cfg["env"]["num_envs"] = self.num_envs

        env = og.Environment(configs=cfg)
        env._eval_robot_config = self.robot_eval_config
        return instantiate(env_wrapper, env=env)

    def _create_instance_eval_states(self) -> List[InstanceEvaluationState]:
        return [
            InstanceEvaluationState(env_accessor=InstanceEnvAccessor(shared_env=self.env, env_idx=env_idx))
            for env_idx in range(self.num_envs)
        ]

    def _validate_robot_eval_config(self) -> None:
        missing_camera_sensors = []
        for camera_id, camera_name in self.robot_camera_names.items():
            sensor_name = camera_name.split("::", 1)[1]
            if sensor_name not in self.instance_eval_states[0].env_accessor.robot.sensors:
                missing_camera_sensors.append(f"{camera_id}: {sensor_name}")
        if missing_camera_sensors:
            raise ValueError(
                "Configured eval.camera_sensor_names entries were not found in robot.sensors: "
                f"{missing_camera_sensors}"
            )

        if self.cfg.get("write_video", False):
            required_camera_ids = {"left_wrist", "right_wrist", "head"}
            missing_camera_ids = sorted(required_camera_ids - set(self.robot_camera_names))
            if missing_camera_ids:
                raise ValueError(
                    "--write-video requires eval.camera_sensor_names roles "
                    f"{sorted(required_camera_ids)}; missing {missing_camera_ids}"
                )

    def _build_robot_config(self, task_name: str, task_cfg: dict) -> dict:
        robot_cfg = _to_plain_dict(OmegaConf.select(self.cfg, "robot"))
        if robot_cfg is None:
            robot_cfg = _to_plain_dict(OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH))
        else:
            robot_cfg = dict(robot_cfg)
        assert "model" in robot_cfg, "Robot config must include canonical 'model'"
        assert "type" not in robot_cfg, "Robot config must use canonical 'model', not 'type'"
        robot_cfg["model"] = robot_cfg["model"].lower()
        assert "name" in robot_cfg, "Robot config must include 'name'"
        self.robot_eval_config = _to_plain_dict(robot_cfg.pop("eval", None)) or {}
        robot_cfg["position"] = task_cfg["robot_start_position"]
        robot_cfg["orientation"] = task_cfg["robot_start_orientation"]

        return robot_cfg

    def _apply_robot_eval_settings(self) -> None:
        # Heavier base so contact with the (heavy) world does not shove the robot around during eval.
        # base mass writes require the sim stopped; do all robots inside a single stop/play (global op).
        robots = [instance_eval_state.env_accessor.robot for instance_eval_state in self.instance_eval_states]
        if any(robot.model in ("r1", "r1pro") for robot in robots):
            og.sim.stop()
            for robot in robots:
                if robot.model in ("r1", "r1pro"):
                    robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
            og.sim.play()

        head_camera_name = self.robot_camera_names.get("head")
        if head_camera_name is not None:
            head_sensor_name = head_camera_name.split("::")[1]
            for robot in robots:
                if head_sensor_name in robot.sensors:
                    robot.sensors[head_sensor_name].horizontal_aperture = EVAL_HEAD_HORIZONTAL_APERTURE

    def load_policy(self) -> Any:
        # A single policy serves all logical environments in one batched call. It accepts observations
        # with a leading num_envs dimension and keeps per-environment state positionally aligned.
        policy = instantiate(self.cfg.model)
        if hasattr(policy, "set_action_dim"):
            policy.set_action_dim(self.instance_eval_states[0].env_accessor.robot.action_dim)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded {self.num_envs} policy instance(s) (batched): {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self, env_accessor: InstanceEnvAccessor) -> List[MetricBase]:
        return [
            AgentMetric(self.human_stats, env_accessor=env_accessor),
            TaskMetric(self.human_stats, env_accessor=env_accessor),
        ]

    def _apply_actions(self, actions: th.Tensor, active_env_indices: List[int]):
        """
        Step the complete environment batch with the given ``(num_envs, action_dim)`` actions, then
        update observations and metrics only for @active_env_indices. Finished environments are still
        advanced by the shared simulator, but their evaluation state is frozen.
        """
        obs_list, _, terminated, truncated, info = self.env.step(actions, n_render_iterations=1)
        if self.should_sync_lights:
            # Re-derive visual light state (toggles drive lights that aren't directly serialized) for
            # active environments, then re-read obs so recorded frames match the synced lights.
            for env_idx in active_env_indices:
                light_synchronizer = self.instance_eval_states[env_idx].light_synchronizer
                if light_synchronizer is not None:
                    light_synchronizer.sync_from_current_state()
            for _ in range(3):
                og.sim.render()
            obs_list, _ = self.env.get_obs()
        for env_idx in active_env_indices:
            instance_eval_state = self.instance_eval_states[env_idx]
            instance_eval_state.obs = self._preprocess_obs(obs_list[env_idx], instance_eval_state)
            for metric in instance_eval_state.metrics:
                metric.step(
                    action=actions[env_idx],
                    obs=obs_list[env_idx],
                    reward=0.0,
                    terminated=bool(terminated[env_idx]),
                    truncated=bool(truncated[env_idx]),
                    info=info[env_idx],
                )
        return terminated, truncated, info

    def _batch_obs(self) -> dict:
        """
        Stack every instance's preprocessed observation into one ``(num_envs, ...)`` batch for a
        single policy call. Finished instances keep their last observation so the batch size remains
        fixed and the policy's per-environment state remains positionally aligned.
        """
        keys = self.instance_eval_states[0].obs.keys()
        return {
            key: th.stack([instance_eval_state.obs[key] for instance_eval_state in self.instance_eval_states], dim=0)
            for key in keys
        }

    def _step_fn(self, active_env_indices: List[int]) -> Tuple[th.Tensor, th.Tensor]:
        """
        ``step_fn`` consumed by :func:`evaluate_instances_batched`: query the policy once with all
        logical environments batched together, then step. Finished environments get a zero action but are
        still advanced by the shared simulator.
        """
        action_dim = self.instance_eval_states[0].env_accessor.robot.action_dim
        batched_action = self.policy.forward(obs=self._batch_obs())  # (num_envs, action_dim)
        if batched_action.ndim == 1:
            batched_action = batched_action.unsqueeze(0)
        actions = th.zeros((self.num_envs, action_dim), dtype=th.float32)
        for env_idx in active_env_indices:
            actions[env_idx] = batched_action[env_idx].to(actions.dtype)
        terminated, truncated, _ = self._apply_actions(actions, active_env_indices)
        return terminated, truncated

    def _set_video_writer(self, instance_eval_state: InstanceEvaluationState, video_writer) -> None:
        existing = instance_eval_state.video_writer
        if existing is not None:
            container, stream = existing
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        instance_eval_state.video_writer = video_writer

    def _load_instance_state(self, instance_id: int, instance_eval_state: InstanceEvaluationState) -> None:
        """
        Load a task instance into its assigned logical environment. Does NOT settle physics or
        finalize the scene -- that happens once for the whole batch in :meth:`_settle_and_finalize` so
        settling does not disturb the other logical environments.
        """
        env_accessor = instance_eval_state.env_accessor
        env_idx = env_accessor.env_idx
        robot = env_accessor.robot
        scene = env_accessor.scene
        # Start every instance from a CLEAN robot: reset joints / controllers / velocities (and release
        # grasp) to the default config -- the instance TRO only restores the robot's base pose below.
        # 2285 got this for free from evaluator.reset() (env.reset) before each load_task_instance; in
        # the batched path we must do it explicitly, otherwise the previous episode's/batch's robot
        # joint+grasp state carries into the snapshot taken in _settle_and_finalize.
        robot.reset()
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        mode = self.cfg.get("mode", "public_test")
        tro_file_path = get_task_instance_path(
            scene_model,
            f"{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state",
            mode=mode,
        )
        if tro_file_path is None:
            raise FileNotFoundError(
                f"Could not find 2026 {mode} task instance {instance_id} for "
                f"{self.env.task.activity_name} in scene {scene_model}."
            )

        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = {key.lower(): value for key, value in tro_state.items()}
                if "robot" in presampled_robot_poses:
                    available_poses = presampled_robot_poses["robot"]
                elif robot.model in presampled_robot_poses:
                    logger.info("No generic presampled robot pose found, using robot-specific pose.")
                    available_poses = presampled_robot_poses[robot.model]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose found for {robot.model}!")
                # Presampled poses are scene-relative; frame="scene" puts environment N's robot in its own
                # scene rather than env 0's geometry (cf. #2257).
                robot.set_position_orientation(
                    available_poses[0]["position"], available_poses[0]["orientation"], frame="scene"
                )
                scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                # load_state applies the TRO root-link pose in WORLD frame, but the TRO is authored in
                # scene-relative (env 0) coordinates. For multi envs (env_idx != 0) the scene prim is
                # shifted by its world offset, so add that offset to the root-link position up front --
                # load_state then places the object correctly in a single write.
                obj = env_accessor.object_scope[tro_key]
                if env_idx != 0 and isinstance(tro_state, dict) and "root_link" in tro_state:
                    scene_offset = scene._scene_prim.get_position_orientation()[0]
                    tro_state["root_link"]["pos"] = tro_state["root_link"]["pos"] + scene_offset
                obj.load_state(tro_state, serialized=False)

        if self.should_sync_lights:
            set_light_control_toggles(env_accessor.object_scope.values(), True)
        og.sim.update_handles()

    def _settle_and_finalize(self, env_indices: List[int]) -> None:
        """
        Settle physics for all freshly-loaded environments together and finalize each scene. Loading
        state can introduce jitter, so keep all loaded task-relevant entities (including the robot,
        which _load_instance_state just reset to a clean config) still for a few sub-steps before
        snapshotting each scene's initial state -- otherwise the robot drifts under gravity before the
        snapshot.
        """
        for _ in range(25):
            og.sim.step_physics()
            for env_idx in env_indices:
                for inst, entity in self.instance_eval_states[env_idx].env_accessor.object_scope.items():
                    if not is_system_bddl_inst(inst) and entity is not None:
                        entity.keep_still()
        for env_idx in env_indices:
            self.instance_eval_states[env_idx].env_accessor.scene.update_initial_file()
        # load_batch immediately calls env.reset(), whose task reset restores every selected scene
        # from this initial file. Avoid doing the same hard restore and physics step twice.

    def load_batch(
        self,
        env_idx_to_instance: dict,
        write_video: bool = False,
        video_path: str = None,
        rollout_id: int = 0,
        video_fps: int = 30,
    ) -> None:
        """
        Load one instance per logical environment, settle the complete batch, refresh observations,
        and reset policy, metrics, light synchronizers, and video writers.
        """
        env_indices = sorted(env_idx_to_instance)
        for env_idx in env_indices:
            instance_eval_state = self.instance_eval_states[env_idx]
            instance_eval_state.instance_id = int(env_idx_to_instance[env_idx])
            instance_eval_state.active = True
            self._load_instance_state(instance_eval_state.instance_id, instance_eval_state)
        self._settle_and_finalize(env_indices)
        obs_list, _ = self.env.reset(env_indices=th.tensor(env_indices, dtype=th.long))
        for env_idx in env_indices:
            self._reset_light_synchronizer(self.instance_eval_states[env_idx])
        # Light toggles change visibility after the reset obs was captured; re-render + re-fetch so the
        # first obs reflects the synced lights (mirrors single-env _sync_lights_and_get_obs). No-op for
        # non-light tasks.
        if self.should_sync_lights:
            for _ in range(3):
                og.sim.render()
            obs_list, _ = self.env.get_obs(env_indices=th.tensor(env_indices, dtype=th.long))
        task_name = self.cfg.task.name
        # One batched policy for the complete batch: reset its per-environment state once.
        self.policy.reset()
        for obs_idx, env_idx in enumerate(env_indices):
            instance_eval_state = self.instance_eval_states[env_idx]
            instance_eval_state.obs = self._preprocess_obs(obs_list[obs_idx], instance_eval_state)
            for metric in instance_eval_state.metrics:
                metric.reset()
            if write_video:
                video_name = os.path.join(video_path, f"{task_name}_{instance_eval_state.instance_id}_{rollout_id}.mp4")
                self._set_video_writer(
                    instance_eval_state,
                    create_video_writer(fpath=video_name, resolution=(448, 672), rate=video_fps),
                )

    def _preprocess_obs(self, obs: dict, instance_eval_state: InstanceEvaluationState) -> dict:
        robot = instance_eval_state.env_accessor.robot
        obs = flatten_obs_dict(obs)
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []
        # Camera extrinsics come directly from the sensor prim. Querying camera_parameters just for
        # cameraViewTransform lazily creates an annotator and forces global renders per camera.
        for camera_name in self.robot_camera_names.values():
            sensor_name = camera_name.split("::")[1]
            if sensor_name not in robot.sensors:
                continue
            camera = robot.sensors[sensor_name]
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose)))
        if cam_rel_poses:
            obs[f"{robot.name}::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self, instance_eval_state: InstanceEvaluationState) -> None:
        obs = instance_eval_state.obs
        required_camera_ids = ("left_wrist", "right_wrist", "head")
        if not all(camera_id in self.robot_camera_names for camera_id in required_camera_ids):
            return
        if obs is None or self.robot_camera_names["head"] + "::rgb" not in obs:
            return
        # .detach().cpu() is required for num_envs>1: robot cameras come from the TiledVisionSensor,
        # whose obs are CUDA tensors (a plain .numpy() would raise). No-op for single-env CPU tensors.
        left_wrist_rgb = cv2.resize(
            obs[self.robot_camera_names["left_wrist"] + "::rgb"].detach().cpu().numpy(), (224, 224)
        )
        right_wrist_rgb = cv2.resize(
            obs[self.robot_camera_names["right_wrist"] + "::rgb"].detach().cpu().numpy(), (224, 224)
        )
        head_rgb = cv2.resize(obs[self.robot_camera_names["head"] + "::rgb"].detach().cpu().numpy(), (448, 448))
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=instance_eval_state.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        """Reset all logical environments to their loaded instances and reset policy and metrics."""
        obs_list, _ = self.env.reset()
        for instance_eval_state in self.instance_eval_states:
            self._reset_light_synchronizer(instance_eval_state)
        # Refresh obs after light visibility changes so the first obs is in sync (see load_batch).
        if self.should_sync_lights:
            for _ in range(3):
                og.sim.render()
            obs_list, _ = self.env.get_obs()
        self.policy.reset()
        for instance_eval_state in self.instance_eval_states:
            env_idx = instance_eval_state.env_idx
            instance_eval_state.obs = self._preprocess_obs(obs_list[env_idx], instance_eval_state)
            instance_eval_state.active = True
            for metric in instance_eval_state.metrics:
                metric.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def run(
        self,
        instances_to_run: List[int],
        write_video: bool = False,
        video_path: str = None,
        metrics_dir: str = None,
        rollout_id: int = 0,
        video_fps: int = 30,
    ) -> dict:
        """
        Offline driver for one batch containing exactly ``num_envs`` instances. An environment whose
        instance finishes early waits idle rather than receiving a new instance. Writes one result JSON per instance to
        @metrics_dir if given. Returns {instance id -> result dict} (result is score_utils-compatible:
        q_score/time/agent_distance plus task/instance/success/steps).
        """
        task_name = self.cfg.task.name

        def load_fn(env_idx_to_instance):
            self.load_batch(
                env_idx_to_instance,
                write_video=write_video,
                video_path=video_path,
                rollout_id=rollout_id,
                video_fps=video_fps,
            )

        def step_fn(active_env_indices):
            terminated, truncated = self._step_fn(active_env_indices)
            if write_video:
                for env_idx in active_env_indices:
                    self._write_video(self.instance_eval_states[env_idx])
            return terminated, truncated

        def record_fn(env_idx, instance, step, terminated, truncated):
            instance_eval_state = self.instance_eval_states[env_idx]
            instance_eval_state.active = False
            self.n_trials += 1
            success = instance_eval_state.env_accessor.success
            if success:
                self.n_success_trials += 1
            result = {"task": task_name, "instance_id": int(instance), "rollout_id": rollout_id, "steps": step}
            result["success"] = success
            for metric in instance_eval_state.metrics:
                result.update(metric.aggregate())
            if metrics_dir is not None:
                with open(os.path.join(metrics_dir, f"{task_name}_{instance}_{rollout_id}.json"), "w") as f:
                    json.dump(result, f, indent=2, default=float)
            if write_video:
                self._set_video_writer(instance_eval_state, None)
            q_score = result.get("q_score", {}).get("final")
            logger.info(
                f"Instance {instance} (env {env_idx}) finished at step {step}: success={success} q_score={q_score}"
            )
            return result

        return evaluate_instances_batched(
            list(instances_to_run),
            self.num_envs,
            load_fn=load_fn,
            step_fn=step_fn,
            record_fn=record_fn,
        )

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        for instance_eval_state in self.instance_eval_states:
            self._set_video_writer(instance_eval_state, None)
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


# Backward-compatible alias for callers that imported the original class name.
Evaluator = BatchedEvaluator


__all__ = [
    "BatchedEvaluator",
    "Evaluator",
    "InstanceEnvAccessor",
    "InstanceEvaluationState",
    "resolve_instance_ids",
    "evaluate_instances_batched",
]
