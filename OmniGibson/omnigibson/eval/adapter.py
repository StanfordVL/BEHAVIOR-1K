import csv
import cv2
import json
import logging
import os
import sys
import traceback
from signal import SIGINT, signal
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.vector.utils import batch_space
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import omnigibson as og
import omnigibson.utils.transform_utils as T
from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.utils.og_teleop_utils import (
    augment_rooms,
    generate_robot_config,
    get_task_relevant_room_types,
    load_available_tasks,
)
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
    generate_basic_environment_config,
)
from omnigibson.eval.utils.obs_utils import create_video_writer, write_video
from omnigibson.eval.utils.vec_eval_scheduler import evaluate_instances_batched
from omnigibson.macros import create_module_macros, gm, macros
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import Robot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.utils.python_utils import recursively_convert_to_torch

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 10

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

with macros.unlocked():
    macros.robots.manipulation_robot.GRASP_WINDOW = 0.75

logger = logging.getLogger("evaluator")
logger.setLevel(logging.INFO)


class Evaluator:
    """
    Vectorized evaluator engine for BEHAVIOR tasks. Holds a single OmniGibson environment with
    ``num_envs`` parallel scene slots (one task instance per slot) and per-slot robots, policies,
    metrics, observations and video writers (index everything by ``env_idx``). num_envs=1 reproduces
    single-env evaluation. All sim interaction lives here; the gym surface (BehaviorEvalVectorEnv) and
    the offline driver (run) are thin layers on top.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0

        # Number of parallel env slots (instances evaluated concurrently). Defaults to 1.
        self.num_envs = int(cfg.get("num_envs", 1))
        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        assert (
            self.env.num_envs == self.num_envs
        ), f"Env created with num_envs={self.env.num_envs} but Evaluator expected {self.num_envs}."
        # Per-slot robots / policies / metrics / observations / video writers.
        self.robots = self.load_robots()
        self.policies = self.load_policies()
        self.metrics = self.load_metrics()
        self.obs = [None] * self.num_envs
        self._video_writers = [None] * self.num_envs

        self.env._current_episode = 0

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"

        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for key in self.human_stats:
                    self.human_stats[key].append(episode[key])
        for key in self.human_stats:
            self.human_stats[key] = sum(self.human_stats[key]) / len(self.human_stats[key])

        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        robot_model = robot_type.lower()
        assert robot_model == "r1pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."

        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                robot_type=robot_model,
                robot_name="robot_r1",
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        cfg["robots"][0]["model"] = cfg["robots"][0].pop("type")
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(
                OmegaConf.to_container(self.cfg.robot.controllers, resolve=True)
            )

        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False

        # Run num_envs instances of the same scene model + task in parallel slots.
        cfg.setdefault("env", {})
        cfg["env"]["num_envs"] = self.num_envs

        env = og.Environment(configs=cfg)
        return instantiate(env_wrapper, env=env)

    def load_robots(self) -> List[Robot]:
        # env.robots is list[list[Robot]] (one inner list per scene). The eval pipeline assumes a single
        # robot per scene (load_env configures exactly one "robot_r1"; _preprocess_obs hardcodes its
        # name) -- assert it so a multi-robot config fails loudly instead of silently using robot 0.
        for scene_robots in self.env.robots:
            assert len(scene_robots) == 1, f"Eval assumes one robot per scene, got {len(scene_robots)}."
        return [scene_robots[0] for scene_robots in self.env.robots]

    def load_policies(self) -> List[Any]:
        # One independent (possibly stateful) policy instance per env slot.
        policies = []
        for _ in range(self.num_envs):
            policy = instantiate(self.cfg.model)
            if hasattr(policy, "set_action_dim"):
                policy.set_action_dim(self.robots[0].action_dim)
            policies.append(policy)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded {self.num_envs} policy instance(s): {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policies

    def load_metrics(self) -> List[List[MetricBase]]:
        return [
            [AgentMetric(self.human_stats, env_idx=i), TaskMetric(self.human_stats, env_idx=i)]
            for i in range(self.num_envs)
        ]

    def _apply_actions(self, actions: th.Tensor, active_slots: List[int]):
        """
        Step all env slots one step with the given ``(num_envs, action_dim)`` actions, then update the
        observations and metrics for the @active_slots only (frozen slots are still stepped by the
        shared simulator but their obs/metrics are not advanced). Returns ``(terminated, truncated,
        info)`` straight from the env (``terminated``/``truncated`` are ``(num_envs,)`` tensors, ``info``
        a per-slot list).
        """
        obs_list, _, terminated, truncated, info = self.env.step(actions, n_render_iterations=1)
        for slot in active_slots:
            self.obs[slot] = self._preprocess_obs(obs_list[slot], env_idx=slot)
            for metric in self.metrics[slot]:
                metric.step(
                    self.env,
                    actions[slot],
                    obs_list[slot],
                    0.0,
                    bool(terminated[slot]),
                    bool(truncated[slot]),
                    info[slot],
                )
        return terminated, truncated, info

    def _step_fn(self, active_slots: List[int]) -> Tuple[th.Tensor, th.Tensor]:
        """
        ``step_fn`` consumed by ``evaluate_instances_batched``: drive active slots with their own
        policies (frozen slots get a zero action).
        """
        action_dim = self.robots[0].action_dim
        actions = th.zeros((self.num_envs, action_dim), dtype=th.float32)
        for slot in active_slots:
            actions[slot] = self.policies[slot].forward(obs=self.obs[slot])
        terminated, truncated, _ = self._apply_actions(actions, active_slots)
        return terminated, truncated

    def _set_video_writer(self, env_idx: int, video_writer) -> None:
        existing = self._video_writers[env_idx]
        if existing is not None:
            container, stream = existing
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._video_writers[env_idx] = video_writer

    def _load_instance_state(self, instance_id: int, env_idx: int, test_hidden: bool = False) -> None:
        """
        Load a task instance's object/robot state into slot @env_idx. Does NOT settle physics or
        finalize the scene -- that happens once for the whole batch in :meth:`_settle_and_finalize` so
        settling does not disturb already-loaded sibling slots.
        """
        robot = self.robots[env_idx]
        scene = self.env.scenes[env_idx]
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(
                    scene_model,
                    f"{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state",
                )
            )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = {key.lower(): value for key, value in tro_state.items()}
                if "robot" in presampled_robot_poses:
                    available_poses = presampled_robot_poses["robot"]
                elif robot.model in presampled_robot_poses:
                    print("No generic presampled robot pose found, using robot-specific pose.")
                    available_poses = presampled_robot_poses[robot.model]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose found for {robot.model}!")
                # Presampled poses are scene-relative; frame="scene" puts slot N's robot in its own
                # scene rather than env 0's geometry (cf. #2257).
                robot.set_position_orientation(
                    available_poses[0]["position"], available_poses[0]["orientation"], frame="scene"
                )
                scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[env_idx][tro_key].load_state(tro_state, serialized=False)
        og.sim.update_handles()

    def _settle_and_finalize(self, slots: List[int]) -> None:
        """
        Settle physics for all freshly-loaded @slots together and finalize each one's scene. Loading
        state can introduce jitter, so keep loaded task-relevant objects (not the robot) still for a
        few sub-steps before snapshotting each scene's initial state.
        """
        for _ in range(25):
            og.sim.step_physics()
            for slot in slots:
                for inst, entity in self.env.task.object_scope[slot].items():
                    if not is_system_bddl_inst(inst) and entity is not None and not isinstance(entity, Robot):
                        entity.keep_still()
        for slot in slots:
            self.env.scenes[slot].update_initial_file()
            self.env.scenes[slot].reset()

    def load_batch(
        self,
        slot_to_instance: dict,
        test_hidden: bool = False,
        write_video: bool = False,
        video_path: str = None,
    ) -> None:
        """
        Load a batch of instances (one per slot) into their slots, settle the whole batch together,
        then refresh observations and reset per-slot policy / metrics (and video writers). Shared by
        :meth:`run` and the gym ``BehaviorEvalVectorEnv``.
        """
        ordered_slots = sorted(slot_to_instance)
        for slot in ordered_slots:
            self._load_instance_state(slot_to_instance[slot], env_idx=slot, test_hidden=test_hidden)
        self._settle_and_finalize(ordered_slots)
        obs_list, _ = self.env.reset(env_indices=th.tensor(ordered_slots, dtype=th.long))
        task_name = self.cfg.task.name
        for i, slot in enumerate(ordered_slots):
            self.obs[slot] = self._preprocess_obs(obs_list[i], env_idx=slot)
            self.policies[slot].reset()
            for metric in self.metrics[slot]:
                metric.reset(self.env)
            if write_video:
                video_name = f"{video_path}/{task_name}_{slot_to_instance[slot]}_0.mp4"
                self._set_video_writer(slot, create_video_writer(fpath=video_name, resolution=(448, 672)))

    def _preprocess_obs(self, obs: dict, env_idx: int = 0) -> dict:
        robot = self.robots[env_idx]
        obs = flatten_obs_dict(obs)
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []
        # The first camera-parameter query returns zeros; fall back to get_position_orientation() then.
        # We prefer camera parameters because they stay in sync with the visual obs under
        # n_render_iterations=1 (camera.get_position_orientation() returns the most-recent pose).
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self, env_idx: int) -> None:
        obs = self.obs[env_idx]
        if obs is None or ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb" not in obs:
            return
        left_wrist_rgb = cv2.resize(obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(), (224, 224))
        right_wrist_rgb = cv2.resize(obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(), (224, 224))
        head_rgb = cv2.resize(obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(), (448, 448))
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self._video_writers[env_idx],
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        """Generic reset of all slots to the currently-loaded (seed) instance + reset policies/metrics."""
        obs_list, _ = self.env.reset()
        self.obs = [self._preprocess_obs(obs_list[i], env_idx=i) for i in range(self.num_envs)]
        for env_idx in range(self.num_envs):
            for metric in self.metrics[env_idx]:
                metric.reset(self.env)
            self.policies[env_idx].reset()
        self.n_success_trials, self.n_trials = 0, 0

    def run(
        self,
        instances_to_run: List[int],
        test_hidden: bool = False,
        write_video: bool = False,
        video_path: str = None,
        metrics_dir: str = None,
    ) -> dict:
        """
        Offline driver: evaluate every instance in @instances_to_run, processing them num_envs at a
        time. The whole group finishes before the next group loads, and an early-finishing slot waits
        idle rather than getting a new instance -- loading an instance settles physics for ALL scenes
        at once, so refilling one slot mid-group would disturb the slots still running. Writes one
        result JSON per instance to @metrics_dir if given. Returns {instance id -> result metrics dict}.
        """
        task_name = self.cfg.task.name

        def load_fn(slot_to_instance):
            self.load_batch(slot_to_instance, test_hidden=test_hidden, write_video=write_video, video_path=video_path)

        def step_fn(active_slots):
            terminated, truncated = self._step_fn(active_slots)
            if write_video:
                for slot in active_slots:
                    self._write_video(slot)
            return terminated, truncated

        def record_fn(slot, instance, step, terminated, truncated):
            self.n_trials += 1
            success = bool(self.env.task.success[slot])
            if success:
                self.n_success_trials += 1
            result = {}
            for metric in self.metrics[slot]:
                result.update(metric.aggregate(self.env))
            if metrics_dir is not None:
                with open(os.path.join(metrics_dir, f"{task_name}_{instance}_0.json"), "w") as f:
                    json.dump(result, f)
            if write_video:
                self._set_video_writer(slot, None)
            logger.info(f"Instance {instance} (slot {slot}) finished at step {step}: success={success}")
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
        for env_idx in range(self.num_envs):
            self._set_video_writer(env_idx, None)
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


def _get_cfg_value(cfg, name, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _resolve_instance_ids(task_name: str, instance_indices: List[int]) -> List[int]:
    task_instance_csv_path = os.path.join(
        gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
    )
    with open(task_instance_csv_path, newline="", encoding="utf-8") as f:
        row = next(row for row in csv.DictReader(f) if row["Task"] == task_name)
    test_instances = [int(instance_id.strip()) for instance_id in row["Public Test Instance IDs"].split(",")]
    return [int(test_instances[i]) for i in instance_indices]


def _space_from_value(value):
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.integer):
        if value.dtype == np.uint8:
            return gym.spaces.Box(low=0, high=255, shape=value.shape, dtype=value.dtype)
        dtype_info = np.iinfo(value.dtype)
        return gym.spaces.Box(low=dtype_info.min, high=dtype_info.max, shape=value.shape, dtype=value.dtype)
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float32)


def _to_numpy(obs: dict) -> dict:
    converted = {}
    for key, value in obs.items():
        if isinstance(value, th.Tensor):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32, copy=False)
        converted[key] = value
    return converted


class BehaviorEvalVectorEnv(gym.vector.VectorEnv):
    """
    Gymnasium vector-env surface over the vectorized :class:`Evaluator`. It is a fixed-width vector env
    of ``num_envs`` slots; each ``reset`` loads one batch of ``num_envs`` task instances (one per slot)
    and ``step`` advances them with caller-supplied actions. A slot that terminates/truncates *freezes*
    (its action is ignored and it keeps reporting done) until the next ``reset`` -- there is NO per-slot
    autoreset, because loading a new instance settles physics globally and would disturb still-running
    slots. To walk a longer instance list, call ``reset(options={"instance_ids": [...]})`` per batch (or
    rely on the internal batch cursor).

    This is what ``make_env`` returns, so EnvHub drives our internal num_envs parallelism through the
    standard batched gym.vector API. (Note: OmniGibson is a singleton simulator, so parallelism is N
    scenes inside ONE env, not N separate envs -- hence a custom VectorEnv rather than SyncVectorEnv.)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_name: str = "turning_on_radio",
        instance_indices: List[int] | None = None,
        num_envs: int = 1,
        log_path: str = "/tmp/og-envhub",
        max_steps: int | None = 1,
        partial_scene_load: bool = True,
        headless: bool = True,
        write_video: bool = False,
    ) -> None:
        instance_indices = list(instance_indices) if instance_indices is not None else list(range(num_envs))
        assert (
            len(instance_indices) == num_envs
        ), f"Expected one instance index per slot (num_envs={num_envs}), got {len(instance_indices)}."
        self.task_name = task_name
        self.write_video = write_video
        self.log_path = log_path
        self._closed = False

        gm.HEADLESS = headless
        self.evaluator = Evaluator(
            OmegaConf.create(
                {
                    "env_wrapper": {"_target_": "omnigibson.eval.wrappers.RGBLowResWrapper"},
                    "policy_name": "local",
                    "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
                    "num_envs": num_envs,
                    "headless": headless,
                    "write_video": write_video,
                    "partial_scene_load": partial_scene_load,
                    "max_steps": max_steps,
                    "log_path": log_path,
                    "test_hidden": False,
                    "task": {"name": task_name},
                    "robot": {"type": "R1Pro", "controllers": None},
                }
            )
        )
        self.num_envs = num_envs
        self._slot_instances = _resolve_instance_ids(task_name, instance_indices)
        self._active = [False] * num_envs

        self.single_action_space = self.evaluator.robots[0].action_space
        self.action_space = batch_space(self.single_action_space, num_envs)

        # Probe one batch to discover the (per-slot) observation structure, then batch it.
        obs, _ = self.reset(options={"instance_ids": self._slot_instances})
        self.single_observation_space = gym.spaces.Dict(
            {key: _space_from_value(value[0]) for key, value in obs.items()}
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)

    def _batch_obs(self) -> dict:
        per_slot = [_to_numpy(self.evaluator.obs[i]) for i in range(self.num_envs)]
        return {key: np.stack([slot_obs[key] for slot_obs in per_slot]) for key in per_slot[0]}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = options or {}
        instance_ids = options.get("instance_ids", self._slot_instances)
        assert (
            len(instance_ids) == self.num_envs
        ), f"reset expects one instance id per slot (num_envs={self.num_envs}), got {len(instance_ids)}."
        self._slot_instances = list(instance_ids)
        slot_to_instance = {slot: int(inst) for slot, inst in enumerate(self._slot_instances)}
        self.evaluator.load_batch(
            slot_to_instance,
            write_video=self.write_video,
            video_path=os.path.join(self.log_path, "videos") if self.write_video else None,
        )
        self.evaluator.n_trials = self.evaluator.n_success_trials = 0
        self._active = [True] * self.num_envs
        infos = {"instance_id": np.asarray(self._slot_instances, dtype=np.int64), "task_name": self.task_name}
        return self._batch_obs(), infos

    def step(self, actions):
        actions = th.as_tensor(np.asarray(actions), dtype=th.float32)
        # Freeze finished slots: zero their action so the shared sim does not advance them.
        for slot in range(self.num_envs):
            if not self._active[slot]:
                actions[slot] = 0.0
        active_slots = [slot for slot in range(self.num_envs) if self._active[slot]]
        terminated, truncated, info = self.evaluator._apply_actions(actions, active_slots)

        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        per_env_metrics: List[dict] = [None] * self.num_envs
        for slot in range(self.num_envs):
            if not self._active[slot]:
                # Already-frozen slots keep reporting terminated.
                terminations[slot] = True
                continue
            term, trunc = bool(terminated[slot]), bool(truncated[slot])
            terminations[slot] = term
            truncations[slot] = trunc
            if self.write_video:
                self.evaluator._write_video(slot)
            if term or trunc:
                self.evaluator.n_trials += 1
                if bool(self.evaluator.env.task.success[slot]):
                    self.evaluator.n_success_trials += 1
                metrics = {}
                for metric in self.evaluator.metrics[slot]:
                    metrics.update(metric.aggregate(self.evaluator.env))
                per_env_metrics[slot] = metrics
                self._active[slot] = False

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        infos = {
            "instance_id": np.asarray(self._slot_instances, dtype=np.int64),
            "metrics": per_env_metrics,
        }
        return self._batch_obs(), rewards, terminations, truncations, infos

    def close(self, **kwargs):
        if self._closed:
            return
        self._closed = True
        for slot in range(self.num_envs):
            self.evaluator._set_video_writer(slot, None)
        self.evaluator.env.close()
        og.shutdown()


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg=None):
    if use_async_envs:
        raise ValueError("BEHAVIOR-1K EnvHub currently supports only synchronous execution.")

    instance_indices = _get_cfg_value(cfg, "instance_indices", list(range(n_envs)))
    env = BehaviorEvalVectorEnv(
        task_name=_get_cfg_value(cfg, "task_name", "turning_on_radio"),
        instance_indices=instance_indices,
        num_envs=n_envs,
        log_path=_get_cfg_value(cfg, "log_path", "/tmp/og-envhub"),
        max_steps=_get_cfg_value(cfg, "max_steps", 1),
        partial_scene_load=_get_cfg_value(cfg, "partial_scene_load", True),
        headless=_get_cfg_value(cfg, "headless", True),
        write_video=_get_cfg_value(cfg, "write_video", False),
    )
    return {"behavior_1k": {0: env}}


__all__ = ["BehaviorEvalVectorEnv", "Evaluator", "make_env"]
