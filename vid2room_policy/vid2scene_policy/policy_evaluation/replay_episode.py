from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch as th
import yaml

import omnigibson as og
from omnigibson.objects import DatasetObject
from vid2scene_policy.data_collection.lerobot_datasets.datasets.lerobot_dataset import LeRobotDataset


def _step_sim(num_steps: int) -> None:
    for _ in range(num_steps):
        og.sim.step()


def _load_episode_metadata(dataset_root: Path, episode_idx: int) -> dict:
    metadata_path = dataset_root / "meta" / f"episode_metadata_{episode_idx}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing episode metadata file: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_episode_actions(dataset_root: Path, episode_idx: int) -> np.ndarray:
    dataset = LeRobotDataset(
        repo_id=dataset_root.resolve().name,
        root=dataset_root.resolve(),
        force_cache_sync=False,
        revision="v3.0",
        episodes=[episode_idx],
    )
    actions = dataset.hf_dataset["action"]
    if len(actions) == 0:
        raise RuntimeError(f"No actions found for episode {episode_idx} in dataset {dataset_root}")
    return np.stack([np.asarray(a, dtype=np.float32) for a in actions], axis=0)


def _build_og_config(scene_model: str, dataset_name: str, sticky_grasp: bool, config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["scene"]["scene_model"] = scene_model
    cfg["scene"]["dataset_name"] = dataset_name
    # Disable scene filtering for deterministic replay.
    cfg["scene"]["not_load_object_categories"] = None
    cfg["scene"]["load_room_types"] = None
    cfg["scene"]["load_room_instances"] = None

    if dataset_name != "behavior-1k-assets":
        cfg["scene"]["scene_instance"] = f"{scene_model}_best"

    if "robots" in cfg and cfg["robots"] and sticky_grasp:
        cfg["robots"][0]["grasping_mode"] = "sticky"
        # Disable controller scaling to match collection-time actions.
        controller_cfg = cfg["robots"][0].get("controller_config", {})
        for key in (
            "base",
            "trunk",
            "arm_left",
            "arm_right",
            "gripper_left",
            "gripper_right",
            "arm_0",
            "gripper_0",
            "camera",
        ):
            if key in controller_cfg and isinstance(controller_cfg[key], dict):
                controller_cfg[key]["command_input_limits"] = None
                controller_cfg[key]["command_output_limits"] = None

    return cfg


def _find_or_spawn_target(scene, metadata: dict):
    target_name = metadata["target_object_name"]
    target_obj = scene.object_registry("name", target_name)
    if target_obj is not None:
        return target_obj

    if not metadata.get("spawned_target_object", False):
        raise RuntimeError(
            f"Target object '{target_name}' is missing from scene and was not marked as spawned in metadata."
        )

    category = metadata.get("target_object_category")
    model = metadata.get("target_object_model")
    dataset_name = metadata.get("target_object_dataset_name")
    if category is None or model is None or dataset_name is None:
        raise RuntimeError("Spawned target metadata is incomplete; cannot reconstruct object.")

    spawned = DatasetObject(
        name=target_name,
        category=category,
        model=model,
        dataset_name=dataset_name,
    )
    scene.add_object(spawned)
    _step_sim(5)
    return spawned


def _setup_scene_state(scene, robot, metadata: dict) -> None:
    start_x, start_y, start_z, start_theta = metadata["robot_start_x_y_z_theta"]
    start_quat = [0.0, 0.0, math.sin(start_theta / 2.0), math.cos(start_theta / 2.0)]
    robot.set_position_orientation(position=[start_x, start_y, start_z], orientation=start_quat)
    robot.keep_still()

    target_obj = _find_or_spawn_target(scene, metadata)
    target_pos = th.as_tensor(metadata["target_object_setup_position"], dtype=th.float32)
    target_orn = th.as_tensor(metadata["target_object_setup_orientation"], dtype=th.float32)
    target_obj.set_position_orientation(position=target_pos, orientation=target_orn)

    _step_sim(10)


def _fit_action_to_robot_dim(action_np: np.ndarray, robot_action_dim: int) -> np.ndarray:
    """Pad or truncate dataset action vector to robot action dimension."""
    action_np = np.asarray(action_np, dtype=np.float32).reshape(-1)
    if action_np.shape[0] == robot_action_dim:
        return action_np

    full_action = np.zeros((robot_action_dim,), dtype=np.float32)
    n_copy = min(action_np.shape[0], robot_action_dim)
    full_action[:n_copy] = action_np[:n_copy]
    return full_action


def _should_print_debug(step_idx: int, num_steps: int, debug: bool) -> bool:
    return step_idx < 5 or step_idx % 10 == 0 or step_idx == num_steps - 1 or debug


def replay_episode(
    dataset_root: Path,
    episode_idx: int,
    realtime: bool,
    sticky_grasp: bool,
    hold_seconds: float,
    debug: bool,
    env_config_path: Path,
) -> None:
    metadata = _load_episode_metadata(dataset_root, episode_idx)
    actions = _load_episode_actions(dataset_root, episode_idx)
    scene_model = metadata["scene_name"]
    dataset_name = metadata["dataset_name"]

    env_cfg = _build_og_config(
        scene_model=scene_model,
        dataset_name=dataset_name,
        sticky_grasp=sticky_grasp,
        config_path=env_config_path,
    )
    env = og.Environment(configs=env_cfg)
    scene = env.scene
    robot = env.robots[0]

    try:
        env.reset()
        robot.reset()
        _setup_scene_state(scene=scene, robot=robot, metadata=metadata)

        print(f"[Replay] Episode {episode_idx}: {len(actions)} action steps")
        print(f"[Replay] first_action={actions[0].tolist()} last_action={actions[-1].tolist()}", flush=True)
        robot_action_dim = int(robot.action_dim)

        print(
            f"[Replay] action dims: dataset={int(actions.shape[1])}, "
            f"robot={robot_action_dim}",
            flush=True,
        )
        action_freq = int(env_cfg.get("env", {}).get("action_frequency", 10))
        replay_dt = 1.0 / max(1, action_freq)
        print(f"[Replay] action_frequency={action_freq}Hz, step_dt={replay_dt:.3f}s", flush=True)

        arm_name = robot.default_arm
        eef_pos_prev, _ = robot.get_eef_pose(arm_name)
        joint_prev = robot.get_joint_positions()[robot.arm_control_idx[arm_name]].clone()
        max_eef_delta = 0.0
        max_joint_delta = 0.0
        for i in range(len(actions)):
            action_np = _fit_action_to_robot_dim(actions[i], robot_action_dim)

            print(f"[Replay][STEP] pre env.step i={i} action={action_np.tolist()}", flush=True)
            try:
                env.step({robot.name: action_np})
            except Exception as step_exc:
                print(f"[Replay][ERROR] env.step failed at i={i}: {step_exc}", flush=True)
                raise
            print(f"[Replay][STEP] post env.step i={i}", flush=True)
            eef_pos_now, _ = robot.get_eef_pose(arm_name)
            joint_now = robot.get_joint_positions()[robot.arm_control_idx[arm_name]]
            eef_delta = float(th.norm(eef_pos_now - eef_pos_prev).item())
            joint_delta = float(th.norm(joint_now - joint_prev).item())
            max_eef_delta = max(max_eef_delta, eef_delta)
            max_joint_delta = max(max_joint_delta, joint_delta)
            action_norm = float(np.linalg.norm(action_np))
            if _should_print_debug(i, len(actions), debug):
                print(
                    f"[Replay][DEBUG] step={i:03d}/{len(actions)-1:03d} "
                    f"action={action_np.tolist()} action_norm={action_norm:.5f} "
                    f"eef_delta={eef_delta:.5f}m joint_delta={joint_delta:.5f} "
                    f"eef_pos={eef_pos_now.tolist()}",
                    flush=True,
                )
            eef_pos_prev = eef_pos_now
            joint_prev = joint_now.clone()
            if realtime:
                time.sleep(replay_dt)

        print(
            f"[Replay] Done. max_eef_delta={max_eef_delta:.5f}m, "
            f"max_joint_delta={max_joint_delta:.5f}",
            flush=True,
        )
        if hold_seconds > 0:
            print(f"[Replay] Holding simulator open for {hold_seconds:.1f}s", flush=True)
            time.sleep(hold_seconds)
    except Exception as replay_exc:
        print(f"[Replay][FATAL] replay failed: {replay_exc}", flush=True)
        raise
    finally:
        og.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one saved LeRobot pick episode.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to saved dataset root (e.g. ./lerobot_datasets/<repo_id>)",
    )
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--realtime", action="store_true", help="Sleep between frames to approximate capture FPS")
    parser.add_argument(
        "--no-sticky-grasp",
        action="store_true",
        help="Disable forcing sticky grasp in replay environment",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Keep simulator open for N seconds after replay",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-step replay diagnostics",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("/home/behavior/workspace/BEHAVIOR-1K/OmniGibson/omnigibson/configs/fetch_vid2scene.yaml"),
        help="Path to OmniGibson env yaml to load",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    env_config_path = args.env_config.resolve()
    if not env_config_path.exists():
        raise FileNotFoundError(f"Env config yaml not found: {env_config_path}")

    replay_episode(
        dataset_root=dataset_root,
        episode_idx=args.episode_idx,
        realtime=args.realtime,
        sticky_grasp=not args.no_sticky_grasp,
        hold_seconds=max(0.0, args.hold_seconds),
        debug=args.debug,
        env_config_path=env_config_path,
    )


if __name__ == "__main__":
    main()

