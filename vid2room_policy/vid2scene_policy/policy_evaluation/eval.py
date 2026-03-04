from __future__ import annotations

import argparse
import collections
import json
import math
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import omnigibson as og
from omnigibson.objects import DatasetObject

from vid2scene_policy.data_collection.lerobot_datasets.datasets.lerobot_dataset import LeRobotDataset
from vid2scene_policy.data_collection.omnigibson_lerobot_wrapper import (
    OmniGibsonLeRobotConfig,
    OmniGibsonLeRobotWrapper,
)
from vid2scene_policy.policy_training.diffusion_policy import (
    ActionNormalizer,
    ConditionalUnet1D,
    get_resnet,
    replace_bn_with_gn,
)


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
    cfg["scene"]["not_load_object_categories"] = None
    cfg["scene"]["load_room_types"] = None
    cfg["scene"]["load_room_instances"] = None

    if dataset_name != "behavior-1k-assets":
        cfg["scene"]["scene_instance"] = f"{scene_model}_best"

    if "robots" in cfg and cfg["robots"] and sticky_grasp:
        cfg["robots"][0]["grasping_mode"] = "sticky"
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


def _find_support(scene, metadata: dict):
    support_name = metadata.get("support_object_name")
    if support_name is None:
        return None
    support_obj = scene.object_registry("name", support_name)
    if support_obj is None:
        return None
    support_pos = metadata.get("support_object_position")
    support_orn = metadata.get("support_object_orientation")
    if support_pos is not None and support_orn is not None:
        support_obj.set_position_orientation(
            position=torch.as_tensor(support_pos, dtype=torch.float32),
            orientation=torch.as_tensor(support_orn, dtype=torch.float32),
        )
    return support_obj


def _setup_scene_state(scene, robot, metadata: dict):
    start_x, start_y, start_z, start_theta = metadata["robot_start_x_y_z_theta"]
    start_quat = [0.0, 0.0, math.sin(start_theta / 2.0), math.cos(start_theta / 2.0)]
    robot.set_position_orientation(position=[start_x, start_y, start_z], orientation=start_quat)
    robot.keep_still()

    support_obj = _find_support(scene=scene, metadata=metadata)
    target_obj = _find_or_spawn_target(scene=scene, metadata=metadata)
    target_pos = torch.as_tensor(metadata["target_object_setup_position"], dtype=torch.float32)
    target_orn = torch.as_tensor(metadata["target_object_setup_orientation"], dtype=torch.float32)
    target_obj.set_position_orientation(position=target_pos, orientation=target_orn)

    _step_sim(10)
    return target_obj, support_obj


def _build_policy_nets(obs_horizon: int, action_dim: int, device: torch.device) -> nn.ModuleDict:
    wrist_encoder = replace_bn_with_gn(get_resnet("resnet18", in_channels=6))
    head_encoder = replace_bn_with_gn(get_resnet("resnet18", in_channels=6))
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=(512 + 512) * obs_horizon,
        hidden_dim=512,
    )
    return nn.ModuleDict(
        {
            "wrist_encoder": wrist_encoder,
            "head_encoder": head_encoder,
            "noise_pred_net": noise_pred_net,
        }
    ).to(device)


def _load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.ModuleDict, ActionNormalizer, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    obs_horizon = int(ckpt_args.get("obs_horizon", 2))

    norm_payload = ckpt.get("action_normalizer", {})
    if "mean" not in norm_payload or "std" not in norm_payload:
        raise RuntimeError("Checkpoint missing action_normalizer.mean/std.")
    mean = torch.as_tensor(norm_payload["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(norm_payload["std"], dtype=torch.float32, device=device)
    action_dim = int(mean.numel())

    nets = _build_policy_nets(obs_horizon=obs_horizon, action_dim=action_dim, device=device)
    nets.load_state_dict(ckpt["model"], strict=True)
    nets.eval()

    if "ema" in ckpt:
        ema = EMAModel(parameters=nets.parameters(), power=0.75)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(nets.parameters())
        nets.eval()

    action_normalizer = ActionNormalizer(mean=mean, std=std)
    return nets, action_normalizer, ckpt_args


def _extract_fused_obs(lerobot_obs: dict) -> dict:
    wrist_rgb = np.asarray(lerobot_obs["observation.images.wrist"], dtype=np.uint8)
    wrist_seg = np.asarray(lerobot_obs["observation.images.wrist_seg_depth"], dtype=np.uint8)
    head_rgb = np.asarray(lerobot_obs["observation.images.head"], dtype=np.uint8)
    head_seg = np.asarray(lerobot_obs["observation.images.head_seg_depth"], dtype=np.uint8)

    wrist = np.concatenate([wrist_rgb, wrist_seg], axis=2).transpose(2, 0, 1)
    head = np.concatenate([head_rgb, head_seg], axis=2).transpose(2, 0, 1)
    return {"wrist": wrist, "head": head}


def _predict_action_sequence(
    nets: nn.ModuleDict,
    action_normalizer: ActionNormalizer,
    noise_scheduler: DDPMScheduler,
    obs_deque: collections.deque,
    pred_horizon: int,
    device: torch.device,
) -> np.ndarray:
    wrist = np.stack([x["wrist"] for x in obs_deque], axis=0).astype(np.float32) / 255.0
    head = np.stack([x["head"] for x in obs_deque], axis=0).astype(np.float32) / 255.0

    with torch.no_grad():
        wrist_t = torch.from_numpy(wrist).to(device=device, dtype=torch.float32)
        head_t = torch.from_numpy(head).to(device=device, dtype=torch.float32)

        wrist_feat = nets["wrist_encoder"](wrist_t)
        head_feat = nets["head_encoder"](head_t)
        obs_features = torch.cat([wrist_feat, head_feat], dim=-1)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        action_dim = int(action_normalizer.mean.numel())
        naction = torch.randn((1, pred_horizon, action_dim), device=device)
        for timestep in noise_scheduler.timesteps:
            timestep_batch = torch.full((1,), int(timestep), device=device, dtype=torch.long)
            noise_pred = nets["noise_pred_net"](
                sample=naction,
                timestep=timestep_batch,
                global_cond=obs_cond,
            )
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=naction,
            ).prev_sample

        action_pred = action_normalizer.denormalize(naction)[0].detach().cpu().numpy()
    return action_pred


def _parse_reset_output(reset_out):
    if isinstance(reset_out, tuple) and len(reset_out) >= 1:
        return reset_out[0]
    return reset_out


def _parse_step_output(step_out):
    if isinstance(step_out, tuple) and len(step_out) >= 5:
        return step_out
    if isinstance(step_out, tuple) and len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, reward, bool(done), False, info
    return step_out, 0.0, False, False, {}


def _load_episode_indices(episode_idx: int | None, episodes_file: Path | None) -> list[int]:
    if episode_idx is not None and episodes_file is not None:
        raise ValueError("Specify either --episode-idx or --episodes-file, not both.")
    if episode_idx is not None:
        return [episode_idx]
    if episodes_file is not None:
        text = episodes_file.read_text(encoding="utf-8")
        tokens = [tok.strip() for tok in text.replace(",", "\n").splitlines() if tok.strip()]
        if not tokens:
            raise ValueError(f"Episodes file is empty: {episodes_file}")
        return [int(tok) for tok in tokens]
    return [0]


def evaluate_episode(
    dataset_root: Path,
    episode_idx: int,
    env_config_path: Path,
    sticky_grasp: bool,
    nets: nn.ModuleDict,
    action_normalizer: ActionNormalizer,
    obs_horizon: int,
    pred_horizon: int,
    num_diffusion_iters: int,
    max_steps: int | None,
    output_dir: Path,
    debug: bool,
    device: torch.device,
) -> dict:
    metadata = _load_episode_metadata(dataset_root=dataset_root, episode_idx=episode_idx)
    gt_actions = _load_episode_actions(dataset_root=dataset_root, episode_idx=episode_idx)

    scene_model = metadata["scene_name"]
    dataset_name = metadata["dataset_name"]
    env_cfg = _build_og_config(
        scene_model=scene_model,
        dataset_name=dataset_name,
        sticky_grasp=sticky_grasp,
        config_path=env_config_path,
    )

    target_steps = len(gt_actions) if max_steps is None else max_steps
    episode_out_dir = output_dir / f"episode_{episode_idx:04d}"
    episode_out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = episode_out_dir / "steps.jsonl"
    summary_path = episode_out_dir / "episode_summary.json"

    print(f"[Eval] Episode {episode_idx}: target_steps={target_steps} gt_steps={len(gt_actions)}", flush=True)
    env = og.Environment(configs=env_cfg)
    wrapper = OmniGibsonLeRobotWrapper(env, OmniGibsonLeRobotConfig())
    robot = env.robots[0]
    arm_name = robot.default_arm
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    noise_scheduler.set_timesteps(num_diffusion_iters)

    steps_run = 0
    terminated = False
    truncated = False
    final_reward = 0.0
    final_obj_in_hand = None
    action_l2_errors: list[float] = []

    try:
        _ = _parse_reset_output(env.reset())
        robot.reset()
        target_obj, support_obj = _setup_scene_state(scene=env.scene, robot=robot, metadata=metadata)

        og_obs, _ = env.get_obs()
        wrapper._infer_observation_structure(og_obs)
        wrapper.set_target_objects(target_object=target_obj, support_object=support_obj)
        lerobot_obs = wrapper._convert_observation(og_obs)

        obs0 = _extract_fused_obs(lerobot_obs)
        obs_deque = collections.deque([obs0] * obs_horizon, maxlen=obs_horizon)

        robot_action_dim = int(robot.action_dim)
        print(
            f"[Eval] action dims: predicted_no_gripper={int(action_normalizer.mean.numel())}, robot={robot_action_dim}",
            flush=True,
        )

        with open(steps_path, "w", encoding="utf-8") as steps_file:
            while steps_run < target_steps:
                action_seq = _predict_action_sequence(
                    nets=nets,
                    action_normalizer=action_normalizer,
                    noise_scheduler=noise_scheduler,
                    obs_deque=obs_deque,
                    pred_horizon=pred_horizon,
                    device=device,
                )
                start = obs_horizon - 1
                action_no_gripper = action_seq[start].astype(np.float32)
                action_with_gripper = np.concatenate([action_no_gripper, np.array([-1.0], dtype=np.float32)])

                if action_with_gripper.shape[0] != robot_action_dim:
                    full_action = np.zeros((robot_action_dim,), dtype=np.float32)
                    n_copy = min(robot_action_dim, action_with_gripper.shape[0])
                    full_action[:n_copy] = action_with_gripper[:n_copy]
                    action_with_gripper = full_action

                step_out = env.step({robot.name: action_with_gripper})
                og_obs, reward, terminated, truncated, _ = _parse_step_output(step_out)
                final_reward = float(reward)

                lerobot_obs = wrapper._convert_observation(og_obs)
                obs_deque.append(_extract_fused_obs(lerobot_obs))

                eef_pos, _ = robot.get_eef_pose(arm_name)
                in_hand_obj = robot._ag_obj_in_hand.get(arm_name)
                final_obj_in_hand = None if in_hand_obj is None else in_hand_obj.name
                target_name = metadata.get("target_object_name")
                success_now = final_obj_in_hand == target_name

                gt_action = gt_actions[steps_run] if steps_run < len(gt_actions) else None
                l2_error = None
                if gt_action is not None:
                    n = min(len(gt_action), len(action_with_gripper))
                    l2_error = float(np.linalg.norm(action_with_gripper[:n] - gt_action[:n]))
                    action_l2_errors.append(l2_error)

                step_record = {
                    "step": steps_run,
                    "pred_action": action_with_gripper.tolist(),
                    "pred_action_norm": float(np.linalg.norm(action_with_gripper)),
                    "gt_action": None if gt_action is None else gt_action.tolist(),
                    "action_l2_error": l2_error,
                    "reward": final_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "eef_pos": [float(x) for x in eef_pos.tolist()],
                    "obj_in_hand": final_obj_in_hand,
                    "target_name": target_name,
                    "target_in_hand": bool(success_now),
                }
                steps_file.write(json.dumps(step_record) + "\n")
                if debug and (steps_run < 5 or steps_run % 10 == 0 or terminated or truncated):
                    print(
                        f"[Eval][STEP] i={steps_run} action_norm={step_record['pred_action_norm']:.4f} "
                        f"l2={l2_error} in_hand={final_obj_in_hand}",
                        flush=True,
                    )

                steps_run += 1
                if terminated or truncated:
                    break

        mean_l2 = None if not action_l2_errors else float(np.mean(action_l2_errors))
        summary = {
            "episode_idx": episode_idx,
            "scene_name": scene_model,
            "dataset_name": dataset_name,
            "steps_run": steps_run,
            "gt_steps": int(len(gt_actions)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_reward": final_reward,
            "target_object_name": metadata.get("target_object_name"),
            "final_obj_in_hand": final_obj_in_hand,
            "target_in_hand": final_obj_in_hand == metadata.get("target_object_name"),
            "mean_action_l2_error": mean_l2,
            "steps_log_path": str(steps_path),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Eval] Wrote {summary_path}", flush=True)
        return summary
    except Exception as exc:
        err_payload = {
            "episode_idx": episode_idx,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        err_path = episode_out_dir / "error.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(err_payload, f, indent=2)
        print(f"[Eval][ERROR] Episode {episode_idx} failed: {type(exc).__name__}: {exc}", flush=True)
        print(f"[Eval][ERROR] Wrote traceback to {err_path}", flush=True)
        raise
    finally:
        og.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate diffusion policy in OmniGibson scenes from dataset metadata.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to saved LeRobot dataset root")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/behavior/workspace/BEHAVIOR-1K/runs/diffusion_policy_100/checkpoints/epoch_0020.pt"),
        help="Path to diffusion policy checkpoint .pt file",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("/home/behavior/workspace/BEHAVIOR-1K/OmniGibson/omnigibson/configs/fetch_vid2scene.yaml"),
        help="Path to OmniGibson env config yaml",
    )
    parser.add_argument("--episode-idx", type=int, default=None, help="Single episode index to evaluate")
    parser.add_argument(
        "--episodes-file",
        type=Path,
        default=None,
        help="Path to text/csv file containing episode indices",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Max rollout steps (default: GT episode length)")
    parser.add_argument("--num-diffusion-iters", type=int, default=100, help="Reverse diffusion iterations")
    parser.add_argument("--obs-horizon", type=int, default=None, help="Override checkpoint obs_horizon")
    parser.add_argument("--pred-horizon", type=int, default=None, help="Override checkpoint pred_horizon")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--debug", action="store_true", help="Print per-step debug logs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for eval logs (default: outputs/policy_eval/<timestamp>)",
    )
    parser.add_argument(
        "--no-sticky-grasp",
        action="store_true",
        help="Disable forcing sticky grasp mode for replay/eval environment",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    env_config_path = args.env_config.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not env_config_path.exists():
        raise FileNotFoundError(f"Env config not found: {env_config_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "policy_eval" / stamp
    else:
        output_dir = args.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_episode_indices(episode_idx=args.episode_idx, episodes_file=args.episodes_file)
    nets, action_normalizer, ckpt_args = _load_policy(checkpoint_path=checkpoint_path, device=device)
    ckpt_obs_horizon = int(ckpt_args.get("obs_horizon", 2))
    if args.obs_horizon is not None and int(args.obs_horizon) != ckpt_obs_horizon:
        raise ValueError(
            f"--obs-horizon={args.obs_horizon} does not match checkpoint obs_horizon={ckpt_obs_horizon}. "
            "Use the checkpoint horizon for this model."
        )
    obs_horizon = ckpt_obs_horizon
    pred_horizon = int(args.pred_horizon if args.pred_horizon is not None else ckpt_args.get("pred_horizon", 16))

    run_summaries = []
    for epi in episodes:
        try:
            summary = evaluate_episode(
                dataset_root=dataset_root,
                episode_idx=epi,
                env_config_path=env_config_path,
                sticky_grasp=not args.no_sticky_grasp,
                nets=nets,
                action_normalizer=action_normalizer,
                obs_horizon=obs_horizon,
                pred_horizon=pred_horizon,
                num_diffusion_iters=args.num_diffusion_iters,
                max_steps=args.max_steps,
                output_dir=output_dir,
                debug=args.debug,
                device=device,
            )
            run_summaries.append(summary)
        except Exception:
            print(f"[Eval][FATAL] Episode {epi} crashed.", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    run_summary_path = output_dir / "run_summary.json"
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summaries, f, indent=2)
    print(f"[Eval] Completed {len(run_summaries)} episode(s). Run summary: {run_summary_path}", flush=True)


if __name__ == "__main__":
    main()

