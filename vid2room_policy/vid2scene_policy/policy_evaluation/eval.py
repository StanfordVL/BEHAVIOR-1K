#!/usr/bin/env python
"""Evaluate a LeRobot policy on BEHAVIOR-1K OmniGibson environments.

This script integrates LeRobot's policy evaluation with OmniGibson environments
for pick-and-place tasks. Supports any LeRobot-compatible policy.

Usage:
    python -m vid2scene_policy.policy_evaluation.eval \
        --policy.path=outputs/train/diffusion_pusht/checkpoints/final/pretrained_model \
        --episodes_file episodes.json \
        --eval.n_episodes=10 \
        --output_dir eval_results
"""

import argparse
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.io_utils import write_video

from .omnigibson_eval_env import OmniGibsonEvalEnv, EpisodeMetadata, EnvConfig

logger = logging.getLogger(__name__)

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}


@dataclass
class EvalConfig:
    """Configuration for policy evaluation."""
    # Policy
    policy_path: str = ""
    policy_device: str = "cuda"
    use_amp: bool = True

    # Environment
    max_steps: int = 500
    fps: int = 30
    image_height: int = 224
    image_width: int = 224

    # Evaluation
    n_episodes: int = 10
    episodes_file: str | None = None
    seed: int = 42
    max_episodes_rendered: int = 5

    # Output
    output_dir: str = "./eval_results"
    save_videos: bool = True


def load_policy(
    policy_path: str,
    device: str = "cuda",
    dataset_stats: dict | None = None,
) -> tuple[PreTrainedPolicy, Any, Any]:
    """Load any LeRobot policy from checkpoint or hub.

    Args:
        policy_path: Path to policy checkpoint or HuggingFace model ID
        device: Device to load policy on
        dataset_stats: Optional dataset statistics for normalization

    Returns:
        Tuple of (policy, preprocessor, postprocessor)
    """
    from lerobot.policies.factory import get_policy_class, make_policy_config

    policy_path_obj = Path(policy_path)

    # Check if it's a local path or hub model
    if policy_path_obj.exists():
        pretrained_path = str(policy_path_obj.resolve())
        config_path = policy_path_obj / "config.json"
    else:
        # Assume it's a HuggingFace hub model ID
        pretrained_path = policy_path
        config_path = None

    # Load config to determine policy type
    if config_path and config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        policy_type = config_dict.get("type")
        logger.info("Detected policy type: %s", policy_type)
    else:
        # For hub models, we'll need to download config first
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(repo_id=pretrained_path, filename="config.json")
        with open(config_file) as f:
            config_dict = json.load(f)
        policy_type = config_dict.get("type")
        logger.info("Detected policy type from hub: %s", policy_type)

    # Convert feature dicts to PolicyFeature objects
    from lerobot.configs.types import PolicyFeature, FeatureType

    def convert_features(features_dict):
        if not features_dict:
            return None
        result = {}
        for key, feat in features_dict.items():
            if isinstance(feat, dict):
                feat_type = FeatureType(feat["type"])
                feat_shape = tuple(feat["shape"])
                result[key] = PolicyFeature(type=feat_type, shape=feat_shape)
            else:
                result[key] = feat
        return result

    # Filter and convert config dict
    filtered_config = {k: v for k, v in config_dict.items()
                       if k not in ["type", "pretrained_path", "device", "input_features", "output_features"]}

    # Convert features
    input_features = convert_features(config_dict.get("input_features"))
    output_features = convert_features(config_dict.get("output_features"))

    # Create policy config with pretrained_path set
    policy_cfg = make_policy_config(
        policy_type=policy_type,
        pretrained_path=pretrained_path,
        device=device,
        input_features=input_features,
        output_features=output_features,
        **filtered_config
    )

    # Get policy class and load from pretrained
    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=pretrained_path,
        config=policy_cfg,
    )

    # Create preprocessors from pretrained path
    preprocessor_overrides = {
        "device_processor": {"device": device},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides=preprocessor_overrides,
    )

    policy.to(device)
    policy.eval()

    logger.info("Loaded policy: %s", type(policy).__name__)
    logger.info("Policy config: n_obs_steps=%s, n_action_steps=%s",
                getattr(policy.config, 'n_obs_steps', 'N/A'),
                getattr(policy.config, 'n_action_steps', 'N/A'))

    return policy, preprocessor, postprocessor


def rollout_episode(
    env: OmniGibsonEvalEnv,
    policy: PreTrainedPolicy,
    preprocessor: Any,
    postprocessor: Any,
    episode_metadata: EpisodeMetadata | None = None,
    seed: int | None = None,
    return_frames: bool = False,
    device: str = "cuda",
) -> dict:
    """Run a single rollout episode.

    Args:
        env: OmniGibson evaluation environment
        policy: LeRobot policy
        preprocessor: Policy preprocessor
        postprocessor: Policy postprocessor
        episode_metadata: Episode setup metadata
        seed: Random seed
        return_frames: Whether to return rendered frames
        device: Device for policy inference

    Returns:
        Dictionary with rollout results
    """
    if episode_metadata is not None:
        env.set_episode_metadata(episode_metadata)

    policy.reset()
    observation, info = env.reset(seed=seed)

    all_observations = []
    all_actions = []
    all_rewards = []
    frames = [] if return_frames else None

    done = False
    step = 0
    max_steps = env.config.max_steps

    # Build observation history buffer for temporal stacking
    obs_history = {key: [] for key in observation.keys()}
    n_obs_steps = getattr(policy.config, 'n_obs_steps', 1)

    while not done and step < max_steps:
        # Add current observation to history
        for key, value in observation.items():
            obs_history[key].append(value)
            # Keep only last n_obs_steps
            if len(obs_history[key]) > n_obs_steps:
                obs_history[key] = obs_history[key][-n_obs_steps:]

        # Pad history if needed (for early steps)
        policy_obs = {}
        for key, values in obs_history.items():
            padded_values = list(values)
            while len(padded_values) < n_obs_steps:
                padded_values.insert(0, padded_values[0])  # Pad with first observation

            # For n_obs_steps=1, just use the latest observation with batch dimension
            # For n_obs_steps>1, stack observations along temporal dimension
            if n_obs_steps == 1:
                latest = padded_values[-1]
                if "images" in key:
                    # Images: (H, W, C) -> (1, C, H, W)
                    img = np.transpose(latest, (2, 0, 1))  # (C, H, W)
                    policy_obs[key] = torch.from_numpy(img).unsqueeze(0).float()
                else:
                    # State: (state_dim,) -> (1, state_dim)
                    policy_obs[key] = torch.from_numpy(latest).unsqueeze(0).float()
            else:
                # Stack observations for temporal models
                if "images" in key:
                    # Images: (n_obs, H, W, C) -> (1, n_obs, C, H, W)
                    stacked = np.stack(padded_values, axis=0)
                    stacked = np.transpose(stacked, (0, 3, 1, 2))  # (n, C, H, W)
                    policy_obs[key] = torch.from_numpy(stacked).unsqueeze(0).float()
                else:
                    # State: (n_obs, state_dim) -> (1, n_obs, state_dim)
                    stacked = np.stack(padded_values, axis=0)
                    policy_obs[key] = torch.from_numpy(stacked).unsqueeze(0).float()

        # Add task description for Pi0.5 (required for the tokenizer)
        # Use a generic manipulation task description
        # Pi0.5 requires a task description for tokenization
        task_description = "pick_and_place_rs_int"
        policy_obs["task"] = [task_description]  # List for batch dimension

        # Preprocess and run policy
        policy_obs = preprocessor(policy_obs)
        policy_obs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in policy_obs.items()
        }

        with torch.inference_mode():
            action = policy.select_action(policy_obs)

        action = postprocessor(action)

        #print(f"Shape: {action.shape} MODE: {action[0][11]}")

        # Convert to numpy and take first action from chunk if needed
        action_numpy = action.cpu().numpy()
        if action_numpy.ndim == 3:
            action_numpy = action_numpy[0, 0]  # Take first action
        elif action_numpy.ndim == 2:  # (batch, action_dim)
            action_numpy = action_numpy[0]

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        done = terminated or truncated

        all_observations.append(observation)
        all_actions.append(action_numpy)
        all_rewards.append(reward)

        if return_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        step += 1

    return {
        "observations": all_observations,
        "actions": all_actions,
        "rewards": all_rewards,
        "success": info.get("is_success", False),
        "n_steps": step,
        "sum_reward": sum(all_rewards),
        "frames": frames,
    }


def evaluate_policy(
    config: EvalConfig,
    episodes: list[EpisodeMetadata] | None = None,
) -> dict:
    """Evaluate a policy on OmniGibson environment.

    Args:
        config: Evaluation configuration
        episodes: Optional list of episode metadata. If not provided,
                  loads from config.episodes_file

    Returns:
        Dictionary with evaluation results
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos" if config.save_videos else None
    if videos_dir:
        videos_dir.mkdir(parents=True, exist_ok=True)

    # Load episodes
    if episodes is None:
        episodes = _load_episodes(config)

    logger.info("Loaded %d episodes for evaluation", len(episodes))

    # Create environment
    env_config = EnvConfig(
        scene_model=episodes[0].scene_name,
        dataset_name=episodes[0].dataset_name,
        max_steps=config.max_steps,
        fps=config.fps,
        image_height=config.image_height,
        image_width=config.image_width,
    )
    env = OmniGibsonEvalEnv(config=env_config, render_mode="rgb_array")

    # Load policy
    logger.info("Loading policy from: %s", config.policy_path)

    # Try to load dataset stats if available
    dataset_stats = _get_default_stats()

    policy, preprocessor, postprocessor = load_policy(
        config.policy_path,
        device=config.policy_device,
        dataset_stats=dataset_stats,
    )
    logger.info("Policy loaded successfully")

    # Run evaluation
    start_time = time.time()
    results = []
    n_success = 0
    n_episodes_rendered = 0

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    amp_context = (
        torch.autocast(device_type=config.policy_device.split(":")[0])
        if config.use_amp else nullcontext()
    )

    n_eval = min(config.n_episodes, len(episodes)) if episodes[0] is not None else config.n_episodes
    progress = trange(n_eval, desc="Evaluating")

    for i in progress:
        episode_meta = episodes[i] if i < len(episodes) and episodes[i] is not None else None

        should_render = (
            config.save_videos and
            n_episodes_rendered < config.max_episodes_rendered
        )

        with amp_context:
            result = rollout_episode(
                env=env,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                episode_metadata=episode_meta,
                seed=config.seed + i,
                return_frames=should_render,
                device=config.policy_device,
            )

        results.append({
            "episode_idx": i,
            "success": result["success"],
            "n_steps": result["n_steps"],
            "sum_reward": result["sum_reward"],
            "metadata": episode_meta.__dict__ if episode_meta else None,
        })

        if result["success"]:
            n_success += 1

        # Save video
        if should_render and result["frames"]:
            video_path = videos_dir / f"episode_{i}.mp4"
            write_video(
                str(video_path),
                np.stack(result["frames"]),
                config.fps,
            )
            n_episodes_rendered += 1
            logger.info("Saved video: %s", video_path)

        progress.set_postfix({
            "success_rate": f"{n_success / (i + 1) * 100:.1f}%"
        })

    # Compute final metrics
    eval_time = time.time() - start_time
    success_rate = n_success / n_eval * 100
    avg_steps = np.mean([r["n_steps"] for r in results])
    avg_reward = np.mean([r["sum_reward"] for r in results])

    summary = {
        "n_episodes": n_eval,
        "success_rate": success_rate,
        "n_success": n_success,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "eval_time_s": eval_time,
        "eval_time_per_episode_s": eval_time / n_eval,
        "config": {
            "policy_path": config.policy_path,
            "max_steps": config.max_steps,
            "seed": config.seed,
        },
    }

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "summary": summary,
            "episodes": results,
        }, f, indent=2)

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info("  Episodes: %d", n_eval)
    logger.info("  Success Rate: %.1f%%", success_rate)
    logger.info("  Avg Steps: %.1f", avg_steps)
    logger.info("  Avg Reward: %.3f", avg_reward)
    logger.info("  Total Time: %.1fs", eval_time)
    logger.info("Results saved to: %s", results_path)

    env.close()

    return {
        "summary": summary,
        "episodes": results,
    }


def _load_episodes(config: EvalConfig) -> list[EpisodeMetadata | None]:
    """Load episode metadata from config."""
    episodes = []

    if config.episodes_file:
        episodes_path = Path(config.episodes_file)
        if episodes_path.exists():
            with open(episodes_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                episodes = [EpisodeMetadata.from_dict(d) for d in data]
            else:
                episodes = [EpisodeMetadata.from_dict(data)]
            logger.info("Loaded %d episodes from %s", len(episodes), episodes_path)
        else:
            logger.warning("Episodes file not found: %s", episodes_path)

    assert episodes, "Could not find episodes file: %s" % config.episodes_file

    return episodes


def _get_default_stats() -> dict:
    """Get default dataset statistics for normalization."""
    stats = {}

    for camera in ["wrist", "head", "wrist_seg_depth", "head_seg_depth"]:
        key = f"observation.images.{camera}"
        stats[key] = {
            "mean": torch.tensor(IMAGENET_STATS["mean"], dtype=torch.float32),
            "std": torch.tensor(IMAGENET_STATS["std"], dtype=torch.float32),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeRobot policy on OmniGibson")

    # Policy arguments
    parser.add_argument("--policy_path", type=str, required=True,
                       help="Path to policy checkpoint or HuggingFace model ID")
    parser.add_argument("--policy_device", type=str, default="cuda",
                       help="Device for policy inference")
    parser.add_argument("--use_amp", default=False,
                        type=bool,
                       help="Use automatic mixed precision")

    # Environment arguments
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Maximum steps per episode")
    parser.add_argument("--fps", type=int, default=30,
                       help="Environment FPS")

    # Evaluation arguments
    parser.add_argument("--n_episodes", type=int, default=1,
                       help="Number of episodes to evaluate")
    parser.add_argument("--episodes_file", type=str, default=None,
                       help="JSON file with episode metadata")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max_episodes_rendered", type=int, default=5,
                       help="Maximum number of episodes to render as videos")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--no_save_videos", action="store_true",
                       help="Disable video saving")

    args = parser.parse_args()

    config = EvalConfig(
        policy_path=args.policy_path,
        policy_device=args.policy_device,
        use_amp=args.use_amp,
        max_steps=args.max_steps,
        fps=args.fps,
        n_episodes=args.n_episodes,
        episodes_file=args.episodes_file,
        seed=args.seed,
        max_episodes_rendered=args.max_episodes_rendered,
        output_dir=args.output_dir,
        save_videos=not args.no_save_videos,
    )

    evaluate_policy(config)


if __name__ == "__main__":
    main()
