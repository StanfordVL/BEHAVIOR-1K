#!/usr/bin/env python
import argparse
import logging
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature


IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}


def get_delta_timestamps(fps: int, n_obs: int = 4, horizon: int = 32) -> dict[str, list[float]]:
    obs_times = [-3.0, -1.0, -0.5, -0.1][:n_obs]
    n_obs_frames = len(obs_times)
    action_times = [i / fps for i in range(1 - n_obs_frames, 1 - n_obs_frames + horizon)]
    return {
        "observation.state": obs_times,
        "observation.images.wrist": obs_times,
        "observation.images.head": obs_times,
        "observation.images.wrist_seg_depth": obs_times,
        "observation.images.head_seg_depth": obs_times,
        "action": action_times,
    }


def make_dataset_with_custom_timestamps(
    repo_id: str,
    root: str,
    episodes: list[int] | None,
    fps: int,
    n_obs: int = 4,
    horizon: int = 32,
    tolerance_s: float = 0.1,
) -> LeRobotDataset:
    delta_timestamps = get_delta_timestamps(fps, n_obs, horizon)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        tolerance_s=tolerance_s,
    )
    for key in dataset.meta.camera_keys:
        for stats_type, stats in IMAGENET_STATS.items():
            dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    return dataset


def create_policy_config(ds_meta: LeRobotDatasetMetadata) -> DiffusionConfig:
    input_features = {}
    output_features = {}

    for key, ft in ds_meta.features.items():
        if key == "observation.state":
            input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=tuple(ft["shape"]))
        elif key.startswith("observation.images."):
            shape = (ft["shape"][2], ft["shape"][0], ft["shape"][1])
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
        elif key == "action":
            output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(11,))

    config = DiffusionConfig(
        n_obs_steps=4,
        horizon=32,
        n_action_steps=8,
        input_features=input_features,
        output_features=output_features,
        vision_backbone="resnet18",
        crop_shape=(84, 84),
        crop_is_random=True,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        use_separate_rgb_encoder_per_camera=False,
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
        use_film_scale_modulation=True,
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        num_inference_steps=None,
        do_mask_loss_for_padding=False,
        drop_n_last_frames=23,
    )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="lerobot_diffusion")
    parser.add_argument("--train_episodes", type=int, default=10)
    parser.add_argument("--val_episodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    logging.getLogger().handlers[0].flush = lambda: None
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset_path)
    repo_id = dataset_path.name

    ds_meta = LeRobotDatasetMetadata(repo_id, root=str(dataset_path.resolve()))
    fps = ds_meta.fps
    logging.info(f"Dataset: {repo_id}, FPS: {fps}, Total episodes: {ds_meta.total_episodes}")

    train_episodes = list(range(args.train_episodes))
    val_episodes = list(range(args.train_episodes, args.train_episodes + args.val_episodes))
    logging.info(f"Train episodes: {train_episodes}, Val episodes: {val_episodes}")

    logging.info("Loading train dataset...")
    train_dataset = make_dataset_with_custom_timestamps(
        repo_id=repo_id,
        root=str(dataset_path.resolve()),
        episodes=train_episodes,
        fps=fps,
        tolerance_s=0.1,
    )
    logging.info(f"Train dataset: {len(train_dataset)} frames")

    logging.info("Loading val dataset...")
    val_dataset = make_dataset_with_custom_timestamps(
        repo_id=repo_id,
        root=str(dataset_path.resolve()),
        episodes=val_episodes,
        fps=fps,
        tolerance_s=0.1,
    )
    logging.info(f"Val dataset: {len(val_dataset)} frames")

    logging.info("Creating policy...")
    config = create_policy_config(ds_meta)

    policy = make_policy(
        cfg=config,
        ds_meta=train_dataset.meta,
    )
    policy.to(args.device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=None,
        dataset_stats=train_dataset.meta.stats,
    )

    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"Policy params: {num_params:,} total, {num_trainable:,} trainable")

    drop_n_last_frames = config.drop_n_last_frames
    train_sampler = EpisodeAwareSampler(
        train_dataset.meta.episodes["dataset_from_index"],
        train_dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=train_dataset.episodes,
        drop_n_last_frames=drop_n_last_frames,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

    logging.info(f"Starting training for {args.steps} steps")

    policy.train()
    train_iter = iter(train_loader)
    step = 0
    running_loss = 0.0
    start_time = time.time()

    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = preprocessor(batch)
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()
        loss, _ = policy.forward(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % args.log_freq == 0:
            avg_loss = running_loss / args.log_freq
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            logging.info(
                f"Step {step}/{args.steps} | Loss: {avg_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | Grad: {grad_norm:.3f} | "
                f"Speed: {steps_per_sec:.2f} steps/s"
            )
            running_loss = 0.0

        if step % args.eval_freq == 0:
            policy.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = preprocessor(val_batch)
                    val_batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                    loss, _ = policy.forward(val_batch)
                    val_loss += loss.item()
                    val_count += 1
                    if val_count >= 10:
                        break
            val_loss /= max(val_count, 1)
            logging.info(f"Step {step} | Val Loss: {val_loss:.4f}")
            policy.train()

        if step % args.save_freq == 0:
            ckpt_path = output_dir / f"{args.run_name}_step{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "args": vars(args),
            }, ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")

    final_path = output_dir / f"{args.run_name}_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": policy.state_dict(),
        "config": config,
        "args": vars(args),
    }, final_path)
    logging.info(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
