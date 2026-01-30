#!/usr/bin/env python
import argparse
import logging
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}


def get_delta_timestamps(fps: int, n_obs: int = 5, horizon: int = 32) -> dict[str, list[float]]:
    dt = 1 / fps
    def _round(timestamp):
        return round(timestamp / dt) * dt
    obs_times = [-3.0, -1.0, -0.5, -0.1, 0.0]
    obs_times = [_round(t) for t in obs_times][:n_obs]
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
    fps: int,
    episodes: list[int] | None = None,
    n_obs: int = 5,
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
        video_backend="pyav",
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
        n_obs_steps=5,
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
        num_train_timesteps=100000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        num_inference_steps=None,
        do_mask_loss_for_padding=False,
        drop_n_last_frames=23,
        use_amp=True
    )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=96)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from (e.g. output_dir/checkpoints/005000 or output_dir/checkpoints/last)")
    args = parser.parse_args()

    # Create Accelerator for distributed training
    # It automatically detects if running in distributed mode or single-process mode
    # When run with `python script.py`, it works as single-GPU
    # When run with `accelerate launch script.py`, it enables DDP
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    # Determine if this is the main process (for logging and checkpointing)
    # In single-process mode, this is always True
    is_main_process = accelerator.is_main_process

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    logging.getLogger().handlers[0].flush = lambda: None
    
    # Only log on main process to avoid duplicate outputs in DDP mode
    if not is_main_process:
        logging.getLogger().setLevel(logging.WARNING)
    
    torch.manual_seed(args.seed)

    # Use accelerator's device - in single-process mode this respects CUDA_VISIBLE_DEVICES
    # or defaults to cuda:0 if available, otherwise CPU
    device = accelerator.device
    
    if is_main_process:
        logging.info(f"Running with {accelerator.num_processes} process(es) on device: {device}")

    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset_path = Path(args.train_dataset_path)
    val_dataset_path = Path(args.val_dataset_path)
    train_repo_id = train_dataset_path.name
    val_repo_id = val_dataset_path.name

    # Dataset loading synchronization: main process loads first to avoid race conditions
    if is_main_process:
        train_ds_meta = LeRobotDatasetMetadata(train_repo_id, root=str(train_dataset_path.resolve()))
        val_ds_meta = LeRobotDatasetMetadata(val_repo_id, root=str(val_dataset_path.resolve()))
        fps = train_ds_meta.fps
        logging.info(f"Train Dataset: {train_repo_id}, FPS: {fps}, Total episodes: {train_ds_meta.total_episodes}")
        logging.info(f"Val Dataset: {val_repo_id}, FPS: {fps}, Total episodes: {val_ds_meta.total_episodes}")

        logging.info("Loading train dataset...")
        train_dataset = make_dataset_with_custom_timestamps(
            repo_id=train_repo_id,
            root=str(train_dataset_path.resolve()),
            fps=fps,
            tolerance_s=0.1,
        )
        logging.info(f"Train dataset: {len(train_dataset)} frames")

        logging.info("Loading val dataset...")
        val_dataset = make_dataset_with_custom_timestamps(
            repo_id=val_repo_id,
            root=str(val_dataset_path.resolve()),
            fps=fps,
            tolerance_s=0.1,
        )
        logging.info(f"Val dataset: {len(val_dataset)} frames")

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        train_ds_meta = LeRobotDatasetMetadata(train_repo_id, root=str(train_dataset_path.resolve()))
        val_ds_meta = LeRobotDatasetMetadata(val_repo_id, root=str(val_dataset_path.resolve()))
        fps = train_ds_meta.fps
        train_dataset = make_dataset_with_custom_timestamps(
            repo_id=train_repo_id,
            root=str(train_dataset_path.resolve()),
            fps=fps,
            tolerance_s=0.1,
        )
        val_dataset = make_dataset_with_custom_timestamps(
            repo_id=val_repo_id,
            root=str(val_dataset_path.resolve()),
            fps=fps,
            tolerance_s=0.1,
        )

    # Create a minimal TrainPipelineConfig for checkpointing
    dataset_cfg = DatasetConfig(
        repo_id=train_repo_id,
        root=str(train_dataset_path.resolve()),
    )
    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        output_dir=output_dir,
        job_name=args.run_name,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
    )

    if is_main_process:
        logging.info("Creating policy...")
    config = create_policy_config(train_ds_meta)

    # Handle checkpoint resumption - load policy first before creating optimizer
    start_step = 0
    resume_path = None
    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.is_symlink():
            resume_path = resume_path.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {resume_path}")
        
        if is_main_process:
            logging.info(f"Resuming from checkpoint: {resume_path}")
        
        # Load policy weights from checkpoint
        pretrained_path = resume_path / PRETRAINED_MODEL_DIR
        if pretrained_path.exists():
            policy = PreTrainedPolicy.from_pretrained(pretrained_path)
            policy.to(device)
            if is_main_process:
                logging.info(f"Loaded policy weights from {pretrained_path}")
        else:
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")
    else:
        # Create fresh policy
        policy = make_policy(
            cfg=config,
            ds_meta=train_dataset.meta,
        )
        policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=None,
        dataset_stats=train_dataset.meta.stats,
    )

    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if is_main_process:
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
        pin_memory=device.type == "cuda",
        drop_last=True,
        prefetch_factor=2,  # Queue up more batches to hide network latency
        persistent_workers=True,  # Keep workers alive to avoid respawn overhead
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=device.type == "cuda",
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Create optimizer and scheduler with the (possibly resumed) policy's parameters
    optimizer = AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

    # Load optimizer/scheduler state if resuming
    if resume_path is not None:
        start_step, optimizer, scheduler = load_training_state(resume_path, optimizer, scheduler)
        if is_main_process:
            logging.info(f"Resuming from step {start_step}")

    # Prepare everything with accelerator for distributed training
    accelerator.wait_for_everyone()
    policy, optimizer, train_loader, scheduler = accelerator.prepare(
        policy, optimizer, train_loader, scheduler
    )
    dl_iter = cycle(train_loader)

    # Calculate effective batch size for distributed training
    effective_batch_size = args.batch_size * accelerator.num_processes
    if is_main_process:
        logging.info(f"Starting training for {args.steps} steps (from step {start_step})")
        logging.info(f"Effective batch size: {args.batch_size} x {accelerator.num_processes} = {effective_batch_size}")

    # Initialize wandb only on main process
    if is_main_process:
        wandb.init(
            project="vid2room-policies-bigrun2",
            name=args.run_name,
            dir=output_dir,
            config={**vars(args), "effective_batch_size": effective_batch_size, "num_processes": accelerator.num_processes},
            save_code=False,
        )

    policy.train()
    step = start_step
    running_loss = 0.0
    total_data_time = 0.0
    total_model_time = 0.0
    last_log_start_time = time.time()
    last_log_step = step

    if is_main_process:
        logging.info("Starting training loop, fetching first batch...")
    
    batch_start_time = time.time()
    
    for _ in range(step, args.steps):
        batch = next(dl_iter)
        data_time = time.time() - batch_start_time
        total_data_time += data_time
        
        if step == start_step and is_main_process:
            logging.info(f"First batch loaded in {data_time:.2f}s, beginning training...")
        
        model_start_time = time.time()
        batch = preprocessor(batch)

        optimizer.zero_grad()
        
        # Use accelerator's autocast for mixed precision
        with accelerator.autocast():
            loss, _ = policy.forward(batch)

        # Use accelerator's backward for proper gradient handling
        accelerator.backward(loss)

        # Use accelerator's gradient clipping
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), 10.0)
        
        optimizer.step()
        scheduler.step()
        
        model_time = time.time() - model_start_time
        total_model_time += model_time

        running_loss += loss.item()
        step += 1

        if step % args.log_freq == 0 and is_main_process:
            avg_loss = running_loss / args.log_freq
            avg_data_time = total_data_time / args.log_freq
            avg_model_time = total_model_time / args.log_freq
            elapsed = time.time() - last_log_start_time
            step_elapsed = step - last_log_step
            steps_per_sec = step_elapsed / elapsed
            current_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"Step {step}/{args.steps} | Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | Grad: {grad_norm:.3f} | "
                f"Speed: {steps_per_sec:.2f} steps/s | "
                f"Data: {avg_data_time*1000:.1f}ms | Model: {avg_model_time*1000:.1f}ms"
            )
            wandb.log({
                "train/loss": avg_loss,
                "train/learning_rate": current_lr,
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/steps_per_sec": steps_per_sec,
                "train/data_time_ms": avg_data_time * 1000,
                "train/model_time_ms": avg_model_time * 1000,
            }, step=step)
            running_loss = 0.0
            total_data_time = 0.0
            total_model_time = 0.0
            last_log_start_time = time.time()
            last_log_step = step
        
        if step % args.eval_freq == 0:
            policy.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = preprocessor(val_batch)
                    val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                    with accelerator.autocast():
                        loss, _ = accelerator.unwrap_model(policy).forward(val_batch)
                    val_loss += loss.item()
                    val_count += 1
                    if val_count >= 10:
                        break
            val_loss /= max(val_count, 1)
            if is_main_process:
                logging.info(f"Step {step} | Val Loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss}, step=step)
            policy.train()

        if step % args.save_freq == 0:
            if is_main_process:
                checkpoint_dir = get_step_checkpoint_dir(output_dir, args.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=train_cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                logging.info(f"Saved checkpoint: {checkpoint_dir}")
            
            accelerator.wait_for_everyone()

        if step >= args.steps:
            break

        batch_start_time = time.time()

    # Save final checkpoint
    if is_main_process:
        final_checkpoint_dir = get_step_checkpoint_dir(output_dir, args.steps, step)
        save_checkpoint(
            checkpoint_dir=final_checkpoint_dir,
            step=step,
            cfg=train_cfg,
            policy=accelerator.unwrap_model(policy),
            optimizer=optimizer,
            scheduler=scheduler,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        update_last_checkpoint(final_checkpoint_dir)
        logging.info(f"Training complete. Final model: {final_checkpoint_dir}")
        wandb.finish()

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
