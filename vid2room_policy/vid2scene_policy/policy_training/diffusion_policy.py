import argparse
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torchvision
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vid2scene_policy.policy_training.data_loader import PickDataset


def get_resnet(name: str, weights=None, in_channels: int = 3, **kwargs) -> nn.Module:
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    if in_channels != 3:
        resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=False,
        )
    resnet.fc = nn.Identity()
    return resnet


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    keys = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, key in keys:
        parent_module = (
            root_module if not parent else root_module.get_submodule(".".join(parent))
        )
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(key)] = func(parent_module[int(key)])
        else:
            setattr(parent_module, key, func(getattr(parent_module, key)))
    return root_module


def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    return replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features // features_per_group),
            num_channels=x.num_features,
        ),
    )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ConvBlock1D(nn.Module):
    """Conv1d → GroupNorm → SiLU with an optional conditioning injection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, cond_dim: int = 0):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(max(1, out_ch // 16), out_ch)
        self.act = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2) if cond_dim > 0 else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(self.conv(x))
        if self.cond_proj is not None and cond is not None:
            scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
            x = x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.act(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim: int, global_cond_dim: int, hidden_dim: int = 512):
        super().__init__()
        H = hidden_dim

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(H),
            nn.Linear(H, H * 2),
            nn.SiLU(),
            nn.Linear(H * 2, H),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(global_cond_dim, H * 2),
            nn.SiLU(),
            nn.Linear(H * 2, H),
        )

        cond_dim = H
        self.in_proj = nn.Conv1d(input_dim, H, 1)

        self.enc1 = ConvBlock1D(H,     H,     3, cond_dim)
        self.enc2 = ConvBlock1D(H,     H * 2, 3, cond_dim)
        self.enc3 = ConvBlock1D(H * 2, H * 4, 3, cond_dim)

        self.down1 = nn.Conv1d(H,     H,     4, stride=2, padding=1)
        self.down2 = nn.Conv1d(H * 2, H * 2, 4, stride=2, padding=1)
        self.down3 = nn.Conv1d(H * 4, H * 4, 4, stride=2, padding=1)

        self.bot1 = ConvBlock1D(H * 4, H * 4, 3, cond_dim)
        self.bot2 = ConvBlock1D(H * 4, H * 4, 3, cond_dim)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(H * 4, H * 2, 3, padding=1),
        )
        self.dec3 = ConvBlock1D(H * 6, H * 2, 3, cond_dim)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(H * 2, H, 3, padding=1),
        )
        self.dec2 = ConvBlock1D(H * 3, H, 3, cond_dim)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(H, H, 3, padding=1),
        )
        self.dec1 = ConvBlock1D(H * 2, H, 3, cond_dim)

        self.out_proj = nn.Conv1d(H, input_dim, 1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.time_embed(timestep) + self.cond_proj(global_cond)
        x = self.in_proj(sample.permute(0, 2, 1))
        s1 = self.enc1(x, cond)
        x = self.down1(s1)
        s2 = self.enc2(x, cond)
        x = self.down2(s2)
        s3 = self.enc3(x, cond)
        x = self.down3(s3)
        x = self.bot1(x, cond)
        x = self.bot2(x, cond)
        x = self.up3(x)
        x = self.dec3(torch.cat([x, s3[..., :x.shape[-1]]], dim=1), cond)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, s2[..., :x.shape[-1]]], dim=1), cond)
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1[..., :x.shape[-1]]], dim=1), cond)
        out = self.out_proj(x)
        return out.permute(0, 2, 1)


class ActionNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def from_dataset(cls, dataset, max_samples: int = 50_000) -> "ActionNormalizer":
        """Prefer dataset.compute_action_stats() when available (no video load)."""
        if hasattr(dataset, "compute_action_stats"):
            print("Computing action normalization statistics (from action column only) …")
            mean, std = dataset.compute_action_stats(max_samples=max_samples)
            print(f"  action mean: {mean.tolist()}")
            print(f"  action std : {std.tolist()}")
            return cls(mean, std)
        print("Computing action normalization statistics …")
        all_actions = []
        step = max(1, len(dataset) // max_samples)
        for i in range(0, len(dataset), step):
            sample = dataset[i]
            mask = sample["action_mask"].bool()
            all_actions.append(sample["action"][mask, :-1])
        all_actions = torch.cat(all_actions, dim=0)
        mean = all_actions.mean(0)
        std = all_actions.std(0).clamp(min=1e-5)
        print(f"  action mean: {mean.tolist()}")
        print(f"  action std : {std.tolist()}")
        return cls(mean, std)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.mean.to(actions.device)) / self.std.to(actions.device)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.std.to(actions.device) + self.mean.to(actions.device)


def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Training")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers (more = better throughput)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--pred_horizon", type=int, default=16)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=2048, help="Per-worker frame cache size (larger = fewer video decodes)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm (0 = disabled)")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision (faster on GPU)")
    parser.add_argument("--output_dir", type=str, default="runs/diffusion_policy_100")
    parser.add_argument("--save_every_epochs", type=int, default=20, help="0 disables periodic checkpointing")
    parser.add_argument("--overfit_single_batch", action="store_true", help="Train repeatedly on one fixed batch for debugging")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PickDataset(
        dataset_path=args.dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.pred_horizon,
        return_debug_indices=False,
        frame_cache_size=args.cache_size,
    )

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    first_sample = dataset[0]
    print("wrist:", tuple(first_sample["wrist"].shape))
    print("head:", tuple(first_sample["head"].shape))
    print("action:", tuple(first_sample["action"].shape))
    print("action_mask:", tuple(first_sample["action_mask"].shape))

    action_normalizer = ActionNormalizer.from_dataset(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = first_sample["action"].shape[-1] - 1  # skip the last action dim (constant gripper)

    wrist_encoder = replace_bn_with_gn(get_resnet("resnet18", in_channels=6))
    head_encoder = replace_bn_with_gn(get_resnet("resnet18", in_channels=6))
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=(512 + 512) * args.obs_horizon,
        hidden_dim=512,
    )

    nets = nn.ModuleDict({
        "wrist_encoder": wrist_encoder,
        "head_encoder": head_encoder,
        "noise_pred_net": noise_pred_net,
    }).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * args.num_epochs,
    )

    fixed_batch = None
    if args.overfit_single_batch:
        fixed_batch = next(iter(dataloader))
        print("Overfit mode: using one fixed batch for all updates")

    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    def save_checkpoint(epoch: int, mean_loss: float, tag: str):
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{tag}.pt"
        payload = {
            "epoch": epoch,
            "mean_loss": float(mean_loss),
            "args": vars(args),
            "model": nets.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": (scaler.state_dict() if scaler is not None else None),
            "action_normalizer": {
                "mean": action_normalizer.mean.detach().cpu(),
                "std": action_normalizer.std.detach().cpu(),
            },
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    for epoch_idx in range(args.num_epochs):
        epoch_losses = []
        iter_source = [fixed_batch] * len(dataloader) if fixed_batch is not None else dataloader
        batch_bar = tqdm(
            iter_source,
            desc=f"Epoch {epoch_idx + 1}/{args.num_epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for nbatch in batch_bar:
            wrist = nbatch["wrist"][:, :args.obs_horizon].to(device).float()
            head = nbatch["head"][:, :args.obs_horizon].to(device).float()
            actions = nbatch["action"].to(device).float()[..., :-1]  # drop constant gripper dim

            wrist = wrist / 255.0
            head = head / 255.0

            batch_size = actions.shape[0]

            actions = action_normalizer.normalize(actions)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                wrist_feat = nets["wrist_encoder"](wrist.flatten(end_dim=1))
                head_feat = nets["head_encoder"](head.flatten(end_dim=1))
                wrist_feat = wrist_feat.reshape(batch_size, args.obs_horizon, -1)
                head_feat = head_feat.reshape(batch_size, args.obs_horizon, -1)
                obs_cond = torch.cat([wrist_feat, head_feat], dim=-1).flatten(start_dim=1)

                noise = torch.randn_like(actions)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=device,
                ).long()

                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                noise_pred = nets["noise_pred_net"](
                    sample=noisy_actions,
                    timestep=timesteps,
                    global_cond=obs_cond,
                )

                loss = ((noise_pred - noise) ** 2).mean()

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(nets.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(nets.parameters(), args.max_grad_norm)
                optimizer.step()
            lr_scheduler.step()
            ema.step(nets.parameters())

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            batch_bar.set_postfix(loss=f"{loss_value:.4f}")

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        tqdm.write(f"Epoch {epoch_idx + 1}/{args.num_epochs} - mean loss: {mean_loss:.4f}")
        if args.save_every_epochs > 0 and ((epoch_idx + 1) % args.save_every_epochs == 0):
            ckpt_path = save_checkpoint(epoch=epoch_idx + 1, mean_loss=mean_loss, tag=f"epoch_{epoch_idx + 1:04d}")
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

    ckpt_path = save_checkpoint(epoch=args.num_epochs, mean_loss=mean_loss, tag="final")
    tqdm.write(f"Saved final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()