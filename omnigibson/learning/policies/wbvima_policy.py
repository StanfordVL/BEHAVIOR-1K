try:
    from brs_algo.optim import CosineScheduleFunction
except ImportError:
    raise ImportError("Please install brs_algo to run WB-VIMA")

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque, defaultdict
from hydra.utils import instantiate
from omnigibson.learning.datas.dataset import ACTION_QPOS_INDICES, PROPRIO_BASED_VEL_INDICES, PROPRIO_QPOS_INDICES
from omnigibson.learning.policies.policy_base import BasePolicy
from omnigibson.learning.utils.array_tensor_utils import any_concat, any_slice, get_batch_size
from omnigibson.learning.utils.pcd_utils import process_fused_point_cloud
from omnigibson.learning.utils.functional_utils import unstack_sequence_fields
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from typing import Any, List, Optional


class WBVIMA(BasePolicy):
    def __init__(
        self,
        # ====== Base class ======
        eval: DictConfig,
        # ====== policy model ======
        policy: DictConfig,
        action_prediction_horizon: int,
        obs_window_size: int = 1,
        # ====== learning ======
        lr: float = 1e-5,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: float = 5e-6,
        lr_layer_decay: float = 1.0,
        weight_decay: float = 0.0,
        action_keys: List[str] = list(),
        loss_on_latest_obs_only: bool = False,
    ) -> None:
        super().__init__(eval=eval)
        self.policy = instantiate(policy)
        self._action_keys = action_keys
        self.action_prediction_horizon = action_prediction_horizon
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.weight_decay = weight_decay
        self.loss_on_latest_obs_only = loss_on_latest_obs_only

        self.obs_window_size = obs_window_size
        self._obs_history = deque(maxlen=obs_window_size)
        self._action_traj_pred = None
        self._action_idx = 0

    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        obs = self.process_data(data_batch=obs, extract_action=False)
        if len(self._obs_history) == 0:
            for _ in range(self.obs_window_size):
                self._obs_history.append(obs)
        else:
            self._obs_history.append(obs)
        obs = any_concat(self._obs_history, dim=1)

        need_inference = self._action_idx % self.action_prediction_horizon == 0
        if need_inference:
            self._action_traj_pred = self.policy.act(obs)  # dict of (B = 1, T_A, ...)
            self._action_traj_pred = {
                k: v[0].detach().cpu() for k, v in self._action_traj_pred.items()
            }  # dict of (T_A, ...)
            self._action_idx = 0
        action = any_slice(self._action_traj_pred, np.s_[self._action_idx])
        self._action_idx += 1
        return torch.cat(list(action.values()))

    def reset(self) -> None:
        pass

    def policy_training_step(self, batch, batch_idx) -> Any:
        batch = self.process_data(data_batch=batch, extract_action=True)
        B = get_batch_size(
            any_slice(batch["action"], np.s_[0]),
            strict=True,
        )
        # obs data is dict of (N_chunks, B, window_size, ...)
        # action chunks is (N_chunks, B, window_size, action_prediction_horizon, A)
        # we loop over chunk dim
        main_data = unstack_sequence_fields(batch, batch_size=get_batch_size(batch, strict=True))
        all_loss, all_mask_sum = [], 0
        for i, main_data_chunk in enumerate(main_data):
            # get padding mask
            pad_mask = main_data_chunk.pop("pad_mask")  # (B, window_size, L_pred_horizon)
            target_action = main_data_chunk.pop("action")  # (B, window_size, L_pred_horizon, A)
            gt_action = torch.cat([target_action[k] for k in self._action_keys], dim=-1)
            transformer_output = self.policy(
                main_data_chunk
            )  # (B, L, E), where L is interleaved time and modality tokens
            loss = self.policy.compute_loss(
                transformer_output=transformer_output,
                gt_action=gt_action,
            )  # (B, T_obs, T_act)
            if self.loss_on_latest_obs_only:
                mask = torch.zeros_like(pad_mask)
                mask[:, -1] = 1
                pad_mask = pad_mask * mask
            loss = loss * pad_mask
            all_loss.append(loss)
            all_mask_sum += pad_mask.sum()
        action_loss = torch.sum(torch.stack(all_loss)) / all_mask_sum
        # sum over action_prediction_horizon dim instead of avg
        action_loss = action_loss * self.action_prediction_horizon
        log_dict = {"diffusion_loss": action_loss}
        loss = action_loss
        return loss, log_dict, B

    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        """
        Will denoise as if it is in rollout
        but no env interaction
        """
        batch = self.process_data(data_batch=batch, extract_action=True)
        B = get_batch_size(
            any_slice(batch["action"], np.s_[0]),
            strict=True,
        )
        # obs data is dict of (N_chunks, B, window_size, ...)
        # action chunks is (N_chunks, B, window_size, action_prediction_horizon, A)
        # we loop over chunk dim
        main_data = unstack_sequence_fields(batch, batch_size=get_batch_size(batch, strict=True))
        all_l1, all_mask_sum = defaultdict(list), 0
        for i, main_data_chunk in enumerate(main_data):
            # get padding mask
            pad_mask = main_data_chunk.pop("pad_mask")  # (B, window_size, L_pred_horizon)
            target_action = main_data_chunk.pop("action")  # (B, window_size, L_pred_horizon, A)
            transformer_output = self.policy(
                main_data_chunk
            )  # (B, L, E), where L is interleaved time and modality tokens
            pred_actions = self.policy.inference(
                transformer_output=transformer_output,
                return_last_timestep_only=False,
            )  # dict of (B, window_size, L_pred_horizon, A)
            for action_k in pred_actions:
                pred = pred_actions[action_k]
                gt = target_action[action_k]
                l1 = F.l1_loss(pred, gt, reduction="none")  # (B, window_size, L_pred_horizon, A)
                # sum over action dim
                l1 = l1.sum(dim=-1).reshape(pad_mask.shape)  # (B, window_size, L_pred_horizon)
                if self.loss_on_latest_obs_only:
                    mask = torch.zeros_like(pad_mask)
                    mask[:, -1] = 1
                    pad_mask = pad_mask * mask
                all_l1[action_k].append(l1 * pad_mask)
            all_mask_sum += pad_mask.sum()
        # avg on chunks dim, batch dim, and obs window dim so we can compare under different training settings
        all_loss = {
            f"l1_{k}": torch.sum(torch.stack(v)) / all_mask_sum * self.action_prediction_horizon
            for k, v in all_l1.items()
        }
        summed_l1 = sum(all_loss.values())
        all_loss["l1"] = summed_l1
        return summed_l1, all_loss, B

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_groups = self.policy.get_optimizer_groups(
            weight_decay=self.weight_decay,
            lr_layer_decay=self.lr_layer_decay,
            lr_scale=1.0,
        )

        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.use_cosine_lr:
            scheduler_kwargs = dict(
                base_value=1.0,  # anneal from the original LR value
                final_value=self.lr_cosine_min / self.lr,
                epochs=self.lr_cosine_steps,
                warmup_start_value=self.lr_cosine_min / self.lr,
                warmup_epochs=self.lr_warmup_steps,
                steps_per_epoch=1,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=CosineScheduleFunction(**scheduler_kwargs),
            )
            return ([optimizer], [{"scheduler": scheduler, "interval": "step"}])

        return optimizer

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        # process observation data
        proprio = data_batch["obs"]["robot_r1::proprio"]
        if proprio.ndim == 1:
            # if proprio is 1D, we need to expand it to 3D
            proprio = proprio[None, None, :].to(self.device)
        if "robot_r1::fused_pcd" in data_batch["obs"]:
            fused_pcd = data_batch["obs"]["robot_r1::fused_pcd"]
        else:
            fused_pcd = process_fused_point_cloud(data_batch["obs"])
        # if fused_pcd is 1D, we need to expand it to 3D
        if fused_pcd.ndim == 2:
            fused_pcd = torch.from_numpy(fused_pcd[None, None, :]).to(self.device)
        data = {
            "pointcloud": {
                "rgb": fused_pcd[..., :3],
                "xyz": fused_pcd[..., 3:],
            },
            "qpos": {key: proprio[..., PROPRIO_QPOS_INDICES[key]] for key in PROPRIO_QPOS_INDICES},
            "odom": {"base_velocity": proprio[..., PROPRIO_BASED_VEL_INDICES]},
        }
        if extract_action:
            # extract action from data_batch
            data.update(
                {
                    "action": {
                        key: data_batch["action_chunks"][..., ACTION_QPOS_INDICES[key]] for key in ACTION_QPOS_INDICES
                    },
                    "pad_mask": data_batch["action_chunk_masks"],
                }
            )
        else:
            # remove action from data_batch
            data.pop("action", None)
            data.pop("pad_mask", None)
        return data
