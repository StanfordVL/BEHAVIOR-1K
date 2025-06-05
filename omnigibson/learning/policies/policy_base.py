import omnigibson as og
import torch
from abc import ABC, abstractmethod
from omnigibson.learning.eval import Evaluator
from typing import Any
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler


class BasePolicy(LightningModule, ABC):
    """
    Base class for policies that is used for training and rollout
    """

    def __init__(self, eval: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # require evaluator for online testing
        self.eval_config = eval
        OmegaConf.resolve(self.eval_config)
        self.evaluator = None

    @abstractmethod
    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the policy.
        This is used for inference and should return the action.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the policy
        """
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Get optimizers, which are subsequently used to train.
        """
        raise NotImplementedError

    @abstractmethod
    def policy_training_step(self, batch, batch_idx) -> Any:
        raise NotImplementedError

    @abstractmethod
    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        raise NotImplementedError

    @abstractmethod
    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        """
        Process the observation data before passing it to the policy for training/eval/prediction.
        Args:
            data_batch (dict): Observation data dictionary.
            extract_action (bool): Whether to extract action from the batch.
                If True, the action will be extracted from the batch and returned along with the processed observation.
                If False, only the processed observation will be returned.
        Returns:
            Any: Processed data that can be used by the policy.
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        loss, log_dict, batch_size = self.policy_training_step(*args, **kwargs)
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict["train/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        loss, log_dict, real_batch_size = self.policy_evaluation_step(*args, **kwargs)
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        log_dict["val/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=real_batch_size,
        )
        return log_dict

    def test_step(self, *args, **kwargs):
        self.evaluator.reset()
        self.evaluator.env._current_episode = 0
        done = False
        while not done:
            terminated, truncated = self.evaluator.step()
            if terminated:
                self.evaluator.env.reset()
            if truncated:
                done = True
        results = {"eval/success_rate": self.evaluator.n_success_trials / self.evaluator.n_trials}
        return results

    def on_test_start(self):
        # evaluator for online evaluation should only be created once
        if self.evaluator is None:
            self.evaluator = self.create_evaluator()
        assert self.evaluator is not None, "evaluator is not created!"

    def on_test_end(self):
        og.shutdown()

    def on_validation_start(self):
        if self.eval_config.eval_on_validation:
            self.on_test_start()

    def on_validation_end(self):
        if self.eval_config.eval_on_validation and not self.trainer.sanity_checking:
            self.test_step(None, None)

    def create_evaluator(self):
        """
        Create a evaluator parameter config containing vectorized distributed envs.
        This will be used to spawn the OmniGibson environments for online evaluation in self.imitation_evaluation_step()
        """
        # update parameters with policy cfg file
        evaluator = Evaluator(self.eval_config)
        # set the policy for the evaluator
        evaluator.policy = self
        return evaluator
