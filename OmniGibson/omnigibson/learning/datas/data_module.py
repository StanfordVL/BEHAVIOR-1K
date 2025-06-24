import numpy as np

from omnigibson.learning.datas.dataset import BehaviorDataset
from omnigibson.learning.utils.array_tensor_utils import any_stack, make_recursive_func
from omnigibson.learning.utils.convert_utils import any_to_torch_tensor
from omnigibson.learning.utils.training_utils import sequential_split_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple


class BehaviorDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        val_batch_size: Optional[int],
        val_split_ratio: float,
        dataloader_num_workers: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self._dataloader_num_workers = dataloader_num_workers
        self._val_split_ratio = val_split_ratio
        # store args and kwargs for dataset initialization
        self._args = args
        self._kwargs = kwargs

        self._train_dataset, self._val_dataset = None, None

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            all_dataset = BehaviorDataset(*self._args, **self._kwargs)
            self._train_dataset, self._val_dataset = sequential_split_dataset(
                all_dataset,
                split_portions=[1 - self._val_split_ratio, self._val_split_ratio],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=min(self._batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=_seq_chunk_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            num_workers=min(self._val_batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=_seq_chunk_collate_fn,
        )


def _seq_chunk_collate_fn(sample_list: List[Tuple]) -> dict:
    """
    sample_list: list of (T, ...). PyTorch's native collate_fn can stack all data.
    But here we also add a leading singleton dimension, so it won't break the compatibility with episode data format.
    """
    stacked_list = any_stack(sample_list, dim=0)  # (B, T, ...)
    expanded_list = _nested_np_expand_dims(stacked_list, axis=0)  # (1, B, T, ...)
    # convert to tensor
    return _any_to_torch_tensor(expanded_list)


@make_recursive_func
def _nested_np_expand_dims(x, axis):
    if isinstance(x, np.ndarray):
        return np.expand_dims(x, axis=axis)
    else:
        raise ValueError(f"Input ({type(x)}) must be a numpy array.")


def _any_to_torch_tensor(x):
    if isinstance(x, dict):
        return {k: _any_to_torch_tensor(v) for k, v in x.items()}
    return any_to_torch_tensor(x)
