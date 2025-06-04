from typing import List
import logging
import time
from copy import deepcopy
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig
import omnigibson.learning.utils.file_utils as FU
import omnigibson.learning.utils.config_utils as CU
import omnigibson.learning.utils.print_utils as PU
from omnigibson.learning.utils.training_utils import load_torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_debug as rank_zero_debug_pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info as rank_zero_info_pl
from pytorch_lightning.callbacks import ModelCheckpoint


__all__ = [
    "Trainer",
    "rank_zero_info",
    "rank_zero_debug",
    "rank_zero_warn",
    "rank_zero_info_pl",
    "rank_zero_debug_pl",
]

logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
logging.getLogger("torch.distributed.nn.jit.instantiator").setLevel(logging.WARNING)
PU.logging_exclude_pattern("root", patterns="*Reducer buckets have been rebuilt in this iteration*")


class Trainer:
    def __init__(self, cfg: DictConfig, eval_only=False):
        """
        Args:
            eval_only: if True, will not save any model dir
        """
        cfg = deepcopy(cfg)
        OmegaConf.set_struct(cfg, False)
        self.cfg = cfg
        CU.register_omegaconf_resolvers()
        run_name = self.generate_run_name(cfg)
        self.run_dir = FU.f_join(cfg.exp_root_dir, run_name)
        self._eval_only = eval_only
        self._resume_mode = None  # 'full state' or 'model only'
        if eval_only:
            rank_zero_info("Eval only, will not save any model dir")
        else:
            if "resume" in cfg and "ckpt_path" in cfg.resume and cfg.resume.ckpt_path:
                cfg.resume.ckpt_path = FU.f_expand(
                    cfg.resume.ckpt_path.replace("_RUN_DIR_", self.run_dir).replace("_RUN_NAME_", run_name)
                )
                self._resume_mode = "full state" if cfg.resume.get("full_state", False) else "model only"
                rank_zero_info(
                    "=" * 80,
                    "=" * 80 + "\n",
                    f"Resume training from {cfg.resume.ckpt_path}",
                    f"\t({self._resume_mode})\n",
                    "=" * 80,
                    "=" * 80,
                    sep="\n",
                    end="\n\n",
                )
                time.sleep(3)
                assert FU.f_exists(cfg.resume.ckpt_path), "resume ckpt_path does not exist"

            rank_zero_print("Run name:", run_name, "\nExp dir:", self.run_dir)
            FU.f_mkdir(self.run_dir)
            FU.f_mkdir(FU.f_join(self.run_dir, "tb"))
            FU.f_mkdir(FU.f_join(self.run_dir, "logs"))
            FU.f_mkdir(FU.f_join(self.run_dir, "ckpt"))
            CU.omegaconf_save(cfg, self.run_dir, "conf.yaml")
            rank_zero_print("Checkpoint cfg:", CU.omegaconf_to_dict(cfg.trainer.checkpoint))
        self.cfg = cfg
        self.run_name = run_name
        self.ckpt_cfg = cfg.trainer.pop("checkpoint")
        self.data_module = self.create_data_module(cfg)
        self._monkey_patch_add_info(self.data_module)
        self.trainer = self.create_trainer(cfg)
        self.module = self.create_module(cfg)
        self.module.data_module = self.data_module
        self._monkey_patch_add_info(self.module)

        if not eval_only and self._resume_mode == "model only":
            ret = self.module.load_state_dict(
                load_torch(cfg.resume.ckpt_path)["state_dict"],
                strict=cfg.resume.strict,
            )
            rank_zero_warn("state_dict load status:", ret)

    def create_module(self, cfg):
        return instantiate(cfg.module, _recursive_=False)

    def create_data_module(self, cfg):
        return instantiate(cfg.data_module)

    def generate_run_name(self, cfg):
        return cfg.run_name + "_" + time.strftime("%Y%m%d-%H%M%S")

    def _monkey_patch_add_info(self, obj):
        """
        Add useful info to module and data_module so they can access directly
        """
        # our own info
        obj.run_config = self.cfg
        obj.run_name = self.run_name
        # add properties from trainer
        for attr in [
            "global_rank",
            "local_rank",
            "world_size",
            "num_nodes",
            "num_processes",
            "node_rank",
            "num_gpus",
            "data_parallel_device_ids",
        ]:
            if hasattr(obj, attr):
                continue
            setattr(
                obj.__class__,
                attr,
                # force capture 'attr'
                property(lambda self, attr=attr: getattr(self.trainer, attr)),
            )

    def create_loggers(self, cfg) -> List[pl_loggers.Logger]:
        if self._eval_only:
            loggers = []
        else:
            loggers = [
                pl_loggers.TensorBoardLogger(self.run_dir, name="tb", version=""),
                pl_loggers.CSVLogger(self.run_dir, name="logs", version=""),
            ]
        if cfg.use_wandb:
            loggers.append(
                pl_loggers.WandbLogger(
                    name=cfg.wandb_run_name,
                    project=cfg.wandb_project,
                    group=cfg.wandb_group,
                    id=self.run_name,
                    save_dir=self.run_dir,
                )
            )
        return loggers

    def create_callbacks(self, cfg) -> List[Callback]:
        ModelCheckpoint.FILE_EXTENSION = ".pth"
        callbacks = []
        # Construct ModelCheckpoint callback
        if isinstance(self.ckpt_cfg, DictConfig):
            ckpt = ModelCheckpoint(dirpath=FU.f_join(self.run_dir, "ckpt"), **self.ckpt_cfg)
            callbacks.append(ckpt)
        else:
            assert isinstance(self.ckpt_cfg, ListConfig)
            for _cfg in self.ckpt_cfg:
                ckpt = ModelCheckpoint(dirpath=FU.f_join(self.run_dir, "ckpt"), **_cfg)
                callbacks.append(ckpt)
        if "callbacks" in cfg.trainer:
            extra_callbacks = [instantiate(callback) for callback in cfg.trainer.pop("callbacks")]
            callbacks.extend(extra_callbacks)
        rank_zero_print("Lightning callbacks:", [c.__class__.__name__ for c in callbacks])
        return callbacks

    def create_trainer(self, cfg) -> pl.Trainer:
        return pl.Trainer(logger=self.create_loggers(cfg), callbacks=self.create_callbacks(cfg), **cfg.trainer)

    def fit(self):
        return self.trainer.fit(
            self.module,
            datamodule=self.data_module,
            ckpt_path=(self.cfg.resume.ckpt_path if self._resume_mode == "full state" else None),
        )

    def validate(self):
        return self.trainer.validate(self.module, datamodule=self.data_module, ckpt_path=None)


@rank_zero_only
def rank_zero_print(*msg, **kwargs):
    PU.pprint_(*msg, **kwargs)


@rank_zero_only
def rank_zero_info(*msg, **kwargs):
    PU.pprint_(
        PU.color_text("[INFO]", color="green", styles=["reverse", "bold"]),
        *msg,
        **kwargs,
    )


@rank_zero_only
def rank_zero_warn(*msg, **kwargs):
    PU.pprint_(
        PU.color_text("[WARN]", color="yellow", styles=["reverse", "bold"]),
        *msg,
        **kwargs,
    )


@rank_zero_only
def rank_zero_debug(*msg, **kwargs):
    if rank_zero_debug.enabled:
        PU.pprint_(PU.color_text("[DEBUG]", color="blue", bg_color="on_grey"), *msg, **kwargs)


rank_zero_debug.enabled = True
