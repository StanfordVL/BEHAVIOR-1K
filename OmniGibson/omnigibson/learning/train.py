import hydra

from omnigibson.learning.utils.training_utils import seed_everywhere
from omnigibson.learning.utils.config_utils import omegaconf_to_dict
from omnigibson.learning.training.trainer import Trainer


@hydra.main(config_name="base_config", config_path="configs", version_base="1.1")
def main(cfg):
    cfg.seed = seed_everywhere(cfg.seed)
    trainer_ = Trainer(cfg)
    trainer_.trainer.loggers[-1].log_hyperparams(omegaconf_to_dict(cfg))
    trainer_.fit()


if __name__ == "__main__":
    main()
