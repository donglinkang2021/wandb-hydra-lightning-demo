import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import hydra
import wandb
from pathlib import Path
from omegaconf import OmegaConf, DictConfig # use DictConfig to type default hint
from src.config.transferlearning_cifar10 import Config
from src.module.transferlearning import ImagenetTransferLearning
from src.dataset.cifar10 import CIFAR10DataModule

def set_env(cfg: Config) -> None:
    DATA_ROOT = cfg.env.data_root
    TORCHHOME = cfg.env.torch_home
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(TORCHHOME).mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = TORCHHOME
    os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(config_path="configs", config_name="default", version_base=None)
def my_app(cfg: Config) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # return
    set_env(cfg)

    dm = CIFAR10DataModule(
        data_dir = cfg.env.data_root, 
        batch_size = cfg.train.batch_size,
        num_workers = cfg.hardware.num_workers
    )

    model = ImagenetTransferLearning(
        backbone = cfg.module.backbone,
        num_classes = cfg.dataset.num_classes, 
        lr = cfg.module.learning_rate
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        log_model=cfg.wandb.log_model
    )

    trainer = L.Trainer(
        logger=wandb_logger,
        accelerator = cfg.hardware.accelerator,
        devices = cfg.hardware.devices,
        precision = cfg.hardware.precision,
        num_nodes = cfg.hardware.num_nodes,
        max_epochs = cfg.train.num_epochs
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)

    wandb.finish()


if __name__ == "__main__":
    my_app()