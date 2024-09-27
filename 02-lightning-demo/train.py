import torch
import lightning as L
from model import ImagenetTransferLearning
from dataset import CIFAR10DataModule
import config
import os
from pathlib import Path

# set data root and cache dir
DATA_ROOT = "/root/autodl-tmp/data"
Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
outdir = "/root/autodl-tmp/.cache/torch"
Path(outdir).mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = outdir

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    # dataset
    dm = CIFAR10DataModule(
        data_dir=DATA_ROOT, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS
    )

    # model
    model = ImagenetTransferLearning(
        num_target_classes=config.NUM_CLASSES, 
        lr=config.LEARNING_RATE
    )

    # training
    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        num_nodes=config.NUM_NODES,
        max_epochs=config.NUM_EPOCHS
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)