from omegaconf import MISSING
from dataclasses import dataclass
from typing import List
@dataclass
class Normalize:
    mean: List[float] = MISSING
    std: List[float] = MISSING

@dataclass
class DataAugmentation:
    random_crop: bool = MISSING
    random_horizontal_flip: bool = MISSING
    normalize: Normalize = MISSING

@dataclass
class LRScheduler:
    name: str = MISSING
    T_max: int = MISSING

@dataclass
class Optimizer:
    name: str = MISSING
    lr_scheduler: LRScheduler = MISSING

@dataclass
class Hardware:
    num_workers: int = MISSING
    pin_memory: bool = MISSING

@dataclass
class Logging:
    save_frequency: int = MISSING
    log_frequency: int = MISSING

@dataclass
class Training:
    epochs: int = MISSING
    batch_size: int = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    momentum: float = MISSING

@dataclass
class Model:
    name: str = MISSING
    pretrained: bool = MISSING

@dataclass
class Dataset:
    name: str = MISSING
    num_classes: int = MISSING
    train_split: str = MISSING
    val_split: str = MISSING
    data_augmentation: DataAugmentation = MISSING

@dataclass
class Config:
    dataset: Dataset = MISSING
    model: Model = MISSING
    training: Training = MISSING
    optimizer: Optimizer = MISSING
    hardware: Hardware = MISSING
    logging: Logging = MISSING
