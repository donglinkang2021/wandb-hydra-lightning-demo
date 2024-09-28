from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class EnvConfig:
    data_root: str = MISSING
    torch_home: str = MISSING

@dataclass
class ModuleConfig:
    backbone: str = MISSING
    learning_rate: float = MISSING

@dataclass
class DatasetConfig:
    name: str = MISSING
    num_classes: int = MISSING

@dataclass
class HardwareConfig:
    num_workers: int = MISSING
    accelerator: str = MISSING
    devices: list[int] = MISSING
    precision: int = MISSING
    num_nodes: int = MISSING

@dataclass
class TrainConfig:
    batch_size: int = MISSING
    num_epochs: int = MISSING

@dataclass
class WandbConfig:
    project: str = MISSING
    log_model: str = MISSING

@dataclass
class Config:
    env: EnvConfig = MISSING
    module: ModuleConfig = MISSING
    dataset: DatasetConfig = MISSING
    hardware: HardwareConfig = MISSING
    train: TrainConfig = MISSING