import torch.nn as nn
import torchvision.models as models

def get_backbone(backbone_name: str) -> nn.Module:
    if backbone_name == "resnet50":
        return models.resnet50(weights="DEFAULT")
    elif backbone_name == "resnet18":
        return models.resnet18(weights="DEFAULT")
    elif backbone_name == "resnet34":
        return models.resnet34(weights="DEFAULT")
    elif backbone_name == "resnet101":
        return models.resnet101(weights="DEFAULT")
    elif backbone_name == "resnet152":
        return models.resnet152(weights="DEFAULT")
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")