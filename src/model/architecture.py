from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_model(num_classes: int, *, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def transfer_backbone(model: nn.Module, checkpoint_path: Path) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = payload["model_state"]
    destination = model.state_dict()
    mapped = {
        key: value
        for key, value in source.items()
        if key.startswith("features.") and key in destination and destination[key].shape == value.shape
    }
    model.load_state_dict(mapped, strict=False)
    return len(mapped)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
