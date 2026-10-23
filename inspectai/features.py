"""Frozen ImageNet features; no task-specific fine-tuning is claimed."""
import os
from pathlib import Path
from functools import lru_cache

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent.parent / ".cache/torch"))

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


@lru_cache(maxsize=1)
def backbone():
    weights = ResNet18_Weights.IMAGENET1K_V1
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms(), device


def encode(images):
    model, transform, device = backbone()
    batch = torch.stack([transform(image.convert("RGB")) for image in images]).to(device)
    with torch.inference_mode():
        features = model(batch)
        features = torch.nn.functional.normalize(features, dim=1)
    return features.cpu().numpy().astype(np.float32)
