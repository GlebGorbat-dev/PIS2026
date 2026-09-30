from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def class_names_from_config(dataset_config: dict) -> list[str]:
    return list(dataset_config["classes"].keys())


def class_to_index(names: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


def build_transforms(config: dict, *, train: bool) -> transforms.Compose:
    size = int(config["image_size"])
    mean = list(config["imagenet_mean"])
    std = list(config["imagenet_std"])
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(size * 256 / 224)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class LeafCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        project_root: Path,
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.frame = pd.read_csv(csv_path)
        self.project_root = project_root
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = open_rgb(self.project_root / row.path)
        tensor = self.transform(image)
        label = self.class_to_idx[row.class_name]
        return tensor, label
