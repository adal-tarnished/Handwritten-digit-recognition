"""
Prepares the MNIST data
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import Config

# Mean and standard deviation of MNIST
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_transforms() -> transforms.Compose:
    # Transforms an image into a tensor and normalizes it
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads the dataloaders for train/val/test and returns the three needed
    loaders
    """
    transform = get_transforms()

    # if train is True, then the train data will be downloaded, otherwise, the test data will
    full_train_dataset = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=cfg.data_dir, train=False, download=True, transform=transform
    )

    # A part of the train data must be used for validation
    val_size = int(len(full_train_dataset) * cfg.val_split)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    print(
        f"Datos preparados -> train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
