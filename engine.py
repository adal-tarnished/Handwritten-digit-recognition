"""
Core functions for training and validation
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class History:
    # Stores how the metrics of training/validation change through epochs
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    # Excecutes one epoch and returns average loss and average accuracy
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    progress = tqdm(loader, desc="Entrenando", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=loss.item())

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    # Evaluates the model without changing weights
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    # Creates the selected optimizer: adam or sgd
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Optimizador no soportado: {cfg.optimizer}")
