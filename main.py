"""
For each epoch the model is trained and validated
The model with the best accuracy ist saved in cfg.checkpoint_path
The graph with the training curved is saved

For the test the best model is evaluated only once with the test set (unknown for the model)
Confusion matrix, accuracy graph and some examples are generated
"""

import os
import torch
import torch.nn as nn

from config import CFG
from data import get_dataloaders
from model import DigitCNN
from engine import History, train_one_epoch, evaluate, build_optimizer
from utils import seed_everything, ensure_dir, plot_training_curves
from test import run_test


def train_and_validate(cfg=CFG) -> History:
    """Excecutes the complete cycle of training + validation for each epoch,
    and the one with the best accuracy is saved"""
    seed_everything(cfg.seed)
    ensure_dir(os.path.dirname(cfg.checkpoint_path))
    ensure_dir(cfg.results_dir)

    device = cfg.device
    print(f"Usando dispositivo: {device}")

    train_loader, val_loader, _ = get_dataloaders(cfg)

    model = DigitCNN(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    history = History()
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        # Training-
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        print(
            f"Época {epoch:02d}/{cfg.epochs} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4%} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4%}"
        )

        # The best model until now is saved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                },
                cfg.checkpoint_path,
            )
            print(f"  -> Nuevo mejor modelo guardado (val_acc={val_acc:.4%})")

    plot_training_curves(history, cfg.results_dir)
    print(f"\nEntrenamiento finalizado. Mejor accuracy de validación: {best_val_acc:.4%}")
    return history


if __name__ == "__main__":
    train_and_validate(CFG)
    run_test(CFG)
