"""
Helpers
"""

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from engine import History


def seed_everything(seed: int) -> None:
    # Fixes the seed for all modules
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    # Creates directories if necessary
    if path:
        os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# Training and validation graphs
# ----------------------------------------------------------------------

def plot_training_curves(history: History, results_dir: str) -> None:
    """Graphs the evolution of loss and accuracy in training and validation
    through epochs"""
    ensure_dir(results_dir)
    epochs = range(1, len(history.train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history.train_loss, marker="o", label="Entrenamiento")
    axes[0].plot(epochs, history.val_loss, marker="o", label="Validación")
    axes[0].set_title("Pérdida (Loss) por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.train_acc, marker="o", label="Entrenamiento")
    axes[1].plot(epochs, history.val_acc, marker="o", label="Validación")
    axes[1].set_title("Precisión (Accuracy) por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Graphs for test
# ----------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, results_dir: str) -> None:
    """Creates and stores the confusion matrix"""
    ensure_dir(results_dir)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusión (Test)")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(y_true, y_pred, results_dir: str) -> None:
    """Bar graph for obtained results"""
    ensure_dir(results_dir)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracies = []
    for digit in range(10):
        mask = y_true == digit
        acc = (y_pred[mask] == digit).mean() if mask.sum() > 0 else 0.0
        accuracies.append(acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(10), accuracies, color="steelblue")
    ax.set_xticks(range(10))
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Precisión")
    ax.set_title("Precisión por clase (Test)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, acc + 0.01,
            f"{acc:.2%}", ha="center", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "per_class_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_sample_predictions(images, y_true, y_pred, results_dir: str, n: int = 16) -> None:
    """Test images with their predictions, green for correct and red for incorrect."""
    ensure_dir(results_dir)
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.array(axes).reshape(-1)

    for idx in range(n):
        ax = axes[idx]
        img = np.squeeze(images[idx])
        ax.imshow(img, cmap="gray")
        correct = y_true[idx] == y_pred[idx]
        color = "green" if correct else "red"
        ax.set_title(f"Real: {y_true[idx]} / Pred: {y_pred[idx]}", color=color, fontsize=9)
        ax.axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Ejemplos de predicciones en el set de prueba")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "sample_predictions.png"), dpi=150)
    plt.close(fig)
