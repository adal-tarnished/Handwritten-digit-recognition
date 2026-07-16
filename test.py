"""
test.py
=======
Fase de PRUEBA (test).

Carga el mejor modelo guardado durante el entrenamiento y lo evalúa
sobre el conjunto de test de MNIST, que el modelo NUNCA vio durante
entrenamiento ni validación. Genera además varias gráficas de
desempeño en `cfg.results_dir`:

    - confusion_matrix.png     : matriz de confusión 10x10
    - per_class_accuracy.png   : precisión por cada dígito
    - sample_predictions.png   : ejemplos visuales de predicciones
"""

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from config import CFG
from model import DigitCNN
from data import get_dataloaders
from utils import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_sample_predictions,
)


@torch.no_grad()
def run_test(cfg=CFG):
    """Carga el mejor checkpoint y evalúa el desempeño final en el set de test.

    Returns
    -------
    (test_loss, test_acc) : Tuple[float, float]
    """
    device = cfg.device

    # Load data and model
    _, _, test_loader = get_dataloaders(cfg)

    model = DigitCNN(cfg).to(device)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # Have predictions about all test set
    all_preds, all_labels, all_images = [], [], []
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in test_loader:
        images_dev, labels_dev = images.to(device), labels.to(device)
        outputs = model(images_dev)
        loss = criterion(outputs, labels_dev)

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        correct += (preds == labels_dev).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        all_images.extend(images.cpu().numpy())

    test_loss = running_loss / total
    test_acc = correct / total

    print("\n===== RESULTADOS DE PRUEBA =====")
    print(f"Pérdida en test:   {test_loss:.4f}")
    print(f"Precisión en test: {test_acc:.4%}\n")
    print("Reporte de clasificación por dígito:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Generate graphs
    plot_confusion_matrix(all_labels, all_preds, cfg.results_dir)
    plot_per_class_accuracy(all_labels, all_preds, cfg.results_dir)
    plot_sample_predictions(all_images, all_labels, all_preds, cfg.results_dir)

    print(f"Gráficas de prueba guardadas en: {cfg.results_dir}")
    return test_loss, test_acc


if __name__ == "__main__":
    run_test()
