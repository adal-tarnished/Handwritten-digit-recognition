"""
Here are the hyperparameters, those which are not trained and are rather a structural question
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir: str = "./data"          # Where MNIST database will be stored
    val_split: float = 0.1            # Percentage of data that will be used for validation
    batch_size: int = 64              # Number of images processed before updating the weights 
    num_workers: int = 2

    # ------------------------------------------------------------------
    # CNN architecture
    # ------------------------------------------------------------------
    conv1_out_channels: int = 32      # Filters for the first convolutional block
    conv2_out_channels: int = 64      # Filters for the second convolutional block
    kernel_size: int = 3
    pool_size: int = 2                # Size for maxpool window
    fc1_out_features: int = 128       # Number of neurons for the first all-to-all layer
    dropout_rate: float = 0.25
    num_classes: int = 10             # Since there are ten digits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    epochs: int = 10
    learning_rate: float = 1e-3       # How much the weights change after learning
    weight_decay: float = 1e-4        # To prevent overfitting
    optimizer: str = "adam"           # I chose adam because it converges a little faster than sgd
    momentum: float = 0.9

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "./checkpoints/best_model.pth"  # Dónde se guarda el mejor modelo
    results_dir: str = "./results"    # Carpeta donde se guardan las gráficas de salida


# Global instance used in all other modules
CFG = Config()
