"""
Net model:
    Input (1x28x28)
        -> [Conv2D -> BatchNorm -> ReLU -> MaxPool2D]   (bloque 1)
        -> [Conv2D -> BatchNorm -> ReLU -> MaxPool2D]   (bloque 2)
        -> Flatten
        -> Linear -> ReLU -> Dropout
        -> Linear (salida, 10 clases)
"""

import torch
import torch.nn as nn

from config import Config


class DigitCNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        # For having the same dimension (28x28)
        padding = cfg.kernel_size // 2

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,                       # Since images are on a gray scale, the number of channels is only 1
                out_channels=cfg.conv1_out_channels,
                kernel_size=cfg.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(cfg.conv1_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=cfg.pool_size),  # Applying maxpool
        )

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.conv1_out_channels,   # The number of input channels must match the number of output channels of the first layer
                out_channels=cfg.conv2_out_channels,
                kernel_size=cfg.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(cfg.conv2_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=cfg.pool_size),
        )

        # Flatten
        self.flatten = nn.Flatten()

        # The flattened size must be dinamically computed
        flattened_size = self._compute_flattened_size(cfg)

        # Fully conected classifier
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, cfg.fc1_out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.fc1_out_features, cfg.num_classes),
        )

    def _compute_flattened_size(self, cfg: Config) -> int:
        # The size of the first layer is computed by using the kernel size, pool size and the numbers of channels
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            out = self.conv_block1(dummy)
            out = self.conv_block2(out)
            return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
        ----------
        A tensor representing the gray-scale image (size of 1x28x28)

        Output
        -------
        A tensor where the greatest logit represents the predicted number
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
