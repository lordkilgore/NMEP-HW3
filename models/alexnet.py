import torch
from torch import nn


class AlexNet(nn.Module):
    """AlexNet with 32x32 color images and 200 classes"""

    def __init__(self, num_classes: int = 200) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(3, 2),
            nn.ReLU(),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.MaxPool2d(3, 2),
            nn.ReLU(),

            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(3, 2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(36, 4096),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
