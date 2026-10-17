import torch
import torch.nn as nn

class MyModel(nn.Module):
    """Five convolution stages, initialized randomly, for 224px landmark images."""
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7):
        super().__init__()
        layers = []
        in_channels = 3
        for channels in [32, 64, 128, 256, 512]:
            layers.extend([
                nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(2048, 256),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, num_classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
