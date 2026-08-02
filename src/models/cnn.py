"""Compact from-scratch CNN for PV patch classification. See Task 14's
design-choice note in plan.md for why this is from-scratch rather than a
pretrained ImageNet backbone."""
import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class PVPatchCNN(nn.Module):
    def __init__(self, in_channels: int = 6, input_size: int = 21, n_classes: int = 2):
        super().__init__()
        self.block1 = _conv_block(in_channels, 32)
        self.block2 = _conv_block(32, 64)
        self.block3 = _conv_block(64, 128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)
