"""
ResNet-18 Architecture Implementation for CUDA GPU Acceleration Engine (Phase 9).
Provides configurable ResNet-18 optimized for CIFAR-10 (32x32 resolution)
as well as standard vision classification benchmarks, with parameter counting,
memory footprint analysis, and deterministic initialization.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock with two 3x3 convolutions, BatchNorm, and residual skip connection.
    """
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 implementation with configurable stem:
    - cifar_stem=True (default): 3x3 conv stride 1, no initial maxpool (preserves 32x32 spatial resolution)
    - cifar_stem=False: 7x7 conv stride 2 followed by 3x3 maxpool (standard ImageNet stem)
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        cifar_stem: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.cifar_stem = cifar_stem
        self.dropout_rate = dropout_rate

        if cifar_stem:
            # CIFAR-10 stem: 3x3 conv, stride 1, no maxpool
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
        else:
            # Standard ImageNet stem: 7x7 conv stride 2 + maxpool
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 residual stages [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for conv layers, unit variance for batch norms."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax probability distribution without gradient graph."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class predictions without gradient graph."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def count_parameters(self) -> Dict[str, int]:
        """Return total, trainable, and non-trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
        }

    def get_model_size_mb(self) -> float:
        """Return total memory footprint in megabytes (FP32)."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        return round((param_bytes + buffer_bytes) / (1024.0 * 1024.0), 3)


def create_resnet18(
    num_classes: int = 10,
    in_channels: int = 3,
    cifar_stem: bool = True,
    dropout_rate: float = 0.0,
) -> ResNet18:
    """Factory function for creating ResNet-18."""
    return ResNet18(
        num_classes=num_classes,
        in_channels=in_channels,
        cifar_stem=cifar_stem,
        dropout_rate=dropout_rate,
    )
