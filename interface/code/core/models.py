
"""Neural-network architecture used by training and interface inference."""
import sys

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from config import EFFICIENT_KAN_PATH

if EFFICIENT_KAN_PATH not in sys.path:
    sys.path.append(EFFICIENT_KAN_PATH)

from efficient_kan.kan import KAN

class SpatialPyramidPooling(nn.Module):
    def __init__(self, output_sizes=[1, 2, 4]):
        super().__init__()
        self.pool_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((size, size*2)) for size in output_sizes
        ])

    def forward(self, x):
        all_features = []
        for pool in self.pool_layers:
            pooled = pool(x)
            all_features.append(pooled.view(pooled.size(0), -1))
        return torch.cat(all_features, dim=1)


class LightResNet18_EEG(nn.Module):
    def __init__(self, num_channels, time_steps, dropout_rate=0.3, spp_output_sizes=[1,2,4]):
        super().__init__()
        self.inplanes = 16
        self.in_channels = 4

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inplanes,
            kernel_size=(3, 5),
            stride=(1, 2),
            padding=(1, 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1)
        )

        self.layer1 = self._make_layer(BasicBlock, 32, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 4, stride=(1, 2))
        self.layer3 = self._make_layer(BasicBlock, 128, 6, stride=(1, 2))
        self.layer4 = self._make_layer(BasicBlock, 128, 3, stride=1)

        self.spp = SpatialPyramidPooling(output_sizes=spp_output_sizes)

        with torch.no_grad():
            dummy = torch.randn(1, self.in_channels, num_channels, time_steps)
            out = self.conv1(dummy)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            self.feature_dim = self.spp(out).size(1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_spp_input=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        spp_input_flat = x.view(x.size(0), -1)
        x = self.spp(x)

        if return_spp_input:
            return x, spp_input_flat
        return x


class KANClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.kan = KAN([feature_dim, 128, 64, num_classes])

    def forward(self, x):
        return self.kan(x)


class FusionModel(nn.Module):
    def __init__(self, num_channels, time_steps, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.resnet_extractor = LightResNet18_EEG(
            num_channels=num_channels,
            time_steps=time_steps,
            dropout_rate=dropout_rate
        )
        self.classifier = KANClassifier(
            self.resnet_extractor.feature_dim,
            num_classes=num_classes
        )

    def forward(self, X):
        feat = self.resnet_extractor(X)
        return self.classifier(feat)
