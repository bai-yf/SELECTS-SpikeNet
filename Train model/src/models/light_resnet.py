"""
Lightweight ResNet model definition
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

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
        
        # Input layer
        self.conv1 = nn.Conv2d(
            in_channels=4, 
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
        
        # Residual blocks
        self.layer1 = self._make_layer(BasicBlock, 32, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=(1, 2))
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=(1, 2))
        self.layer4 = self._make_layer(BasicBlock, 128, 2, stride=1)
        
        # SPP
        self.spp = SpatialPyramidPooling(output_sizes=spp_output_sizes)
        
        # Compute feature dimensions
        self._compute_feature_dims(num_channels, time_steps)
    
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
    
    def _compute_feature_dims(self, num_channels, time_steps):
        with torch.no_grad():
            dummy = torch.randn(1, 4, num_channels, time_steps)
            out = self.forward_features(dummy, return_spp_input=False)
            self.feature_dim = out.size(1)
    
    def forward_features(self, x, return_spp_input=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if return_spp_input:
            spp_input = x
            spp_input_flat = spp_input.view(spp_input.size(0), -1)
            spp_output = self.spp(x)
            return spp_output, spp_input_flat
        
        return self.spp(x)
    
    def forward(self, x, return_spp_input=False):
        return self.forward_features(x, return_spp_input)