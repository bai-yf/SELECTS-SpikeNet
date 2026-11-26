"""
Fusion Model: LightResNet + KAN Classifier
"""
import torch.nn as nn
from .light_resnet import LightResNet34_EEG
from .kan_classifier import KANClassifier

class FusionModel(nn.Module):
    """Complete Fusion Model"""
    def __init__(self, num_channels, time_steps, num_classes, 
                 in_channels=4, channel_plan=(32, 64, 128, 128),
                 block_plan=(3, 4, 6, 3), pooling="spp",
                 dropout_rate=0.3, spp_output_sizes=(1, 2, 4),
                 kan_layers=(128, 64)):
        super().__init__()
        
        # ResNet feature extractor
        self.resnet_extractor = LightResNet34_EEG(
            num_channels=num_channels,
            time_steps=time_steps,
            in_channels=in_channels,
            channel_plan=channel_plan,
            block_plan=block_plan,
            pooling=pooling,
            dropout_rate=dropout_rate,
            spp_output_sizes=spp_output_sizes
        )
        
        # KAN classifier
        self.classifier = KANClassifier(
            feature_dim=self.resnet_extractor.feature_dim,
            num_classes=num_classes,
            hidden_layers=kan_layers
        )
    
    def forward(self, x):
        """Forward pass"""
        features = self.resnet_extractor(x)
        outputs = self.classifier(features)
        return outputs
    
    def extract_features(self, x):
        """Extract features (for further analysis)"""
        return self.resnet_extractor(x)

    @property
    def backbone(self):
        """Notebook-compatible name without changing saved weight keys."""
        return self.resnet_extractor
