"""
Models module
"""
from .light_resnet import LightResNet34_EEG, SpatialPyramidPooling
from .kan_classifier import KANClassifier
from .fusion_model import FusionModel

__all__ = [
    'LightResNet34_EEG',
    'SpatialPyramidPooling',
    'KANClassifier',
    'FusionModel'
]
