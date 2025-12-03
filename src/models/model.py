"""
Simple CNN model for face attribute classification
"""
import torch
import torch.nn as nn


class FaceAttributeClassifier(nn.Module):
    """Multi-task CNN for face attribute classification"""
    
    def __init__(self, num_attributes=40):
        super(FaceAttributeClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_attributes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
