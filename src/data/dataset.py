"""
Dataset for face attribute classification
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class FaceAttributeDataset(Dataset):
    """Dataset for face attributes - dummy implementation for testing"""
    
    def __init__(self, num_samples=1000, num_attributes=40):
        """
        Args:
            num_samples: Number of samples in dataset
            num_attributes: Number of attributes to predict
        """
        self.num_samples = num_samples
        self.num_attributes = num_attributes
        
        # Generate dummy data
        np.random.seed(42)
        self.images = torch.randn(num_samples, 3, 64, 64)
        self.labels = torch.randint(0, 2, (num_samples, num_attributes)).float()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
