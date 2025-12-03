"""
Pytest configuration and fixtures
"""
import pytest
import torch
import numpy as np


@pytest.fixture
def sample_image():
    """Create a sample image tensor (3, 64, 64)"""
    return torch.randn(3, 64, 64)


@pytest.fixture
def sample_batch():
    """Create a batch of sample images (batch_size=4, 3, 64, 64)"""
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def sample_labels():
    """Create sample labels for all attributes"""
    return {
        "beard": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "mustache": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "glasses": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "hair_color": torch.tensor([0, 1, 2, 3]),
        "hair_length": torch.tensor([0, 1, 2, 0])
    }


@pytest.fixture
def device():
    """Get the device for testing (CPU)"""
    return torch.device('cpu')


@pytest.fixture
def sample_numpy_images():
    """Create sample numpy images in (N, H, W, C) format"""
    return np.random.rand(10, 64, 64, 3).astype(np.float32)


@pytest.fixture
def sample_numpy_labels():
    """Create sample numpy labels (N, 5)"""
    return np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 2],
        [1, 1, 1, 2, 0],
        [0, 0, 0, 3, 1],
        [1, 0, 1, 4, 2],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 2, 2],
        [1, 0, 1, 3, 0],
        [0, 1, 0, 4, 1],
    ], dtype=np.int64)
