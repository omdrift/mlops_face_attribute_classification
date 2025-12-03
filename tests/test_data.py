"""
Tests for data preprocessing and dataset
"""
import pytest
import torch
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import ImageDataset


class TestImageDataset:
    """Test cases for ImageDataset"""

    def test_dataset_initialization_not_preprocessed(self, sample_numpy_images, sample_numpy_labels):
        """Test dataset initialization with non-preprocessed data"""
        dataset = ImageDataset(
            images=sample_numpy_images,
            labels=sample_numpy_labels,
            is_preprocessed=False
        )

        assert len(dataset) == len(sample_numpy_images)

    def test_dataset_initialization_preprocessed(self, sample_numpy_labels):
        """Test dataset initialization with preprocessed data (C, H, W format)"""
        # Preprocessed format: (N, C, H, W)
        preprocessed_images = np.random.rand(10, 3, 64, 64).astype(np.float32)

        dataset = ImageDataset(
            images=preprocessed_images,
            labels=sample_numpy_labels,
            is_preprocessed=True
        )

        assert len(dataset) == len(preprocessed_images)

    def test_dataset_getitem(self, sample_numpy_images, sample_numpy_labels):
        """Test getting a single item from dataset"""
        dataset = ImageDataset(
            images=sample_numpy_images,
            labels=sample_numpy_labels,
            is_preprocessed=False
        )

        image, labels = dataset[0]

        # Check image shape (should be C, H, W)
        assert image.shape == (3, 64, 64)

        # Check labels structure
        assert "beard" in labels
        assert "mustache" in labels
        assert "glasses" in labels
        assert "hair_color" in labels
        assert "hair_length" in labels

    def test_dataset_label_types(self, sample_numpy_images, sample_numpy_labels):
        """Test that labels have correct types"""
        dataset = ImageDataset(
            images=sample_numpy_images,
            labels=sample_numpy_labels,
            is_preprocessed=False
        )

        image, labels = dataset[0]

        # Binary labels should be float
        assert isinstance(labels["beard"].item(), float)
        assert isinstance(labels["mustache"].item(), float)
        assert isinstance(labels["glasses"].item(), float)

        # Multi-class labels should be integers
        assert isinstance(labels["hair_color"].item(), int)
        assert isinstance(labels["hair_length"].item(), int)

    def test_dataset_length(self, sample_numpy_images, sample_numpy_labels):
        """Test dataset length"""
        dataset = ImageDataset(
            images=sample_numpy_images,
            labels=sample_numpy_labels,
            is_preprocessed=False
        )

        assert len(dataset) == 10

    def test_dataset_shape_conversion(self, sample_numpy_labels):
        """Test that non-preprocessed images are converted correctly"""
        # Create images in (N, H, W, C) format
        images_hwc = np.random.rand(5, 64, 64, 3).astype(np.float32)

        dataset = ImageDataset(
            images=images_hwc,
            labels=sample_numpy_labels[:5],
            is_preprocessed=False
        )

        image, _ = dataset[0]

        # Should be converted to (C, H, W)
        assert image.shape == (3, 64, 64)

    def test_dataset_all_items(self, sample_numpy_images, sample_numpy_labels):
        """Test accessing all items in dataset"""
        dataset = ImageDataset(
            images=sample_numpy_images,
            labels=sample_numpy_labels,
            is_preprocessed=False
        )

        for i in range(len(dataset)):
            image, labels = dataset[i]
            assert image.shape == (3, 64, 64)
            assert len(labels) == 5

    def test_dataset_with_transform(self, sample_numpy_images, sample_numpy_labels):
        """Test dataset with transforms (even if None)"""
        dataset = ImageDataset(
            images=sample_numpy_images,
            labels=sample_numpy_labels,
            transform=None,
            is_preprocessed=False
        )

        image, labels = dataset[0]
        assert image is not None