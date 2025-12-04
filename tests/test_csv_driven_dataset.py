"""
Tests for CSV-driven dataset building from make_dataset.py
"""
import pytest
import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.make_dataset import find_image_in_raw, preprocess_image


class TestCSVDrivenDataset:
    """Test cases for CSV-driven dataset building"""
    
    def test_find_image_in_raw_with_subfolder(self, tmp_path):
        """Test finding images in lot subdirectories with lot prefix extraction"""
        raw_dir = tmp_path / "data" / "raw"
        lot_dir = raw_dir / "s1"
        lot_dir.mkdir(parents=True)
        
        # Create a test image with lot prefix in filename
        test_img = lot_dir / "s1_00000.png"
        test_img.touch()
        
        # Test finding the image with just the filename (lot extracted from filename)
        result = find_image_in_raw("s1_00000.png", str(raw_dir))
        assert result is not None
        assert "s1_00000.png" in result
        assert str(lot_dir) in result
        
    def test_find_image_in_raw_not_found(self, tmp_path):
        """Test handling of missing images"""
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        result = find_image_in_raw("nonexistent_image.png", str(raw_dir))
        assert result is None
    
    def test_find_image_in_raw_at_root(self, tmp_path):
        """Test finding images directly at raw root (fallback)"""
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        # Create a test image at root
        test_img = raw_dir / "test_image.png"
        test_img.touch()
        
        # Test finding with just filename (no subfolder)
        result = find_image_in_raw("test_image.png", str(raw_dir))
        assert result is not None
        assert "test_image.png" in result
    
    def test_preprocess_image_with_valid_image(self):
        """Test image preprocessing with a valid image"""
        # Create a simple test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Process it
        result = preprocess_image(test_img, size=64)
        
        assert result is not None
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_preprocess_image_with_none(self):
        """Test that preprocess_image handles None input"""
        result = preprocess_image(None, size=64)
        assert result is None
    
    def test_csv_structure_validation(self, tmp_path):
        """Test that we can validate CSV has required columns"""
        # Create a test CSV with required columns
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'filename': ['s1_00000.png', 's1_00001.png'],
            'beard': [1, 0],
            'mustache': [0, 1],
            'glasses_binary': [1, 0],
            'hair_color_label': [0, 1],
            'hair_length': [1, 2]
        })
        df.to_csv(csv_path, index=False)
        
        # Load and validate
        loaded_df = pd.read_csv(csv_path)
        required_cols = ['filename', 'beard', 'mustache', 'glasses_binary', 
                        'hair_color_label', 'hair_length']
        
        for col in required_cols:
            assert col in loaded_df.columns
        
        assert len(loaded_df) == 2
        assert loaded_df['filename'][0] == 's1_00000.png'
    
    def test_csv_with_multiple_lots(self, tmp_path):
        """Test CSV can contain images from multiple lots"""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'filename': ['s1_00000.png', 's2_00001.png', 's3_00002.png'],
            'beard': [1, 0, 1],
            'mustache': [0, 1, 0],
            'glasses_binary': [1, 0, 1],
            'hair_color_label': [0, 1, 2],
            'hair_length': [1, 2, 0]
        })
        df.to_csv(csv_path, index=False)
        
        loaded_df = pd.read_csv(csv_path)
        
        # Check we have entries from different lots
        # Extract lot prefix from filenames (s1, s2, s3)
        lots = loaded_df['filename'].str.split('_').str[0].unique()
        assert len(lots) == 3
        assert 's1' in lots
        assert 's2' in lots
        assert 's3' in lots
