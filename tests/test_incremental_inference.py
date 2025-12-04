"""
Tests for incremental lot-based batch inference
"""
import pytest
import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.batch_inference import (
    get_lot_directories,
    get_training_images,
    get_existing_predictions,
    get_images_in_lot
)


class TestIncrementalInference:
    """Test cases for incremental lot-based inference"""
    
    def test_get_lot_directories_finds_valid_lots(self, tmp_path):
        """Test that get_lot_directories finds valid lot folders"""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        
        # Create valid lot directories
        (raw_dir / "s1").mkdir()
        (raw_dir / "s2").mkdir()
        (raw_dir / "s10").mkdir()
        
        # Create invalid directories (should be ignored)
        (raw_dir / "invalid").mkdir()
        (raw_dir / "s").mkdir()
        (raw_dir / "s1a").mkdir()
        
        lots = get_lot_directories(str(raw_dir))
        
        assert len(lots) == 3
        assert 's1' in lots
        assert 's2' in lots
        assert 's10' in lots
        assert 'invalid' not in lots
    
    def test_get_lot_directories_empty(self, tmp_path):
        """Test behavior when no lot directories exist"""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        
        lots = get_lot_directories(str(raw_dir))
        assert lots == []
    
    def test_get_lot_directories_nonexistent_dir(self, tmp_path):
        """Test behavior when raw directory doesn't exist"""
        raw_dir = tmp_path / "nonexistent"
        
        lots = get_lot_directories(str(raw_dir))
        assert lots == []
    
    def test_get_training_images(self, tmp_path):
        """Test loading training image filenames from CSV"""
        csv_path = tmp_path / "train.csv"
        df = pd.DataFrame({
            'filename': ['s1/img1.png', 's1/img2.png', 's2/img1.png'],
            'beard': [1, 0, 1],
            'mustache': [0, 1, 0],
            'glasses_binary': [1, 0, 1],
            'hair_color_label': [0, 1, 2],
            'hair_length': [1, 2, 0]
        })
        df.to_csv(csv_path, index=False)
        
        training_imgs = get_training_images(str(csv_path))
        
        assert len(training_imgs) == 3
        assert 's1/img1.png' in training_imgs
        assert 's1/img2.png' in training_imgs
        assert 's2/img1.png' in training_imgs
    
    def test_get_training_images_missing_file(self, tmp_path):
        """Test handling when training CSV doesn't exist"""
        csv_path = tmp_path / "nonexistent.csv"
        
        training_imgs = get_training_images(str(csv_path))
        assert training_imgs == set()
    
    def test_get_existing_predictions(self, tmp_path):
        """Test detection of existing prediction files"""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        
        # Create some prediction files
        (output_dir / "predictions_s1.csv").touch()
        (output_dir / "predictions_s3.csv").touch()
        (output_dir / "other_file.csv").touch()  # Should be ignored
        
        existing = get_existing_predictions(str(output_dir))
        
        assert len(existing) == 2
        assert 's1' in existing
        assert 's3' in existing
        assert 'other_file' not in existing
    
    def test_get_existing_predictions_empty_dir(self, tmp_path):
        """Test behavior when output directory is empty"""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        
        existing = get_existing_predictions(str(output_dir))
        assert existing == set()
    
    def test_get_existing_predictions_nonexistent_dir(self, tmp_path):
        """Test behavior when output directory doesn't exist"""
        output_dir = tmp_path / "nonexistent"
        
        existing = get_existing_predictions(str(output_dir))
        assert existing == set()
    
    def test_get_images_in_lot(self, tmp_path):
        """Test getting all images from a lot directory"""
        raw_dir = tmp_path / "raw"
        lot_dir = raw_dir / "s1"
        lot_dir.mkdir(parents=True)
        
        # Create some test image files
        (lot_dir / "img1.png").touch()
        (lot_dir / "img2.jpg").touch()
        (lot_dir / "img3.PNG").touch()
        (lot_dir / "readme.txt").touch()  # Should be ignored
        
        images = get_images_in_lot(str(raw_dir), "s1")
        
        assert len(images) == 3
        image_names = [img.name for img in images]
        assert 'img1.png' in image_names
        assert 'img2.jpg' in image_names
        assert 'img3.PNG' in image_names
        assert 'readme.txt' not in image_names
    
    def test_get_images_in_lot_empty(self, tmp_path):
        """Test behavior when lot has no images"""
        raw_dir = tmp_path / "raw"
        lot_dir = raw_dir / "s1"
        lot_dir.mkdir(parents=True)
        
        images = get_images_in_lot(str(raw_dir), "s1")
        assert images == []
    
    def test_get_images_in_lot_nonexistent(self, tmp_path):
        """Test behavior when lot directory doesn't exist"""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        
        images = get_images_in_lot(str(raw_dir), "s99")
        assert images == []
    
    def test_incremental_logic(self, tmp_path):
        """Test the incremental inference logic end-to-end"""
        # Setup
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "outputs"
        
        # Create lot directories
        (raw_dir / "s1").mkdir(parents=True)
        (raw_dir / "s2").mkdir(parents=True)
        (raw_dir / "s3").mkdir(parents=True)
        output_dir.mkdir()
        
        # Simulate that s1 and s2 already have predictions
        (output_dir / "predictions_s1.csv").touch()
        (output_dir / "predictions_s2.csv").touch()
        
        # Get lots and check which need processing
        all_lots = get_lot_directories(str(raw_dir))
        existing = get_existing_predictions(str(output_dir))
        lots_to_process = [lot for lot in all_lots if lot not in existing]
        
        # Should only need to process s3
        assert len(lots_to_process) == 1
        assert 's3' in lots_to_process
        assert 's1' not in lots_to_process
        assert 's2' not in lots_to_process
    
    def test_prediction_file_naming(self, tmp_path):
        """Test that prediction files follow correct naming pattern"""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        
        # Create predictions with correct naming
        (output_dir / "predictions_s1.csv").touch()
        (output_dir / "predictions_s2.csv").touch()
        (output_dir / "predictions_s10.csv").touch()
        
        # These should be found
        existing = get_existing_predictions(str(output_dir))
        assert 's1' in existing
        assert 's2' in existing
        assert 's10' in existing
        
        # These should not match the pattern
        (output_dir / "prediction_s3.csv").touch()  # Wrong prefix
        (output_dir / "predictions_3.csv").touch()   # Missing 's'
        
        existing = get_existing_predictions(str(output_dir))
        assert '3' not in existing
        assert 'prediction_s3' not in existing
