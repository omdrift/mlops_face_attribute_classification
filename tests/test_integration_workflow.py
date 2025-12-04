"""
Integration test demonstrating the complete workflow for CSV-driven training
and incremental inference.

This test creates a realistic directory structure and validates the end-to-end behavior.
"""
import pytest
import os
import sys
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import tempfile

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.make_dataset import find_image_in_raw
from src.inference.batch_inference import (
    get_lot_directories,
    get_existing_predictions,
    get_training_images
)


class TestIntegrationWorkflow:
    """Integration tests for the complete workflow"""
    
    def test_complete_workflow_structure(self, tmp_path):
        """
        Test a complete workflow:
        1. Setup directory structure with multiple lots
        2. Create CSV with training annotations
        3. Verify CSV-driven approach selects correct images
        4. Verify incremental inference detects lots correctly
        """
        # 1. Create directory structure
        raw_dir = tmp_path / "data" / "raw"
        
        # Create 3 lots with images
        for lot_num in [1, 2, 3]:
            lot_dir = raw_dir / f"s{lot_num}"
            lot_dir.mkdir(parents=True)
            
            # Create some dummy images
            for img_num in range(5):
                img_file = lot_dir / f"s{lot_num}_{img_num:05d}.png"
                # Create a simple image
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(img_file), img)
        
        # 2. Create training CSV with mixed lots
        csv_path = tmp_path / "data" / "annotations" / "mapped_train.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        training_data = []
        # Select some images from s1 and s2 for training
        for lot_num in [1, 2]:
            for img_num in [0, 1, 2]:  # Only first 3 images from each lot
                training_data.append({
                    'filename': f's{lot_num}_{img_num:05d}.png',
                    'beard': np.random.randint(0, 2),
                    'mustache': np.random.randint(0, 2),
                    'glasses_binary': np.random.randint(0, 2),
                    'hair_color_label': np.random.randint(0, 5),
                    'hair_length': np.random.randint(0, 3)
                })
        
        df_train = pd.DataFrame(training_data)
        df_train.to_csv(csv_path, index=False)
        
        # 3. Verify CSV reading and image finding
        training_images = get_training_images(str(csv_path))
        assert len(training_images) == 6  # 3 from s1 + 3 from s2
        assert 's1_00000.png' in training_images
        assert 's2_00002.png' in training_images
        
        # Verify we can find these images
        for filename in training_images:
            img_path = find_image_in_raw(filename, str(raw_dir))
            assert img_path is not None
            assert os.path.exists(img_path)
        
        # 4. Verify lot detection
        lots = get_lot_directories(str(raw_dir))
        assert len(lots) == 3
        assert 's1' in lots
        assert 's2' in lots
        assert 's3' in lots
        
        # 5. Simulate inference workflow
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        
        # Initially, no predictions exist
        existing = get_existing_predictions(str(output_dir))
        assert len(existing) == 0
        
        # Simulate creating predictions for s1
        (output_dir / "predictions_s1.csv").touch()
        
        existing = get_existing_predictions(str(output_dir))
        assert len(existing) == 1
        assert 's1' in existing
        
        # Determine lots to process
        lots_to_process = [lot for lot in lots if lot not in existing]
        assert len(lots_to_process) == 2
        assert 's2' in lots_to_process
        assert 's3' in lots_to_process
        assert 's1' not in lots_to_process  # Already processed
        
    def test_adding_new_lot_workflow(self, tmp_path):
        """
        Test workflow when adding a new lot after initial processing:
        1. Start with lots s1, s2
        2. Process them
        3. Add lot s3
        4. Verify only s3 needs processing
        """
        raw_dir = tmp_path / "data" / "raw"
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True)
        
        # Initial state: 2 lots
        for lot_num in [1, 2]:
            lot_dir = raw_dir / f"s{lot_num}"
            lot_dir.mkdir(parents=True)
            (lot_dir / f"img_{lot_num}.png").touch()
        
        # Process initial lots
        lots = get_lot_directories(str(raw_dir))
        assert len(lots) == 2
        
        for lot in lots:
            (output_dir / f"predictions_{lot}.csv").touch()
        
        existing = get_existing_predictions(str(output_dir))
        assert existing == {'s1', 's2'}
        
        # Add new lot s3
        lot3_dir = raw_dir / "s3"
        lot3_dir.mkdir()
        (lot3_dir / "img_3.png").touch()
        
        # Check updated state
        lots = get_lot_directories(str(raw_dir))
        assert len(lots) == 3
        
        existing = get_existing_predictions(str(output_dir))
        lots_to_process = [lot for lot in lots if lot not in existing]
        
        # Only s3 should need processing
        assert len(lots_to_process) == 1
        assert lots_to_process[0] == 's3'
    
    def test_csv_update_workflow(self, tmp_path):
        """
        Test workflow when updating training CSV:
        1. Start with some images in CSV
        2. Add more images to CSV
        3. Verify new images are recognized
        """
        raw_dir = tmp_path / "data" / "raw"
        csv_path = tmp_path / "data" / "annotations" / "mapped_train.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create lots
        for lot_num in [1, 2]:
            lot_dir = raw_dir / f"s{lot_num}"
            lot_dir.mkdir(parents=True)
            for img_num in range(5):
                (lot_dir / f"s{lot_num}_{img_num:05d}.png").touch()
        
        # Initial CSV with 2 images
        df_initial = pd.DataFrame({
            'filename': ['s1_00000.png', 's1_00001.png'],
            'beard': [1, 0],
            'mustache': [0, 1],
            'glasses_binary': [1, 0],
            'hair_color_label': [0, 1],
            'hair_length': [1, 2]
        })
        df_initial.to_csv(csv_path, index=False)
        
        # Load and verify
        training_imgs = get_training_images(str(csv_path))
        assert len(training_imgs) == 2
        
        # Update CSV to add more images
        df_updated = pd.DataFrame({
            'filename': ['s1_00000.png', 's1_00001.png', 's2_00000.png', 's2_00001.png'],
            'beard': [1, 0, 1, 1],
            'mustache': [0, 1, 0, 0],
            'glasses_binary': [1, 0, 1, 1],
            'hair_color_label': [0, 1, 2, 3],
            'hair_length': [1, 2, 0, 1]
        })
        df_updated.to_csv(csv_path, index=False)
        
        # Load and verify updated
        training_imgs = get_training_images(str(csv_path))
        assert len(training_imgs) == 4
        assert 's2_00000.png' in training_imgs
        assert 's2_00001.png' in training_imgs
    
    def test_reprocessing_workflow(self, tmp_path):
        """
        Test workflow for reprocessing a specific lot:
        1. Have predictions for all lots
        2. Delete prediction for one lot
        3. Verify only that lot is marked for processing
        """
        raw_dir = tmp_path / "data" / "raw"
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True)
        
        # Create 3 lots
        for lot_num in [1, 2, 3]:
            lot_dir = raw_dir / f"s{lot_num}"
            lot_dir.mkdir(parents=True)
            (lot_dir / f"img.png").touch()
        
        # Create predictions for all
        for lot_num in [1, 2, 3]:
            (output_dir / f"predictions_s{lot_num}.csv").touch()
        
        lots = get_lot_directories(str(raw_dir))
        existing = get_existing_predictions(str(output_dir))
        
        # All are processed
        assert len(lots) == 3
        assert len(existing) == 3
        lots_to_process = [lot for lot in lots if lot not in existing]
        assert len(lots_to_process) == 0
        
        # Delete prediction for s2
        os.remove(output_dir / "predictions_s2.csv")
        
        existing = get_existing_predictions(str(output_dir))
        lots_to_process = [lot for lot in lots if lot not in existing]
        
        # Only s2 should need reprocessing
        assert len(lots_to_process) == 1
        assert lots_to_process[0] == 's2'
