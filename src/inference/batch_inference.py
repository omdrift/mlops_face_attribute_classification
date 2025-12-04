"""
Batch inference script for incremental lot-based predictions.

This script performs inference on images organized in lot subdirectories under data/raw/
(e.g., s1/, s2/, s3/, etc.) and generates separate prediction CSVs for each lot.

Key features:
- Detects all lot subdirectories under data/raw/
- For each lot, checks if outputs/predictions_sX.csv already exists
- Only processes lots that don't have existing prediction files
- Optionally skips images already in the training set (mapped_train.csv)

Usage:
    python src/inference/batch_inference.py
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

# Add root directory to PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.architecture import CustomMultiHeadCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lot_directories(raw_dir):
    """
    Detect all lot subdirectories under data/raw/.
    
    Returns a list of lot names (e.g., ['s1', 's2', 's3']).
    Only includes directories that start with 's' followed by a digit.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    
    lot_dirs = []
    for item in raw_path.iterdir():
        if item.is_dir() and item.name.startswith('s') and item.name[1:].isdigit():
            lot_dirs.append(item.name)
    
    return sorted(lot_dirs)


def get_training_images(csv_path):
    """
    Load the set of images already in the training set from mapped_train.csv.
    
    Returns a set of filenames that should be skipped during inference.
    """
    if not os.path.exists(csv_path):
        return set()
    
    try:
        df = pd.read_csv(csv_path)
        if 'filename' in df.columns:
            return set(df['filename'].tolist())
    except Exception as e:
        print(f"⚠️  Warning: Could not read training CSV: {e}")
    
    return set()


def get_existing_predictions(output_dir):
    """
    Check which lot predictions already exist.
    
    Returns a set of lot names that already have prediction CSVs.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return set()
    
    existing_lots = set()
    for csv_file in output_path.glob("predictions_s*.csv"):
        # Extract lot name from filename (e.g., 'predictions_s1.csv' -> 's1')
        lot_name = csv_file.stem.replace('predictions_', '')
        existing_lots.add(lot_name)
    
    return existing_lots


def get_images_in_lot(raw_dir, lot_name):
    """
    Get all image files in a specific lot directory.
    
    Returns a list of full paths to images.
    """
    lot_path = Path(raw_dir) / lot_name
    if not lot_path.exists():
        return []
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    images = []
    
    for ext in image_extensions:
        images.extend(list(lot_path.glob(f"*{ext}")))
    
    return images


def preprocess_image(img_path, size=64):
    """Preprocess a single image (same as training)"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Resize and normalize
    img = cv2.resize(img, (size, size)).astype("float32") / 255.0

    return img


def process_lot_inference(model, lot_name, lot_images, training_images, batch_size=64, skip_training=False):
    """
    Run inference on images from a specific lot.
    
    Args:
        model: The trained model
        lot_name: Name of the lot (e.g., 's1')
        lot_images: List of image paths for this lot
        training_images: Set of filenames already in training
        batch_size: Batch size for inference
        skip_training: Whether to skip images in the training set
    
    Returns:
        DataFrame with predictions
    """
    predictions = []
    batch_imgs = []
    batch_filenames = []
    
    print(f"\n🔄 Processing lot {lot_name} ({len(lot_images)} images)...")
    
    for img_path in tqdm(lot_images, desc=f"Lot {lot_name}"):
        # Check if we should skip this image (already in training)
        relative_filename = f"{lot_name}/{img_path.name}"
        if skip_training and relative_filename in training_images:
            continue
        
        # Load and preprocess image
        img = preprocess_image(str(img_path))
        
        if img is not None:
            batch_imgs.append(img)
            batch_filenames.append(img_path.name)
        
        # Process batch when full or at end
        if len(batch_imgs) >= batch_size or img_path == lot_images[-1]:
            if len(batch_imgs) == 0:
                continue
            
            # Convert to tensor (N, H, W, C) -> (N, C, H, W)
            tensor_batch = torch.tensor(np.array(batch_imgs)).permute(0, 3, 1, 2).to(DEVICE)
            
            # Run inference
            with torch.no_grad():
                outputs = model(tensor_batch)
            
            # Process outputs
            beards = (torch.sigmoid(outputs['beard']) > 0.5).cpu().numpy()
            mustaches = (torch.sigmoid(outputs['mustache']) > 0.5).cpu().numpy()
            glasses = (torch.sigmoid(outputs['glasses']) > 0.5).cpu().numpy()
            hair_colors = torch.argmax(outputs['hair_color'], dim=1).cpu().numpy()
            hair_lengths = torch.argmax(outputs['hair_length'], dim=1).cpu().numpy()
            
            # Store predictions
            for i in range(len(batch_imgs)):
                predictions.append({
                    'filename': batch_filenames[i],
                    'beard': int(beards[i]),
                    'mustache': int(mustaches[i]),
                    'glasses': int(glasses[i]),
                    'hair_color': int(hair_colors[i]),
                    'hair_length': int(hair_lengths[i])
                })
            
            # Reset batch
            batch_imgs = []
            batch_filenames = []
    
    return pd.DataFrame(predictions)


def batch_inference():
    """
    Run incremental lot-based batch inference.
    
    Detects all lot subdirectories under data/raw/, checks which lots already
    have prediction files, and only processes new lots without existing predictions.
    """
    print(f"{'='*60}")
    print(" INCREMENTAL LOT-BASED BATCH INFERENCE")
    print(f"{'='*60}")
    print(f" Device: {DEVICE}")
    
    # Configuration
    raw_dir = "data/raw"
    output_dir = "outputs"
    training_csv = "data/annotations/mapped_train.csv"
    batch_size = 64
    skip_training_images = True  # Skip images already in training set
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run: dvc repro train"
        )
    
    print(f"\n📦 Loading model from {model_path}...")
    model = CustomMultiHeadCNN(n_color=5, n_length=3).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded successfully!")
    
    # Detect lot directories
    print(f"\n📁 Detecting lot directories in {raw_dir}...")
    lot_dirs = get_lot_directories(raw_dir)
    
    if not lot_dirs:
        print(f"⚠️  No lot directories found in {raw_dir}")
        print(f"   Expected directories like s1/, s2/, s3/, etc.")
        return
    
    print(f"   Found {len(lot_dirs)} lot directories: {lot_dirs}")
    
    # Check existing predictions
    print(f"\n🔍 Checking for existing predictions in {output_dir}...")
    existing_predictions = get_existing_predictions(output_dir)
    
    if existing_predictions:
        print(f"   Found existing predictions for: {sorted(existing_predictions)}")
    else:
        print(f"   No existing predictions found")
    
    # Determine which lots need processing
    lots_to_process = [lot for lot in lot_dirs if lot not in existing_predictions]
    
    if not lots_to_process:
        print(f"\n✅ All lots already have predictions. Nothing to do!")
        print(f"   To reprocess a lot, delete its predictions_sX.csv file.")
        return
    
    print(f"\n🎯 Lots to process: {lots_to_process}")
    lots_skipped = [lot for lot in lot_dirs if lot in existing_predictions]
    if lots_skipped:
        print(f"   Skipping (already processed): {lots_skipped}")
    
    # Load training images (to optionally skip them)
    training_images = set()
    if skip_training_images:
        print(f"\n📋 Loading training images from {training_csv}...")
        training_images = get_training_images(training_csv)
        if training_images:
            print(f"   Will skip {len(training_images)} images already in training set")
    
    # Process each lot
    total_predictions = 0
    for lot_name in lots_to_process:
        print(f"\n{'='*60}")
        print(f" Processing Lot: {lot_name}")
        print(f"{'='*60}")
        
        # Get images for this lot
        lot_images = get_images_in_lot(raw_dir, lot_name)
        
        if not lot_images:
            print(f"⚠️  No images found in {lot_name}, skipping...")
            continue
        
        # Run inference
        df_predictions = process_lot_inference(
            model=model,
            lot_name=lot_name,
            lot_images=lot_images,
            training_images=training_images,
            batch_size=batch_size,
            skip_training=skip_training_images
        )
        
        # Save predictions for this lot
        output_path = os.path.join(output_dir, f"predictions_{lot_name}.csv")
        df_predictions.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved {len(df_predictions)} predictions to: {output_path}")
        total_predictions += len(df_predictions)
        
        # Print statistics for this lot
        if len(df_predictions) > 0:
            print(f"\n📊 Statistics for {lot_name}:")
            print(f"   Beard:        {df_predictions['beard'].sum()} / {len(df_predictions)} ({df_predictions['beard'].mean()*100:.1f}%)")
            print(f"   Mustache:     {df_predictions['mustache'].sum()} / {len(df_predictions)} ({df_predictions['mustache'].mean()*100:.1f}%)")
            print(f"   Glasses:      {df_predictions['glasses'].sum()} / {len(df_predictions)} ({df_predictions['glasses'].mean()*100:.1f}%)")
    
    # Final summary
    print(f"\n{'='*60}")
    print(" INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f" Lots processed: {len(lots_to_process)}")
    print(f" Total predictions generated: {total_predictions}")
    print(f" Output directory: {output_dir}/")
    print(f"{'='*60}")
    
    return


if __name__ == '__main__':
    batch_inference()