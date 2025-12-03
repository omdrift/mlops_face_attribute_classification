#!/usr/bin/env python
"""
Script to generate annotations CSV from images using the face attribute model.
This can be used to pre-compute attributes for faster search.

Usage:
    python scripts/generate_annotations.py --images-dir data/images --output data/annotations.csv
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model.face_attribute_model import FaceAttributeModel


def get_image_transform():
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def generate_annotations(images_dir: str, output_path: str, model_path: str = None):
    """
    Generate annotations CSV for all images in a directory.
    
    Args:
        images_dir: Directory containing face images
        output_path: Path to save the annotations CSV
        model_path: Optional path to trained model weights
    """
    print("Initializing model...")
    model = FaceAttributeModel(model_path=model_path)
    transform = get_image_transform()
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    print(f"Scanning for images in {images_dir}...")
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                # Get relative path from images_dir
                rel_path = os.path.relpath(full_path, images_dir)
                image_files.append((full_path, rel_path))
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found. Exiting.")
        return
    
    # Process images
    results = []
    attributes = model.get_attribute_list()
    
    print("Processing images...")
    for full_path, rel_path in tqdm(image_files):
        try:
            # Load and preprocess image
            image = Image.open(full_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # Predict attributes
            attr_probs = model.predict_attributes(image_tensor)
            
            # Create row
            row = {'filename': rel_path}
            row.update(attr_probs)
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
            continue
    
    # Create DataFrame and save
    print(f"Saving annotations to {output_path}...")
    df = pd.DataFrame(results)
    
    # Ensure all attributes are in the DataFrame
    for attr in attributes:
        if attr not in df.columns:
            df[attr] = 0.0
    
    # Reorder columns: filename first, then all attributes
    columns = ['filename'] + attributes
    df = df[columns]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Successfully generated annotations for {len(results)} images")
    print(f"Annotations saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate annotations CSV from face images'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        required=True,
        help='Directory containing face images'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for annotations CSV'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model weights (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        sys.exit(1)
    
    generate_annotations(args.images_dir, args.output, args.model_path)


if __name__ == '__main__':
    main()
