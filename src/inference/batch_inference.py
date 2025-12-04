"""
Batch inference script for unlabeled data (lots 2-9)
Generates predictions CSV for all images in data/raw
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


def preprocess_image(img_path, size=64):
    """Preprocess a single image (same as training)"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Resize and normalize
    img = cv2.resize(img, (size, size)).astype("float32") / 255.0

    return img


def batch_inference():
    """Run inference on all images in data/raw and generate predictions CSV"""
    print(f"{'='*60}")
    print(" BATCH INFERENCE")
    print(f"{'='*60}")
    print(f" Device: {DEVICE}")

    # Load model
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run: dvc repro train"
        )

    print(f"\n Loading model from {model_path}...")
    model = CustomMultiHeadCNN(n_color=5, n_length=3).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f" Model loaded successfully!")

    # Find all images in data/raw (excluding those with labels from lot 1)
    raw_dir = Path("data/raw")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    all_images = []

    for ext in image_extensions:
        all_images.extend(list(raw_dir.glob(f"*{ext}")))

    print(f"\n Found {len(all_images)} images in {raw_dir}")

    if len(all_images) == 0:
        print(" Warning: No images found to process")
        # Create empty output
        os.makedirs('outputs', exist_ok=True)
        pd.DataFrame(columns=['filename', 'beard', 'mustache', 'glasses', 'hair_color', 'hair_length']).to_csv(
            'outputs/predictions.csv', index=False
        )
        return

    # Process in batches
    batch_size = 64
    predictions = []

    print(f"\n Processing images in batches of {batch_size}...")

    batch_imgs = []
    batch_filenames = []

    for img_path in tqdm(all_images, desc="Inference"):
        # Load and preprocess image
        img = preprocess_image(str(img_path))

        if img is not None:
            batch_imgs.append(img)
            batch_filenames.append(img_path.name)

        # Process batch when full or at end
        if len(batch_imgs) >= batch_size or img_path == all_images[-1]:
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

    # Create DataFrame and save
    print(f"\n Creating predictions DataFrame...")
    df_predictions = pd.DataFrame(predictions)

    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/predictions.csv'

    df_predictions.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(" INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f" Total predictions: {len(predictions)}")
    print(f" Output saved to: {output_path}")
    print(f"\n Sample predictions:")
    print(df_predictions.head(10))
    print(f"{'='*60}")

    # Print summary statistics
    print(f"\n Prediction statistics:")
    print(f"   Beard:        {df_predictions['beard'].sum()} / {len(df_predictions)} ({df_predictions['beard'].mean()*100:.1f}%)")
    print(f"   Mustache:     {df_predictions['mustache'].sum()} / {len(df_predictions)} ({df_predictions['mustache'].mean()*100:.1f}%)")
    print(f"   Glasses:      {df_predictions['glasses'].sum()} / {len(df_predictions)} ({df_predictions['glasses'].mean()*100:.1f}%)")
    print(f"   Hair color distribution:")
    for color, count in df_predictions['hair_color'].value_counts().sort_index().items():
        print(f"     Color {color}: {count} ({count/len(df_predictions)*100:.1f}%)")
    print(f"   Hair length distribution:")
    for length, count in df_predictions['hair_length'].value_counts().sort_index().items():
        print(f"     Length {length}: {count} ({count/len(df_predictions)*100:.1f}%)")

    return df_predictions


if __name__ == '__main__':
    batch_inference()