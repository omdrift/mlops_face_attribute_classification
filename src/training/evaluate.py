"""
Evaluation script for the trained model
Generates detailed metrics on a test set
"""
import os
import sys
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# Add root directory to PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.architecture import CustomMultiHeadCNN
from src.data.dataset import ImageDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model():
    """Evaluate the best model and generate detailed metrics"""
    print(f"{'='*60}")
    print(" MODEL EVALUATION")
    print(f"{'='*60}")
    print(f" Device: {DEVICE}")
    
    # Load processed data
    processed_data_path = "data/processed/train_data_s1.pt"
    
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"Processed data not found: {processed_data_path}\n"
            f"Run: dvc repro prepare_train"
        )
    
    print(f"\n Loading data from {processed_data_path}...")
    data = torch.load(processed_data_path)
    X = data['X'].numpy()
    y = data['y'].numpy()
    
    print(f" Data loaded: {len(X)} images")
    
    # Split with same random_state as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f" Test set size: {len(X_test)} images")
    
    # Create test dataset
    test_dataset = ImageDataset(X_test, y_test, transform=None, is_preprocessed=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=2
    )
    
    # Load best model
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
    
    # Evaluate
    print(f"\n Evaluating...")
    
    all_preds = {
        'beard': [], 'mustache': [], 'glasses': [],
        'hair_color': [], 'hair_length': []
    }
    all_labels = {
        'beard': [], 'mustache': [], 'glasses': [],
        'hair_color': [], 'hair_length': []
    }
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Binary attributes (beard, mustache, glasses)
            for attr in ['beard', 'mustache', 'glasses']:
                preds = (torch.sigmoid(outputs[attr]) > 0.5).cpu().numpy()
                all_preds[attr].extend(preds)
                all_labels[attr].extend(labels[attr].cpu().numpy())
            
            # Multi-class attributes (hair_color, hair_length)
            for attr in ['hair_color', 'hair_length']:
                preds = torch.argmax(outputs[attr], dim=1).cpu().numpy()
                all_preds[attr].extend(preds)
                all_labels[attr].extend(labels[attr].cpu().numpy())
    
    # Calculate metrics
    print(f"\n Calculating metrics...")
    
    metrics = {}
    
    # Binary attributes
    for attr in ['beard', 'mustache', 'glasses']:
        y_true = np.array(all_labels[attr], dtype=int)
        y_pred = np.array(all_preds[attr], dtype=int)
        
        acc = np.mean(y_true == y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        metrics[attr] = {
            'accuracy': float(acc),
            'confusion_matrix': cm,
            'support': int(len(y_true))
        }
    
    # Multi-class attributes
    for attr in ['hair_color', 'hair_length']:
        y_true = np.array(all_labels[attr], dtype=int)
        y_pred = np.array(all_preds[attr], dtype=int)
        
        acc = np.mean(y_true == y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        # Per-class accuracy
        n_classes = len(cm)
        per_class_acc = []
        for i in range(n_classes):
            if sum(cm[i]) > 0:
                per_class_acc.append(cm[i][i] / sum(cm[i]))
            else:
                per_class_acc.append(0.0)
        
        metrics[attr] = {
            'accuracy': float(acc),
            'confusion_matrix': cm,
            'per_class_accuracy': [float(x) for x in per_class_acc],
            'support': int(len(y_true))
        }
    
    # Overall metrics
    avg_accuracy = np.mean([metrics[attr]['accuracy'] for attr in metrics])
    
    metrics['overall'] = {
        'average_accuracy': float(avg_accuracy),
        'test_size': len(X_test),
        'model_path': model_path
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(" EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f" Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"\n Per-attribute accuracy:")
    for attr in ['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']:
        acc = metrics[attr]['accuracy']
        print(f"   {attr:15}: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*60}")
    
    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    output_path = 'metrics/eval_metrics.json'
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Metrics saved to: {output_path}")
    
    return metrics


if __name__ == '__main__':
    evaluate_model()
