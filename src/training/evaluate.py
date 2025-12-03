"""
Script d'évaluation du modèle sur l'ensemble de test
"""
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
import seaborn as sns

# Ajouter le répertoire racine au PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.architecture import CustomMultiHeadCNN
from src.data.dataset import ImageDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, device):
    """Charge le modèle entraîné"""
    checkpoint = torch.load(model_path, map_location=device)
    model = CustomMultiHeadCNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, dataloader, device):
    """Évalue le modèle et retourne les prédictions et labels"""
    model.eval()
    
    all_preds = {
        'beard': [], 'mustache': [], 'glasses': [],
        'hair_color': [], 'hair_length': []
    }
    all_labels = {
        'beard': [], 'mustache': [], 'glasses': [],
        'hair_color': [], 'hair_length': []
    }
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Collect predictions and labels
            for attr in all_preds.keys():
                if attr in ['beard', 'mustache', 'glasses']:
                    # Binary classification
                    preds = torch.sigmoid(outputs[attr]).cpu().numpy()
                    all_preds[attr].extend(preds.flatten())
                    all_labels[attr].extend(labels[attr].cpu().numpy().flatten())
                else:
                    # Multi-class classification
                    preds = torch.softmax(outputs[attr], dim=1).cpu().numpy()
                    all_preds[attr].extend(preds)
                    all_labels[attr].extend(labels[attr].cpu().numpy())
    
    return all_preds, all_labels

def compute_metrics(all_preds, all_labels):
    """Calcule les métriques d'évaluation"""
    metrics = {}
    
    for attr in all_preds.keys():
        if attr in ['beard', 'mustache', 'glasses']:
            # Binary classification metrics
            preds_binary = (np.array(all_preds[attr]) > 0.5).astype(int)
            labels_binary = np.array(all_labels[attr])
            
            accuracy = (preds_binary == labels_binary).mean()
            
            metrics[attr] = {
                'accuracy': float(accuracy),
                'num_samples': len(labels_binary)
            }
        else:
            # Multi-class classification metrics
            preds_class = np.argmax(all_preds[attr], axis=1)
            labels_class = np.array(all_labels[attr])
            
            accuracy = (preds_class == labels_class).mean()
            
            metrics[attr] = {
                'accuracy': float(accuracy),
                'num_samples': len(labels_class)
            }
    
    # Overall accuracy
    all_accuracies = [m['accuracy'] for m in metrics.values()]
    metrics['overall'] = {
        'mean_accuracy': float(np.mean(all_accuracies)),
        'total_attributes': len(all_accuracies)
    }
    
    return metrics

def plot_confusion_matrices(all_preds, all_labels, save_path='plots/confusion_matrices.png'):
    """Génère les matrices de confusion pour tous les attributs"""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    attributes = ['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']
    
    for idx, attr in enumerate(attributes):
        if attr in ['beard', 'mustache', 'glasses']:
            preds = (np.array(all_preds[attr]) > 0.5).astype(int)
            labels = np.array(all_labels[attr])
            cm = confusion_matrix(labels, preds)
            classes = ['No', 'Yes']
        else:
            preds = np.argmax(all_preds[attr], axis=1)
            labels = np.array(all_labels[attr])
            cm = confusion_matrix(labels, preds)
            classes = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        axes[idx].set_title(f'{attr.capitalize()} Confusion Matrix')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide the last subplot if not used
    if len(attributes) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {save_path}")

def plot_roc_curves(all_preds, all_labels, save_path='plots/roc_curves.png'):
    """Génère les courbes ROC pour les attributs binaires"""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    binary_attrs = ['beard', 'mustache', 'glasses']
    
    for idx, attr in enumerate(binary_attrs):
        preds = np.array(all_preds[attr])
        labels = np.array(all_labels[attr])
        
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        
        axes[idx].plot(fpr, tpr, color='darkorange', lw=2,
                      label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].set_title(f'{attr.capitalize()} ROC Curve')
        axes[idx].legend(loc="lower right")
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {save_path}")

def main():
    """Fonction principale d'évaluation"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Évaluation du modèle')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth',
                       help='Chemin vers le modèle entraîné')
    parser.add_argument('--data-path', type=str, default='data/processed/train_data_s1.pt',
                       help='Chemin vers les données')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Proportion de données pour le test (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Taille du batch pour l\'évaluation')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Seed pour la reproductibilité')
    args = parser.parse_args()
    
    print("=" * 60)
    print("EVALUATION DU MODELE")
    print("=" * 60)
    
    # Set random seed for reproducibility
    import random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Create directories
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    print("\n1. Chargement des données...")
    data = torch.load(args.data_path)
    
    # Split into train and test
    # NOTE: Using the last portion as test set. For production, consider using
    # sklearn.model_selection.train_test_split with shuffle=True to avoid bias
    # if the data is ordered by any characteristic.
    total_samples = len(data['images'])
    test_size = int(args.test_split * total_samples)
    
    # Using last samples as test set (data is assumed to be shuffled during preparation)
    test_images = data['images'][-test_size:]
    test_labels = {k: v[-test_size:] for k, v in data['labels'].items()}
    
    test_dataset = ImageDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"   Nombre d'échantillons de test: {len(test_dataset)}")
    print(f"   Taille du batch: {args.batch_size}")
    
    # Load model
    print("\n2. Chargement du modèle...")
    model = load_model(args.model_path, DEVICE)
    print(f"   Modèle chargé depuis: {args.model_path}")
    print(f"   Device: {DEVICE}")
    
    # Evaluate
    print("\n3. Évaluation du modèle...")
    all_preds, all_labels = evaluate_model(model, test_loader, DEVICE)
    
    # Compute metrics
    print("\n4. Calcul des métriques...")
    metrics = compute_metrics(all_preds, all_labels)
    
    # Save metrics
    with open('metrics/eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("   Métriques sauvegardées dans metrics/eval_metrics.json")
    
    # Print metrics
    print("\n" + "=" * 60)
    print("RESULTATS DE L'EVALUATION")
    print("=" * 60)
    for attr, metric in metrics.items():
        if attr != 'overall':
            print(f"{attr.capitalize():15s}: Accuracy = {metric['accuracy']:.4f}")
    print("-" * 60)
    print(f"{'Overall':15s}: Mean Accuracy = {metrics['overall']['mean_accuracy']:.4f}")
    print("=" * 60)
    
    # Generate plots
    print("\n5. Génération des visualisations...")
    plot_confusion_matrices(all_preds, all_labels)
    plot_roc_curves(all_preds, all_labels)
    
    print("\n✓ Évaluation terminée avec succès!")

if __name__ == "__main__":
    main()
