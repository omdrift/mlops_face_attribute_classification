"""
Training script with proper path handling
"""
import os
import sys

# Ajouter le répertoire racine au PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import mlflow
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

# Imports depuis src (maintenant ça marche)
from src.models.architecture import CustomMultiHeadCNN
from src.data.dataset import ImageDataset
from src.training.loops import train_one_epoch, validate

import json
import matplotlib.pyplot as plt
from datetime import datetime

def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Génère des courbes d'entraînement"""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_accuracy_curves(acc_history, save_path='accuracy_curves.png'):
    """Génère des courbes d'accuracy par attribut"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    attributes = ['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']
    
    for idx, attr in enumerate(attributes):
        epochs = range(1, len(acc_history[attr]) + 1)
        axes[idx].plot(epochs, acc_history[attr], 'g-', linewidth=2)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Accuracy')
        axes[idx].set_title(f'{attr.replace("_", " ").title()} Accuracy')
        axes[idx].grid(True)
        axes[idx].set_ylim([0, 1])
    
    axes[-1].axis('off')  # Dernier subplot vide
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Configuration
DATA_DIR = "data/raw"
CSV_PATH = "data/annotations/mapped_train.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10

def load_data(csv_path, data_dir):
    """Charge les données prétraitées"""
    import pandas as pd
    from src.data.make_dataset import preprocess_image
    from tqdm import tqdm
    
    df = pd.read_csv(csv_path)
    filenames = df["filename"].values
    
    images = []
    for filename in tqdm(filenames, desc="Loading images"):
        img_path = os.path.join(data_dir, filename)
        img = preprocess_image(img_path, size=64)
        if img is not None:
            images.append(img)
    
    images = np.array(images)
    labels = df[["beard", "mustache", "glasses_binary", 
                 "hair_color_label", "hair_length"]].values
    
    return images, labels

import json 

def load_hyperopt_params():
    """Charge les hyperparamètres optimisés si disponibles"""
    hyperopt_path = 'src/training/hyperopt_params.json'
    
    if os.path.exists(hyperopt_path):
        print(f" Chargement des hyperparamètres optimisés: {hyperopt_path}")
        with open(hyperopt_path, 'r') as f:
            params = json.load(f)
        return params
    else:
        print(" Pas d'hyperparamètres optimisés, utilisation des valeurs par défaut")
        return None

def main():
    """Pipeline d'entraînement"""
    print(f" Device: {DEVICE}")
    print(f" ROOT_DIR: {ROOT_DIR}")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    hyperopt_params = load_hyperopt_params()
    
    if hyperopt_params:
        BATCH_SIZE = hyperopt_params.get('batch_size', 32)
        lr = hyperopt_params.get('lr', 1e-3)
        weight_decay = hyperopt_params.get('weight_decay', 1e-4)
        glasses_weight = hyperopt_params.get('glasses_weight', 2.0)
        hair_color_weight = hyperopt_params.get('hair_color_weight', 0.8)
        dropout = hyperopt_params.get('dropout', 0.3)
    else:
        BATCH_SIZE = 32
        lr = 5e-3
        weight_decay = 5e-4
        glasses_weight = 2.0
        hair_color_weight = 0.8
        dropout = 0.4

    # Charger les données prétraitées si elles existent
    processed_data_path = "data/processed/train_data_s1.pt"

    
    if os.path.exists(processed_data_path):
        print(" Chargement des données prétraitées...")
        data = torch.load(processed_data_path)
        X = data['X'].numpy()
        y = data['y'].numpy()

    
    print(f" {len(X)} images chargées")
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f" Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Transforms
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # ← Augmenté
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ← NOUVEAU
        transforms.RandomErasing(p=0.1),  # ← NOUVEAU (masque aléatoire)

    ])


    
    
    # Datasets
    train_dataset = ImageDataset(X_train, y_train, transform=train_transforms, is_preprocessed=True)
    val_dataset = ImageDataset(X_val, y_val, transform=None, is_preprocessed=True)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    # Modèle
    print(" Initialisation du modèle...")
    model = CustomMultiHeadCNN(n_color=5, n_length=3, dropout=dropout).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f" Paramètres: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Divise par 2 le LR
        patience=3,  # Attend 3 epochs sans amélioration
        min_lr=1e-6,
    )
    

    # MLflow
    mlflow.set_experiment("Projet_Visage_MultiHead")
    
    # Historique pour les graphiques
    train_losses_history = []
    val_losses_history = []
    acc_history = {
        'beard': [], 'mustache': [], 'glasses': [], 
        'hair_color': [], 'hair_length': []
    }
    
    with mlflow.start_run(run_name=f"multi_head_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log params
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "scheduler": "CosineAnnealingWarmRestarts",
            "patience": 5,
            "device": str(DEVICE),
            "using_hyperopt": hyperopt_params is not None,
            "num_params": sum(p.numel() for p in model.parameters()),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset)
        })

        
        # Log model architecture as text
        with open('model_summary.txt', 'w') as f:
            f.write(str(model))
            f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        mlflow.log_artifacts('.', artifact_path='model_info')  # ✅ Nouvelle API
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience = 7
        patience_counter = 0
        
        print("\n Début de l'entraînement...\n")
        
        for epoch in range(1, EPOCHS + 1):
            print(f"{'='*60}")
            print(f"EPOCH {epoch}/{EPOCHS}")
            print(f"{'='*60}")
            
            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, DEVICE, epoch
            )
            
            # Validate
            val_loss, val_accs = validate(model, val_loader, DEVICE)
            
            # Sauvegarder l'historique
            train_losses_history.append(train_loss)
            val_losses_history.append(val_loss)
            for k in acc_history:
                acc_history[k].append(val_accs[k])
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            
            # Log accuracies et losses par attribut
            for k, v in val_accs.items():
                mlflow.log_metric(f"acc_{k}", v, step=epoch)
            
            # Calculer l'accuracy moyenne
            avg_acc = sum(val_accs.values()) / len(val_accs)
            mlflow.log_metric("avg_accuracy", avg_acc, step=epoch)
            
            # Log overfitting indicator
            overfitting = train_loss - val_loss
            mlflow.log_metric("overfitting", overfitting, step=epoch)
            
            # Print results
            print(f"\n Résultats:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Avg Acc:    {avg_acc:.4f} ({avg_acc*100:.1f}%)")
            print(f"   LR:         {optimizer.param_groups[0]['lr']:.2e}")
            
            print(f"\n Accuracies détaillées:")
            for attr, acc in val_accs.items():
                print(f"   {attr:12}: {acc:.4f} ({acc*100:.1f}%)")
            
            # Save best model
            if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    if improvement > 0.001:  # Au moins 0.1% d'amélioration
                        best_val_loss = val_loss
                        best_epoch = epoch
                        patience_counter = 0
                        
                        os.makedirs('models', exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'val_loss': val_loss,
                            'val_accs': val_accs,
                            'train_loss': train_loss
                        }, 'models/best_model.pth')
                
                        print(f"\n Nouveau meilleur modèle! Amélioration: {improvement:.4f}")

                    else:
                        patience_counter += 1
                        print(f"\n Amélioration < 0.001, patience: {patience_counter}/{patience}")
            else:
                patience_counter += 1
                print(f"\n Pas d'amélioration, patience: {patience_counter}/{patience}")
            
            if train_loss < 0.1 and val_loss > 1.0:
                    print(f"\n OVERFITTING DÉTECTÉ!")
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss: {val_loss:.4f}")
                    print(f"   Arrêt précoce")
                    
                    break
            
            if patience_counter >= patience:
                    print(f"\n Early stopping à l'epoch {epoch}")
                    
                    break
        # Générer et logger les graphiques
        print(f"\n Génération des graphiques...")
        plot_training_curves(train_losses_history, val_losses_history, 'training_curves.png')
        mlflow.log_artifacts('.', artifact_path='plots')  # Log tous les .png
        
        plot_accuracy_curves(acc_history, 'accuracy_curves.png')
        mlflow.log_artifacts('.', artifact_path='plots')  # Log tous les .png
        
        print(f"\n{'='*60}")
        print(" Entraînement terminé!")
        print(f"{'='*60}")
        print(f" Meilleur val_loss: {best_val_loss:.4f} (epoch {best_epoch})")
        print(f" Total epochs: {epoch}")
        
        # Save detailed metrics
        final_metrics = {
            'best_val_loss': float(best_val_loss),
            'best_epoch': int(best_epoch),
            'final_epoch': int(epoch),
            'best_accuracies': {k: float(v) for k, v in val_accs.items()},
            'avg_best_accuracy': float(sum(val_accs.values()) / len(val_accs)),
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'early_stopped': patience_counter >= patience
        }
        
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/train_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        mlflow.log_artifacts('metrics', artifact_path='metrics')
        
        # Log final metrics summary
        mlflow.log_metric("final_best_val_loss", float(best_val_loss))
        mlflow.log_metric("final_avg_accuracy", float(sum(val_accs.values()) / len(val_accs)))
        mlflow.log_metric("total_epochs", int(epoch))

if __name__ == '__main__':
    main()