"""
Training script for face attribute classification
This version has the problems that need to be fixed:
1. Uses mlflow.log_artifacts('./', ...) which logs entire directory
2. Uses mlflow.log_artifacts('.', ...) which logs entire directory  
3. Uses mlflow.log_artifacts('models/', ...) incorrectly
4. No system info logging
5. No advanced visualizations
6. No Model Registry usage
7. Limited metrics
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os

from src.models.model import FaceAttributeClassifier
from src.data.dataset import FaceAttributeDataset


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = torch.zeros(40)
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Calculate per-attribute accuracy
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == target).sum(dim=0).cpu()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracies = (correct / total).numpy()
    
    return avg_loss, accuracies


def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_curves.png')
    plt.close()


def plot_accuracy_curves(accuracies_history):
    """Plot accuracy curves for all attributes"""
    plt.figure(figsize=(12, 8))
    for i in range(min(10, len(accuracies_history[0]))):  # Plot first 10 attributes
        accs = [epoch_accs[i] for epoch_accs in accuracies_history]
        plt.plot(accs, label=f'Attr {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy per Attribute')
    plt.savefig('accuracy_curves.png')
    plt.close()


def save_model_summary(model):
    """Save model summary to file"""
    os.makedirs('models', exist_ok=True)
    with open('model_summary.txt', 'w') as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters())}")


def main():
    # Configuration
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    num_attributes = 40
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datasets
    train_dataset = FaceAttributeDataset(num_samples=800, num_attributes=num_attributes)
    val_dataset = FaceAttributeDataset(num_samples=200, num_attributes=num_attributes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = FaceAttributeClassifier(num_attributes=num_attributes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("face_attribute_classification")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_attributes", num_attributes)
        
        # Training loop
        train_losses = []
        val_losses = []
        accuracies_history = []
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accs = validate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            accuracies_history.append(val_accs)
            
            # Log metrics - LIMITED (problem #7)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("avg_accuracy", val_accs.mean(), step=epoch)
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), 'best_model.pth')
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Avg Acc: {val_accs.mean():.4f}")
        
        # Plot curves
        plot_training_curves(train_losses, val_losses)
        plot_accuracy_curves(accuracies_history)
        
        # Save model summary
        save_model_summary(model)
        
        # PROBLEMS: These log_artifacts calls are incorrect
        # Problem #1: Logs entire current directory
        mlflow.log_artifacts('./', artifact_path='plots')
        
        # Problem #2: Logs entire current directory again
        mlflow.log_artifacts('.', artifact_path='plots')
        
        # Problem #3: Tries to log models/ directory
        mlflow.log_artifacts('models/', artifact_path='model_info')
        
        # Log final metrics
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")


if __name__ == "__main__":
    main()
