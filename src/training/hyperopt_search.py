"""
Script d'optimisation des hyperparamètres avec Hyperopt
Utilise les données prétraitées de DVC (déjà normalisées)
"""
import os
import sys
import json
import torch
import mlflow
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Ajouter le répertoire racine au PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.architecture import CustomMultiHeadCNN
from src.data.dataset import ImageDataset
from src.training.loops import train_one_epoch, validate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Espace de recherche des hyperparamètres
SEARCH_SPACE = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(5e-3)),
    'batch_size': hp.choice('batch_size', [32, 64]),  # Retiré 128 (trop pour GPU)
    'weight_decay': hp.loguniform('weight_decay', np.log(1e-4), np.log(5e-3)),
    'dropout': hp.uniform('dropout', 0.1, 0.6),
    'glasses_weight': hp.uniform('glasses_weight', 1.5, 3.0),
    'hair_color_weight': hp.uniform('hair_color_weight', 0.5, 1.2),
}

# Variable globale pour le compteur de trials
trial_counter = {'count': 0}



_cached_data = None

def load_data():
    """ Charge les données prétraitées UNE SEULE FOIS"""
    global _cached_data
    
    if _cached_data is not None:
        print(f"  Utilisation des données en cache")
        return _cached_data
    
    processed_data_path = "data/processed/train_data_s1.pt"
    
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f" Données prétraitées introuvables: {processed_data_path}\n"
            f"   Exécutez d'abord: dvc repro prepare_train"
        )
    
    print(f" Chargement des données prétraitées de DVC...")
    data = torch.load(processed_data_path)
    
    X = data['X'].numpy()
    y = data['y'].numpy()
    
    print(f" {len(X)} images chargées")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    

    
    # Split et cache
    _cached_data = train_test_split(X, y, test_size=0.2, random_state=42)
    return _cached_data

def cleanup_cuda():
    """🔥 Nettoie complètement le GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection
        import gc
        gc.collect()

def objective(params):
    """Fonction objectif pour Hyperopt"""
    trial_counter['count'] += 1
    trial_num = trial_counter['count']
    
    print(f"\n{'='*60}")
    print(f" HYPEROPT TRIAL #{trial_num}")
    print(f"{'='*60}")
    for k, v in params.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"{'='*60}\n")
    
    # Charger les données prétraitées
    X_train, X_val, y_train, y_val = load_data()
    

    # Datasets avec is_preprocessed=True (pas de conversion PIL)
    train_dataset = ImageDataset(
        X_train, 
        y_train, 
        transform=None,  #  Pas de transform, déjà prétraité
        is_preprocessed=True
    )
    val_dataset = ImageDataset(
        X_val, 
        y_val, 
        transform=None,
        is_preprocessed=True
    )
    
    # DataLoaders
    batch_size = params['batch_size']
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Modèle avec dropout variable
    model = CustomMultiHeadCNN(
        n_color=5, 
        n_length=3,
        dropout=params['dropout']
    ).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Entraînement rapide (3 epochs pour hyperopt)
    QUICK_EPOCHS = 3
    best_val_loss = float('inf')
    best_val_accs = {}
    
    with mlflow.start_run(nested=True, run_name=f"trial_{trial_num:03d}"):
        # Log les hyperparamètres testés
        print(f" Logging params to MLflow...")
        mlflow.log_param("trial_number", trial_num)
        mlflow.log_param("lr", float(params['lr']))
        mlflow.log_param("batch_size", int(params['batch_size']))
        mlflow.log_param("weight_decay", float(params['weight_decay']))
        mlflow.log_param("dropout", float(params['dropout']))
        mlflow.log_param("glasses_weight", float(params['glasses_weight']))
        mlflow.log_param("hair_color_weight", float(params['hair_color_weight']))
        mlflow.log_param("quick_epochs", QUICK_EPOCHS)
        
        print(f" Entraînement du trial {trial_num} ({QUICK_EPOCHS} epochs)...\n")
        
        for epoch in range(1, QUICK_EPOCHS + 1):
            try:
                # Train
                train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
                
                # Validate
                val_loss, val_accs = validate(model, val_loader, DEVICE)
                
                # Log metrics par epoch
                mlflow.log_metric("train_loss", float(train_loss), step=epoch)
                mlflow.log_metric("val_loss", float(val_loss), step=epoch)
                mlflow.log_metric("learning_rate", float(optimizer.param_groups[0]['lr']), step=epoch)
                
                for k, v in val_accs.items():
                    mlflow.log_metric(f"acc_{k}", float(v), step=epoch)
                
                # Moyenne des accuracies
                avg_acc = sum(val_accs.values()) / len(val_accs)
                mlflow.log_metric("avg_accuracy", float(avg_acc), step=epoch)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_accs = val_accs.copy()
                    
            except RuntimeError as e:
                #  Gestion des erreurs CUDA
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    print(f" CUDA ERROR dans le trial {trial_num}: {e}")
                    print(f"   Réinitialisation du GPU...")
                    torch.cuda.empty_cache()
                    # Retourner une très mauvaise loss pour que Hyperopt évite ces params
                    mlflow.log_metric("cuda_error", 1.0)
                    return {'loss': 999.0, 'status': STATUS_OK}
                else:
                    raise
        
        #  Métrique combinée: val_loss + pénalité si acc glasses < 0.7
        penalty = 0
        if 'glasses' in best_val_accs and best_val_accs['glasses'] < 0.70:
            penalty = (0.70 - best_val_accs['glasses']) * 2.0
        
        objective_metric = best_val_loss + penalty
        
        # Log métriques finales
        mlflow.log_metric("best_val_loss", float(best_val_loss))
        mlflow.log_metric("objective_metric", float(objective_metric))
        mlflow.log_metric("penalty", float(penalty))
        
        # Log summary des accuracies finales
        if best_val_accs:
            best_avg_acc = sum(best_val_accs.values()) / len(best_val_accs)
            mlflow.log_metric("best_avg_accuracy", float(best_avg_acc))
            
            for k, v in best_val_accs.items():
                mlflow.log_metric(f"best_acc_{k}", float(v))
        else:
            best_avg_acc = 0.0
        
        print(f"\n{'='*60}")
        print(f" RÉSULTATS TRIAL #{trial_num}")
        print(f"{'='*60}")
        print(f"   Best Val Loss:  {best_val_loss:.4f}")
        print(f"   Objective:      {objective_metric:.4f}")
        print(f"   Penalty:        {penalty:.4f}")
        print(f"   Avg Accuracy:   {best_avg_acc:.4f} ({best_avg_acc*100:.1f}%)")
        
        if best_val_accs:
            print(f"\n   Accuracies finales:")
            for k, v in best_val_accs.items():
                print(f"     {k:12}: {v:.4f} ({v*100:.1f}%)")
        
        print(f"{'='*60}\n")
    
    return {'loss': objective_metric, 'status': STATUS_OK}

def run_hyperopt(max_evals=20):
    """Lance l'optimisation des hyperparamètres"""
    print(f"\n{'='*60}")
    print(f" HYPEROPT - RECHERCHE D'HYPERPARAMÈTRES")
    print(f"{'='*60}")
    print(f"   Nombre d'essais: {max_evals}")
    print(f"   Epochs/essai: 3")
    print(f"   Device: {DEVICE}")
    print(f"   Durée estimée: ~{max_evals * 2} minutes")
    print(f"{'='*60}\n")
    
    # Configuration MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Hyperopt_MultiHead")
    
    print(f" MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f" MLflow experiment: Hyperopt_MultiHead\n")
    
    # RUN PARENT qui contient tous les trials
    with mlflow.start_run(run_name=f"hyperopt_search_{max_evals}_trials") as parent_run:
        print(f" MLflow parent run ID: {parent_run.info.run_id}")
        print(f" MLflow parent run name: {parent_run.info.run_name}\n")
        
        # Log params du parent
        mlflow.log_param("max_evals", max_evals)
        mlflow.log_param("device", str(DEVICE))
        mlflow.log_param("quick_epochs", 3)
        mlflow.log_param("search_algorithm", "TPE")
        
        # Log l'espace de recherche
        search_space_str = {
            'lr': 'loguniform(1e-5, 5e-3)',
            'batch_size': 'choice([32, 64])',
            'weight_decay': 'loguniform(1e-4, 5e-3)',
            'dropout': 'uniform(0.3, 0.6)',
            'glasses_weight': 'uniform(1.5, 3.0)',
            'hair_color_weight': 'uniform(0.5, 1.2)'
        }
        mlflow.log_dict(search_space_str, "search_space.json")
        
        trials = Trials()
        trial_counter['count'] = 0  # Reset counter
        
        print(f" Démarrage de l'optimisation...\n")
        
        best_params = fmin(
            fn=objective,
            space=SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=True
        )
        
        #  Convertir les indices en valeurs réelles
        if 'batch_size' in best_params:
            best_params['batch_size'] = [32, 64][int(best_params['batch_size'])]
        
        # Sauvegarder les meilleurs paramètres
        os.makedirs('src/training', exist_ok=True)
        with open('src/training/hyperopt_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Log artifact (nouvelle API)
        mlflow.log_artifacts('src/training', artifact_path='best_params')
        
        # Log les meilleurs params dans le run parent
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)
        
        #  Log statistiques de l'optimisation
        all_losses = [trial['result']['loss'] for trial in trials.trials]
        best_loss = min(all_losses)
        worst_loss = max(all_losses)
        avg_loss = sum(all_losses) / len(all_losses)
        
        mlflow.log_metric("best_objective", float(best_loss))
        mlflow.log_metric("worst_objective", float(worst_loss))
        mlflow.log_metric("avg_objective", float(avg_loss))
        mlflow.log_metric("std_objective", float(np.std(all_losses)))
        
        # Trouver le meilleur trial
        best_trial_idx = np.argmin(all_losses)
        mlflow.log_metric("best_trial_number", int(best_trial_idx + 1))
        
        print(f"\n{'='*60}")
        print(" OPTIMISATION TERMINÉE!")
        print(f"{'='*60}")
        print(f"\n Statistiques globales:")
        print(f"   Total trials:    {len(trials.trials)}")
        print(f"   Best objective:  {best_loss:.4f} (trial #{best_trial_idx + 1})")
        print(f"   Worst objective: {worst_loss:.4f}")
        print(f"   Average:         {avg_loss:.4f}")
        print(f"   Std dev:         {np.std(all_losses):.4f}")
        
        print(f"\n MEILLEURS HYPERPARAMÈTRES:")
        print(f"{'='*60}")
        for k, v in best_params.items():
            print(f"   {k:18}: {v:.6f}" if isinstance(v, float) else f"   {k:18}: {v}")
        
        print(f"\n Sauvegardés dans: src/training/hyperopt_params.json")
        print(f" Lancez maintenant: dvc repro train")
        print(f"{'='*60}\n")
        
        return best_params

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-evals', type=int, default=10, help='Nombre d\'essais Hyperopt')
    args = parser.parse_args()
    
    run_hyperopt(max_evals=args.max_evals)