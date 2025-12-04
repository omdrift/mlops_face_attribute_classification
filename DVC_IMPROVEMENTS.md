# Guide d'Amélioration DVC

Ce guide explique comment améliorer et optimiser votre pipeline DVC pour ce projet.

## 🎯 État Actuel du Pipeline

Le pipeline DVC actuel comprend 5 stages :
1. `prepare_train` - Préparation des données d'entraînement
2. `hyperopt` - Optimisation des hyperparamètres
3. `train` - Entraînement du modèle
4. `evaluate` - Évaluation sur le test set
5. `inference_batches` - Prédictions batch sur lots 2-9

## 🚀 Améliorations Recommandées

### 1. Gestion des Versions de Données

**Problème actuel :** Les données brutes ne sont pas versionnées proprement.

**Solution :**
```bash
# Versionner les données brutes avec DVC
dvc add data/raw

# Pousser vers le remote storage (à configurer)
dvc remote add -d myremote s3://mybucket/dvcstore
# ou avec Google Drive, Azure, etc.
dvc push
```

**Bénéfices :**
- Reproductibilité complète
- Partage facile entre équipes
- Historique des versions de données

### 2. Paramétrage Avancé

**Amélioration dans `params.yaml` :**

```yaml
# Ajoutez plus de paramètres pour une meilleure reproductibilité
data:
  seed: 42
  train_split: 0.8
  val_split: 0.2

hyperopt:
  max_evals: 10
  algorithm: tpe  # tree-structured parzen estimator
  timeout: 3600   # 1 heure max

train:
  epochs: 10
  batch_size: 32
  early_stopping_patience: 7
  learning_rate: 0.001
  optimizer: AdamW
  scheduler: ReduceLROnPlateau

evaluate:
  test_size: 0.2
  random_state: 42
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score

inference:
  batch_size: 64
  output_path: outputs/predictions.csv
  confidence_threshold: 0.5
```

### 3. Ajout de Stages de Validation

**Créer `dvc.yaml` amélioré :**

```yaml
stages:
  # ... stages existants ...

  # NOUVEAU: Validation des données
  validate_data:
    cmd: python src/data/validate_data.py
    deps:
      - data/raw
      - data/annotations/mapped_train.csv
      - src/data/validate_data.py
    metrics:
      - metrics/data_quality.json:
          cache: false

  # NOUVEAU: Comparaison de modèles
  compare_models:
    cmd: python src/training/compare_models.py
    deps:
      - models/best_model.pth
      - metrics/eval_metrics.json
      - src/training/compare_models.py
    metrics:
      - metrics/model_comparison.json:
          cache: false

  # NOUVEAU: Tests de performance
  performance_test:
    cmd: python src/testing/test_performance.py
    deps:
      - models/best_model.pth
      - src/testing/test_performance.py
    metrics:
      - metrics/performance.json:
          cache: false
```

### 4. Gestion des Expérimentations

**Utiliser DVC Experiments :**

```bash
# Lancer une expérimentation avec différents paramètres
dvc exp run --set-param train.learning_rate=0.01

# Lancer plusieurs expérimentations en parallèle
dvc exp run --queue --set-param train.batch_size=64
dvc exp run --queue --set-param train.batch_size=128
dvc exp run --queue --set-param train.batch_size=256
dvc exp run --queue --run-all --jobs 4

# Comparer les résultats
dvc exp show --only-changed

# Appliquer la meilleure expérimentation
dvc exp apply exp-12345
```

### 5. Pipelines Conditionnels

**Ajouter des conditions dans `dvc.yaml` :**

```yaml
stages:
  train:
    cmd: python src/training/train.py
    deps:
      - data/processed/train_data_s1.pt
      - src/training/train.py
      - src/training/hyperopt_params.json
    params:
      - train
    outs:
      - models/best_model.pth
    metrics:
      - metrics/train_metrics.json:
          cache: false
    # NOUVEAU: Ne ré-entraîner que si la précision est insuffisante
    frozen: false
```

### 6. Monitoring et Alertes

**Créer `src/monitoring/check_metrics.py` :**

```python
#!/usr/bin/env python
"""Vérifie les métriques et envoie des alertes si nécessaire"""
import json
import sys

def check_metrics():
    with open('metrics/eval_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    avg_acc = metrics['overall']['average_accuracy']
    
    # Seuils d'alerte
    if avg_acc < 0.70:
        print(f"❌ ALERTE: Précision trop faible: {avg_acc:.2%}")
        sys.exit(1)
    elif avg_acc < 0.80:
        print(f"⚠️  ATTENTION: Précision moyenne: {avg_acc:.2%}")
    else:
        print(f"✅ Précision bonne: {avg_acc:.2%}")
    
    return 0

if __name__ == '__main__':
    sys.exit(check_metrics())
```

**Ajouter dans `dvc.yaml` :**

```yaml
stages:
  # ... après evaluate ...
  
  check_quality:
    cmd: python src/monitoring/check_metrics.py
    deps:
      - metrics/eval_metrics.json
      - src/monitoring/check_metrics.py
```

### 7. Documentation Automatique

**Générer des rapports avec DVC :**

```bash
# Générer un rapport HTML des métriques
dvc metrics diff --show-md > reports/metrics_report.md

# Créer un graphe du pipeline
dvc dag --md > reports/pipeline_graph.md

# Exporter les paramètres
dvc params diff --all --show-md > reports/params_report.md
```

### 8. Intégration CI/CD

**Créer `.github/workflows/dvc-pipeline.yml` :**

```yaml
name: DVC Pipeline

on:
  push:
    branches: [ main, development ]
  pull_request:
    branches: [ main ]

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup DVC
        uses: iterative/setup-dvc@v1
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Pull data
        run: dvc pull
        
      - name: Run pipeline
        run: dvc repro
        
      - name: Check metrics
        run: |
          dvc metrics show --show-md >> $GITHUB_STEP_SUMMARY
      
      - name: Publish metrics
        if: github.event_name == 'pull_request'
        uses: iterative/cml-action@v1
        with:
          publish_metrics: true
```

## 📊 Commandes DVC Utiles

### Exécution du Pipeline

```bash
# Exécuter le pipeline complet
dvc repro

# Exécuter jusqu'à un stage spécifique
dvc repro evaluate

# Forcer la ré-exécution d'un stage
dvc repro -f train

# Exécuter en mode dry-run (simulation)
dvc repro --dry
```

### Gestion des Expérimentations

```bash
# Lister les expérimentations
dvc exp list --all

# Montrer les différences entre expérimentations
dvc exp diff exp-1 exp-2

# Supprimer les expérimentations inutiles
dvc exp remove exp-old-*

# Créer une branche à partir d'une expérimentation
dvc exp branch exp-best my-best-model
```

### Métriques et Paramètres

```bash
# Afficher toutes les métriques
dvc metrics show

# Comparer avec une version précédente
dvc metrics diff HEAD~1

# Afficher les paramètres
dvc params show

# Diff des paramètres
dvc params diff main
```

### Gestion des Données

```bash
# Vérifier le statut DVC
dvc status

# Pousser les données vers le remote
dvc push

# Récupérer les données depuis le remote
dvc pull

# Mettre à jour le cache
dvc gc --workspace --cloud
```

## 🔧 Optimisations Avancées

### 1. Cache Intelligent

```bash
# Configurer le cache local
dvc cache dir .dvc/cache

# Partager le cache entre projets
dvc cache dir /shared/dvc-cache

# Configurer la protection du cache
dvc config cache.type hardlink,symlink
```

### 2. Remote Storage

```bash
# Configurer plusieurs remotes
dvc remote add -d production s3://prod-bucket/dvc
dvc remote add backup gs://backup-bucket/dvc

# Configurer les credentials
dvc remote modify production access_key_id XXX
dvc remote modify production secret_access_key YYY
```

### 3. Parallélisation

```bash
# Exécuter les stages en parallèle (si indépendants)
dvc repro --jobs 4

# Configuration permanente
dvc config core.jobs 4
```

## 📈 Métriques de Performance du Pipeline

Pour suivre la performance de votre pipeline DVC :

1. **Temps d'exécution** : Mesuré automatiquement par DVC
2. **Utilisation du cache** : `dvc status` montre les hits/misses
3. **Taille des artefacts** : `du -sh .dvc/cache`
4. **Reproductibilité** : Score basé sur les paramètres versionnés

## 🎓 Ressources

- [Documentation DVC](https://dvc.org/doc)
- [DVC Experiments](https://dvc.org/doc/user-guide/experiment-management)
- [CML pour CI/CD](https://cml.dev/)
- [DVC Studio](https://studio.iterative.ai/) - Interface web pour DVC

## 💡 Prochaines Étapes

1. ✅ Configurer un remote storage (S3, GCS, Azure)
2. ✅ Implémenter les stages de validation
3. ✅ Ajouter le monitoring des métriques
4. ✅ Configurer l'intégration CI/CD
5. ✅ Documenter les expérimentations
6. ✅ Optimiser le cache et la parallélisation
