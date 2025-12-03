# Guide des Bonnes Pratiques DVC

## 📚 Introduction

Ce guide présente les bonnes pratiques pour utiliser DVC (Data Version Control) dans ce projet MLOps.

## 🎯 Principes Fondamentaux

### 1. Versionner les Données, Pas le Code

- ✅ **À FAIRE**: Utiliser DVC pour les données, modèles et artefacts lourds
- ❌ **À ÉVITER**: Commiter directement de gros fichiers dans Git

```bash
# Bon
dvc add data/raw
git add data/raw.dvc .gitignore

# Mauvais
git add data/raw/*
```

### 2. Toujours Utiliser un Remote

Configurez un remote storage pour partager les données avec l'équipe:

```bash
# Local (pour tests)
dvc remote add -d local /tmp/dvc-storage

# S3 (production)
dvc remote add -d s3storage s3://my-bucket/dvc-storage
dvc remote modify s3storage region us-east-1

# Google Cloud Storage
dvc remote add -d gcs gs://my-bucket/dvc-storage

# Azure Blob Storage
dvc remote add -d azure azure://my-container/path
```

### 3. Paramétrer avec params.yaml

Tous les hyperparamètres doivent être dans `params.yaml`:

```yaml
train:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
```

Puis utilisez-les dans le code:

```python
import yaml

with open('params.yaml') as f:
    params = yaml.safe_load(f)

epochs = params['train']['epochs']
```

## 🔄 Workflow DVC

### Workflow Quotidien

```bash
# 1. Récupérer les dernières données
dvc pull

# 2. Modifier le code ou les paramètres
nano src/training/train.py
# ou
nano params.yaml

# 3. Reproduire le pipeline
dvc repro

# 4. Vérifier les changements
dvc status
dvc metrics diff

# 5. Commiter les changements
git add dvc.lock params.yaml src/
git commit -m "Amélioration du modèle"

# 6. Pousser les données et le code
dvc push
git push
```

### Expérimentation

DVC permet de tester rapidement différentes configurations:

```bash
# Lancer une expérience
dvc exp run -S train.epochs=20 -S train.batch_size=64

# Comparer les expériences
dvc exp show

# Comparer les métriques
dvc exp diff

# Appliquer une expérience
dvc exp apply exp-12345

# Créer une branche depuis une expérience
dvc exp branch exp-12345 feature/new-model
```

## 📊 Structure du Pipeline

### dvc.yaml Bien Structuré

```yaml
stages:
  stage_name:
    desc: "Description claire du stage"
    cmd: python script.py
    deps:
      - input_file.csv
      - script.py
    params:
      - section.param1
      - section.param2
    outs:
      - output_file.csv:
          desc: "Description de l'output"
          cache: true
    metrics:
      - metrics.json:
          cache: false
    plots:
      - plot.png:
          cache: false
```

### Bonnes Pratiques pour les Stages

1. **Un stage = Une responsabilité**
   - Chaque stage fait une seule chose
   - Facilite le debugging et la réutilisation

2. **Déclarer toutes les dépendances**
   ```yaml
   deps:
     - data/input.csv      # Données
     - src/script.py       # Code
     - src/utils/helper.py # Modules importés
   ```

3. **Typer correctement les outputs**
   - `cache: true` pour les données/modèles
   - `cache: false` pour les métriques/plots

4. **Ajouter des descriptions**
   - Aide à comprendre le pipeline
   - Utile pour la documentation

## 💾 Gestion du Cache

### Nettoyage du Cache

```bash
# Voir l'utilisation du cache
dvc cache size

# Nettoyer les fichiers non utilisés
dvc gc

# Nettoyer en gardant uniquement le workspace actuel
dvc gc -w

# Nettoyer de force
dvc gc -f
```

### Optimisation du Cache

Dans `.dvc/config`:

```ini
[cache]
    type = symlink  # Plus rapide, utilise des liens symboliques
    dir = .dvc/cache
```

## 📈 Métriques et Plots

### Sauvegarder les Métriques

```python
import json

metrics = {
    'accuracy': 0.95,
    'loss': 0.05
}

with open('metrics/train_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

### Visualiser les Métriques

```bash
# Voir les métriques actuelles
dvc metrics show

# Comparer avec la version précédente
dvc metrics diff

# Comparer avec un commit spécifique
dvc metrics diff HEAD~1
```

### Plots

DVC supporte plusieurs formats:

```yaml
plots:
  - plots/accuracy.csv:
      x: epoch
      y: accuracy
  - plots/confusion_matrix.png
```

Visualiser:

```bash
dvc plots show
# Ouvre un fichier HTML avec les plots
```

## 🔐 Sécurité et Secrets

### Ne JAMAIS commiter de secrets

```bash
# ❌ MAUVAIS
dvc remote modify myremote access_key_id AKIAIOSFODNN7EXAMPLE

# ✅ BON - Utiliser des variables d'environnement
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Ou configurer localement (non versionné)
dvc remote modify --local myremote access_key_id $AWS_ACCESS_KEY_ID
```

### Fichiers Sensibles

Ajoutez-les à `.dvcignore`:

```
# .dvcignore
secrets/
*.key
*.pem
credentials.json
```

## 🚀 CI/CD avec DVC

### GitHub Actions Exemple

```yaml
name: DVC Pipeline

on: [push]

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup DVC
        uses: iterative/setup-dvc@v1
        
      - name: Pull data
        run: dvc pull
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      
      - name: Run pipeline
        run: dvc repro
        
      - name: Push results
        run: dvc push
```

## 🐛 Debugging

### Problèmes Courants

#### "Output already exists"

```bash
# Supprimer l'output et refaire
rm -rf output_file
dvc repro stage_name
```

#### "Cannot reproduce"

```bash
# Forcer la reproduction
dvc repro -f stage_name

# Voir les détails
dvc repro -v
```

#### "File not found"

```bash
# Vérifier le statut
dvc status

# Récupérer les fichiers manquants
dvc pull
```

## 📝 Checklist Avant Commit

- [ ] Le pipeline fonctionne: `dvc repro`
- [ ] Les métriques sont meilleures: `dvc metrics diff`
- [ ] Les données sont poussées: `dvc push`
- [ ] Le code est propre: `git status`
- [ ] dvc.lock est à jour: `git add dvc.lock`
- [ ] Le commit est descriptif

## 🔗 Ressources

- [Documentation DVC](https://dvc.org/doc)
- [DVC Tutorial](https://dvc.org/doc/start)
- [DVC with MLflow](https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial)
- [DVC Command Reference](https://dvc.org/doc/command-reference)

## 💡 Tips & Tricks

### 1. Utiliser des Alias

```bash
# Dans .bashrc ou .zshrc
alias dvcr='dvc repro'
alias dvcs='dvc status'
alias dvcm='dvc metrics show'
alias dvcp='dvc pull && dvc repro && dvc push'
```

### 2. Pre-commit Hooks

Créer `.git/hooks/pre-commit`:

```bash
#!/bin/sh
# Vérifier que dvc.lock est à jour
if ! git diff --cached --name-only | grep -q "dvc.lock"; then
    if dvc status | grep -q "changed"; then
        echo "Error: dvc.lock is not up to date"
        exit 1
    fi
fi
```

### 3. Watch Mode

Pour développement rapide:

```bash
# Dans un terminal
watch -n 5 dvc status
```

### 4. Profiling

Mesurer le temps de chaque stage:

```bash
time dvc repro stage_name
```
