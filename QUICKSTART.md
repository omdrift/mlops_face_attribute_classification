# Quick Start Guide - MLOps Face Attribute Classification

## 🚀 Démarrage Rapide

Ce guide vous permettra de démarrer rapidement avec le projet.

## 📋 Prérequis

- Python 3.10+
- Docker et Docker Compose
- Git
- 4GB+ RAM disponible

## ⚡ Installation en 5 Minutes

### 1. Cloner le Repository

```bash
git clone https://github.com/omdrift/mlops_face_attribute_classification.git
cd mlops_face_attribute_classification
```

### 2. Installer les Dépendances Python

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configurer DVC (Optionnel)

```bash
# Configurer un remote local (pour tests)
dvc remote add -d storage /tmp/dvc-storage

# Ou configurer S3 (pour production)
# dvc remote add -d s3storage s3://my-bucket/dvc-storage
# dvc remote modify s3storage region us-east-1
```

### 4. Démarrer Airflow

```bash
# Initialiser Airflow
./scripts/init_airflow.sh

# Démarrer les services
docker-compose up -d

# Vérifier que tout fonctionne
docker-compose ps
```

### 5. Accéder à Airflow

Ouvrez votre navigateur: http://localhost:8080

- Username: `airflow`
- Password: `airflow`

## 🎯 Premiers Pas

### Exécuter le Pipeline ML

#### Option 1: Avec Airflow (Recommandé)

1. Ouvrez l'interface Airflow: http://localhost:8080
2. Activez le DAG `ml_pipeline_face_attributes`
3. Cliquez sur "Trigger DAG" pour démarrer

#### Option 2: Avec DVC (Manuel)

```bash
# Reproduire tout le pipeline
dvc repro

# Ou étape par étape
dvc repro prepare_train
dvc repro hyperopt
dvc repro train
dvc repro evaluate
```

### Voir les Résultats

```bash
# Métriques
cat metrics/train_metrics.json
cat metrics/eval_metrics.json

# Visualisations
ls plots/

# Avec DVC
dvc metrics show
dvc plots show
```

## 📊 Structure du Projet (Simplifié)

```
.
├── airflow/              # Orchestration Airflow
│   └── dags/             # DAGs Airflow
├── data/                 # Données (géré par DVC)
├── src/                  # Code source
│   ├── data/             # Préparation des données
│   ├── models/           # Architecture du modèle
│   └── training/         # Entraînement et évaluation
├── models/               # Modèles entraînés
├── metrics/              # Métriques JSON
├── plots/                # Visualisations
├── scripts/              # Scripts utiles
├── dvc.yaml              # Pipeline DVC
├── params.yaml           # Paramètres
└── docker-compose.yml    # Configuration Airflow
```

## 🔧 Configuration Rapide

### Modifier les Hyperparamètres

Éditez `params.yaml`:

```yaml
train:
  epochs: 20        # Changer de 10 à 20
  batch_size: 64    # Changer de 32 à 64
```

Puis relancez:

```bash
dvc repro
# ou via Airflow
```

### Configurer les Notifications Email

Dans `docker-compose.yml`:

```yaml
environment:
  AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
  AIRFLOW__SMTP__SMTP_USER: your_email@gmail.com
  AIRFLOW__SMTP__SMTP_PASSWORD: your_app_password
```

## 📖 Documentation Complète

- [README Principal](README.md) - Vue d'ensemble complète
- [Guide DVC](docs/DVC_BEST_PRACTICES.md) - Bonnes pratiques DVC
- [Guide Airflow](docs/AIRFLOW_GUIDE.md) - Documentation Airflow

## 🛠️ Commandes Utiles

### DVC

```bash
# Helper DVC
./scripts/dvc_helper.sh status       # Statut du pipeline
./scripts/dvc_helper.sh repro        # Reproduire le pipeline
./scripts/dvc_helper.sh metrics      # Voir les métriques
./scripts/dvc_helper.sh plots        # Générer les plots
```

### Airflow

```bash
# Voir les logs
docker-compose logs -f

# Redémarrer un service
docker-compose restart airflow-scheduler

# Arrêter tout
docker-compose down
```

### Pipeline ML

```bash
# Entraîner le modèle
python src/training/train.py

# Évaluer le modèle
python src/training/evaluate.py

# Optimiser les hyperparamètres
python src/training/hyperopt_search.py --max-evals 10
```

## 🐛 Dépannage Rapide

### Airflow ne démarre pas

```bash
docker-compose down -v
./scripts/init_airflow.sh
docker-compose up -d
```

### Erreur "DVC not found"

```bash
pip install dvc
```

### Erreur "Data not found"

```bash
# Si vous avez configuré un remote
dvc pull

# Sinon, assurez-vous que data/raw existe
```

### Port 8080 déjà utilisé

Modifiez dans `docker-compose.yml`:

```yaml
ports:
  - "8081:8080"  # Utilisez 8081 au lieu de 8080
```

## 💡 Prochaines Étapes

1. **Explorer les DAGs Airflow**
   - `ml_pipeline_face_attributes`: Pipeline principal
   - `model_monitoring_and_retraining`: Surveillance automatique

2. **Expérimenter avec DVC**
   ```bash
   # Tester différents paramètres
   dvc exp run -S train.epochs=20 -S train.batch_size=64
   
   # Comparer les résultats
   dvc exp show
   ```

3. **Configurer le Remote Storage**
   - Pour partager les données avec l'équipe
   - S3, GCS, Azure, ou stockage local partagé

4. **Personnaliser les DAGs**
   - Ajouter vos propres étapes
   - Configurer les notifications
   - Ajuster les planifications

## 📞 Support

Pour plus d'informations:

- [Documentation DVC](https://dvc.org/doc)
- [Documentation Airflow](https://airflow.apache.org/docs/)
- Issues GitHub: https://github.com/omdrift/mlops_face_attribute_classification/issues

## ✅ Checklist de Vérification

- [ ] Python installé et environnement virtuel créé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Docker et Docker Compose installés
- [ ] Airflow démarré et accessible (http://localhost:8080)
- [ ] DAG visible dans l'interface Airflow
- [ ] Pipeline DVC testé (`dvc status`)
- [ ] Documentation lue (README.md)

Vous êtes prêt! 🎉
