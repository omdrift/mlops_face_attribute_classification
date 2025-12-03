# MLOps Face Attribute Classification

Projet de classification d'attributs faciaux utilisant les meilleures pratiques MLOps avec DVC et Apache Airflow.

## 🎯 Objectif du Projet

Ce projet implémente un pipeline complet de Machine Learning pour la classification d'attributs faciaux:
- Barbe (beard)
- Moustache (mustache)
- Lunettes (glasses)
- Couleur des cheveux (hair_color)
- Longueur des cheveux (hair_length)

## 🏗️ Architecture

```
├── airflow/                    # Configuration Apache Airflow
│   ├── dags/                   # DAGs Airflow
│   │   ├── ml_pipeline_dag.py      # Pipeline ML principal
│   │   └── monitoring_dag.py       # Surveillance et ré-entraînement
│   ├── logs/                   # Logs Airflow
│   ├── plugins/                # Plugins personnalisés
│   └── config/                 # Configuration Airflow
├── data/
│   ├── raw/                    # Données brutes (géré par DVC)
│   ├── processed/              # Données prétraitées
│   └── annotations/            # Fichiers d'annotations
├── src/
│   ├── data/                   # Scripts de préparation des données
│   ├── models/                 # Architecture du modèle
│   ├── training/               # Scripts d'entraînement
│   ├── inference/              # Scripts d'inférence
│   └── utils/                  # Utilitaires
├── models/                     # Modèles entraînés
├── metrics/                    # Métriques d'évaluation
├── plots/                      # Visualisations
├── notebooks/                  # Notebooks Jupyter
├── dvc.yaml                    # Pipeline DVC
├── params.yaml                 # Paramètres du projet
├── docker-compose.yml          # Configuration Docker pour Airflow
└── requirements.txt            # Dépendances Python
```

## 🚀 Installation

### Prérequis

- Python 3.10+
- pip
- Docker et Docker Compose (pour Airflow)
- Git

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/omdrift/mlops_face_attribute_classification.git
cd mlops_face_attribute_classification

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Utilisation de DVC

### Configuration initiale

```bash
# Initialiser DVC (déjà fait)
dvc init

# Configurer un remote storage (optionnel mais recommandé)
# Option 1: Stockage local
dvc remote add -d storage /path/to/storage

# Option 2: Stockage S3
dvc remote add -d s3storage s3://my-bucket/dvc-storage
dvc remote modify s3storage region us-east-1

# Option 3: Google Drive
dvc remote add -d gdrive gdrive://your-folder-id
```

### Gestion des données

```bash
# Télécharger les données depuis le remote
dvc pull

# Ajouter de nouvelles données
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Add new raw data"

# Pousser les données vers le remote
dvc push
```

### Exécution du pipeline

```bash
# Reproduire tout le pipeline
dvc repro

# Reproduire une étape spécifique
dvc repro prepare_train
dvc repro hyperopt
dvc repro train
dvc repro evaluate

# Voir le graphe du pipeline
dvc dag

# Comparer les métriques entre versions
dvc metrics show
dvc metrics diff
```

### Visualisation des expériences

```bash
# Voir les différentes expériences
dvc exp show

# Comparer plusieurs expériences
dvc exp diff

# Visualiser les métriques et plots
dvc plots show

# Comparer les plots entre expériences
dvc plots diff
```

## 🔄 Utilisation d'Apache Airflow

### Démarrage d'Airflow avec Docker

```bash
# Démarrer tous les services Airflow
docker-compose up -d

# Vérifier l'état des services
docker-compose ps

# Voir les logs
docker-compose logs -f

# Arrêter les services
docker-compose down
```

### Accès à l'interface Web

1. Ouvrez votre navigateur: http://localhost:8080
2. Connectez-vous avec:
   - Username: `airflow`
   - Password: `airflow`

### DAGs disponibles

#### 1. ml_pipeline_face_attributes
Pipeline principal qui exécute:
- ✓ Vérification de l'environnement
- ✓ Vérification des données brutes
- ✓ Préparation des données (DVC)
- ✓ Optimisation des hyperparamètres
- ✓ Entraînement du modèle
- ✓ Évaluation du modèle
- ✓ Archivage des artifacts

**Planification**: Quotidienne (`@daily`)

#### 2. model_monitoring_and_retraining
Pipeline de surveillance qui:
- ✓ Vérifie les performances du modèle
- ✓ Détecte le data drift
- ✓ Déclenche automatiquement le ré-entraînement si nécessaire
- ✓ Log les résultats de la surveillance

**Planification**: Hebdomadaire (`@weekly`)

### Exécution manuelle d'un DAG

```bash
# Via l'interface web: cliquez sur le bouton "Trigger DAG"

# Via CLI (dans le container)
docker-compose exec airflow-scheduler airflow dags trigger ml_pipeline_face_attributes

# Vérifier l'état d'exécution
docker-compose exec airflow-scheduler airflow dags list-runs -d ml_pipeline_face_attributes
```

### Configuration personnalisée

Modifiez les variables d'environnement dans `docker-compose.yml`:

```yaml
environment:
  PROJECT_DIR: /opt/airflow/project
  PYTHONPATH: /opt/airflow/project
  # Ajoutez vos variables personnalisées ici
```

## 📝 Paramètres du Projet

Tous les paramètres sont définis dans `params.yaml`:

```yaml
hyperopt:
  max_evals: 10          # Nombre d'évaluations Hyperopt
  timeout: 3600          # Timeout en secondes
  
train:
  epochs: 10             # Nombre d'époques
  batch_size: 32         # Taille du batch
  learning_rate: 0.001   # Taux d'apprentissage
  
data:
  image_size: 64         # Taille des images
  train_split: 0.8       # Proportion train
  val_split: 0.1         # Proportion validation
  test_split: 0.1        # Proportion test
```

Pour modifier les paramètres:

```bash
# Éditer params.yaml
nano params.yaml

# Rejouer le pipeline avec les nouveaux paramètres
dvc repro

# Comparer les résultats
dvc metrics diff
```

## 📈 Métriques et Visualisations

### Métriques générées

- `metrics/data_stats.json`: Statistiques sur les données
- `metrics/hyperopt_results.json`: Résultats de l'optimisation
- `metrics/train_metrics.json`: Métriques d'entraînement
- `metrics/eval_metrics.json`: Métriques d'évaluation

### Visualisations générées

- `plots/data_distribution.png`: Distribution des attributs
- `plots/training_curves.png`: Courbes de loss
- `plots/accuracy_curves.png`: Courbes d'accuracy
- `plots/confusion_matrices.png`: Matrices de confusion
- `plots/roc_curves.png`: Courbes ROC

## 🔍 Surveillance et Monitoring

### Logs de surveillance

Les logs de surveillance sont sauvegardés dans:
- `logs/monitoring_log.json`: Historique des vérifications
- `airflow/logs/`: Logs des DAGs Airflow

### Seuils de performance

Le modèle est considéré comme nécessitant un ré-entraînement si:
- Accuracy moyenne < 85% (configurable dans `monitoring_dag.py`)
- Détection de data drift

## 🛠️ Bonnes Pratiques

### DVC

1. **Toujours versionner avec Git**: Commitez les fichiers `.dvc` et `dvc.lock`
2. **Utiliser un remote**: Configurez un remote storage pour partager les données
3. **Paramétrer avec params.yaml**: Évitez les valeurs hardcodées
4. **Documenter les métriques**: Ajoutez des descriptions dans `dvc.yaml`
5. **Utiliser dvc experiments**: Pour tester rapidement différentes configurations

### Airflow

1. **Idempotence**: Les tâches doivent pouvoir être rejouées sans effets de bord
2. **Logging**: Loggez abondamment pour faciliter le debugging
3. **Task Groups**: Organisez les tâches liées dans des groupes
4. **Sensors**: Utilisez des sensors pour attendre les dépendances
5. **Retry Strategy**: Configurez des retries appropriés pour les tâches

### Développement

1. **Environnements virtuels**: Toujours utiliser un venv
2. **Tests**: Testez chaque étape du pipeline individuellement
3. **Documentation**: Documentez les changements importants
4. **Versioning**: Utilisez des tags Git pour les versions de production

## 🐛 Dépannage

### Problème: DVC ne trouve pas les données

```bash
# Vérifier la configuration du remote
dvc remote list

# Télécharger les données
dvc pull -v
```

### Problème: Airflow ne démarre pas

```bash
# Vérifier les logs
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler

# Réinitialiser la base de données
docker-compose down -v
docker-compose up -d
```

### Problème: Pipeline échoue

```bash
# Voir les détails de l'erreur
dvc repro -v

# Nettoyer le cache DVC
dvc gc
dvc repro -f  # Force la reproduction
```

## 📚 Ressources

- [Documentation DVC](https://dvc.org/doc)
- [Documentation Apache Airflow](https://airflow.apache.org/docs/)
- [Best Practices MLOps](https://ml-ops.org/)

## 👥 Contributeurs

- MLOps Team

## 📄 Licence

Ce projet est sous licence MIT.
