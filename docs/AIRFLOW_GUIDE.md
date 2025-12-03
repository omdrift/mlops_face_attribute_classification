# Guide Apache Airflow pour le Projet MLOps

## 📚 Introduction

Ce guide explique comment utiliser Apache Airflow pour orchestrer le pipeline de Machine Learning.

## 🏗️ Architecture Airflow

### Composants

- **Webserver**: Interface web pour visualiser et gérer les DAGs
- **Scheduler**: Planifie et déclenche les tâches
- **Database**: Stocke l'état des DAGs et des tâches (PostgreSQL)
- **Executor**: Exécute les tâches (LocalExecutor dans notre cas)

### Structure des Fichiers

```
airflow/
├── dags/                       # DAGs Airflow
│   ├── ml_pipeline_dag.py      # Pipeline ML principal
│   └── monitoring_dag.py       # Surveillance automatique
├── logs/                       # Logs d'exécution
├── plugins/                    # Plugins personnalisés (si nécessaire)
└── config/                     # Configuration additionnelle
```

## 🚀 Installation et Démarrage

### Première Installation

```bash
# 1. Initialiser Airflow
./scripts/init_airflow.sh

# 2. Démarrer les services
docker-compose up -d

# 3. Vérifier l'état
docker-compose ps
```

### Accès à l'Interface Web

- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

### Commandes Utiles

```bash
# Démarrer Airflow
docker-compose up -d

# Arrêter Airflow
docker-compose down

# Voir les logs
docker-compose logs -f

# Redémarrer un service
docker-compose restart airflow-scheduler

# Nettoyer tout (⚠️ supprime la BDD)
docker-compose down -v
```

## 📊 DAGs Disponibles

### 1. ml_pipeline_face_attributes

**Description**: Pipeline complet de ML pour la classification d'attributs faciaux.

**Planification**: Quotidienne (`@daily` à minuit)

**Étapes**:

1. **check_environment**: Vérifie que DVC et les fichiers nécessaires sont présents
2. **check_raw_data**: Attend que les données brutes soient disponibles
3. **data_preparation**: 
   - Pull des données avec DVC
   - Préparation des données d'entraînement
4. **hyperparameter_optimization**: Optimisation avec Hyperopt
5. **model_training**: 
   - Entraînement du modèle
   - Push du modèle vers DVC remote
6. **model_evaluation**: 
   - Évaluation du modèle
   - Archivage des métriques et plots
7. **notify_success**: Notification de fin

**Déclenchement manuel**:

```bash
# Via l'interface web
# DAGs → ml_pipeline_face_attributes → Trigger DAG

# Via CLI
docker-compose exec airflow-scheduler \
  airflow dags trigger ml_pipeline_face_attributes
```

### 2. model_monitoring_and_retraining

**Description**: Surveille les performances et déclenche un ré-entraînement si nécessaire.

**Planification**: Hebdomadaire (`@weekly`)

**Étapes**:

1. **check_data_drift**: Détecte le data drift (à implémenter avec Evidently)
2. **check_performance**: Compare la performance actuelle au seuil requis (85%)
3. **trigger_retraining** OU **skip_retraining**: Décision basée sur la performance
4. **log_monitoring_results**: Sauvegarde les résultats
5. **send_notification**: Notifie l'équipe

**Seuils configurables**:

Dans `monitoring_dag.py`:
```python
ACCURACY_THRESHOLD = 0.85  # 85% de précision minimale
```

Dans `.env`:
```bash
ACCURACY_THRESHOLD=0.85
```

## 🛠️ Création de DAGs Personnalisés

### Structure d'un DAG

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_custom_dag',
    default_args=default_args,
    description='Description du DAG',
    schedule_interval='@daily',
    catchup=False,
    tags=['custom', 'ml'],
)

# Définir les tâches
task1 = BashOperator(
    task_id='task1',
    bash_command='echo "Hello World"',
    dag=dag,
)

task2 = BashOperator(
    task_id='task2',
    bash_command='echo "Task 2"',
    dag=dag,
)

# Définir les dépendances
task1 >> task2
```

### Types d'Opérateurs Utilisés

#### BashOperator

Pour exécuter des commandes shell:

```python
from airflow.operators.bash import BashOperator

run_dvc = BashOperator(
    task_id='run_dvc_repro',
    bash_command='cd /opt/airflow/project && dvc repro',
    dag=dag,
)
```

#### PythonOperator

Pour exécuter du code Python:

```python
from airflow.operators.python import PythonOperator

def my_function(**context):
    print("Exécution de ma fonction")
    return "success"

task = PythonOperator(
    task_id='my_task',
    python_callable=my_function,
    dag=dag,
)
```

#### BranchPythonOperator

Pour des décisions conditionnelles:

```python
from airflow.operators.python import BranchPythonOperator

def decide_branch(**context):
    if condition:
        return 'task_if_true'
    else:
        return 'task_if_false'

branch = BranchPythonOperator(
    task_id='branch_task',
    python_callable=decide_branch,
    dag=dag,
)
```

#### FileSensor

Pour attendre qu'un fichier existe:

```python
from airflow.sensors.filesystem import FileSensor

wait_file = FileSensor(
    task_id='wait_for_file',
    filepath='/path/to/file',
    poke_interval=30,  # Vérifier toutes les 30 secondes
    timeout=600,       # Timeout après 10 minutes
    dag=dag,
)
```

### Task Groups

Organiser des tâches liées:

```python
from airflow.utils.task_group import TaskGroup

with TaskGroup('data_processing', dag=dag) as group:
    task1 = BashOperator(...)
    task2 = BashOperator(...)
    task1 >> task2
```

## 📅 Planification

### Expressions Schedule

```python
# Quotidien à minuit
schedule_interval='@daily'

# Hebdomadaire le dimanche à minuit
schedule_interval='@weekly'

# Mensuel le 1er à minuit
schedule_interval='@monthly'

# Toutes les heures
schedule_interval='@hourly'

# Cron custom (tous les jours à 6h30)
schedule_interval='30 6 * * *'

# Manuel uniquement
schedule_interval=None
```

### Catchup

```python
# Ne pas rattraper les exécutions manquées
catchup=False

# Rattraper toutes les exécutions manquées
catchup=True
```

## 🔍 Monitoring et Debugging

### Voir les Logs

#### Via l'Interface Web

1. DAGs → Sélectionner le DAG
2. Graph → Cliquer sur une tâche
3. Logs

#### Via CLI

```bash
# Logs d'une tâche spécifique
docker-compose exec airflow-scheduler \
  airflow tasks logs ml_pipeline_face_attributes task_id 2024-01-01

# Logs du scheduler
docker-compose logs -f airflow-scheduler

# Logs du webserver
docker-compose logs -f airflow-webserver
```

### États des Tâches

- ⚪ **None**: Pas encore planifiée
- 🟡 **Scheduled**: Planifiée
- 🔵 **Queued**: En file d'attente
- 🟢 **Running**: En cours d'exécution
- ✅ **Success**: Réussie
- ❌ **Failed**: Échouée
- 🔴 **Upstream Failed**: Échec d'une dépendance
- ⏭️ **Skipped**: Sautée

### Commandes de Debug

```bash
# Tester une tâche sans l'exécuter
docker-compose exec airflow-scheduler \
  airflow tasks test ml_pipeline_face_attributes check_environment 2024-01-01

# Lister les DAGs
docker-compose exec airflow-scheduler airflow dags list

# Voir l'état d'un DAG
docker-compose exec airflow-scheduler \
  airflow dags state ml_pipeline_face_attributes 2024-01-01

# Marquer une tâche comme réussie
docker-compose exec airflow-scheduler \
  airflow tasks clear ml_pipeline_face_attributes -t task_id -s 2024-01-01
```

## ⚙️ Configuration Avancée

### Variables Airflow

Stocker des configurations:

```bash
# Via CLI
docker-compose exec airflow-scheduler \
  airflow variables set my_variable my_value

# Via l'interface: Admin → Variables
```

Utiliser dans un DAG:

```python
from airflow.models import Variable

my_var = Variable.get("my_variable")
```

### Connexions

Pour se connecter à des services externes:

```bash
# Via l'interface: Admin → Connections
# Type: Amazon Web Services
# Conn Id: aws_default
# Extra: {"region_name": "us-east-1"}
```

### XComs

Partager des données entre tâches:

```python
# Pousser une valeur
def push_function(**context):
    context['task_instance'].xcom_push(key='my_key', value='my_value')

# Tirer une valeur
def pull_function(**context):
    value = context['task_instance'].xcom_pull(
        task_ids='push_task',
        key='my_key'
    )
```

## 🔐 Sécurité

### Secrets

**NE JAMAIS** hardcoder de secrets dans les DAGs!

```python
# ❌ MAUVAIS
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"

# ✅ BON - Variables d'environnement
import os
AWS_KEY = os.getenv('AWS_ACCESS_KEY_ID')

# ✅ BON - Airflow Variables (avec encryption)
from airflow.models import Variable
AWS_KEY = Variable.get("aws_key")

# ✅ BON - Airflow Connections
from airflow.hooks.base import BaseHook
conn = BaseHook.get_connection('aws_default')
```

### Permissions

Configurer les rôles dans l'interface:
Security → List Roles

## 📧 Notifications

### Email

Configurer dans `docker-compose.yml`:

```yaml
environment:
  AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
  AIRFLOW__SMTP__SMTP_PORT: 587
  AIRFLOW__SMTP__SMTP_USER: your_email@gmail.com
  AIRFLOW__SMTP__SMTP_PASSWORD: your_password
  AIRFLOW__SMTP__SMTP_MAIL_FROM: your_email@gmail.com
```

Dans le DAG:

```python
default_args = {
    'email': ['team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': False,
}
```

### Slack

Utiliser le SlackWebhookOperator:

```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

notify_slack = SlackWebhookOperator(
    task_id='notify_slack',
    http_conn_id='slack_webhook',
    message='Pipeline terminé!',
    dag=dag,
)
```

## 🔄 Bonnes Pratiques

### 1. Idempotence

Les tâches doivent pouvoir être réexécutées:

```python
# ✅ BON - Idempotent
def process_data():
    # Supprimer l'output s'il existe
    if os.path.exists(output_file):
        os.remove(output_file)
    # Créer le nouveau fichier
    create_file(output_file)

# ❌ MAUVAIS - Non idempotent
def process_data():
    # Ajoute à un fichier existant
    with open(output_file, 'a') as f:
        f.write(data)
```

### 2. Task Size

Gardez les tâches petites et focalisées:

```python
# ✅ BON - Tâches séparées
extract_task >> transform_task >> load_task

# ❌ MAUVAIS - Tâche monolithique
big_task_that_does_everything
```

### 3. Logging

Loggez abondamment:

```python
import logging

def my_function(**context):
    logging.info("Début du traitement")
    result = process_data()
    logging.info(f"Traitement terminé: {result}")
    return result
```

### 4. Timeouts

Configurez des timeouts:

```python
task = BashOperator(
    task_id='long_task',
    bash_command='long_running_command',
    execution_timeout=timedelta(hours=2),
    dag=dag,
)
```

### 5. Retries

Configurez des retries appropriés:

```python
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
}
```

## 📝 Checklist Déploiement

- [ ] Tester le DAG localement: `airflow dags test`
- [ ] Vérifier la syntaxe Python
- [ ] Documenter le DAG (docstring)
- [ ] Configurer les retries
- [ ] Ajouter du logging
- [ ] Tester l'idempotence
- [ ] Configurer les notifications
- [ ] Définir les SLAs si nécessaire
- [ ] Tester avec de petits datasets

## 🔗 Ressources

- [Documentation Airflow](https://airflow.apache.org/docs/)
- [Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Airflow Concepts](https://airflow.apache.org/docs/apache-airflow/stable/concepts/index.html)

## 💡 Tips & Tricks

### 1. Développement Local

Tester un DAG sans Docker:

```bash
# Installer Airflow localement
pip install apache-airflow

# Tester la syntaxe
python airflow/dags/my_dag.py

# Tester une tâche
airflow tasks test my_dag task_id 2024-01-01
```

### 2. Pauser des DAGs

```bash
# Via CLI
airflow dags pause my_dag
airflow dags unpause my_dag
```

### 3. Backfill

Exécuter le DAG sur une période passée:

```bash
airflow dags backfill my_dag \
  --start-date 2024-01-01 \
  --end-date 2024-01-31
```

### 4. Clear Tasks

Réinitialiser des tâches pour les réexécuter:

```bash
airflow tasks clear my_dag \
  --start-date 2024-01-01 \
  --end-date 2024-01-31
```
