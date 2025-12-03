"""
DAG Airflow pour orchestrer le pipeline MLOps de classification d'attributs faciaux

Ce DAG exécute les étapes suivantes:
1. Préparation des données
2. Optimisation des hyperparamètres
3. Entraînement du modèle
4. Évaluation du modèle
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
import os

# Configuration du projet
# Use current directory by default, can be overridden with PROJECT_DIR env var
PROJECT_DIR = os.getenv('PROJECT_DIR', os.getcwd())

# Arguments par défaut pour le DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Définition du DAG
dag = DAG(
    'ml_pipeline_face_attributes',
    default_args=default_args,
    description='Pipeline MLOps complet pour la classification d\'attributs faciaux',
    schedule_interval='@daily',  # Exécution quotidienne
    catchup=False,
    tags=['machine-learning', 'computer-vision', 'dvc'],
)

# Fonction de vérification de l'environnement
def check_environment(**context):
    """Vérifie que l'environnement est correctement configuré"""
    import subprocess
    
    # Vérifier que DVC est installé
    try:
        result = subprocess.run(['dvc', 'version'], capture_output=True, text=True)
        print(f"DVC version: {result.stdout}")
    except FileNotFoundError:
        raise Exception("DVC n'est pas installé. Veuillez installer DVC.")
    
    # Vérifier que les fichiers nécessaires existent
    required_files = [
        'dvc.yaml',
        'params.yaml',
        'data/annotations/mapped_train.csv'
    ]
    
    for file in required_files:
        filepath = os.path.join(PROJECT_DIR, file)
        if not os.path.exists(filepath):
            raise Exception(f"Fichier requis manquant: {file}")
    
    print("✓ Environnement vérifié avec succès")

# Tâche 1: Vérification de l'environnement
check_env = PythonOperator(
    task_id='check_environment',
    python_callable=check_environment,
    dag=dag,
)

# Tâche 2: Vérifier que les données brutes sont disponibles
check_raw_data = FileSensor(
    task_id='check_raw_data',
    filepath=os.path.join(PROJECT_DIR, 'data/annotations/mapped_train.csv'),
    poke_interval=30,
    timeout=300,
    mode='poke',
    dag=dag,
)

# Groupe de tâches pour la préparation des données
with TaskGroup('data_preparation', dag=dag) as data_prep_group:
    
    # Pull des données avec DVC
    dvc_pull = BashOperator(
        task_id='dvc_pull_data',
        bash_command=f'cd {PROJECT_DIR} && dvc pull data/raw.dvc || echo "No remote configured, skipping pull"',
        dag=dag,
    )
    
    # Exécuter la préparation des données
    prepare_data = BashOperator(
        task_id='prepare_training_data',
        bash_command=f'cd {PROJECT_DIR} && dvc repro prepare_train',
        dag=dag,
    )
    
    dvc_pull >> prepare_data

# Groupe de tâches pour l'optimisation des hyperparamètres
with TaskGroup('hyperparameter_optimization', dag=dag) as hyperopt_group:
    
    run_hyperopt = BashOperator(
        task_id='run_hyperopt',
        bash_command=f'cd {PROJECT_DIR} && dvc repro hyperopt',
        dag=dag,
    )

# Groupe de tâches pour l'entraînement
with TaskGroup('model_training', dag=dag) as training_group:
    
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'cd {PROJECT_DIR} && dvc repro train',
        dag=dag,
    )
    
    # Push du modèle entraîné vers le remote DVC
    dvc_push_model = BashOperator(
        task_id='dvc_push_model',
        bash_command=f'cd {PROJECT_DIR} && dvc push models/best_model.pth.dvc || echo "No remote configured, skipping push"',
        dag=dag,
    )
    
    train_model >> dvc_push_model

# Groupe de tâches pour l'évaluation
with TaskGroup('model_evaluation', dag=dag) as evaluation_group:
    
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command=f'cd {PROJECT_DIR} && dvc repro evaluate',
        dag=dag,
    )
    
    # Archiver les métriques
    archive_metrics = BashOperator(
        task_id='archive_metrics',
        bash_command=f'''
            cd {PROJECT_DIR} && 
            mkdir -p artifacts/$(date +%Y%m%d_%H%M%S) &&
            cp -r metrics plots artifacts/$(date +%Y%m%d_%H%M%S)/ &&
            echo "Metrics archived successfully"
        ''',
        dag=dag,
    )
    
    evaluate_model >> archive_metrics

# Notification de succès
notify_success = BashOperator(
    task_id='notify_success',
    bash_command='echo "✓ Pipeline ML exécuté avec succès!"',
    dag=dag,
)

# Définition du flux de tâches
check_env >> check_raw_data >> data_prep_group >> hyperopt_group >> training_group >> evaluation_group >> notify_success
