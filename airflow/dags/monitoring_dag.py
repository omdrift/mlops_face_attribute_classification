"""
DAG de surveillance et ré-entraînement automatique

Ce DAG surveille les performances du modèle et déclenche un ré-entraînement
si nécessaire.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
import json
import os

# Configuration
# Use current directory by default, can be overridden with PROJECT_DIR env var
PROJECT_DIR = os.getenv('PROJECT_DIR', os.getcwd())
ACCURACY_THRESHOLD = float(os.getenv('ACCURACY_THRESHOLD', '0.85'))  # Seuil de performance minimal

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'model_monitoring_and_retraining',
    default_args=default_args,
    description='Surveillance des performances et ré-entraînement automatique',
    schedule_interval='@weekly',  # Vérification hebdomadaire
    catchup=False,
    tags=['monitoring', 'retraining', 'mlops'],
)

def check_model_performance(**context):
    """
    Vérifie les performances du modèle et décide s'il faut ré-entraîner
    """
    metrics_path = os.path.join(PROJECT_DIR, 'metrics/eval_metrics.json')
    
    if not os.path.exists(metrics_path):
        print("⚠️  Pas de métriques disponibles, déclenchement du ré-entraînement")
        return 'trigger_retraining'
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    mean_accuracy = metrics['overall']['mean_accuracy']
    print(f"Performance actuelle: {mean_accuracy:.4f}")
    print(f"Seuil requis: {ACCURACY_THRESHOLD:.4f}")
    
    if mean_accuracy < ACCURACY_THRESHOLD:
        print("⚠️  Performance en dessous du seuil, ré-entraînement nécessaire")
        return 'trigger_retraining'
    else:
        print("✓ Performance satisfaisante, pas de ré-entraînement nécessaire")
        return 'skip_retraining'

def check_data_drift(**context):
    """
    Vérifie s'il y a du data drift (à implémenter avec Evidently)
    """
    # TODO: Implémenter la détection de data drift avec Evidently
    print("Vérification du data drift (à implémenter)")
    return True

def log_monitoring_results(**context):
    """
    Log les résultats de la surveillance
    """
    ti = context['task_instance']
    performance_check = ti.xcom_pull(task_ids='check_performance')
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'performance_check': performance_check,
        'action': 'retraining' if performance_check == 'trigger_retraining' else 'no_action'
    }
    
    # Sauvegarder le log
    log_file = os.path.join(PROJECT_DIR, 'logs/monitoring_log.json')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"Log entry saved: {log_entry}")

# Tâche 1: Vérifier les performances
check_performance = BranchPythonOperator(
    task_id='check_performance',
    python_callable=check_model_performance,
    dag=dag,
)

# Tâche 2: Vérifier le data drift
check_drift = PythonOperator(
    task_id='check_data_drift',
    python_callable=check_data_drift,
    dag=dag,
)

# Tâche 3a: Déclencher le ré-entraînement
trigger_retraining = BashOperator(
    task_id='trigger_retraining',
    bash_command=f'cd {PROJECT_DIR} && dvc repro',
    dag=dag,
)

# Tâche 3b: Passer le ré-entraînement
skip_retraining = BashOperator(
    task_id='skip_retraining',
    bash_command='echo "Ré-entraînement non nécessaire"',
    dag=dag,
)

# Tâche 4: Logger les résultats
log_results = PythonOperator(
    task_id='log_monitoring_results',
    python_callable=log_monitoring_results,
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# Tâche 5: Notification
notify = BashOperator(
    task_id='send_notification',
    bash_command='echo "Surveillance terminée - voir logs/monitoring_log.json"',
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# Flux du DAG
check_drift >> check_performance
check_performance >> [trigger_retraining, skip_retraining]
[trigger_retraining, skip_retraining] >> log_results >> notify
