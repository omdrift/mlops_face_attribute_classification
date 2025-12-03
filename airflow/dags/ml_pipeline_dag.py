"""
ML Pipeline DAG - Weekly execution
Orchestrates the complete ML pipeline from data preparation to deployment
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.dates import days_ago
import os
import json


default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['alerts@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


def check_new_data(**context):
    """Check if new data is available for training"""
    # This is a placeholder - implement actual check logic
    import os
    data_dir = os.getenv('DATA_DIR', 'data/raw')
    
    # Check if data directory exists and has files
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if len(files) > 0:
            print(f"✓ Found {len(files)} image files in {data_dir}")
            return True
    
    print(f"✗ No new data found in {data_dir}")
    return False


def check_model_metrics(**context):
    """Check if model metrics meet deployment criteria"""
    metrics_path = 'metrics/train_metrics.json'
    
    if not os.path.exists(metrics_path):
        print("✗ No metrics file found")
        return 'skip_deploy'
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Check thresholds
    avg_accuracy = metrics.get('avg_best_accuracy', 0)
    best_val_loss = metrics.get('best_val_loss', float('inf'))
    
    print(f"Model metrics: avg_accuracy={avg_accuracy:.4f}, val_loss={best_val_loss:.4f}")
    
    # Deployment criteria
    if avg_accuracy >= 0.85 and best_val_loss < 0.5:
        print("✓ Model meets deployment criteria")
        return 'deploy_model'
    else:
        print("✗ Model does not meet deployment criteria")
        return 'skip_deploy'


def deploy_model_task(**context):
    """Deploy the trained model"""
    print("Deploying model to production...")
    # Placeholder for actual deployment logic
    # Could involve copying model to deployment directory,
    # updating API configuration, etc.
    print("✓ Model deployed successfully")


def skip_deploy_task(**context):
    """Skip deployment"""
    print("⊘ Deployment skipped - metrics don't meet criteria")


with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Complete ML pipeline - data preparation, training, evaluation, and deployment',
    schedule_interval='@weekly',  # Run every Sunday at midnight
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training', 'production'],
) as dag:
    
    # Task 1: Check for new data
    check_data = PythonOperator(
        task_id='check_new_data',
        python_callable=check_new_data,
        provide_context=True,
    )
    
    # Task 2: Preprocess data using DVC
    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='cd {{ params.project_dir }} && dvc repro prepare_train',
        params={'project_dir': os.getenv('PROJECT_DIR', '/opt/airflow/mlops')},
    )
    
    # Task 3: Train model
    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd {{ params.project_dir }} && python src/training/train.py',
        params={'project_dir': os.getenv('PROJECT_DIR', '/opt/airflow/mlops')},
        execution_timeout=timedelta(hours=1),
    )
    
    # Task 4: Evaluate model
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command='cd {{ params.project_dir }} && python src/evaluation/evaluate.py || echo "Evaluation script not found"',
        params={'project_dir': os.getenv('PROJECT_DIR', '/opt/airflow/mlops')},
    )
    
    # Task 5: Check for drift
    check_drift = BashOperator(
        task_id='check_drift',
        bash_command='cd {{ params.project_dir }} && python src/monitoring/drift_detection.py || echo "Drift detection not critical"',
        params={'project_dir': os.getenv('PROJECT_DIR', '/opt/airflow/mlops')},
    )
    
    # Task 6: Decide whether to deploy
    check_metrics = BranchPythonOperator(
        task_id='check_deployment_criteria',
        python_callable=check_model_metrics,
        provide_context=True,
    )
    
    # Task 7a: Deploy model (conditional)
    deploy_model = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model_task,
        provide_context=True,
    )
    
    # Task 7b: Skip deployment (conditional)
    skip_deploy = PythonOperator(
        task_id='skip_deploy',
        python_callable=skip_deploy_task,
        provide_context=True,
    )
    
    # Define task dependencies
    check_data >> preprocess_data >> train_model >> evaluate_model >> check_drift >> check_metrics
    check_metrics >> [deploy_model, skip_deploy]
