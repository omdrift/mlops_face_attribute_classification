"""
Model Retraining DAG

This DAG is triggered when drift is detected or manually.
It retrains the model, compares with production model, and deploys if better.

Schedule: On demand (triggered by drift detection)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'model_retraining_dag',
    default_args=default_args,
    description='Automatic model retraining triggered by drift',
    schedule_interval=None,  # Triggered manually or by other DAGs
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'retraining', 'automation'],
)


def backup_current_model(**context):
    """Backup the current production model"""
    import shutil
    from pathlib import Path
    from datetime import datetime
    
    model_path = Path('/app/models/best_model.pth')
    if model_path.exists():
        backup_path = Path(f'/app/models/backup/best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, backup_path)
        print(f"✓ Model backed up to {backup_path}")
        return str(backup_path)
    else:
        print("⚠️ No existing model to backup")
        return None


def evaluate_new_model(**context):
    """Evaluate the newly trained model"""
    import json
    from pathlib import Path
    
    metrics_path = Path('/app/metrics/train_metrics.json')
    
    if not metrics_path.exists():
        raise ValueError("Metrics file not found")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract key metrics
    accuracy = metrics.get('test_accuracy', 0)
    
    context['ti'].xcom_push(key='new_model_accuracy', value=accuracy)
    
    print(f"New model accuracy: {accuracy:.4f}")
    return accuracy


def compare_models(**context):
    """Compare new model with production model"""
    import json
    from pathlib import Path
    
    new_accuracy = context['ti'].xcom_pull(key='new_model_accuracy', task_ids='evaluate_new_model')
    
    # Load production model metrics if available
    prod_metrics_path = Path('/app/models/production_metrics.json')
    
    if prod_metrics_path.exists():
        with open(prod_metrics_path, 'r') as f:
            prod_metrics = json.load(f)
        prod_accuracy = prod_metrics.get('accuracy', 0)
    else:
        # No production model yet, use the new one
        prod_accuracy = 0
    
    print(f"Production model accuracy: {prod_accuracy:.4f}")
    print(f"New model accuracy: {new_accuracy:.4f}")
    
    # Deploy if new model is better by at least 0.5%
    if new_accuracy > prod_accuracy + 0.005:
        print("✓ New model is better - deploying")
        return 'deploy_new_model'
    else:
        print("⚠️ New model is not better - keeping production model")
        return 'rollback_model'


def deploy_new_model(**context):
    """Deploy the new model to production"""
    import json
    from pathlib import Path
    
    new_accuracy = context['ti'].xcom_pull(key='new_model_accuracy', task_ids='evaluate_new_model')
    
    # Update production metrics
    prod_metrics_path = Path('/app/models/production_metrics.json')
    prod_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(prod_metrics_path, 'w') as f:
        json.dump({
            'accuracy': new_accuracy,
            'deployment_date': datetime.now().isoformat(),
            'model_version': 'retrained'
        }, f, indent=2)
    
    print("✓ New model deployed to production")
    return "deployed"


def rollback_model(**context):
    """Rollback to previous model"""
    import shutil
    from pathlib import Path
    
    backup_path = context['ti'].xcom_pull(key='backup_path', task_ids='backup_current_model')
    
    if backup_path:
        model_path = Path('/app/models/best_model.pth')
        shutil.copy(backup_path, model_path)
        print(f"✓ Model rolled back from {backup_path}")
    
    return "rollback_complete"


# Task 1: Backup current model
backup_task = PythonOperator(
    task_id='backup_current_model',
    python_callable=backup_current_model,
    provide_context=True,
    dag=dag,
)

# Task 2: Retrain model
retrain_task = BashOperator(
    task_id='retrain_model',
    bash_command='cd /app && dvc repro train',
    dag=dag,
)

# Task 3: Evaluate new model
evaluate_task = PythonOperator(
    task_id='evaluate_new_model',
    python_callable=evaluate_new_model,
    provide_context=True,
    dag=dag,
)

# Task 4: Compare models and decide
compare_task = BranchPythonOperator(
    task_id='compare_models',
    python_callable=compare_models,
    provide_context=True,
    dag=dag,
)

# Task 5a: Deploy new model
deploy_task = PythonOperator(
    task_id='deploy_new_model',
    python_callable=deploy_new_model,
    provide_context=True,
    dag=dag,
)

# Task 5b: Rollback to previous model
rollback_task = PythonOperator(
    task_id='rollback_model',
    python_callable=rollback_model,
    provide_context=True,
    dag=dag,
)

# Define dependencies
backup_task >> retrain_task >> evaluate_task >> compare_task
compare_task >> [deploy_task, rollback_task]
