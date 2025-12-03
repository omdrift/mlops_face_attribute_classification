"""
ML Pipeline DAG - Main orchestration for face attribute classification

This DAG orchestrates the complete ML pipeline including:
- Data validation
- Preprocessing with DVC
- Training with MLflow
- Model evaluation
- Drift detection
- Conditional deployment

Schedule: Weekly (@weekly)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='Main ML pipeline for face attribute classification',
    schedule_interval='@weekly',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'face-attributes', 'production'],
)


def check_new_data(**context):
    """Check if new data is available"""
    import os
    from pathlib import Path
    
    data_dir = Path('/app/data/raw')
    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    # Check for new images
    images = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
    
    if len(images) == 0:
        raise ValueError("No new data found")
    
    print(f"Found {len(images)} images")
    return len(images)


def validate_data(**context):
    """Validate data using Evidently"""
    import sys
    sys.path.insert(0, '/app/src')
    
    from monitoring.data_drift import check_data_quality
    
    report = check_data_quality()
    
    # Push validation results to XCom
    context['ti'].xcom_push(key='validation_report', value=report)
    
    return report


def check_drift(**context):
    """Check for data and model drift"""
    import sys
    sys.path.insert(0, '/app/src')
    
    from monitoring.evidently_monitoring import EvidentlyMonitor
    
    monitor = EvidentlyMonitor()
    drift_detected = monitor.check_drift_threshold(threshold=0.1)
    
    # Push drift detection result to XCom
    context['ti'].xcom_push(key='drift_detected', value=drift_detected)
    
    if drift_detected:
        print("⚠️ Drift detected! Model retraining recommended.")
    else:
        print("✓ No significant drift detected.")
    
    return drift_detected


def deploy_model(**context):
    """Deploy model if all conditions are met"""
    # Check if drift was detected
    drift_detected = context['ti'].xcom_pull(key='drift_detected', task_ids='check_drift')
    
    if drift_detected:
        print("Drift detected - triggering retraining DAG instead of deployment")
        # Trigger model_retraining_dag
        return "retraining_needed"
    
    # Deploy the model
    print("Deploying model to production...")
    return "deployed"


# Task 1: Check for new data
check_new_data_task = PythonOperator(
    task_id='check_new_data',
    python_callable=check_new_data,
    dag=dag,
)

# Task 2: Validate data with Evidently
validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# Task 3: Preprocess data with DVC
preprocess_task = BashOperator(
    task_id='preprocess',
    bash_command='cd /app && dvc repro prepare_train',
    dag=dag,
)

# Task 4: Train model with MLflow
train_task = BashOperator(
    task_id='train',
    bash_command='cd /app && dvc repro train',
    dag=dag,
)

# Task 5: Evaluate model
evaluate_task = BashOperator(
    task_id='evaluate',
    bash_command='cd /app && python src/training/evaluate.py',
    dag=dag,
)

# Task 6: Check for drift
check_drift_task = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift,
    dag=dag,
)

# Task 7: Deploy model conditionally
deploy_task = PythonOperator(
    task_id='deploy',
    python_callable=deploy_model,
    dag=dag,
)

# Define task dependencies
check_new_data_task >> validate_data_task >> preprocess_task >> train_task >> evaluate_task >> check_drift_task >> deploy_task
