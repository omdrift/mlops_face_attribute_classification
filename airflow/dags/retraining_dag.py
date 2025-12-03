"""
Retraining DAG - Conditional execution
Triggers model retraining when drift or performance degradation is detected
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
import os
import json


default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['alerts@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_retraining_criteria(**context):
    """Determine if model retraining is needed"""
    print("Checking retraining criteria...")
    
    # Check drift reports
    drift_threshold = 0.15
    performance_threshold = 0.80
    
    needs_retraining = False
    reasons = []
    
    # Check for recent drift reports
    reports_dir = 'reports/evidently'
    if os.path.exists(reports_dir):
        # Look for JSON summary files
        import glob
        recent_reports = sorted(glob.glob(f'{reports_dir}/*.json'), reverse=True)[:1]
        
        if recent_reports:
            with open(recent_reports[0], 'r') as f:
                drift_data = json.load(f)
            
            drift_share = drift_data.get('drift_share', 0)
            if drift_share > drift_threshold:
                needs_retraining = True
                reasons.append(f"Data drift detected: {drift_share:.2%} > {drift_threshold:.2%}")
    
    # Check model performance
    metrics_path = 'metrics/train_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        avg_accuracy = metrics.get('avg_best_accuracy', 1.0)
        if avg_accuracy < performance_threshold:
            needs_retraining = True
            reasons.append(f"Performance degradation: {avg_accuracy:.2%} < {performance_threshold:.2%}")
    
    # Check time since last training
    if os.path.exists('models/best_model.pth'):
        import time
        model_age_days = (time.time() - os.path.getmtime('models/best_model.pth')) / 86400
        if model_age_days > 30:
            needs_retraining = True
            reasons.append(f"Model is {model_age_days:.0f} days old (> 30 days)")
    
    if needs_retraining:
        print(f"✓ Retraining needed. Reasons:")
        for reason in reasons:
            print(f"  • {reason}")
        return 'trigger_retraining'
    else:
        print("✗ No retraining needed - all criteria met")
        return 'skip_retraining'


def trigger_retraining_task(**context):
    """Trigger the retraining process"""
    print("Triggering model retraining...")
    print("✓ Retraining pipeline initiated")


def skip_retraining_task(**context):
    """Skip retraining"""
    print("⊘ Retraining skipped - criteria not met")


def notify_retraining_complete(**context):
    """Send notification that retraining is complete"""
    print("Sending retraining completion notification...")
    
    # Placeholder for actual notification
    notification = {
        'timestamp': datetime.now().isoformat(),
        'event': 'retraining_complete',
        'message': 'Model retraining completed successfully'
    }
    
    print(f"✓ Notification sent: {notification['message']}")


with DAG(
    'retraining_pipeline',
    default_args=default_args,
    description='Conditional retraining based on drift and performance',
    schedule_interval='0 2 * * 1',  # Run every Monday at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['retraining', 'conditional', 'drift'],
) as dag:
    
    # Task 1: Check if retraining is needed
    check_criteria = BranchPythonOperator(
        task_id='check_retraining_criteria',
        python_callable=check_retraining_criteria,
        provide_context=True,
    )
    
    # Task 2a: Trigger retraining (conditional)
    trigger_retraining = PythonOperator(
        task_id='trigger_retraining',
        python_callable=trigger_retraining_task,
        provide_context=True,
    )
    
    # Task 2b: Skip retraining (conditional)
    skip_retraining = PythonOperator(
        task_id='skip_retraining',
        python_callable=skip_retraining_task,
        provide_context=True,
    )
    
    # Task 3: Run DVC pipeline (after triggering)
    run_dvc_pipeline = BashOperator(
        task_id='run_dvc_pipeline',
        bash_command='cd {{ params.project_dir }} && dvc repro',
        params={'project_dir': os.getenv('PROJECT_DIR', '/opt/airflow/mlops')},
        trigger_rule='none_failed',
    )
    
    # Task 4: Train model
    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd {{ params.project_dir }} && python src/training/train.py',
        params={'project_dir': os.getenv('PROJECT_DIR', '/opt/airflow/mlops')},
        execution_timeout=timedelta(hours=1),
    )
    
    # Task 5: Notify completion
    notify_complete = PythonOperator(
        task_id='notify_retraining_complete',
        python_callable=notify_retraining_complete,
        provide_context=True,
        trigger_rule='none_failed',
    )
    
    # Define task dependencies
    check_criteria >> [trigger_retraining, skip_retraining]
    trigger_retraining >> run_dvc_pipeline >> train_model >> notify_complete
