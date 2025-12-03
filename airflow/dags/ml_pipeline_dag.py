"""
ML Pipeline DAG - Orchestrates the complete ML workflow using DVC
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['mlops@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

# Project root directory - adjust based on your setup
PROJECT_ROOT = '/opt/airflow/project'  # This should be mounted in docker-compose


def notify_completion(**context):
    """Notification function when pipeline completes"""
    execution_date = context['execution_date']
    print(f"✅ ML Pipeline completed successfully!")
    print(f"📅 Execution date: {execution_date}")
    print(f"🎉 All stages finished: prepare_data → hyperopt → train → evaluate → inference")
    return "Pipeline completed successfully"


# Create the DAG
with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Complete ML pipeline orchestration with DVC',
    schedule_interval='@weekly',  # Run weekly, adjust as needed
    catchup=False,
    tags=['ml', 'dvc', 'face-attributes'],
) as dag:
    
    # Task 1: Prepare training data
    prepare_data = BashOperator(
        task_id='prepare_data',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro prepare_train',
        dag=dag,
    )
    
    # Task 2: Hyperparameter optimization
    hyperopt_search = BashOperator(
        task_id='hyperopt_search',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro hyperopt',
        dag=dag,
    )
    
    # Task 3: Train model with best hyperparameters
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro train',
        dag=dag,
    )
    
    # Task 4: Evaluate model on test set
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro evaluate',
        dag=dag,
    )
    
    # Task 5: Run batch inference on all data
    batch_inference = BashOperator(
        task_id='batch_inference',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro inference_batches',
        dag=dag,
    )
    
    # Task 6: Notify completion
    notify_completion_task = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        provide_context=True,
        dag=dag,
    )
    
    # Define task dependencies
    # Linear flow: prepare → hyperopt → train → [evaluate, inference] → notify
    prepare_data >> hyperopt_search >> train_model
    train_model >> [evaluate_model, batch_inference]
    [evaluate_model, batch_inference] >> notify_completion_task


# Optional: Add a separate DAG for retraining only (no hyperopt)
with DAG(
    'ml_retrain_only',
    default_args=default_args,
    description='Quick retrain pipeline (skips hyperopt)',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['ml', 'dvc', 'retrain'],
) as retrain_dag:
    
    # Task 1: Train model (uses existing hyperopt params)
    retrain_model = BashOperator(
        task_id='train_model',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro train',
    )
    
    # Task 2: Evaluate
    retrain_evaluate = BashOperator(
        task_id='evaluate_model',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro evaluate',
    )
    
    # Task 3: Inference
    retrain_inference = BashOperator(
        task_id='batch_inference',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro inference_batches',
    )
    
    # Task 4: Notify
    retrain_notify = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        provide_context=True,
    )
    
    # Dependencies
    retrain_model >> [retrain_evaluate, retrain_inference] >> retrain_notify


# Optional: Add a DAG for inference only
with DAG(
    'ml_inference_only',
    default_args=default_args,
    description='Run inference only on new data',
    schedule_interval='@daily',  # Run daily for new predictions
    catchup=False,
    tags=['ml', 'inference'],
) as inference_dag:
    
    # Single task: Run inference
    inference_only = BashOperator(
        task_id='batch_inference',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro inference_batches',
    )
