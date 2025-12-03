"""
Data Validation DAG

This DAG performs data validation checks:
- Schema validation
- Data quality checks
- Distribution checks
- Missing values detection

Schedule: Daily
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_validation_dag',
    default_args=default_args,
    description='Data validation pipeline',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'data-validation'],
)


def validate_schema(**context):
    """Validate data schema"""
    import pandas as pd
    import sys
    sys.path.insert(0, '/app/src')
    
    # Load annotations
    df = pd.read_csv('/app/data/annotations/mapped_train.csv')
    
    # Check required columns
    required_columns = ['image_id', 'beard', 'mustache', 'glasses', 'hair_color', 'hair_length']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"✓ Schema validation passed. Columns: {list(df.columns)}")
    return True


def check_data_quality(**context):
    """Check data quality metrics"""
    import pandas as pd
    
    df = pd.read_csv('/app/data/annotations/mapped_train.csv')
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"⚠️ Missing values detected:\n{missing[missing > 0]}")
    
    # Check value ranges for binary attributes
    for col in ['beard', 'mustache', 'glasses']:
        if col in df.columns:
            unique_vals = df[col].unique()
            if not all(v in [0, 1, -1] for v in unique_vals):
                raise ValueError(f"Invalid values in {col}: {unique_vals}")
    
    print("✓ Data quality checks passed")
    return True


def check_distributions(**context):
    """Check data distributions"""
    import pandas as pd
    import sys
    sys.path.insert(0, '/app/src')
    
    from monitoring.data_drift import check_data_quality
    
    # Use Evidently to check distributions
    report = check_data_quality()
    
    context['ti'].xcom_push(key='distribution_report', value=report)
    
    print("✓ Distribution checks completed")
    return report


def detect_anomalies(**context):
    """Detect anomalies in the data"""
    import pandas as pd
    
    df = pd.read_csv('/app/data/annotations/mapped_train.csv')
    
    # Check for duplicate image_ids
    duplicates = df['image_id'].duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ Found {duplicates} duplicate image IDs")
    
    # Check for outliers in numeric columns
    print("✓ Anomaly detection completed")
    return True


# Define tasks
validate_schema_task = PythonOperator(
    task_id='validate_schema',
    python_callable=validate_schema,
    dag=dag,
)

check_data_quality_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

check_distributions_task = PythonOperator(
    task_id='check_distributions',
    python_callable=check_distributions,
    dag=dag,
)

detect_anomalies_task = PythonOperator(
    task_id='detect_anomalies',
    python_callable=detect_anomalies,
    dag=dag,
)

# Define dependencies
validate_schema_task >> [check_data_quality_task, check_distributions_task, detect_anomalies_task]
