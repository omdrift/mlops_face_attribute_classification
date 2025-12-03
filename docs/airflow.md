# Airflow Guide

This guide explains how to set up and use Apache Airflow for orchestrating ML pipelines.

## Overview

Airflow manages the following workflows:

- **ML Pipeline DAG**: Main pipeline (data validation, training, drift detection, deployment)
- **Data Validation DAG**: Data quality and schema checks
- **Model Retraining DAG**: Automatic retraining triggered by drift

## Installation

### Using Docker Compose

```bash
# Start Airflow services
docker-compose -f airflow/docker-compose.airflow.yml up -d

# Or use the full stack
docker-compose -f docker-compose.full.yml up -d
```

### Standalone Installation

```bash
# Install Airflow
pip install apache-airflow==2.7.3

# Install dependencies
pip install -r airflow/requirements-airflow.txt

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start webserver
airflow webserver --port 8080

# Start scheduler (in another terminal)
airflow scheduler
```

## Accessing Airflow

- **Web UI**: http://localhost:8080
- **Username**: `admin`
- **Password**: `admin`

## DAG Descriptions

### 1. ML Pipeline DAG (`ml_pipeline_dag.py`)

**Schedule**: Weekly (@weekly)

**Tasks**:

1. **check_new_data**: Verify new data availability
2. **validate_data**: Run Evidently data validation
3. **preprocess**: Run DVC preprocessing stage
4. **train**: Train model with MLflow tracking
5. **evaluate**: Evaluate model performance
6. **check_drift**: Detect data/model drift
7. **deploy**: Conditionally deploy model

**Dependencies**:
```
check_new_data → validate_data → preprocess → train → evaluate → check_drift → deploy
```

**Usage**:
```bash
# Trigger manually
airflow dags trigger ml_pipeline_dag

# View logs
airflow tasks logs ml_pipeline_dag check_new_data <execution_date>
```

### 2. Data Validation DAG (`data_validation_dag.py`)

**Schedule**: Daily (@daily)

**Tasks**:

1. **validate_schema**: Check required columns exist
2. **check_data_quality**: Validate data ranges and types
3. **check_distributions**: Compare with reference distributions
4. **detect_anomalies**: Find duplicates and outliers

**Dependencies**:
```
validate_schema → [check_data_quality, check_distributions, detect_anomalies]
```

**Usage**:
```bash
# Trigger manually
airflow dags trigger data_validation_dag

# View task status
airflow tasks state data_validation_dag validate_schema <execution_date>
```

### 3. Model Retraining DAG (`model_retraining_dag.py`)

**Schedule**: On-demand (triggered by drift detection)

**Tasks**:

1. **backup_current_model**: Backup production model
2. **retrain_model**: Train new model
3. **evaluate_new_model**: Evaluate new model
4. **compare_models**: Compare with production model
5. **deploy_new_model** or **rollback_model**: Deploy if better, rollback if not

**Dependencies**:
```
backup_current_model → retrain_model → evaluate_new_model → compare_models
                                                                    ├→ deploy_new_model
                                                                    └→ rollback_model
```

**Usage**:
```bash
# Trigger manually
airflow dags trigger model_retraining_dag

# Trigger from another DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger = TriggerDagRunOperator(
    task_id='trigger_retraining',
    trigger_dag_id='model_retraining_dag',
)
```

## Configuration

### Airflow Config (`airflow/config/airflow.cfg`)

Key configurations:

```ini
[core]
dags_folder = /opt/airflow/dags
executor = LocalExecutor
sql_alchemy_conn = postgresql+psycopg2://airflow:airflow@postgres:5432/airflow

[webserver]
base_url = http://localhost:8080
web_server_port = 8080

[scheduler]
dag_dir_list_interval = 300
```

### Environment Variables

Set in docker-compose or `.env`:

```bash
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
MLFLOW_TRACKING_URI=http://mlflow:5000
```

## Creating Custom DAGs

### Basic DAG Template

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_custom_dag',
    default_args=default_args,
    description='Custom DAG description',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['custom', 'ml'],
)

def my_task_function(**context):
    # Task logic
    print("Executing custom task")
    return "Task completed"

task1 = PythonOperator(
    task_id='task1',
    python_callable=my_task_function,
    provide_context=True,
    dag=dag,
)

task2 = BashOperator(
    task_id='task2',
    bash_command='echo "Hello from Airflow"',
    dag=dag,
)

task1 >> task2
```

### Using XCom for Task Communication

```python
def task1(**context):
    result = {'key': 'value'}
    context['ti'].xcom_push(key='my_result', value=result)

def task2(**context):
    result = context['ti'].xcom_pull(key='my_result', task_ids='task1')
    print(f"Retrieved: {result}")
```

### Conditional Tasks (Branching)

```python
from airflow.operators.python import BranchPythonOperator

def choose_branch(**context):
    # Logic to choose branch
    if condition:
        return 'task_a'
    else:
        return 'task_b'

branch = BranchPythonOperator(
    task_id='branch',
    python_callable=choose_branch,
    dag=dag,
)

task_a = PythonOperator(...)
task_b = PythonOperator(...)

branch >> [task_a, task_b]
```

## Sensors and Triggers

### File Sensor

```python
from airflow.sensors.filesystem import FileSensor

wait_for_file = FileSensor(
    task_id='wait_for_file',
    filepath='/path/to/file.csv',
    poke_interval=60,
    timeout=600,
    dag=dag,
)
```

### External Task Sensor

```python
from airflow.sensors.external_task import ExternalTaskSensor

wait_for_other_dag = ExternalTaskSensor(
    task_id='wait_for_other_dag',
    external_dag_id='other_dag_id',
    external_task_id='task_id',
    dag=dag,
)
```

## Connections and Variables

### Setting Connections

Via UI:
1. Go to Admin → Connections
2. Add connection (e.g., AWS, GCP, database)

Via CLI:
```bash
airflow connections add 'my_connection' \
    --conn-type 'postgres' \
    --conn-host 'localhost' \
    --conn-login 'user' \
    --conn-password 'password' \
    --conn-port '5432'
```

### Using Variables

Set via UI or CLI:
```bash
airflow variables set my_variable "my_value"
```

Use in DAG:
```python
from airflow.models import Variable

my_var = Variable.get("my_variable")
```

## Monitoring DAGs

### View DAG Status

```bash
# List all DAGs
airflow dags list

# View DAG details
airflow dags show ml_pipeline_dag

# View task instances
airflow tasks list ml_pipeline_dag
```

### Logs

```bash
# View task logs
airflow tasks logs ml_pipeline_dag task_id execution_date

# Tail logs
tail -f /opt/airflow/logs/ml_pipeline_dag/task_id/execution_date/1.log
```

### Metrics

Airflow exposes metrics at `/admin/metrics` (requires StatsD setup).

## Best Practices

### DAG Design

1. **Idempotency**: Tasks should be rerunnable
2. **Atomicity**: Keep tasks small and focused
3. **Error Handling**: Use retries and failure callbacks
4. **Logging**: Log important information
5. **Testing**: Test tasks independently

### Performance

1. **Parallelism**: Set appropriate `max_active_runs`
2. **Task Instances**: Limit concurrent tasks
3. **Database**: Use production-ready database (PostgreSQL)
4. **Executor**: Use CeleryExecutor for distributed execution

### Security

1. **Connections**: Use Airflow Connections for credentials
2. **Variables**: Store sensitive data as Variables
3. **RBAC**: Enable role-based access control
4. **Fernet Key**: Set a strong encryption key

## Troubleshooting

### DAG not appearing

1. Check DAG file syntax: `python /path/to/dag.py`
2. Check DAG folder: Verify `dags_folder` in config
3. Check for errors: View webserver/scheduler logs
4. Refresh DAGs: Wait for `dag_dir_list_interval` or restart scheduler

### Task failing

1. Check task logs in UI or CLI
2. Verify dependencies are installed
3. Check environment variables
4. Test task function independently

### Scheduler not running tasks

1. Check scheduler is running: `ps aux | grep airflow`
2. Verify DAG is unpaused in UI
3. Check `schedule_interval` and `start_date`
4. Review scheduler logs

### Database issues

```bash
# Reset database
airflow db reset

# Upgrade schema
airflow db upgrade

# Check connections
airflow db check
```

## Advanced Topics

### Custom Operators

```python
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults

class MyCustomOperator(BaseOperator):
    @apply_defaults
    def __init__(self, my_param, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_param = my_param
    
    def execute(self, context):
        # Operator logic
        self.log.info(f"Executing with param: {self.my_param}")
```

### Dynamic DAGs

Generate DAGs programmatically:

```python
for model in ['model_a', 'model_b', 'model_c']:
    dag_id = f'train_{model}_dag'
    
    dag = DAG(
        dag_id,
        default_args=default_args,
        schedule_interval='@daily',
    )
    
    # Add tasks to DAG
    globals()[dag_id] = dag
```

### Integration with MLflow

```python
import mlflow

def train_with_mlflow(**context):
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    with mlflow.start_run():
        # Training logic
        mlflow.log_param("epochs", 10)
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_artifact("model.pkl")
```

## Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Astronomer Guides](https://www.astronomer.io/guides/)
