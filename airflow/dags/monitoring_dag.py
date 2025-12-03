"""
Monitoring DAG - Daily execution
Collects metrics, generates drift reports, and sends alerts
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
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
    'retry_delay': timedelta(minutes=2),
}


def collect_inference_metrics(**context):
    """Collect inference metrics from API logs"""
    print("Collecting inference metrics from API...")
    
    # Placeholder for actual metric collection
    # In production, this would query the API logs, database, or metrics store
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_requests': 1000,
        'avg_latency_ms': 45.2,
        'error_rate': 0.01,
        'predictions_by_attribute': {
            'beard': {'0': 600, '1': 400},
            'mustache': {'0': 750, '1': 250},
            'glasses': {'0': 700, '1': 300},
        }
    }
    
    # Save metrics
    os.makedirs('logs/metrics', exist_ok=True)
    metrics_file = f'logs/metrics/daily_metrics_{datetime.now().strftime("%Y%m%d")}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {metrics_file}")
    return metrics


def check_alerts(**context):
    """Check if any alert thresholds are exceeded"""
    # Pull metrics from XCom
    ti = context['ti']
    metrics = ti.xcom_pull(task_ids='collect_inference_metrics')
    
    alerts = []
    
    # Check latency
    if metrics.get('avg_latency_ms', 0) > 100:
        alerts.append({
            'severity': 'warning',
            'type': 'latency',
            'message': f"High latency detected: {metrics['avg_latency_ms']}ms"
        })
    
    # Check error rate
    if metrics.get('error_rate', 0) > 0.05:
        alerts.append({
            'severity': 'critical',
            'type': 'error_rate',
            'message': f"High error rate: {metrics['error_rate']*100:.2f}%"
        })
    
    context['ti'].xcom_push(key='alerts', value=alerts)
    
    if alerts:
        print(f"⚠ {len(alerts)} alert(s) detected")
        for alert in alerts:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")
    else:
        print("✓ No alerts - all metrics within normal range")
    
    return len(alerts) > 0


def send_notifications(**context):
    """Send notifications for alerts"""
    ti = context['ti']
    alerts = ti.xcom_pull(task_ids='check_alerts', key='alerts')
    
    if not alerts:
        print("No notifications to send")
        return
    
    print(f"Sending notifications for {len(alerts)} alert(s)...")
    
    # Placeholder for actual notification logic
    # In production, this would send emails, Slack messages, etc.
    for alert in alerts:
        print(f"  📧 Notification sent: [{alert['severity']}] {alert['message']}")
    
    print("✓ All notifications sent")


with DAG(
    'monitoring_pipeline',
    default_args=default_args,
    description='Daily monitoring - collect metrics, check drift, send alerts',
    schedule_interval='@daily',  # Run every day at midnight
    start_date=days_ago(1),
    catchup=False,
    tags=['monitoring', 'drift', 'alerts'],
) as dag:
    
    # Task 1: Collect inference metrics
    collect_metrics = PythonOperator(
        task_id='collect_inference_metrics',
        python_callable=collect_inference_metrics,
        provide_context=True,
    )
    
    # Task 2: Generate drift report using Evidently
    generate_drift_report = BashOperator(
        task_id='generate_drift_report',
        bash_command='cd {{ params.project_dir }} && python src/monitoring/report_generator.py || echo "Drift report generation skipped"',
        params={'project_dir': os.getenv('PROJECT_DIR', '/opt/airflow/mlops')},
    )
    
    # Task 3: Check alert thresholds
    check_alert_thresholds = PythonOperator(
        task_id='check_alerts',
        python_callable=check_alerts,
        provide_context=True,
    )
    
    # Task 4: Send notifications if needed
    send_alert_notifications = PythonOperator(
        task_id='send_notifications',
        python_callable=send_notifications,
        provide_context=True,
    )
    
    # Define task dependencies
    collect_metrics >> generate_drift_report >> check_alert_thresholds >> send_alert_notifications
