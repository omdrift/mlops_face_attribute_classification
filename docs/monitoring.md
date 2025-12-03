# Monitoring Guide

This guide explains how to set up and use the monitoring infrastructure for the MLOps face attribute classification system.

## Overview

The monitoring stack includes:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert management and notifications
- **Evidently AI**: Data and model drift detection

## Quick Start

### 1. Start the Monitoring Stack

```bash
# Start only monitoring components
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Or start the full MLOps stack
docker-compose -f docker-compose.full.yml up -d
```

### 2. Access the Dashboards

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093

## Prometheus Configuration

### Metrics Endpoints

The ML API exposes metrics at `/metrics` endpoint:

```bash
curl http://localhost:8000/metrics
```

### Available Metrics

#### API Metrics
- `api_requests_total`: Total API requests by endpoint, method, and status
- `api_request_latency_seconds`: Request latency histogram

#### Model Metrics
- `model_predictions_total`: Total predictions by attribute and value
- `model_inference_latency_seconds`: Inference latency histogram
- `model_confidence_score`: Distribution of confidence scores
- `model_accuracy`: Current accuracy per attribute
- `images_processed_total`: Total images processed
- `model_loaded`: Model loading status (0/1)
- `cache_size`: Prediction cache size

#### Drift Metrics
- `drift_score`: Drift score per attribute (0-1)

### Custom Metrics

To add custom metrics to your API:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metric
custom_metric = Counter('custom_metric', 'Description', ['label1', 'label2'])

# Use metric
custom_metric.labels(label1='value1', label2='value2').inc()
```

## Grafana Dashboards

### Pre-configured Dashboards

1. **ML Model Metrics** (`ml_metrics.json`)
   - Predictions by attribute (pie chart)
   - Confidence distribution (heatmap)
   - Inference latency (P50/P95/P99)
   - Drift score timeline
   - Model accuracy per attribute

2. **API Performance** (`api_performance.json`)
   - Requests per second
   - Latency by endpoint
   - Error rate
   - Status code distribution

3. **System Metrics** (`system_metrics.json`)
   - CPU and memory usage
   - Disk I/O
   - Network traffic
   - Container metrics

### Creating Custom Dashboards

1. Go to Grafana UI: http://localhost:3000
2. Click "+" → "Dashboard"
3. Add panels with Prometheus queries
4. Save and export as JSON
5. Place in `monitoring/grafana/dashboards/`

### Example Queries

```promql
# Average inference latency
histogram_quantile(0.95, rate(model_inference_latency_seconds_bucket[5m]))

# Prediction rate per attribute
sum by (attribute) (rate(model_predictions_total[5m]))

# Error rate
sum(rate(api_requests_total{status="error"}[5m])) / sum(rate(api_requests_total[5m]))

# Drift score for specific attribute
drift_score{attribute="beard"}
```

## Alerting

### Alert Rules

Alerts are defined in `monitoring/prometheus/alert_rules.yml`:

- **HighLatency**: P95 latency > 500ms for 5min
- **HighErrorRate**: Error rate > 5% for 5min
- **ModelNotLoaded**: Model not loaded for 1min
- **DriftDetected**: Drift score > 0.2 for 10min
- **LowModelAccuracy**: Accuracy < 0.7 for 15min

### Configuring Alertmanager

Edit `monitoring/alertmanager/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@yourdomain.com'
  smtp_auth_username: 'your-email@gmail.com'
  smtp_auth_password: 'your-app-password'

receivers:
  - name: 'team'
    email_configs:
      - to: 'team@yourdomain.com'
```

### Slack Integration

Add Slack webhook to Alertmanager:

```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts'
        title: 'MLOps Alert: {{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.summary }}'
```

## Evidently AI - Drift Detection

### Running Drift Checks

```bash
# Run all drift checks
python scripts/run_drift_check.py

# Run only data drift
python scripts/run_drift_check.py --check data --data-threshold 0.1

# Run only model drift
python scripts/run_drift_check.py --check model --model-threshold 0.05

# Run Evidently check
python scripts/run_drift_check.py --check evidently
```

### Drift Reports

HTML reports are saved to `src/monitoring/reports/`:

```bash
# View latest report
ls -lt src/monitoring/reports/*.html | head -1
```

### Reference Data

On first run, create reference data:

```python
from src.monitoring.evidently_monitoring import EvidentlyMonitor
import pandas as pd

monitor = EvidentlyMonitor()
reference_data = pd.read_csv('data/annotations/mapped_train.csv')
monitor.set_reference_data(reference_data)
```

### Automated Drift Detection

Drift checks run automatically via:

1. **GitHub Actions**: Daily at 2 AM UTC (`.github/workflows/monitoring.yml`)
2. **Airflow**: In the main ML pipeline DAG

## Exporting Metrics

Update Prometheus metrics manually:

```bash
# Export all metrics
python scripts/export_metrics.py

# Export specific metrics
python scripts/export_metrics.py --export model
python scripts/export_metrics.py --export drift
```

## Troubleshooting

### Prometheus not scraping metrics

1. Check if API is running: `curl http://localhost:8000/health`
2. Check if metrics endpoint is accessible: `curl http://localhost:8000/metrics`
3. Verify Prometheus targets: http://localhost:9090/targets

### Grafana dashboards not showing data

1. Check Prometheus datasource: Grafana → Configuration → Data Sources
2. Verify Prometheus URL: `http://prometheus:9090`
3. Test queries in Prometheus UI first

### Alerts not firing

1. Check alert rules: http://localhost:9090/alerts
2. Verify Alertmanager config: http://localhost:9093
3. Check Alertmanager logs: `docker logs mlops-alertmanager`

### Drift reports not generating

1. Install Evidently: `pip install evidently`
2. Check reference data exists: `ls src/monitoring/reference_data/`
3. Verify current data path: `ls data/annotations/mapped_train.csv`

## Best Practices

### Metric Naming

Follow Prometheus naming conventions:
- Use `snake_case`
- Add units as suffix: `_seconds`, `_bytes`, `_total`
- Use descriptive labels

### Dashboard Organization

- Group related panels together
- Use consistent time ranges
- Add descriptions to panels
- Set appropriate refresh intervals

### Alert Tuning

- Start with conservative thresholds
- Adjust based on baseline performance
- Use appropriate `for` durations to avoid flapping
- Group related alerts

### Drift Monitoring

- Update reference data periodically
- Set thresholds based on business requirements
- Review drift reports regularly
- Automate retraining when drift detected

## Advanced Topics

### Custom Exporters

Create custom Prometheus exporters for external services:

```python
from prometheus_client import start_http_server, Gauge
import time

# Define metrics
external_metric = Gauge('external_service_status', 'Status of external service')

def collect_metrics():
    # Fetch from external service
    status = get_external_status()
    external_metric.set(status)

if __name__ == '__main__':
    start_http_server(8001)
    while True:
        collect_metrics()
        time.sleep(60)
```

### Long-term Storage

Configure Prometheus remote write for long-term storage:

```yaml
remote_write:
  - url: "https://your-remote-storage/api/v1/write"
    basic_auth:
      username: "user"
      password: "password"
```

### High Availability

For production, run multiple instances:
- Prometheus with remote storage
- Grafana with shared database
- Alertmanager in cluster mode

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
