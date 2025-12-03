# Monitoring & Observability Guide

## Overview

This document describes the monitoring and observability infrastructure for the MLOps Face Attribute Classification system. The system uses a comprehensive stack including:

- **Evidently AI** - Data drift and model performance monitoring
- **Prometheus** - Metrics collection and alerting
- **Grafana** - Visualization and dashboards
- **Airflow** - Pipeline orchestration and scheduling

## Architecture

```
┌─────────────┐
│   ML Model  │
│   (API)     │
└──────┬──────┘
       │
       ├─────► Prometheus (Metrics Collection)
       │              │
       │              ▼
       │       ┌──────────────┐
       │       │   Grafana    │ (Visualization)
       │       └──────────────┘
       │
       ├─────► Evidently AI (Drift Detection)
       │              │
       │              ▼
       │       ┌──────────────┐
       │       │   Reports    │ (HTML/JSON)
       │       └──────────────┘
       │
       └─────► Airflow (Orchestration)
                      │
                      ▼
               ┌──────────────┐
               │  Alert Mgr   │ (Notifications)
               └──────────────┘
```

## Components

### 1. Evidently AI - Drift Monitoring

Evidently AI monitors data drift and model performance degradation.

#### Key Features
- **Data Drift Detection**: Monitors distribution changes in input features
- **Model Performance**: Tracks accuracy degradation over time
- **Target Drift**: Detects changes in label distributions
- **PSI Calculation**: Population Stability Index for each feature

#### Usage

Generate drift report:
```python
from src.monitoring.evidently_monitoring import EvidentlyMonitor, load_reference_data

# Load reference data
ref_data = load_reference_data()

# Initialize monitor
monitor = EvidentlyMonitor(reference_data=ref_data)

# Generate report
monitor.generate_data_drift_report(current_data)
```

Check drift threshold:
```python
drift_detected, metrics = monitor.check_drift_threshold(current_data, threshold=0.1)
print(f"Drift detected: {drift_detected}")
print(f"Drift share: {metrics['drift_share']:.2%}")
```

#### Drift Thresholds

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant drift | Monitor |
| 0.1 - 0.2 | Moderate drift | Investigate |
| > 0.2 | Significant drift | Retrain model |

### 2. Prometheus - Metrics Collection

Prometheus collects real-time metrics from the ML API and system.

#### Exposed Metrics

**Counters:**
- `api_requests_total{method, endpoint, status}` - Total API requests
- `model_predictions_total{attribute, value}` - Predictions by attribute
- `drift_alerts_total{attribute, severity}` - Drift alerts fired

**Histograms:**
- `api_request_latency_seconds{endpoint}` - Request latency distribution
- `model_inference_latency_seconds` - Model inference time
- `batch_processing_time_seconds` - Batch processing duration

**Gauges:**
- `model_loaded{version}` - Model load status (0/1)
- `images_in_cache_total` - Number of cached images
- `model_accuracy{attribute}` - Current model accuracy
- `drift_score{attribute}` - Current drift score per attribute

#### Configuration

Prometheus configuration: `monitoring/prometheus/prometheus.yml`

```yaml
scrape_configs:
  - job_name: 'mlops-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### 3. Grafana - Dashboards

Grafana provides real-time visualization of metrics.

#### Available Dashboards

1. **ML Training Dashboard** (`ml_training_dashboard.json`)
   - Training/validation loss curves
   - Accuracy by attribute
   - Learning rate schedule
   - Best model metrics

2. **API Performance Dashboard** (`api_performance_dashboard.json`)
   - Requests per second
   - Latency percentiles (P50, P95, P99)
   - Error rate tracking
   - Prediction distribution
   - Cache statistics

3. **Drift Monitoring Dashboard** (`drift_monitoring_dashboard.json`)
   - Drift scores by attribute
   - Drift timeline
   - Alert history
   - Feature distributions
   - Model performance over time

#### Access Grafana

Default credentials:
- URL: `http://localhost:3000`
- Username: `admin`
- Password: `admin`

Change password on first login.

### 4. Airflow - Orchestration

Airflow orchestrates the ML pipeline and monitoring tasks.

#### DAGs

1. **ml_pipeline** (Weekly - @weekly)
   - Check for new data
   - Preprocess data
   - Train model
   - Evaluate model
   - Check drift
   - Deploy if criteria met

2. **monitoring_pipeline** (Daily - @daily)
   - Collect inference metrics
   - Generate drift reports
   - Check alert thresholds
   - Send notifications

3. **retraining_pipeline** (Weekly Monday 2 AM)
   - Check retraining criteria
   - Trigger retraining if needed
   - Run DVC pipeline
   - Train new model
   - Notify on completion

#### Access Airflow

Default credentials:
- URL: `http://localhost:8080`
- Username: `airflow`
- Password: `airflow`

## Setup and Configuration

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- At least 4GB RAM available

### Quick Start

1. **Start Monitoring Stack**

```bash
# Start Prometheus, Grafana, and Alertmanager
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Verify services
docker-compose -f docker-compose.monitoring.yml ps
```

2. **Start Airflow**

```bash
# Initialize Airflow database
cd airflow
export AIRFLOW_UID=$(id -u)
docker-compose -f docker-compose.airflow.yml up airflow-init

# Start Airflow services
docker-compose -f docker-compose.airflow.yml up -d

# Check status
docker-compose -f docker-compose.airflow.yml ps
```

3. **Start ML API (with Prometheus metrics)**

```bash
# From project root
cd deployment/api
python main.py
```

Or with Docker:

```bash
docker build -t mlops-api -f deployment/api/Dockerfile .
docker run -p 8000:8000 mlops-api
```

### Configuration

#### Prometheus Alerts

Edit `monitoring/prometheus/alert_rules.yml` to configure alert thresholds:

```yaml
- alert: HighAPILatency
  expr: histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m])) > 1.0
  for: 5m
```

#### Alertmanager Notifications

Edit `monitoring/alertmanager/alertmanager.yml` to configure notifications:

```yaml
receivers:
  - name: 'critical-alerts'
    slack_configs:
      - channel: '#mlops-critical'
        api_url: 'YOUR_SLACK_WEBHOOK_URL'
    email_configs:
      - to: 'oncall@example.com'
```

## Monitoring Best Practices

### 1. Drift Detection

- **Reference Data**: Update reference data every 2-4 weeks
- **Thresholds**: Start with PSI threshold of 0.15, adjust based on business needs
- **Frequency**: Check drift daily for high-traffic models

### 2. Alert Configuration

- **Critical Alerts**: Model unavailable, high error rate (>5%)
- **Warning Alerts**: High latency, moderate drift (PSI > 0.15)
- **Info Alerts**: Model updates, scheduled maintenance

### 3. Dashboard Usage

- **Training Dashboard**: Review after each training run
- **API Dashboard**: Monitor during deployment and high traffic
- **Drift Dashboard**: Check daily, investigate weekly trends

### 4. Retraining Triggers

Retrain model when:
- Drift score > 0.2 for any critical attribute
- Average accuracy drops below 80%
- Model is more than 30 days old
- Manual trigger via Airflow

## Troubleshooting

### Prometheus Not Scraping Metrics

**Problem**: No data in Grafana

**Solutions**:
1. Check API is exposing `/metrics` endpoint:
   ```bash
   curl http://localhost:8000/metrics
   ```

2. Verify Prometheus targets:
   - Open `http://localhost:9090/targets`
   - Check if `mlops-api` target is UP

3. Check Prometheus logs:
   ```bash
   docker logs mlops-prometheus
   ```

### Airflow DAG Not Running

**Problem**: DAG not executing

**Solutions**:
1. Check DAG is not paused in UI
2. Verify schedule interval is correct
3. Check Airflow logs:
   ```bash
   docker logs mlops-airflow-scheduler
   ```

### Drift Reports Not Generating

**Problem**: No HTML reports in `reports/evidently/`

**Solutions**:
1. Verify reference data exists:
   ```python
   from src.monitoring.reference_data import load_reference_from_training
   ref_data = load_reference_from_training()
   ```

2. Check Evidently installation:
   ```bash
   pip install evidently
   ```

3. Run manual test:
   ```bash
   python src/monitoring/evidently_monitoring.py
   ```

### High Memory Usage

**Problem**: Monitoring stack using too much memory

**Solutions**:
1. Reduce Prometheus retention:
   ```yaml
   # In docker-compose.monitoring.yml
   --storage.tsdb.retention.time=15d  # Instead of 30d
   ```

2. Limit Grafana dashboard refresh rates
3. Use sampling for high-volume metrics

## Maintenance

### Daily
- Review critical alerts
- Check Grafana dashboards for anomalies
- Verify Airflow DAGs completed successfully

### Weekly
- Review drift reports
- Analyze performance trends
- Update alert thresholds if needed

### Monthly
- Update reference data
- Review and archive old reports
- Optimize slow queries/dashboards
- Update documentation

## Additional Resources

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)

## Support

For issues or questions:
1. Check this documentation
2. Review logs in respective service
3. Open an issue in the repository
4. Contact the MLOps team
