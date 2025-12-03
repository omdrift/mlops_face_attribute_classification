# Monitoring Infrastructure

This directory contains the monitoring infrastructure configuration for the MLOps face attribute classification system.

## Structure

```
monitoring/
├── prometheus/              # Prometheus configuration
│   ├── prometheus.yml       # Main Prometheus config
│   ├── alert_rules.yml      # Alert definitions
│   └── targets.yml          # Scrape targets
├── grafana/                 # Grafana configuration
│   ├── provisioning/
│   │   ├── dashboards/      # Dashboard provisioning
│   │   └── datasources/     # Datasource config
│   └── dashboards/          # Dashboard JSON files
│       ├── ml_metrics.json
│       ├── api_performance.json
│       └── system_metrics.json
├── alertmanager/            # Alertmanager configuration
│   └── alertmanager.yml
├── docker-compose.monitoring.yml  # Docker Compose for monitoring stack
└── README.md
```

## Quick Start

### Start the Monitoring Stack

```bash
# From the project root
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```

### Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093

### Stop the Stack

```bash
docker-compose -f monitoring/docker-compose.monitoring.yml down
```

## Components

### Prometheus

- Collects metrics from the ML API at `/metrics` endpoint
- Stores time-series data
- Evaluates alert rules
- Web UI for querying metrics

**Configuration**: `prometheus/prometheus.yml`

### Grafana

- Visualizes metrics in dashboards
- Pre-configured dashboards for ML metrics, API performance, and system metrics
- Alerts based on dashboard queries

**Dashboards**:
- ML Model Metrics: Model predictions, confidence, latency, drift
- API Performance: Request rate, latency, error rate
- System Metrics: CPU, memory, disk, network

### Alertmanager

- Manages alerts from Prometheus
- Routes alerts to receivers (email, Slack, PagerDuty)
- Groups and deduplicates alerts

**Configuration**: `alertmanager/alertmanager.yml`

### Optional Components

- **Node Exporter**: System metrics (CPU, memory, disk)
- **cAdvisor**: Container metrics

## Configuration

### Prometheus Targets

Edit `prometheus/prometheus.yml` to add scrape targets:

```yaml
scrape_configs:
  - job_name: 'my-service'
    static_configs:
      - targets: ['my-service:9090']
```

### Alert Rules

Add custom alerts in `prometheus/alert_rules.yml`:

```yaml
groups:
  - name: my_alerts
    rules:
      - alert: MyAlert
        expr: my_metric > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "My alert description"
```

### Grafana Dashboards

1. Create dashboard in Grafana UI
2. Export as JSON
3. Save to `grafana/dashboards/`
4. Restart Grafana to load

## Metrics

The ML API exposes Prometheus metrics at `/metrics`:

### API Metrics
- `api_requests_total`: Total requests by endpoint/status
- `api_request_latency_seconds`: Request latency histogram

### Model Metrics
- `model_predictions_total`: Predictions by attribute/value
- `model_inference_latency_seconds`: Inference latency
- `model_confidence_score`: Confidence distribution
- `model_accuracy`: Current accuracy per attribute
- `drift_score`: Drift score per attribute

### System Metrics
- `images_processed_total`: Total images processed
- `model_loaded`: Model loading status
- `cache_size`: Cache size

## Alerting

### Email Setup

Edit `alertmanager/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@example.com'
  smtp_auth_username: 'your-email@gmail.com'
  smtp_auth_password: 'your-app-password'

receivers:
  - name: 'team'
    email_configs:
      - to: 'team@example.com'
```

### Slack Integration

Add Slack webhook:

```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts'
```

## Maintenance

### Data Retention

Prometheus retains data for 15 days by default. To change:

```yaml
command:
  - '--storage.tsdb.retention.time=30d'
```

### Backup

Backup Prometheus data:

```bash
docker run --rm -v prometheus-data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup.tar.gz /data
```

Restore:

```bash
docker run --rm -v prometheus-data:/data -v $(pwd):/backup alpine tar xzf /backup/prometheus-backup.tar.gz -C /
```

### Clean Up

Remove old data:

```bash
# Remove all monitoring data
docker-compose -f monitoring/docker-compose.monitoring.yml down -v
```

## Troubleshooting

### Prometheus not scraping

1. Check targets: http://localhost:9090/targets
2. Verify API is running and `/metrics` is accessible
3. Check network connectivity between containers

### Grafana no data

1. Verify Prometheus datasource: Configuration → Data Sources
2. Test connection
3. Check if metrics exist in Prometheus

### Alerts not firing

1. Check alert rules: http://localhost:9090/alerts
2. Verify Alertmanager is running: http://localhost:9093
3. Check Alertmanager logs: `docker logs mlops-alertmanager`

## Documentation

See detailed guides in `docs/`:
- [Monitoring Guide](../docs/monitoring.md)
- [Airflow Guide](../docs/airflow.md)
- [CI/CD Guide](../docs/ci-cd.md)

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
