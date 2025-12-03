# MLOps Face Attribute Classification

A complete MLOps pipeline for face attribute classification using deep learning with comprehensive monitoring, orchestration, and CI/CD automation.

## 🎯 Overview

This project implements a production-ready ML system for classifying face attributes:
- **Binary attributes**: beard, mustache, glasses
- **Multi-class attributes**: hair_color (5 classes), hair_length (3 classes)

## 🏗️ Architecture

### Model
- **Architecture**: CustomMultiHeadCNN with ResidualBlocks
- **Framework**: PyTorch
- **Input**: 64x64x3 RGB images
- **Training**: MLflow tracking with hyperparameter optimization

### MLOps Components

#### 1. Data Versioning (DVC)
- Data pipeline orchestration
- Reproducible experiments
- Remote storage support (S3, GCS)

#### 2. Workflow Orchestration (Airflow)
- ML pipeline automation
- Data validation workflows
- Automatic model retraining

#### 3. CI/CD (GitHub Actions)
- Automated testing and linting
- Docker image builds
- DVC pipeline execution
- Daily drift monitoring

#### 4. Monitoring (Prometheus + Grafana)
- Real-time metrics collection
- Pre-built dashboards
- Alert management
- Drift detection

#### 5. Drift Detection (Evidently AI)
- Data distribution monitoring
- Model performance tracking
- Automated reporting

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.10+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/omdrift/mlops_face_attribute_classification.git
cd mlops_face_attribute_classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-monitoring.txt
pip install -r airflow/requirements-airflow.txt
```

3. **Start the full MLOps stack**
```bash
docker-compose -f docker-compose.full.yml up -d
```

### Access Services

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Airflow**: http://localhost:8080 (admin/admin)
- **MLflow**: http://localhost:5000
- **Alertmanager**: http://localhost:9093

## 📊 Dashboards

### ML Model Metrics
- Predictions by attribute
- Confidence distribution
- Inference latency (P50/P95/P99)
- Drift score timeline
- Model accuracy per attribute

### API Performance
- Requests per second
- Latency by endpoint
- Error rate
- Status code distribution

### System Metrics
- CPU and memory usage
- Disk I/O
- Network traffic
- Container metrics

## 🔄 Workflows

### Airflow DAGs

1. **ML Pipeline** (Weekly)
   - Check new data
   - Validate with Evidently
   - Preprocess with DVC
   - Train with MLflow
   - Evaluate model
   - Check drift
   - Deploy conditionally

2. **Data Validation** (Daily)
   - Schema validation
   - Quality checks
   - Distribution monitoring
   - Anomaly detection

3. **Model Retraining** (On-demand)
   - Triggered by drift detection
   - Compare with production model
   - Deploy if better

### GitHub Actions

- **CI**: Lint, test, type-check on every push/PR
- **CD**: Build and push Docker images on releases
- **DVC Pipeline**: Execute data pipeline on push to master
- **Model Training**: Manual training with custom parameters
- **Monitoring**: Daily drift checks with auto-issue creation

## 🔍 Monitoring

### Drift Detection

Run drift checks:
```bash
# All checks
python scripts/run_drift_check.py

# Data drift only
python scripts/run_drift_check.py --check data --data-threshold 0.1

# Model drift only
python scripts/run_drift_check.py --check model --model-threshold 0.05

# Evidently report
python scripts/run_drift_check.py --check evidently
```

### Export Metrics

Update Prometheus metrics:
```bash
# Export all metrics
python scripts/export_metrics.py

# Export specific metrics
python scripts/export_metrics.py --export model
python scripts/export_metrics.py --export drift
```

## 📦 Project Structure

```
mlops_face_attribute_classification/
├── .github/
│   ├── workflows/          # CI/CD pipelines
│   │   ├── ci.yml
│   │   ├── cd.yml
│   │   ├── dvc-pipeline.yml
│   │   ├── model-training.yml
│   │   └── monitoring.yml
│   └── dependabot.yml
├── airflow/
│   ├── dags/               # Airflow DAGs
│   │   ├── ml_pipeline_dag.py
│   │   ├── data_validation_dag.py
│   │   └── model_retraining_dag.py
│   ├── config/
│   ├── Dockerfile.airflow
│   └── docker-compose.airflow.yml
├── monitoring/
│   ├── prometheus/         # Prometheus config
│   ├── grafana/            # Grafana dashboards
│   ├── alertmanager/       # Alert management
│   └── docker-compose.monitoring.yml
├── src/
│   ├── data/               # Data processing
│   ├── models/             # Model architecture
│   ├── training/           # Training scripts
│   ├── inference/          # Inference code
│   └── monitoring/         # Monitoring modules
│       ├── evidently_monitoring.py
│       ├── prometheus_metrics.py
│       ├── data_drift.py
│       └── model_drift.py
├── scripts/
│   ├── run_drift_check.py
│   └── export_metrics.py
├── docs/
│   ├── monitoring.md
│   ├── airflow.md
│   └── ci-cd.md
├── docker-compose.full.yml
├── requirements.txt
└── requirements-monitoring.txt
```

## 🛠️ Development

### Training a Model

```bash
# Using DVC
dvc repro

# Direct training
python src/training/train.py

# With hyperparameter search
python src/training/hyperopt_search.py --max-evals 20
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov flake8 black mypy

# Run linting
flake8 src/

# Check formatting
black --check src/

# Run tests
pytest tests/
```

### Adding Custom Metrics

```python
from src.monitoring.prometheus_metrics import (
    record_prediction,
    update_drift_metrics,
    set_model_loaded_status
)

# Record a prediction
record_prediction('beard', 1, confidence=0.95)

# Update drift scores
update_drift_metrics({'beard': 0.05, 'glasses': 0.12})

# Update model status
set_model_loaded_status(True)
```

## 📚 Documentation

Comprehensive guides available in the `docs/` directory:

- [**Monitoring Guide**](docs/monitoring.md): Prometheus, Grafana, Evidently setup
- [**Airflow Guide**](docs/airflow.md): DAG development and orchestration
- [**CI/CD Guide**](docs/ci-cd.md): GitHub Actions workflows

## 🔐 Security

### Secrets Management

Configure secrets in GitHub repository settings:
- `AWS_ACCESS_KEY_ID`: For DVC S3 remote
- `AWS_SECRET_ACCESS_KEY`: For DVC S3 remote
- `GCS_PROJECT_ID`: For DVC GCS remote
- `MLFLOW_TRACKING_URI`: MLflow server URL

### Alerting

Configure email/Slack alerts in `monitoring/alertmanager/alertmanager.yml`:

```yaml
receivers:
  - name: 'team'
    email_configs:
      - to: 'team@example.com'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK'
```

## 🎓 Best Practices

1. **Reproducibility**: All experiments tracked with DVC and MLflow
2. **Automation**: CI/CD pipelines for testing and deployment
3. **Monitoring**: Real-time metrics and drift detection
4. **Documentation**: Comprehensive guides for all components
5. **Security**: Secrets management and RBAC

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with PyTorch, MLflow, DVC, Airflow, Prometheus, Grafana, and Evidently
- Inspired by MLOps best practices and patterns

---

**Note**: This is a production-ready MLOps system with enterprise-grade monitoring, orchestration, and automation. Perfect for learning MLOps or deploying to production.
