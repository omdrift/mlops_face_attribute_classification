# MLOps Face Attribute Classification

A complete MLOps pipeline for facial attribute classification using deep learning with comprehensive monitoring, orchestration, and CI/CD.

## 🎯 Project Overview

This project implements a multi-output CNN model to classify facial attributes:
- **Beard** (binary: 0/1)
- **Mustache** (binary: 0/1)
- **Glasses** (binary: 0/1)
- **Hair Length** (3 classes: Short/Medium/Long)
- **Hair Color** (5 classes: Black/Blonde/Brown/Gray/Red)

### Model Architecture
- **CustomMultiHeadCNN** - Multi-head architecture with shared backbone
- **Framework**: PyTorch
- **Input**: 64x64 RGB images
- **Optimizer**: AdamW with ReduceLROnPlateau scheduler

## 🏗️ MLOps Infrastructure

### Core Components

1. **DVC (Data Version Control)**
   - Data versioning and pipeline management
   - Reproducible experiments

2. **MLflow**
   - Experiment tracking
   - Model registry
   - Metrics logging

3. **Docker**
   - Containerized deployment
   - Consistent environments

4. **Airflow**
   - Pipeline orchestration
   - Scheduled workflows
   - Conditional retraining

5. **GitHub Actions**
   - CI/CD automation
   - Testing and linting
   - Docker image builds

6. **Evidently AI**
   - Data drift detection
   - Model performance monitoring
   - Automated reporting

7. **Prometheus + Grafana**
   - Real-time metrics
   - Custom dashboards
   - Alerting

## 📁 Project Structure

```
.
├── airflow/                      # Airflow DAGs and configuration
│   ├── dags/
│   │   ├── ml_pipeline_dag.py   # Weekly training pipeline
│   │   ├── monitoring_dag.py    # Daily monitoring
│   │   └── retraining_dag.py    # Conditional retraining
│   ├── docker-compose.airflow.yml
│   └── Dockerfile.airflow
│
├── .github/workflows/            # CI/CD pipelines
│   ├── ci.yml                   # Linting, testing, security
│   ├── cd.yml                   # Docker build and deploy
│   ├── dvc-pipeline.yml         # DVC workflow
│   └── model-training.yml       # Manual training
│
├── src/
│   ├── data/                    # Data processing
│   ├── models/                  # Model architecture
│   ├── training/                # Training scripts
│   ├── monitoring/              # Monitoring components
│   │   ├── evidently_monitoring.py
│   │   ├── drift_detection.py
│   │   ├── report_generator.py
│   │   ├── reference_data.py
│   │   └── prometheus_metrics.py
│   └── inference/               # Inference scripts
│
├── deployment/
│   └── api/
│       ├── main.py              # FastAPI with Prometheus
│       ├── Dockerfile
│       └── requirements-api.txt
│
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml       # Prometheus config
│   │   └── alert_rules.yml      # Alert definitions
│   ├── grafana/
│   │   ├── dashboards/          # 3 custom dashboards
│   │   └── provisioning/
│   ├── alertmanager/
│   │   └── alertmanager.yml     # Alert routing
│   └── docker-compose.monitoring.yml
│
├── tests/
│   ├── test_monitoring.py       # Monitoring tests
│   └── test_airflow_dags.py     # DAG validation tests
│
├── docs/
│   └── monitoring.md            # Comprehensive monitoring guide
│
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Training parameters
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git
- 4GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/omdrift/mlops_face_attribute_classification.git
cd mlops_face_attribute_classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure DVC (optional)**
```bash
dvc remote add -d storage <your-remote-url>
dvc pull
```

### Training

#### Using DVC Pipeline
```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train
```

#### Manual Training
```bash
python src/training/train.py
```

#### Hyperparameter Optimization
```bash
python src/training/hyperopt_search.py --max-evals 20
```

### Deployment

#### Start ML API
```bash
# With Python
cd deployment/api
python main.py

# With Docker
docker build -t mlops-api -f deployment/api/Dockerfile .
docker run -p 8000:8000 mlops-api
```

API will be available at `http://localhost:8000`

**Endpoints:**
- `GET /` - API info
- `POST /predict` - Predict attributes from image
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

#### Example API Usage
```python
import requests

# Upload image for prediction
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

print(response.json())
# {
#   "predictions": {"beard": 1, "mustache": 0, ...},
#   "labels": {"beard": "Yes", "mustache": "No", ...}
# }
```

## 📊 Monitoring & Orchestration

### Start Monitoring Stack

```bash
# Start Prometheus, Grafana, Alertmanager
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

**Access:**
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)
- Alertmanager: `http://localhost:9093`

### Start Airflow

```bash
cd airflow
export AIRFLOW_UID=$(id -u)
docker-compose -f docker-compose.airflow.yml up airflow-init
docker-compose -f docker-compose.airflow.yml up -d
```

**Access:** `http://localhost:8080` (airflow/airflow)

### Airflow DAGs

1. **ml_pipeline** - Runs weekly
   - Data preparation
   - Model training
   - Evaluation
   - Conditional deployment

2. **monitoring_pipeline** - Runs daily
   - Collect metrics
   - Generate drift reports
   - Check alerts
   - Send notifications

3. **retraining_pipeline** - Runs weekly (Monday 2 AM)
   - Check drift/performance
   - Trigger retraining if needed

### Generate Drift Reports

```python
from src.monitoring.evidently_monitoring import EvidentlyMonitor, load_reference_data

# Load reference data
ref_data = load_reference_data()

# Initialize monitor
monitor = EvidentlyMonitor(reference_data=ref_data)

# Generate drift report
monitor.generate_data_drift_report(current_data)

# Check drift threshold
drift_detected, metrics = monitor.check_drift_threshold(current_data, threshold=0.1)
```

Reports are saved to `reports/evidently/`

### Grafana Dashboards

Three pre-configured dashboards:

1. **ML Training Dashboard**
   - Training/validation curves
   - Accuracy by attribute
   - Learning rate schedule

2. **API Performance Dashboard**
   - Requests per second
   - Latency percentiles
   - Error rates
   - Prediction distributions

3. **Drift Monitoring Dashboard**
   - Drift scores by attribute
   - Alert history
   - Model performance trends

## 🔧 CI/CD

### GitHub Actions Workflows

#### CI - Continuous Integration
Runs on push/PR:
- Code linting (flake8, black, isort)
- Unit tests with coverage
- Security scans (bandit, safety)

#### CD - Continuous Deployment
Runs on push to main:
- Build Docker image
- Push to GitHub Container Registry
- Optional deployment

#### DVC Pipeline
Manual trigger:
- Pull data with DVC
- Run DVC pipeline
- Push results

#### Model Training
Manual trigger with parameters:
- Epochs
- Batch size
- Learning rate

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v --cov=src
```

### Run Specific Test Suites
```bash
# Monitoring tests
pytest tests/test_monitoring.py -v

# Airflow DAG tests
pytest tests/test_airflow_dags.py -v
```

### Validate Airflow DAGs
```bash
cd airflow
python -c "from airflow.models import DagBag; db = DagBag('dags/'); print(db.import_errors)"
```

## 📈 Monitoring Metrics

### Key Metrics Tracked

**API Metrics:**
- Request rate and latency
- Error rate
- Prediction distribution

**Model Metrics:**
- Accuracy per attribute
- Inference latency
- Drift scores

**System Metrics:**
- CPU/Memory usage
- Cache hit rate
- Queue depth

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| API Latency (P95) | > 500ms | > 1s |
| Error Rate | > 1% | > 5% |
| Drift Score | > 0.15 | > 0.25 |
| Model Accuracy | < 85% | < 80% |

## 🔐 Security

### Implemented Security Measures

1. **Dependency Scanning** - Safety check in CI
2. **Code Scanning** - Bandit security linting
3. **Container Scanning** - Docker image vulnerability checks
4. **Secret Management** - GitHub secrets for sensitive data
5. **Input Validation** - FastAPI request validation

### Security Best Practices

- Never commit secrets or credentials
- Use environment variables for configuration
- Keep dependencies updated
- Review security scan results
- Implement rate limiting in production

## 📖 Documentation

- [Monitoring Guide](docs/monitoring.md) - Comprehensive monitoring documentation
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)
- [MLflow UI](http://localhost:5000) - Experiment tracking (when running)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes
4. Run tests (`pytest tests/ -v`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Quality Standards

- Follow PEP 8 style guide
- Write tests for new features
- Update documentation
- Pass all CI checks

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **DVC** - Data versioning
- **MLflow** - Experiment tracking
- **Evidently AI** - Drift monitoring
- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **Apache Airflow** - Orchestration
- **FastAPI** - API framework

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check the [monitoring documentation](docs/monitoring.md)
- Review Airflow logs for pipeline issues
- Check Prometheus/Grafana for metrics issues

## 🗺️ Roadmap

- [ ] Add model versioning API
- [ ] Implement A/B testing framework
- [ ] Add feature importance tracking
- [ ] Implement model explainability
- [ ] Add data quality checks
- [ ] Implement canary deployments
- [ ] Add Kubernetes deployment configs

## 📊 Project Status

✅ **Completed:**
- Model training pipeline
- DVC integration
- MLflow tracking
- Airflow orchestration
- GitHub Actions CI/CD
- Evidently AI monitoring
- Prometheus + Grafana
- API deployment
- Comprehensive documentation

🔄 **In Progress:**
- Production deployment
- Advanced alerting rules
- Performance optimization

## Version

**Current Version:** 1.0.0

**Last Updated:** 2024-12-03
