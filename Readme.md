# MLOps Face Attribute Classification

A complete MLOps pipeline for face attribute classification using deep learning. This project implements a multi-head CNN model to predict facial attributes including beard, mustache, glasses, hair color, and hair length from face images.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Experiment Tracking](#experiment-tracking)
- [Deployment](#deployment)
- [Orchestration with Airflow](#orchestration-with-airflow)
- [Testing](#testing)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)

## Project Overview

This project provides an end-to-end machine learning pipeline for facial attribute classification with the following capabilities:

**Predicted Attributes:**
- **Beard**: Binary (0: No, 1: Yes)
- **Mustache**: Binary (0: No, 1: Yes)
- **Glasses**: Binary (0: No, 1: Yes)
- **Hair Length**: 3 classes (0: Bald, 1: Short, 2: Long)
- **Hair Color**: 5 classes (0: Blond, 1: Brown, 2: Red, 3: Dark, 4: Gray/White)

**Key Features:**
- Custom multi-head CNN architecture with residual blocks
- DVC-based data versioning and pipeline management
- MLflow integration for experiment tracking
- Hyperparameter optimization using Hyperopt
- FastAPI-based deployment with Docker support
- Apache Airflow orchestration for automated workflows
- Comprehensive CI/CD pipelines with GitHub Actions

**Expected Outputs:**
- Trained model achieving ~99% accuracy on all attributes
- Prediction CSVs for batch inference
- Training metrics and curves (loss, accuracy per attribute)
- Model artifacts stored with version control

## Repository Structure

```
.
├── airflow/                    # Airflow DAGs for ML pipeline orchestration
│   ├── dags/
│   │   └── ml_pipeline_dag.py # Main ML pipeline DAG
│   └── docker-compose.airflow.yml
├── data/                       # Data directory (versioned with DVC)
│   ├── raw/                    # Raw image data organized in lots (s1/, s2/, etc.)
│   ├── annotations/            # CSV files with labels
│   │   └── mapped_train.csv   # Training annotations
│   └── processed/              # Preprocessed data tensors
├── deployment/                 # Deployment artifacts
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # API endpoints
│   │   ├── inference.py       # Inference logic
│   │   └── models.py          # Pydantic models
│   ├── models/                # Model architecture for deployment
│   ├── docker-compose.yaml    # Docker deployment configuration
│   └── Readme.md              # Deployment documentation
├── metrics/                    # Training and evaluation metrics (JSON)
│   ├── train_metrics.json
│   └── eval_metrics.json
├── models/                     # Trained model artifacts
│   └── best_model.pth         # Best model checkpoint
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── fast_training.ipynb    # Quick training experiments
│   ├── inference.ipynb        # Inference examples
│   └── projet_mlops_(1).ipynb # Main project notebook
├── outputs/                    # Inference outputs
│   └── predictions_sX.csv     # Prediction files per lot
├── src/                        # Source code
│   ├── data/
│   │   ├── dataset.py         # PyTorch dataset class
│   │   ├── make_dataset.py    # Data preprocessing pipeline
│   │   └── update_annotations.py
│   ├── inference/
│   │   ├── batch_inference.py # Batch prediction script
│   │   └── predict_lots.py    # Lot-based prediction
│   ├── models/
│   │   └── architecture.py    # CNN model architecture
│   ├── training/
│   │   ├── train.py           # Training script
│   │   ├── evaluate.py        # Evaluation script
│   │   ├── loops.py           # Training/validation loops
│   │   ├── hyperopt_search.py # Hyperparameter optimization
│   │   └── mlflow_utils.py    # MLflow utilities
│   └── utils/                  # Utility functions
├── tests/                      # Unit tests
│   ├── test_model.py          # Model architecture tests
│   └── test_data.py           # Data processing tests
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml                 # Continuous integration
│   ├── dvc.yml                # DVC pipeline automation
│   └── docker.yml             # Docker build and push
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Hyperparameters and configuration
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                   # This file
```

## Setup Instructions

### Prerequisites

- **Python**: 3.8 or higher (Python 3.10 recommended)
- **pip** or **conda** for package management
- **Git** and **Git LFS** for version control
- **DVC** for data versioning (included in requirements)
- **CUDA-capable GPU** (optional but recommended for training)

### Installation Steps

1. **Clone the repository:**
   ```bash
   # Clone from GitHub (replace with your fork if contributing)
   git clone https://github.com/omdrift/mlops_face_attribute_classification.git
   cd mlops_face_attribute_classification
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n face-attr python=3.10
   conda activate face-attr
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install DVC and pull data (if configured):**
   ```bash
   # Initialize DVC (if not already done)
   dvc pull
   ```

   **Note:** If DVC remote is not configured or data cannot be shared, you will need to:
   - Place your raw face images in `data/raw/` organized in subdirectories (e.g., `s1/`, `s2/`, etc.)
   - Create annotations file at `data/annotations/mapped_train.csv` with columns: `filename,beard,mustache,glasses_binary,hair_color_label,hair_length`

## Data Preparation

### Data Format

The project expects:
- **Images**: Face images (any size, will be resized to 64x64) in `data/raw/` subdirectories
- **Annotations**: CSV file at `data/annotations/mapped_train.csv` with the following columns:
  - `filename`: Image filename (e.g., `s1_00000.png`)
  - `beard`: 0 or 1
  - `mustache`: 0 or 1
  - `glasses_binary`: 0 or 1
  - `hair_color_label`: 0-4 (5 classes)
  - `hair_length`: 0-2 (3 classes)

### Preprocessing Data

Run the data preprocessing pipeline to create processed tensors:

```bash
python src/data/make_dataset.py
```

This script will:
- Load raw images from `data/raw/`
- Apply preprocessing (crop, resize to 64x64, normalize)
- Create PyTorch tensors saved to `data/processed/train_data_s1.pt`

**Alternative:** Use DVC to run the full pipeline:
```bash
dvc repro prepare_train
```

## Training the Model

### Quick Start Training

To train the model with default parameters:

```bash
python src/training/train.py
```

### Hyperparameter Optimization

To find optimal hyperparameters using Hyperopt:

```bash
python src/training/hyperopt_search.py --max-evals 20
```

This will search for the best combination of:
- Learning rate
- Batch size
- Dropout rate
- Number of residual blocks

Results are saved to `src/training/hyperopt_params.json`.

### Training with DVC Pipeline

Run the complete training pipeline (preprocessing + hyperopt + training):

```bash
dvc repro train
```

### Training Configuration

Edit `params.yaml` to modify training parameters:

```yaml
train:
  epochs: 10          # Number of training epochs
  batch_size: 32      # Batch size for training

hyperopt:
  max_evals: 20       # Number of hyperopt trials
```
### Training Outputs
After training completes, you will find:
- **Model checkpoint**: `models/best_model.pth`
- **Training metrics**: `metrics/train_metrics.json`
- **Training curves**: `training_curves.png` and `accuracy_curves.png`
- **MLflow logs**: Check `mlflow.db` or run `mlflow ui` to view experiments

Example training metrics:
```json
{
  "best_val_loss": 0.026,
  "best_epoch": 16,
  "best_accuracies": {
    "beard": 0.998,
    "mustache": 0.997,
    "glasses": 0.999,
    "hair_color": 0.998,
    "hair_length": 1.0
  },
  "avg_best_accuracy": 0.999
}
```

## Running Inference

### Batch Inference

To run inference on all images in `data/raw/`:

```bash
python src/inference/batch_inference.py
```

This will:
- Detect all lot subdirectories (e.g., `s1/`, `s2/`)
- Process images in each lot
- Generate prediction CSVs: `outputs/predictions_s1.csv`, `outputs/predictions_s2.csv`, etc.

### Using DVC Pipeline

```bash
dvc repro inference_batches
```

### Single Image Prediction

For interactive inference, use the Jupyter notebook:

```bash
jupyter notebook notebooks/inference.ipynb
```

### Inference Configuration

Modify inference settings in `params.yaml`:

```yaml
inference:
  batch_size: 64                      # Batch size for inference
  output_path: outputs/predictions.csv # Output CSV path
```

### Inference Output Format

Prediction CSVs contain:
- `filename`: Image filename
- `beard`: Predicted value (0 or 1)
- `mustache`: Predicted value (0 or 1)
- `glasses`: Predicted value (0 or 1)
- `hair_length`: Predicted class (0-2)
- `hair_color`: Predicted class (0-4)

Example:
```csv
filename,beard,mustache,glasses,hair_length,hair_color
s2_00000.png,1,0,1,1,3
s2_00001.png,0,0,0,2,0
```

## Experiment Tracking

### MLflow Integration

This project uses **MLflow** for experiment tracking and model versioning.

**Starting MLflow UI:**
```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

**What is tracked:**
- Hyperparameters (learning rate, batch size, dropout, etc.)
- Training metrics (loss per epoch)
- Validation metrics (loss and accuracy per attribute)
- Model artifacts
- Training curves (PNG files)
- System information (Python version, CUDA availability)

**Viewing experiments programmatically:**
```python
import mlflow
# List all experiments
experiments = mlflow.search_experiments()
# Get runs from an experiment
runs = mlflow.search_runs(experiment_ids=["0"])
```

### Metrics Files

Training and evaluation metrics are also saved as JSON files:
- `metrics/train_metrics.json` - Training results
- `metrics/eval_metrics.json` - Evaluation results

## Deployment

The project includes a complete deployment setup using FastAPI and Docker.

### Local API Deployment

1. **Navigate to deployment directory:**
   ```bash
   cd deployment
   ```

2. **Install deployment dependencies:**
   ```bash
   pip install -r requirements-api.txt
   ```

3. **Start the API server:**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API:**
   - Web interface: `http://localhost:8000`
   - API documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   cd deployment
   docker-compose up -d
   ```

2. **Access the deployed service:**
   - API: `http://localhost:8000`

### API Endpoints

- `GET /` - Web interface for searching images by attributes
- `GET /health` - API health check
- `GET /api/attributes` - List available attributes and values
- `POST /api/search` - Search images by facial attributes
- `POST /api/predict` - Predict attributes for an uploaded image
- `GET /api/images/{filename}` - Retrieve an image

For detailed API documentation, see [deployment/Readme.md](deployment/Readme.md).

## Orchestration with Airflow

The project includes Apache Airflow DAGs for automated ML pipeline orchestration.

### Starting Airflow

1. **Navigate to airflow directory:**
   ```bash
   cd airflow
   ```

2. **Start Airflow with Docker Compose:**
   ```bash
   docker-compose -f docker-compose.airflow.yml up -d
   ```

3. **Access Airflow UI:**
   - URL: `http://localhost:8080`
   - Default credentials: `admin` / `admin` (check `docker-compose.airflow.yml` file)
   - ** Security Note**: Change default credentials in production deployments

### Available DAGs

- **ml_pipeline_dag**: Orchestrates the complete ML workflow
  - Data preparation
  - Hyperparameter optimization
  - Model training
  - Model evaluation
  - Batch inference

The DAG automatically triggers the DVC pipeline stages in the correct order.

## Testing

### Running Tests

The project uses **pytest** for testing. Run all tests:

```bash
pytest tests/ -v
```

**Run tests with coverage:**
```bash
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
```

View coverage report: `htmlcov/index.html`

### Test Structure

- `tests/test_model.py` - Tests for model architecture (ResidualBlock, CustomMultiHeadCNN)
- `tests/test_data.py` - Tests for data processing and dataset classes
- `tests/conftest.py` - Shared fixtures for tests

### Linting

Run code quality checks with Flake8:

```bash
# Check for critical errors
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
# Full linting
flake8 src/ --count --max-complexity=10 --max-line-length=120 --statistics
```

### CI/CD Pipelines

The project uses GitHub Actions for automated testing and deployment:

- **CI Pipeline** (`.github/workflows/ci.yml`):
  - Runs linting with Flake8
  - Executes pytest with coverage
  - Uploads coverage to Codecov

- **DVC Pipeline** (`.github/workflows/dvc.yml`):
  - Automates DVC pipeline execution
  - Runs on data or code changes

- **Docker Pipeline** (`.github/workflows/docker.yml`):
  - Builds and pushes Docker images

All pipelines run automatically on push and pull requests to `master` and `development` branches.

## Contribution Guidelines

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style:
   - Follow PEP 8 style guidelines
   - Maximum line length: 120 characters
   - Add docstrings to functions and classes
   - Add type hints where appropriate

3. **Write tests** for new features:
   - Add tests to the appropriate file in `tests/`
   - Ensure all tests pass: `pytest tests/`
   - Maintain or improve code coverage

4. **Run linting**:
   ```bash
   flake8 src/ --count --max-line-length=120
   ```

5. **Commit your changes** with clear messages:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

6. **Push to your fork** and **create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Code Review Process

- All PRs require at least one review
- CI/CD pipelines must pass (linting, tests)
- Code coverage should not decrease significantly
- Update documentation if you add new features

### Development Setup

For active development:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov flake8
# Run tests in watch mode (requires pytest-watch)
pip install pytest-watch
ptw tests/
```

## License

**TODO**: This repository does not currently have a LICENSE file. Please add a license to clarify usage rights.

For open-source projects, consider using:
- MIT License (permissive)
- Apache 2.0 (permissive with patent grant)
- GPL v3 (copyleft)

## Additional Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **DVC Documentation**: https://dvc.org/doc
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **FastAPI Documentation**: https://fastapi.tiangolo.com/

## Troubleshooting

### Common Issues

**Issue: `dvc pull` fails**
- Solution: Ensure DVC remote is configured or manually place data in `data/raw/`

**Issue: CUDA out of memory during training**
- Solution: Reduce `batch_size` in `params.yaml` or use CPU by setting `CUDA_VISIBLE_DEVICES=""`

**Issue: MLflow UI shows no experiments**
- Solution: Ensure training has been run at least once and `mlflow.db` exists

**Issue: Import errors when running scripts**
- Solution: Ensure you're running scripts from the project root directory and PYTHONPATH is set correctly

### Getting Help

For questions or issues:
1. Check existing GitHub Issues in this repository
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

---
