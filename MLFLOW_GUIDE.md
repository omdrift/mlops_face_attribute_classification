# MLflow Tracking Guide

This project uses MLflow to track experiments, log models, and manage the model registry.

## 🎯 Quick Start

### Starting MLflow UI

The simplest way to view your MLflow experiments:

```bash
# From project root
./start_mlflow_ui.sh
```

Or manually:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Then open: **http://localhost:5000**

## 📊 What Gets Tracked

When you run `dvc repro train`, the following is automatically logged to MLflow:

### Parameters
- Learning rate, batch size, optimizer settings
- Model architecture (dropout, etc.)
- Training configuration (epochs, patience, etc.)

### Metrics (per epoch)
- Training loss, validation loss
- Accuracy per attribute (beard, mustache, glasses, hair_color, hair_length)
- Average accuracy
- Learning rate changes

### Artifacts
- Model summary (`model_summary.txt`)
- Training curves (`training_curves.png`)
- Accuracy curves (`accuracy_curves.png`)
- **Trained model** with signature (in Model Registry)

### Model Registry
- Model name: `face_attributes_multihead`
- Includes input/output signature
- Tags: `model_type`, `dataset`, `stage`

## 🔧 Using MLflow Utils

### Promote Model to Production

```python
from src.training.mlflow_utils import promote_model

# Promote version 1 to Production
promote_model(version="1", stage="Production")

# Archive old production models automatically
```

### Load Model from Registry

```python
from src.training.mlflow_utils import load_best_model_from_mlflow

# Load production model
model = load_best_model_from_mlflow(stage="Production")

# Or load staging model
model = load_best_model_from_mlflow(stage="Staging")
```

### Get Model Information

```python
from src.training.mlflow_utils import get_model_info

# Get all versions and their stages
info = get_model_info()
print(f"Total versions: {info['total_versions']}")
for version in info['versions']:
    print(f"  Version {version['version']}: {version['stage']}")
```

### Register Model Manually

```python
from src.training.mlflow_utils import register_model

# Register a model from a specific run
model_version = register_model(
    run_id="your_run_id_here",
    model_name="face_attributes_multihead"
)
```

## 🚀 Workflow Example

```bash
# 1. Train model (automatically logs to MLflow)
dvc repro train

# 2. View experiments in UI
./start_mlflow_ui.sh

# 3. Find best run in UI, note the run_id or version

# 4. Promote best model to production
python -c "
from src.training.mlflow_utils import promote_model
promote_model(version='2', stage='Production')
"

# 5. Use in deployment
python -c "
from src.training.mlflow_utils import load_best_model_from_mlflow
model = load_best_model_from_mlflow(stage='Production')
print('Production model loaded!')
"
```

## 🔍 Troubleshooting

### Issue: MLflow UI shows no experiments

**Cause**: The `mlflow.db` file doesn't exist yet or you haven't trained a model.

**Solution**:
```bash
# Run training first
dvc repro train

# Then start UI from project root
./start_mlflow_ui.sh
```

### Issue: "No such file or directory: mlflow.db"

**Cause**: You're running `mlflow ui` from the wrong directory.

**Solution**:
```bash
# Always run from project root where mlflow.db is located
cd /path/to/mlops_face_attribute_classification
./start_mlflow_ui.sh
```

### Issue: Models not appearing in registry

**Cause**: You might be running an older version of the training script.

**Solution**:
```bash
# Ensure you have the latest code
git pull

# Run a new training
dvc repro train

# Check MLflow UI - look for "Models" tab
```

## 📁 Storage Location

- **Tracking Database**: `mlflow.db` (SQLite) in project root
- **Artifacts**: `mlruns/` directory (auto-created)
- **Models**: Stored in `mlruns/<experiment_id>/<run_id>/artifacts/model/`

## 🔐 Model Stages

MLflow supports these stages for model lifecycle:
- **None**: Newly registered models
- **Staging**: Models being tested/validated
- **Production**: Models deployed in production
- **Archived**: Old models no longer in use

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
