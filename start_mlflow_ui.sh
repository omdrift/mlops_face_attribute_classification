#!/bin/bash
# Script to launch MLflow UI with the correct tracking URI
# This points to the SQLite database created by train.py

echo "🚀 Starting MLflow UI..."
echo "📍 Tracking URI: sqlite:///mlflow.db"
echo "🌐 UI will be available at: http://localhost:5000"
echo ""

# Set the tracking URI to match train.py configuration
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
