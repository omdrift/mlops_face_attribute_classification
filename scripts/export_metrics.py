#!/usr/bin/env python3
"""
Export Metrics Script

This script exports metrics to Prometheus by updating gauge values.
Can be run periodically to keep metrics up to date.
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.prometheus_metrics import (
    update_drift_metrics,
    update_accuracy_metrics,
    set_model_info,
    set_model_loaded_status
)


def export_model_metrics(metrics_path: str = "metrics/train_metrics.json"):
    """
    Export model metrics from training results
    
    Args:
        metrics_path: Path to metrics JSON file
    """
    metrics_file = Path(metrics_path)
    
    if not metrics_file.exists():
        print(f"⚠️ Metrics file not found: {metrics_file}")
        return
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract accuracy metrics per attribute
        accuracies = {}
        
        # Assuming metrics structure has per-attribute accuracies
        # Adjust based on actual metrics structure
        for key, value in metrics.items():
            if 'accuracy' in key.lower():
                attribute = key.replace('_accuracy', '').replace('accuracy_', '')
                accuracies[attribute] = value
        
        if accuracies:
            update_accuracy_metrics(accuracies)
            print(f"✓ Updated accuracy metrics: {accuracies}")
        else:
            print("⚠️ No accuracy metrics found in file")
            
    except Exception as e:
        print(f"❌ Error exporting model metrics: {e}")


def export_drift_metrics(drift_results_path: str = "src/monitoring/reports/drift_results.json"):
    """
    Export drift metrics
    
    Args:
        drift_results_path: Path to drift results JSON
    """
    drift_file = Path(drift_results_path)
    
    if not drift_file.exists():
        print(f"⚠️ Drift results file not found: {drift_file}")
        return
    
    try:
        with open(drift_file, 'r') as f:
            drift_results = json.load(f)
        
        # Extract drift scores per attribute
        drift_scores = {}
        
        if 'attributes' in drift_results:
            for attr, results in drift_results['attributes'].items():
                # Use p_value as drift score (inverted)
                p_value = results.get('p_value', 1.0)
                drift_score = 1.0 - p_value
                drift_scores[attr] = drift_score
        
        if drift_scores:
            update_drift_metrics(drift_scores)
            print(f"✓ Updated drift metrics: {drift_scores}")
        else:
            print("⚠️ No drift scores found in file")
            
    except Exception as e:
        print(f"❌ Error exporting drift metrics: {e}")


def export_model_info(
    model_name: str = "CustomMultiHeadCNN",
    model_version: str = "1.0",
    architecture: str = "ResNet-based CNN",
    trained_date: str = None
):
    """
    Export model information
    
    Args:
        model_name: Name of the model
        model_version: Version identifier
        architecture: Architecture description
        trained_date: Training date (ISO format)
    """
    from datetime import datetime
    
    if trained_date is None:
        # Try to get from model file modification time
        model_path = Path("models/best_model.pth")
        if model_path.exists():
            trained_date = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        else:
            trained_date = datetime.now().isoformat()
    
    try:
        set_model_info(model_name, model_version, architecture, trained_date)
        print(f"✓ Updated model info: {model_name} v{model_version}")
    except Exception as e:
        print(f"❌ Error exporting model info: {e}")


def check_model_loaded():
    """Check if model is loaded and update status"""
    model_path = Path("models/best_model.pth")
    
    loaded = model_path.exists()
    
    try:
        set_model_loaded_status(loaded)
        status = "loaded" if loaded else "not loaded"
        print(f"✓ Model status: {status}")
    except Exception as e:
        print(f"❌ Error updating model status: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Export metrics to Prometheus"
    )
    parser.add_argument(
        '--metrics-path',
        default='metrics/train_metrics.json',
        help='Path to training metrics file'
    )
    parser.add_argument(
        '--drift-path',
        default='src/monitoring/reports/drift_results.json',
        help='Path to drift results file'
    )
    parser.add_argument(
        '--export',
        choices=['model', 'drift', 'info', 'status', 'all'],
        default='all',
        help='Type of metrics to export (default: all)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("Exporting Metrics to Prometheus")
    print(f"{'='*80}\n")
    
    # Export requested metrics
    if args.export in ['model', 'all']:
        print("Exporting model metrics...")
        export_model_metrics(args.metrics_path)
        print()
    
    if args.export in ['drift', 'all']:
        print("Exporting drift metrics...")
        export_drift_metrics(args.drift_path)
        print()
    
    if args.export in ['info', 'all']:
        print("Exporting model info...")
        export_model_info()
        print()
    
    if args.export in ['status', 'all']:
        print("Checking model status...")
        check_model_loaded()
        print()
    
    print(f"{'='*80}")
    print("✓ Metrics export completed")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
